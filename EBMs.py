import torch
import torch.nn as nn
import torch.optim as optim

# ==============================================================================
# BƯỚC 1: ĐỊNH NGHĨA MẠNG NĂNG LƯỢNG (Scalar Energy Network)
# ==============================================================================
class EnergyNet(nn.Module):
    """
    Mạng nơ-ron nhận vào một mẫu dữ liệu x (D chiều)
    và xuất ra đúng 1 giá trị vô hướng: Năng lượng E_theta(x).
    """
    def __init__(self, in_dim=784, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(), # Hàm kích hoạt mượt (smooth) giúp tính gradient theo x tốt hơn
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Đầu ra là 1 số thực duy nhất (Scalar Energy)
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# BƯỚC 2: BỘ ĐỆM MẪU (REPLAY BUFFER)
# ==============================================================================
class ReplayBuffer:
    """
    Lưu trữ các mẫu đã sinh ra từ các vòng lặp trước để tái sử dụng
    làm điểm khởi đầu cho SGLD, giúp chuỗi Markov hội tụ nhanh hơn.
    """
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = []

    def sample(self, batch_size, data_dim, device):
        # 95% lấy từ buffer cũ, 5% sinh ngẫu nhiên từ phân phối đều để khám phá không gian mới
        if len(self.buffer) > batch_size and torch.rand(1).item() > 0.05:
            indices = torch.randint(0, len(self.buffer), (batch_size,))
            samples = torch.stack([self.buffer[i] for i in indices]).to(device)
        else:
            samples = torch.rand((batch_size, data_dim), device=device) * 2.0 - 1.0 # Uniform [-1, 1]
        return samples

    def add(self, samples):
        for s in samples.detach().cpu():
            if len(self.buffer) < self.max_size:
                self.buffer.append(s)
            else:
                # Thay thế ngẫu nhiên khi buffer đầy
                idx = torch.randint(0, self.max_size, (1,)).item()
                self.buffer[idx] = s


# ==============================================================================
# BƯỚC 3: BỘ SINH MẪU LANGEVIN DYNAMICS (SGLD)
# ==============================================================================
def sample_sgld(energy_net, x_init, num_steps=60, step_size=0.01, noise_std=0.005):
    """
    Sinh mẫu bằng cách lăn dần xuống đáy thung lũng năng lượng.
    """
    x = x_init.clone().detach().requires_grad_(True)
    
    for _ in range(num_steps):
        # 1. Tính năng lượng của trạng thái hiện tại
        energy = energy_net(x).sum()
        
        # 2. Tính gradient của năng lượng theo chính bức ảnh x: ∇_x E(x)
        grad_x = torch.autograd.grad(energy, x, create_graph=False)[0]
        
        # 3. Thành phần nhiễu Brownian motion
        noise = torch.randn_like(x) * noise_std
        
        # 4. Cập nhật trạng thái: Đi ngược hướng gradient (giảm năng lượng) + thêm nhiễu
        x.data = x.data - 0.5 * step_size * grad_x + noise
        
        # Giới hạn giá trị nằm trong miền hợp lệ [-1, 1]
        x.data.clamp_(-1.0, 1.0)
        
    return x.detach()


# ==============================================================================
# BƯỚC 4: VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
# ==============================================================================
def train_ebm(energy_net, dataloader, epochs=10, lr=1e-4, device='cuda'):
    optimizer = optim.Adam(energy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(max_size=10000)
    data_dim = 784

    for epoch in range(epochs):
        for batch_idx, (x_real, _) in enumerate(dataloader):
            x_real = x_real.view(x_real.size(0), -1).to(device)
            batch_size = x_real.size(0)

            # --- Pha 1: Lấy mẫu x_fake thông qua SGLD (Negative Phase) ---
            x_init = buffer.sample(batch_size, data_dim, device)
            x_fake = sample_sgld(energy_net, x_init, num_steps=60, step_size=0.01)
            buffer.add(x_fake)

            # --- Pha 2: Tính năng lượng của ảnh thật và ảnh giả ---
            energy_real = energy_net(x_real) # Mong muốn: Cực tiểu hóa
            energy_fake = energy_net(x_fake) # Mong muốn: Cực đại hóa

            # --- Pha 3: Hàm mất mát (Loss) ---
            # Loss = E_real - E_fake + Regularization (ngăn năng lượng bùng nổ quá lớn)
            reg_loss = 0.1 * (energy_real ** 2 + energy_fake ** 2).mean()
            loss = (energy_real.mean() - energy_fake.mean()) + reg_loss

            # --- Pha 4: Tối ưu hóa trọng số theta ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(energy_net.parameters(), max_norm=1.0)
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f} - E_real: {energy_real.mean().item():.2f} - E_fake: {energy_fake.mean().item():.2f}")

