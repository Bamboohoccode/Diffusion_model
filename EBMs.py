import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# -------------------------------------------------------------
# 1. DATA LOADER
# -------------------------------------------------------------
def get_dataloader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Chuyển về [-1, 1]
    ])
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# -------------------------------------------------------------
# 2. ENERGY NETWORK (CNN cho ảnh 28x28)
# -------------------------------------------------------------
class EnergyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Dùng Swish (SiLU) thay vì ReLU để mặt phẳng năng lượng mượt, dễ lấy đạo hàm
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 7x7
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# 4x4
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.SiLU(),
            nn.Linear(128, 1) # Xuất ra 1 giá trị Năng Lượng vô hướng
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------
# 3. REPLAY BUFFER
# -------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, max_size=8192):
        self.max_size = max_size
        self.buffer = []

    def sample(self, batch_size, device):
        if len(self.buffer) > batch_size and torch.rand(1).item() > 0.05:
            # 95% lấy mẫu cũ trong buffer
            indices = torch.randint(0, len(self.buffer), (batch_size,))
            return torch.stack([self.buffer[i] for i in indices]).to(device)
        else:
            # 5% khởi tạo từ nhiễu ngẫu nhiên Uniform [-1, 1]
            return torch.rand((batch_size, 1, 28, 28), device=device) * 2.0 - 1.0

    def add(self, samples):
        for s in samples.detach().cpu():
            if len(self.buffer) < self.max_size:
                self.buffer.append(s)
            else:
                self.buffer[torch.randint(0, self.max_size, (1,)).item()] = s

# -------------------------------------------------------------
# 4. SGLD / LANGEVIN DYNAMICS SAMPLING
# -------------------------------------------------------------
def sample_sgld(energy_net, x_init, num_steps=60, step_size=10.0, noise_std=0.005):
    energy_net.eval()
    x = x_init.clone().detach().requires_grad_(True)

    for _ in range(num_steps):
        # Tính năng lượng
        energy = energy_net(x).sum()
        # Tính đạo hàm ∇_x E(x)
        grad_x = torch.autograd.grad(energy, x, create_graph=False)[0]
        # Nhiễu Brownian motion
        noise = torch.randn_like(x) * noise_std
        # Bước lăn dốc
        x.data = x.data - 0.5 * step_size * grad_x + noise
        x.data.clamp_(-1.0, 1.0)

    energy_net.train()
    return x.detach()

# -------------------------------------------------------------
# 5. MAIN TRAINING SCRIPT
# -------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Đang chạy trên thiết bị: {device}")

    os.makedirs("./samples", exist_ok=True)
    dataloader = get_dataloader(batch_size=128)
    model = EnergyCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))
    buffer = ReplayBuffer(max_size=8192)

    epochs = 20
    for epoch in range(epochs):
        for step, (x_real, _) in enumerate(dataloader):
            x_real = x_real.to(device)
            batch_size = x_real.size(0)

            # 1. Sinh ảnh giả (Negative Phase) bằng SGLD
            x_init = buffer.sample(batch_size, device)
            x_fake = sample_sgld(model, x_init, num_steps=60, step_size=10.0)
            buffer.add(x_fake)

            # 2. Tính năng lượng
            e_real = model(x_real)
            e_fake = model(x_fake)

            # 3. Hàm Loss: Ép E(real) xuống, đẩy E(fake) lên + Regularization
            loss_reg = 0.1 * (e_real ** 2 + e_fake ** 2).mean()
            loss = (e_real.mean() - e_fake.mean()) + loss_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} | E_real: {e_real.mean().item():.2f} | E_fake: {e_fake.mean().item():.2f}")

        # Lưu ảnh sinh ra sau mỗi Epoch để kiểm tra kết quả
        utils.save_image(
            x_fake[:64] * 0.5 + 0.5, 
            f"./samples/epoch_{epoch+1}.png", 
            nrow=8
        )
        print(f"--> Đã lưu ảnh sinh ra vào ./samples/epoch_{epoch+1}.png")