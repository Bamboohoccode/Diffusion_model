import torch
import torch.nn as nn
import math


class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim, mask):
        super().__init__()
        self.dim = dim
        self.register_buffer('mask', mask)
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, dim * 2) 
        )

    def forward(self, x):
        x1 = x * self.mask
        out = self.net(x1)
        s, t = out.chunk(2, dim=-1)
        
        s = torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        
        z = x1 + (x * (1.0 - self.mask)) * torch.exp(s) + t
        
        log_det = torch.sum(s, dim=-1)
        return z, log_det

    def inverse(self, z):
        z1 = z * self.mask
        out = self.net(z1)
        s, t = out.chunk(2, dim=-1) # .chunk help separate tensor into n pieces.
        
        s = torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)

        x = z1 + ((z * (1.0 - self.mask)) - t) * torch.exp(-s)
        return x



class NormalizingFlow(nn.Module):
    def __init__(self, dim=2, hidden_dim=64, num_layers=6):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            mask = torch.zeros(dim)
            mask[i % 2::2] = 1.0 
            self.layers.append(CouplingLayer(dim, hidden_dim, mask))
            
    def forward(self, x):
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_sum += log_det
        return z, log_det_sum

    def sample(self, num_samples):
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.dim, device=device)
        
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x
    def log_prob(self, x):
        z, log_det = self.forward(x)
        
        log_prior = -0.5 * (
            self.dim * math.log(2.0 * math.pi) + 
            torch.sum(z ** 2, dim=-1))

        log_px = log_prior + log_det
        
        return log_px



def compute_loss(model, x):
    """
    Loss = - E [ log p_X(x) ]
         = - E [ log p_Z(z) + log|det J| ]
    """
    dim = x.shape[1]
    z, log_det = model(x)
    
    log_prior = -0.5 * (dim * math.log(2.0 * math.pi) + torch.sum(z ** 2, dim=-1))
    
    log_likelihood = log_prior + log_det
    
    loss = -torch.mean(log_likelihood)
    return loss



if __name__ == "__main__":
    torch.manual_seed(42)
    
    num_samples = 2000
    cluster1 = torch.randn(num_samples // 2, 2) * 0.5 + torch.tensor([2.0, 2.0])
    cluster2 = torch.randn(num_samples // 2, 2) * 0.5 + torch.tensor([-2.0, -2.0])
    data = torch.cat([cluster1, cluster2], dim=0)
    
    model = NormalizingFlow(dim=2, hidden_dim=64, num_layers=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("--- Bắt đầu huấn luyện ---")
    for epoch in range(1, 1001):
        optimizer.zero_grad()
        loss = compute_loss(model, data)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | NLL Loss: {loss.item():.4f}")
            
    print("\n--- Sinh mẫu mới từ mô hình đã học ---")
    model.eval()
    with torch.no_grad():
        generated_x = model.sample(num_samples=5)
        print("Mẫu sinh ra (generated x):\n", generated_x)