import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import numpy as np
import argparse, os
import math
class EBMs(nn.Module):
    def __init__(self,input_dims = 784,hidden_dims = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten()
            nn.Linear(in_features=input_dims,
                    out_features=hidden_dims),
            nn.SiLU(),
            nn.Linear(in_features=hidden_dims,
                    out_features=hidden_dims),
            nn.SiLU(),
            nn.Linear(in_features=hidden_dims,
                    out_features=hidden_dims),
            nn.SiLU(),
            nn.Linear(in_features=hidden_dims,
                    out_features=1)
        )
    def forward(self,x):
        return self.net(x)

class ReplayBuffer():
    def __init__(self,input_size = 784,max_size = 512):
        self.size = math.isqrt(input_size)
        self.buffer = []
        self.max_size = 512
    def sample(self,batch_size,device):
        if(len(self.buffer) > batch_size and torch.rand(1).item() > 0.05):
            indices = torch.randint(0, len(self.buffer), (batch_size,))
            return torch.stack([self.buffer[i] for i in indices])
        else:
            return torch.rand((batch_size, 1, self.size, self.size), device=device) * 2.0 - 1.0
    def add(self,x):
        x_cpu = x.detach().cpu()
        for img in x_cpu:
            if len(self.buffer) < self.max_size:
                self.buffer.append(img)
            else:
                idx = torch.randint(0, self.max_size, (1,)).item()
                self.buffer[idx] = img

def sample_sgld(model,x_init, num_steps=60,alpha = 0.5):
    model.eval()
    x = x_init.clone().detach().requires_grad_(True)

    for step in range(num_steps):

        grad = torch.autograd.grad(model(x).sum(),x,create_graph=False)[0]
        noise = torch.randn_like(x)
        # using x.data for not tracking grad every different x.
        x.data = x.data - alpha * grad + torch.sqrt(2 * alpha) * noise
        x.data.clamp_(-1.0, 1.0)
    model.train()
    return x.detach()
def get_dataloader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

def train(model,cfg):
    model.train()
    model.to(cfg['device'])
    Buffer = ReplayBuffer()
    optimizer = optim.AdamW(model.parameters(),lr = cfg['lr'])
    dataloader = get_dataloader()
    os.makedirs("./samples", exist_ok=True)
    for epoch in range(cfg['epochs']):
        for step,(x_real,_) in enumerate(dataloader):
            x_real = x_real.to(cfg['device'])
            batch_size = x_real.size(0)

            x_init = Buffer.sample(batch_size,cfg['device'])
            x_fake = sample_sgld(model,x_init)
            Buffer.add(x_fake)

            E_real = model(x_real)
            E_fake = model(x_fake)

            normalized_loss = 0.1 * (E_real ** 2 + E_fake ** 2).mean()
            loss = E_real.mean() - E_fake.mean() + normalized_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            

        
        utils.save_image(
        x_fake[:64] * 0.5 + 0.5, 
        f"./samples/epoch_{epoch+1}.png", 
        nrow=8)
        print(f"--> Đã lưu ảnh sinh ra vào ./samples/epoch_{epoch+1}.png")
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 20
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        default=DEFAULT_LR,
                        type = float)
    parser.add_argument("--epochs",
                        default=DEFAULT_EPOCHS,
                        type=int)
    parser.add_argument("--device",default=DEVICE)
    args = parser.parse_args()
    cfg = vars(args).copy()
    cfg["device"] = torch.device(cfg["device"])
    model = EBMs()
    train(model,cfg)