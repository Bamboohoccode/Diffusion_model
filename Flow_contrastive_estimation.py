from Normalizing_Flow import NormalizingFlow
from NCE import NCE
import torch
import torch.nn as nn
import torch.optim as optim

'''Parameterize the noise distribution with a normalizing flow model p_n,ϕ(x).
Parameterize the discriminator Dθ,Z,ϕ(x) '''


def compute_loss(f_theta,log_p_n,c):
    return f_theta - torch.logaddexp(f_theta,log_p_n + c)

def train(Generator : NormalizingFlow,
          f_theta : NCE,
          data,
          epochs = 10):
    optimizer_G = optim.AdamW(Generator.parameters(),lr = 0.01)
    optimizer_f = optim.AdamW(f_theta.parameters(),lr = 0.01)
    for epoch in range(epochs):
        batch_size = data.shape[0]
        # P1: Update f_theta and c .
        # Maximize: log D(x_real) + log(1 - D(x_fake))
        with torch.no_grad():
            fake_data = Generator.sample(batch_size)
            log_prob_fake = Generator.log_prob(fake_data)
        log_prob_real = Generator.log_prob(data)
        optimizer_f.zero_grad()
        f_real = f_theta(data).squeeze(-1)
        f_fake = f_theta(fake_data).squeeze(-1)
        c = f_theta.get_c
        real_loss_f = f_real - torch.logaddexp(f_real,c + log_prob_real)
        fake_loss_f = c + log_prob_fake - torch.logaddexp(f_fake,c + log_prob_fake)
        loss_f = - (real_loss_f.mean() + fake_loss_f.mean())
        loss_f.backward()
        optimizer_f.step()

        #P2: Update Generator(Normalizing Flow)
        # Maximize: log D(x_fake)
        optimizer_G.zero_grad()
        fake_data = Generator.sample(batch_size)
        log_prob_fake = Generator.log_prob(fake_data)
        f_fake = f_theta(fake_data).squeeze(-1)
        c = f_theta.get_c.detach()
        loss_G = - (f_fake - torch.logaddexp(f_fake,log_prob_fake + c)).mean()
        loss_G.backward()
        optimizer_G.step()
        loss = loss_f.item() + loss_G.item()
        print(f"EPOCHS: {epoch}----LOSS: {round(loss,4)} Generator LOSS: {round(loss_G.item(),4)}---f_theta LOSS: {round(loss_f.item(),4)}")



if __name__ == "__main__":
    input_size = 2
    num_samples = 2000
    cluster1 = torch.randn(num_samples // 2, 2) * 0.5 + torch.tensor([2.0, 2.0])
    cluster2 = torch.randn(num_samples // 2, 2) * 0.5 + torch.tensor([-2.0, -2.0])
    data = torch.cat([cluster1, cluster2], dim=0)
    Generator = NormalizingFlow(dim = input_size)
    f_theta = NCE(hidden_dim=30,input_size = input_size)
    train(Generator,f_theta,data,epochs = 50)
