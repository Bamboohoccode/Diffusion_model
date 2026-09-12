import torch 
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_real_data(batch_size : int = 1024,
                  sigma : int = 0.5):
    points = torch.randint(0,2,(batch_size,1)).float()
    data = points * 8 - 4
    # Adding noise
    data += torch.randn_like(data) * sigma
    return data

def adding_noise(data,sigma):
    return data + torch.randn_like(data) * sigma

class DSM(nn.Module):
    def __init__(self,input_dim : int = 1,
                  hidden_dim : int = 64,
                  output_dim : int = 1): # input_dim == output_dim !!!
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),            
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim),
        )
    def forward(self,X):
        return self.net(X)
    @torch.no_grad()
    def sample(self,batch_size,step_size = 0.5,num_steps = 2000):
        self.eval()
        '''Generate samples using Langevin Dynamics:
        x_{t+1} = x_t + step_size * score(x_t) + sqrt(2 * step_size) * z_t'''
        x = torch.randn((batch_size,self.input_dim))
        inital_x = x.clone()
        for _ in range(num_steps):
            z = torch.randn_like(x)
            x += step_size * self.forward(x) + torch.sqrt(torch.tensor(2 * step_size)) * z
        return inital_x.detach().numpy().flatten(),x.detach().numpy().flatten()


def compute_loss(x,output_model,x_noise,sigma):
    loss = output_model + (x_noise - x) / sigma ** 2
    return 0.5 * torch.pow(loss,2).mean()

def train(model,lr,epochs,sigma_data,sigma_noise,batch_size):
    optimizer = optim.AdamW(model.parameters(),lr = lr)

    for epoch in range(epochs):
        x = get_real_data(batch_size,sigma_data)
        x_noise = adding_noise(x,sigma_noise)
        output_model = model(x_noise)
        loss = compute_loss(x,output_model,x_noise,sigma_noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch} --- loss: {loss.item()}")
    

if __name__ == "__main__":
    batch_size = 500
    # Sigma for defining data distribution !
    sigma_data = 1
    data = get_real_data(batch_size = batch_size,
                         sigma= sigma_data)
    x = np.arange(0,batch_size)
    plt.figure(figsize=(8,4))
    sns.kdeplot(data)
    plt.savefig("output_DSM/plot.png")


    model = DSM()
    lr = 0.01
    epochs = 50
    # sigma_noise too large help the model searching more further but too small help model guide the output closer than original distribution.
    sigma_noise = 1
    train(model,lr,epochs,sigma_data,sigma_noise,batch_size)

    step_size = 0.1
    num_steps = 2000
    inital_x,x = model.sample(batch_size,step_size=step_size,num_steps = num_steps)

    plt.figure(figsize=(12, 6))
    sns.kdeplot(data,color = 'blue')
    sns.kdeplot(inital_x,color = 'gray')
    sns.kdeplot(x,color = 'red')
    plt.legend()
    plt.savefig("output_DSM/compared_plot.png")


    