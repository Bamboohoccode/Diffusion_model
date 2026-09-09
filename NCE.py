import torch 
import torch.nn as nn


def sample_real_data(n):
    # Dữ liệu thật 2 cụm: 50% ở -2 và 50% ở +2
    mask = torch.rand(n) > 0.5
    means = torch.where(mask, torch.tensor(-2.0), torch.tensor(2.0))
    return torch.normal(means, 0.7).unsqueeze(-1)

fake_data = torch.distributions.Normal(0,3) #Fake data dung la N(0,3^2)

class NCE(nn.Module):
    def __init__(self,hidden_dim,input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
        self.c = nn.Parameter(torch.tensor([0.0]))
    def forward(self,X): # Calculate log_prob
        return self.net(X)
    @property
    def get_c(self): # get -logZ
        return self.c

v = 1 # Number of fake samples with every real sample
batch_size = 5 # Number of real samples each epoch
def train(model : NCE,
          epochs : int = 5,
          lr : float = 0.01):

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr = lr)
    for i,epochs in enumerate(range(epochs)):

        #Step 1: Get sample !
        real_sample = sample_real_data(batch_size)
        fake_sample = fake_data.sample((batch_size * v,1))
        # calculate f_theta and -log(Z)
        f_real = model(real_sample)
        f_fake = model(fake_sample)
        c = model.get_c
        # Calculate Loss
        real_loss = f_real - torch.logaddexp(f_real,c + fake_data.log_prob(real_sample))
        # Cautious with fake_loss !!!
        fake_loss = c + fake_data.log_prob(fake_sample) - torch.logaddexp(f_fake,c + fake_data.log_prob(fake_sample))
        # Maximize the Noise Contrastive Estimation.
        loss = -(real_loss.mean() + fake_loss.mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"EPOCH: {i} -------- LOSS: {loss.item()}")

if __name__ == "__main__":
    model = NCE(100,1)
    train(model,epochs = 15,lr = 0.01)





