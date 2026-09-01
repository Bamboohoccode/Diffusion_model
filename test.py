import torch

x = torch.tensor([2.0,0.2], requires_grad=True)

class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x
neu = NN()
dy_dx = torch.autograd.grad(outputs=neu(x), inputs=x)[0]

print("Đạo hàm tại x=2 là:", dy_dx) 
