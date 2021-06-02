import torch
import torch.nn as nn
import torch.optim as optim

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.w

def main():
    batch_size = 256
    lr         = 1e-2

    my_net    = MyNet()
    optimizer = optim.SGD(my_net.parameters(), lr=lr)
    x         = torch.ones((batch_size, 1), requires_grad=True)
    y         = torch.zeros((batch_size, 1), requires_grad=True)

    y_hat = my_net(x)

    loss  = nn.L1Loss()(y, y_hat)
    loss.backward()

    print("loss:            ", loss)
    print("my_net.w.grad:   ", my_net.w.grad)
    print("my_net.w:        ", my_net.w.data)

    optimizer.step()

    print("my_net.w:        ", my_net.w.data)

if __name__ == "__main__":
    main()
