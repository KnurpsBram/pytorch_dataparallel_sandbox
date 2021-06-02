import torch
import torch.nn as nn
import torch.optim as optim

import torch.multiprocessing as mp

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.w

def main(rank):
    batch_size = 16
    lr         = 1e-2

    my_net    = MyNet().to(rank)
    optimizer = optim.SGD(my_net.parameters(), lr=lr)
    x         = torch.ones((batch_size, 1), requires_grad=True).to(rank)
    y         = torch.zeros((batch_size, 1), requires_grad=True).to(rank)

    y_hat = my_net(x)

    loss  = nn.L1Loss()(y, y_hat)
    loss.backward()

    print("rank:", rank, " loss:            ", loss)
    print("rank:", rank, " my_net.w.grad:   ", my_net.w.grad)
    print("rank:", rank, " my_net.w:        ", my_net.w.data)

    optimizer.step()

    print("rank:", rank, " my_net.w:        ", my_net.w.data)


if __name__ == "__main__":
    mp.spawn(main, args=(), nprocs=torch.cuda.device_count(), join=True)
