import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import shared

def main(args, rank=0):
    my_net    = shared.MyNet().to(rank)
    optimizer = optim.SGD(my_net.parameters(), lr=args.lr)

    dataset    = shared.MyDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    for _ in range(args.n_epochs):
        for x, y in dataloader:

            x, y = x.to(rank), y.to(rank)
            
            y_hat = my_net(x)

            loss  = nn.L1Loss()(y, y_hat)
            loss.backward()

            optimizer.step()

    print("my_net.w: ", my_net.w.data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--n_epochs",   type=int,   default=1)

    args = parser.parse_args()


    main(args)
