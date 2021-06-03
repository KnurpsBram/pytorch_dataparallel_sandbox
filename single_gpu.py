import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import shared

def main(args):
    my_net    = shared.MyNet()
    optimizer = optim.SGD(my_net.parameters(), lr=args.lr)

    dataset    = shared.MyDataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    for x, y in dataloader:

        y_hat = my_net(x)

        loss  = nn.L1Loss()(y, y_hat)
        loss.backward()

        optimizer.step()

    print("my_net.w: ", my_net.w.data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    main(args)
