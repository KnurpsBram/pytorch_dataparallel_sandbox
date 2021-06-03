import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import shared

def main(args, rank=0):

    grads = []
    params_after_training = []
    for _ in range(args.n_experiments):

        my_net    = shared.MyNet().to(rank)
        optimizer = optim.SGD(my_net.parameters(), lr=args.lr)

        dataset    = shared.MyDataset(deterministic=args.deterministic)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)

        for _ in range(args.n_epochs):
            for x, y in dataloader:

                optimizer.zero_grad()

                x, y = x.to(rank), y.to(rank)

                y_hat = my_net(x)

                loss  = nn.L1Loss()(y, y_hat)
                loss.backward()

                grads.append(my_net.w.grad.clone())

                optimizer.step()

        params_after_training.append(my_net.w.data.clone())

    print("my_net.w:        ", torch.mean(torch.cat(params_after_training)))
    print("grad variance:   ", torch.std(torch.cat(grads)).squeeze()**2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size",    type=int,                default=16)
    parser.add_argument("--lr",            type=float,              default=1e-4)
    parser.add_argument("--n_epochs",      type=int,                default=1)
    parser.add_argument("--deterministic", type=shared.str_to_bool, default=True)
    parser.add_argument("--n_experiments", type=int,                default=1)

    args = parser.parse_args()


    main(args)
