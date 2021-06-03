import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import shared

world_size = torch.cuda.device_count()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def main(rank, args):

    setup(rank, world_size=world_size)

    grads = []
    params_after_training = []
    for _ in range(args.n_experiments):

        my_net    = shared.MyNet().to(rank)
        optimizer = optim.SGD(my_net.parameters(), lr=args.lr)

        ddp_net   = DDP(my_net, device_ids=[rank])

        dataset     = shared.MyDataset(deterministic=args.deterministic)
        datasampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader  = DataLoader(dataset, batch_size=args.batch_size, sampler=datasampler)

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

    dist.barrier()

    print(rank)
    print(grads)
    if rank != 0:
        tensor = torch.cat(grads)
        dist.send(tensor = tensor, dst=0)

    if rank == 0:
        for i in range(1, world_size):
            tensor = torch.zeros(1)
            dist.recv(tensor=tensor, src=i)
            print(rank)
            print(tensor)

    if rank == 0: # the net will have the same weights on all gpu's, so we only need to print one of them


        print("n_grads", len(grads))

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

    print("World Size: ", world_size)

    mp.spawn(main, args=(args,), nprocs=world_size, join=True)
