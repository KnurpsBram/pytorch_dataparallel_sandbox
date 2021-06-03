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

WORLD_SIZE = torch.cuda.device_count()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)

def main(rank, args):

    setup(rank, world_size=WORLD_SIZE)

    grads = []
    params_after_training = []
    for _ in range(args.n_experiments):

        my_net    = shared.MyNet().to(rank)
        optimizer = optim.SGD(my_net.parameters(), lr=args.lr)

        ddp_net   = DDP(my_net, device_ids=[rank])

        dataset     = shared.MyDataset(deterministic=args.deterministic)
        datasampler = DistributedSampler(dataset, num_replicas=WORLD_SIZE, rank=rank)
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

    tensor = torch.cat(grads)

    # the net will have the same weights on all gpu's, so we only need to print one of them
    if rank != 0:
        dist.gather(tensor, dst=0)
    else:
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(WORLD_SIZE)]
        dist.gather(tensor, gathered_tensors, dst=0)

        print(gathered_tensors)

        print("n_grads", len(grads))

        print("my_net.w:        ", torch.mean(torch.cat(params_after_training)))
        print("grad variance:   ", torch.std(torch.cat(gathered_tensors)).squeeze()**2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size",    type=int,                default=16)
    parser.add_argument("--lr",            type=float,              default=1e-4)
    parser.add_argument("--n_epochs",      type=int,                default=1)
    parser.add_argument("--deterministic", type=shared.str_to_bool, default=True)
    parser.add_argument("--n_experiments", type=int,                default=1)

    args = parser.parse_args()

    print("World Size: ", WORLD_SIZE)

    mp.spawn(main, args=(args,), nprocs=WORLD_SIZE, join=True)
