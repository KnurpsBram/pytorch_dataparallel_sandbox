import os

import torch
import torch.nn as nn
import torch.optim as optim

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import shared

# world_size = torch.cuda.device_count()
world_size = 2

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def main(rank):

    setup(rank, world_size=world_size)

    my_net    = shared.MyNet().to(rank)
    optimizer = optim.SGD(my_net.parameters(), lr=shared.lr)

    ddp_net   = DDP(my_net, device_ids=[rank])

    dataset     = shared.MyDataset()
    datasampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader  = DataLoader(dataset, batch_size=shared.batch_size, sampler=datasampler)

    for x, y in dataloader:

        x, y = x.to(rank), y.to(rank)

        y_hat = my_net(x)

        loss  = nn.L1Loss()(y, y_hat)
        loss.backward()

        optimizer.step()

    print("rank:", rank, "my_net.w: ", my_net.w.data)

if __name__ == "__main__":
    mp.spawn(main, args=(), nprocs=world_size, join=True)
