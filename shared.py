import distutils.util

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset

def str_to_bool(s):
    return bool(distutils.util.strtobool(s))

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.w

class MyDataset(Dataset):
    def __init__(self, deterministic=True):
        super(MyDataset, self).__init__()

        self.deterministic = deterministic

    def __len__(self):
        return 256

    def __getitem__(self, idx):
        if self.deterministic:
            return torch.ones(1), torch.zeros(1)
        else:
            return torch.ones(1) + torch.randn(1), torch.randn(1)
