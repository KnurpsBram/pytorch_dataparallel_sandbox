import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.w

class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()

        self.inputs  = [torch.ones(1)  for _ in range(256)]
        self.targets = [torch.zeros(1) for _ in range(256)]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
