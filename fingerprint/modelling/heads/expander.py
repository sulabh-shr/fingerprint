import torch.nn as nn
import torch.nn.functional as f


class Expander(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.fc3 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        o = self.fc1(x)
        o = self.bn1(o)
        o = f.relu(o)
        o = self.fc2(o)
        o = self.bn2(o)
        o = f.relu(o)
        o = self.fc3(o)
        return o
