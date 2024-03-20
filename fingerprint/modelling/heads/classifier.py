import torch.nn as nn
import torch.nn.functional as f


class BinaryClassifier(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        layers = []
        n = len(channels)
        for i in range(n - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1]))
            layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers[:-2])

    def forward(self, x):
        o = self.layers(x)
        return o
