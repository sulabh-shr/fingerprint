import torch.nn as nn

__all__ = ['Identity']


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
