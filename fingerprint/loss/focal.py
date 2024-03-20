import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

__all__ = ['FocalLoss']


class FocalLoss(nn.Module):
    NAME = 'Focal'

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_fn = sigmoid_focal_loss

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if y.device != x.device:
            y = y.to(x.device)
        loss = self.loss_fn(x, y, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        result = {
            'loss': loss
        }
        return result

    def __str__(self):
        out = (f'{self.NAME} Loss: '
               f'margin: {self.margin} | reduction: {self.reduction} | '
               f'pos_wt: {self.pos_wt} | neg_wt: {self.neg_wt}')
        return out
