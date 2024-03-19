import torch
import torch.nn as nn

__all__ = ['CrossCosineEmbeddingLoss']


class CrossCosineEmbeddingLoss(nn.Module):
    NAME = 'Cosine'

    def __init__(self, margin=0.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin, reduction='none')

    def forward(self, x, y):
        n = len(x)
        losses = []
        for i in range(n):
            x_repeat = x[i:i + 1, :].repeat([n, 1])
            target = torch.ones(n).to(x.device) * (-1.0)
            target[i] = 1.0
            loss_i = self.loss_fn(x_repeat, y, target)
            losses.append(loss_i)
            del target
        losses = torch.cat(losses)
        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = torch.sum(losses)
        elif self.reduction == 'mean':
            loss = torch.sum(losses) / losses.numel()
        else:
            raise ValueError(f'Invalid reduction: {self.reduction}')
        result = {
            'loss': loss
        }
        return result

    def __str__(self):
        out = (f'{self.NAME} Loss: '
               f'margin: {self.margin} | reduction: {self.reduction}')
        return out
