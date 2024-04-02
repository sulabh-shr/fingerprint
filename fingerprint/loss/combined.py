import torch
import torch.nn as nn

__all__ = ['CombinedLoss']


class CombinedLoss(nn.Module):
    NAME = 'combined'

    def __init__(self, losses, weights):
        super().__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, **kwargs):
        result = {}
        loss = None
        for wt, loss_fn in zip(self.weights, self.losses):
            loss_i = loss_fn(**kwargs)['loss']
            if loss is None:
                loss = wt * loss_i
            else:
                loss += wt * loss_i
            result[f'loss-{loss_fn.NAME}'] = loss_i
        result['loss'] = loss
        return result

    def __str__(self):
        out = f'Loss {self.NAME}:\n '
        for idx in range(len(self.losses)):
            weight = self.weights[idx]
            loss = self.losses[idx]
            out += f'Weight: {weight} | ' + str(loss) + '\n '
        return out
