import math
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
        self.dynamic_divisor = None

    def forward(self, **kwargs):
        """
        Keyword Args:
            pred (torch.Tensor): predicted logits
            gt (torch.Tensor): binary gt targets
        Returns:
            result (dict): computed loss in key `loss`
        """

        pred: torch.Tensor = kwargs.get('pred', None)
        gt: torch.Tensor = kwargs.get('gt', None)

        if pred is None and gt is None:
            return {'loss': 0}

        if gt.device != pred.device:
            gt = gt.to(pred.device)

        if self.reduction in ('none', 'sum', 'mean'):
            loss = self.loss_fn(pred, gt, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        elif self.reduction == 'dynamic':
            losses = self.loss_fn(pred, gt, alpha=self.alpha, gamma=self.gamma, reduction='none')
            if self.dynamic_divisor is None:
                loss_sum = torch.sum(losses)
                loss_divisor = 10 ** (math.floor(math.log10(loss_sum.item())))
                self.dynamic_divisor = loss_divisor
                print(f'{self.__class__.__name__}: Initial batch loss {loss_sum}. '
                      f'Setting dynamic divisor to: {loss_divisor}')
            loss = torch.sum(losses) / self.dynamic_divisor
        else:
            raise ValueError(f'Invalid reduction: {self.reduction}')

        result = {
            'loss': loss
        }
        return result

    def __str__(self):
        out = (f'{self.NAME} Loss: '
               f'alpha: {self.alpha} | gamma: {self.gamma} | reduction: {self.reduction} | ')
        return out
