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
        loss = self.loss_fn(pred, gt, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        result = {
            'loss': loss
        }
        return result

    def __str__(self):
        out = (f'{self.NAME} Loss: '
               f'alpha: {self.alpha} | gamma: {self.gamma} | reduction: {self.reduction} | ')
        return out
