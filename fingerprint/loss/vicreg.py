import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

__all__ = ['VICRegLoss']


def sample_unique_labels(labels: torch.Tensor):
    sorted_inds = torch.argsort(labels)
    unique_labels, counts = torch.unique(labels, return_counts=True)
    sorted_inds = sorted_inds.cpu().tolist()
    higher_counts = counts > 1
    # Sample 1 embedding per label
    if torch.any(higher_counts):
        unique_indices = []
        current_idx = 0
        for c in counts:
            random_index = np.random.choice(sorted_inds[current_idx:current_idx + c])
            unique_indices.append(random_index)
            current_idx = current_idx + c
    else:
        unique_indices = list(range(len(labels)))
    return unique_indices


class VICRegLoss(nn.Module):
    NAME = 'VICReg'

    def __init__(self, sim_coeff, std_coeff, cov_coeff):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y, label=None, eps=0.0001, **kwargs):
        """

        Args:
            x: tensor of shape (B, D)
            y: tensor of shape (B, D)
            label: label for x and y that denotes class/identity
            eps: small number added to variance
            **kwargs:

        Returns:

        """
        if dist.is_initialized():
            x = torch.cat(FullGatherLayer.apply(x), dim=0)
            y = torch.cat(FullGatherLayer.apply(y), dim=0)

        batch_size = x.shape[0]
        num_features = x.shape[-1]

        # Invariance Loss
        repr_loss = F.mse_loss(x, y)

        # Variance Loss
        unique_x = x
        unique_y = y
        if label is not None:
            unique_indices = sample_unique_labels(label)
            unique_x = x[unique_indices]
            unique_y = y[unique_indices]
        unique_x = unique_x - unique_x.mean(dim=0)
        unique_y = unique_y - unique_y.mean(dim=0)
        unique_std_x = torch.sqrt(unique_x.var(dim=0) + eps)
        unique_std_y = torch.sqrt(unique_y.var(dim=0) + eps)
        std_loss = torch.mean(F.relu(1 - unique_std_x)) / 2 + torch.mean(F.relu(1 - unique_std_y)) / 2

        # Covariance Loss
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = (
                off_diagonal(cov_x).pow_(2).sum().div(num_features)
                + off_diagonal(cov_y).pow_(2).sum().div(num_features)
        )

        # Total Loss
        loss = (
                self.sim_coeff * repr_loss
                + self.std_coeff * std_loss
                + self.cov_coeff * cov_loss
        )

        result = {
            'loss': loss
        }

        return result

    def __str__(self):
        out = (f'{self.NAME} Loss: '
               f'sim: {self.sim_coeff} | std: {self.std_coeff} | cov: {self.cov_coeff}')
        return out


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
