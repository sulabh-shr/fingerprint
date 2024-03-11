import torch
from .lars import LARS

__all__ = ['build_optimizer']


def build_optimizer(cfg, parameters):
    optim_cfg = cfg.OPTIM.OPTIMIZER
    if optim_cfg.NAME in globals():
        class_ = globals()[optim_cfg.NAME]
    else:
        class_ = getattr(torch.optim, optim_cfg.NAME)
    optimizer = class_(parameters, **optim_cfg.get('KWARGS', {}))
    return optimizer
