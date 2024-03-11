import torch
from fingerprint.utils.structures import DummyClass

__all__ = ['build_lr_scheduler']


def build_lr_scheduler(cfg, optimizer):
    lr_scheduler_cfg = cfg.OPTIM.LR_SCHEDULERS
    schedulers = []
    for cfg_i in lr_scheduler_cfg.SCHEDULERS:
        if cfg_i.NAME in globals():
            class_ = globals()[cfg_i.NAME]
        else:
            class_ = getattr(torch.optim.lr_scheduler, cfg_i.NAME)
        scheduler_i = class_(optimizer, **cfg_i.get('KWARGS', {}))
        schedulers.append(scheduler_i)

    if len(schedulers) == 1:
        final_scheduler = schedulers[0]
    elif len(schedulers) > 1:
        milestones = lr_scheduler_cfg.MILESTONES
        final_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones)
    else:
        final_scheduler = DummyClass()
    return final_scheduler
