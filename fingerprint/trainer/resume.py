import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import *


def resume(path, rank, ft, model, optimizer=None, scheduler=None):
    if path is not None:
        if dist.is_initialized():
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            ckpt = torch.load(path, map_location=map_location)
        else:
            ckpt = torch.load(path)
        load_ckpt_single(ckpt['model'], model)
        del ckpt['model']
        if ft:
            return ckpt
        else:
            if 'optim' in ckpt:
                optimizer.load_state_dict(ckpt['optim'])
                del ckpt['optim']
            if 'lr_scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['lr_scheduler'])
                del ckpt['lr_scheduler']
        print(f'Rank: {rank} | Checkpoint loaded from {path}')
        return ckpt
    return None


def load_ckpt_single(state_dict: Dict, model: torch.nn.Module) -> torch.nn.Module:
    """ Load model ckpt by converting from DDP to single if required.

    Args:
        state_dict:
        model:

    Returns:
        model: Loaded model.
    """
    if isinstance(model, DDP):
        model.load_state_dict(state_dict)
    else:
        keys = list(state_dict.keys())
        for k in keys:
            if k.startswith('module.'):
                state_dict[k[len('module.'):]] = state_dict[k]
                del state_dict[k]
        model.load_state_dict(state_dict)
    return model
