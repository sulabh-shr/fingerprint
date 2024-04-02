from .vicreg import *
from .cosine import *
from .focal import *
from .combined import CombinedLoss


def build_loss(cfg):
    losses = []
    weights = []
    for d in cfg.LOSSES:
        wt = d.get('WEIGHT', 1.0)
        name = d['NAME']
        kwargs = d.get('KWARGS', {})
        loss_fn = build_loss_by_name(name, kwargs)
        losses.append(loss_fn)
        weights.append(wt)

    loss = CombinedLoss(losses=losses, weights=weights)

    return loss


def build_loss_by_name(name, kwargs):
    try:
        class_ = globals()[name]
    except KeyError:
        raise NotImplementedError(f'Loss: {name} not found!')
    loss_fn = class_(**kwargs)
    return loss_fn
