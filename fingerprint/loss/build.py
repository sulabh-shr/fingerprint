from .vicreg import *
from .cosine import *


def build_loss(cfg):
    try:
        class_ = globals()[cfg.LOSS.NAME]
    except KeyError:
        raise NotImplementedError(f'Loss: {cfg.LOSS.NAME} not found!')
    loss_fn = class_(**cfg.LOSS.KWARGS)
    return loss_fn
