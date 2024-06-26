from .expander import *
from .identity import *
from .classifier import *


def build_head(name, **kwargs):
    try:
        class_ = globals()[name]
    except KeyError:
        raise NotImplementedError(f'Head: {name} not found!')
    head = class_(**kwargs)
    return head
