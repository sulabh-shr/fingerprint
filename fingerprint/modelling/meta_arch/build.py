from .contrastive import *


def build_meta_arch(name, **kwargs):
    try:
        class_ = globals()[name]
    except KeyError:
        raise NotImplementedError(f'Meta-Arch: {name} not found!')
    meta_arch = class_(**kwargs)
    return meta_arch
