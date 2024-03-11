import timm
import torch
import easydict

from .heads import build_head
from .meta_arch import build_meta_arch

__all__ = ['build_model']


def build_model(cfg: easydict.EasyDict, device) -> torch.nn.Module:
    backbone_cfg = cfg.MODEL.BACKBONE
    backbone = timm.create_model(backbone_cfg.NAME, **backbone_cfg.get('KWARGS', {}))

    head_cfg = cfg.MODEL.HEAD
    head = build_head(head_cfg.NAME, **head_cfg.get('KWARGS', {}))

    meta_cfg = cfg.MODEL.META_ARCH
    model = build_meta_arch(meta_cfg.NAME, backbone=backbone, head=head,
                            device=device, **meta_cfg.get('KWARGS', {}))

    return model
