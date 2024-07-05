import timm
import torch
import easydict

from .heads import build_head
from .meta_arch import build_meta_arch

__all__ = ['build_model']


def build_model(cfg: easydict.EasyDict, device) -> torch.nn.Module:
    meta_cfg = cfg.MODEL.META_ARCH
    extracted_kwargs = {
        'name': meta_cfg.NAME,
        'device': device
    }

    backbone_cfg = cfg.MODEL.BACKBONE
    backbone = timm.create_model(backbone_cfg.NAME, **backbone_cfg.get('KWARGS', {}))
    extracted_kwargs['backbone'] = backbone

    head_cfg = cfg.MODEL.HEAD
    head = build_head(head_cfg.NAME, **head_cfg.get('KWARGS', {}))
    extracted_kwargs['head'] = head

    if 'CLASSIFIER' in cfg.MODEL:
        classifier_cfg = cfg.MODEL.CLASSIFIER
        classifier = build_head(classifier_cfg.NAME, **classifier_cfg.get('KWARGS', {}))
        extracted_kwargs['classifier'] = classifier

    if 'EXPANDER' in cfg.MODEL:
        expander_cfg = cfg.MODEL.EXPANDER
        expander = build_head(expander_cfg.NAME, **expander_cfg.get('KWARGS', {}))
        extracted_kwargs['expander'] = expander

    extracted_kwargs.update(meta_cfg.get('KWARGS', {}))
    model = build_meta_arch(**extracted_kwargs)

    return model
