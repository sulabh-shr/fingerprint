import torch
import torch.nn as nn

__all__ = ['Vanilla']


class Vanilla(nn.Module):
    def __init__(self, backbone, head, device):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.device = device

    def forward(self, x):
        features = self.backbone.forward_features(x)
        embeddings = self.head(features)
        result = {
            'backbone': features,
            'head': embeddings
        }
        return result

    def forward_backbone(self, x):
        return self.backbone(x)
