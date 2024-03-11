import torch.nn as nn

__all__ = ['Contrastive']


class Contrastive(nn.Module):
    def __init__(self, backbone, head, device):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.device = device

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.head(features)
        result = {
            'backbone': features,
            'head': embeddings
        }
        return result
