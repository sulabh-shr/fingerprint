import torch
import torch.nn as nn

__all__ = ['Contrastive', 'ContrastiveWithLogits']


class Contrastive(nn.Module):
    def __init__(self, backbone, head, device, classifier=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.device = device
        self.classifier = classifier

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.head(features)
        result = {
            'backbone': features,
            'head': embeddings
        }
        return result

    def forward_backbone(self, x):
        return self.backbone(x)

    def has_classifier(self):
        return self.classifier is not None

    def cross_classify(self, x1, x2):
        if not self.has_classifier():
            raise TypeError(f'Model does not have a classifier.')

        n = len(x1)
        output = []
        for i in range(n):
            x1_repeat = x1[i:i + 1, :].repeat([n, 1])
            o = self.classify(x1_repeat, x2)
            output.append(o['logits'])

        gt = torch.zeros((n, n))
        gt[torch.eye(n).bool()] = 1.0
        gt = gt.reshape(-1, 1)

        result = {
            'gt': gt,
            'logits': torch.cat(output)  # (n*n, 1)
        }
        return result

    def classify(self, x1, x2):
        if not self.has_classifier():
            raise TypeError(f'Model does not have a classifier.')
        classifier_input = torch.cat((x1, x2), dim=-1)
        result = {
            'logits': self.classifier(classifier_input)  # (n, 1)
        }
        return result


# Backward Compatibility
ContrastiveWithLogits = Contrastive
