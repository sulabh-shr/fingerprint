import torch
import torch.nn as nn

__all__ = ['Contrastive', 'ContrastiveWithLogits']


class Contrastive(nn.Module):
    def __init__(self, backbone, head, device, expander=None, classifier=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.device = device
        self.classifier = classifier
        self.expander = expander

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.head(features)

        result = {
            'backbone': features,
            'head': embeddings,
        }
        if self.expander is not None:
            expanded = self.expander(features)
            result['expander'] = expanded

        return result

    def forward_backbone(self, x):
        return self.backbone(x)

    def has_expander(self):
        return self.expander is not None

    def has_classifier(self):
        return self.classifier is not None

    def cross_classify(self, x1, x2, label=None):
        if not self.has_classifier():
            raise TypeError(f'Model does not have a classifier.')

        n = len(x1)
        output = []
        gt = []
        for i in range(n):
            x1_repeat = x1[i:i + 1, :].repeat([n, 1])
            o = self.classify(x1_repeat, x2)
            output.append(o['logits'])
            gt_i = torch.zeros(n)
            # Multiple data points may have same class
            if label is not None:
                current_label = label[i]
                gt_i[label == current_label] = 1.0
            # Assume 1 class per data point
            else:
                gt_i[i] = 1
            gt.append(gt_i)

        result = {
            'gt': torch.cat(gt).unsqueeze(1),
            'logits': torch.cat(output)  # (n*n, 1)
        }

        return result

    def classify(self, x1: torch.Tensor, x2: torch.Tensor):
        if not self.has_classifier():
            raise TypeError(f'Model does not have a classifier.')

        if x1.ndim == 1:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)

        classifier_input = torch.cat((x1, x2), dim=-1)  # (n, 2d) <- (n, d) + (n, d)
        result = {
            'logits': self.classifier(classifier_input)  # (n, 1)
        }
        return result


class ContrastiveExpander(Contrastive):
    def __init__(self,
                 backbone,
                 head,
                 device,
                 expander=None,
                 classifier=None):
        super().__init__(
            backbone=backbone,
            head=head,
            device=device,
            classifier=classifier
        )
        self.expander = expander

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.head(features)
        expanded = self.expander(embeddings)
        result = {
            'backbone': features,
            'head': embeddings,
            'expander': expanded
        }
        return result


# Backward Compatibility
ContrastiveWithLogits = Contrastive
