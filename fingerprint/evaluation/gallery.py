import torch
import numpy as np
from typing import List
import torch.nn.functional as F
import torch.distributed as dist
from collections import defaultdict

from .base import BaseEvaluator
from .utils import scores_to_metrics

__all__ = ['FingerprintEvaluator', 'FingerprintEvaluatorWithLogits']


class FingerprintEvaluator(BaseEvaluator):
    def __init__(self, dim=-1, fusion='score-avg', gallery_id='data1', probe_id='data2', device='cpu', verbose=True):
        super().__init__()
        self.FUSIONS = ('score-avg', 'feat-avg')
        self.dim = dim
        assert fusion in self.FUSIONS, f'Invalid fusion: {fusion}. Must be one of {self.FUSIONS}'
        self.fusion = fusion

        self.gallery_id = gallery_id
        self.probe_id = probe_id
        self.device = device
        self.verbose = verbose
        self.similarity_fn = torch.nn.CosineSimilarity(dim=dim)
        self.gallery = defaultdict(list)
        self.probe = defaultdict(list)
        self.reset()

    def reset(self):
        self.gallery = defaultdict(list)
        self.probe = defaultdict(list)

    def process(self, features: List[torch.Tensor], classes: List[str], locations: List[str]):
        for idx in range(len(features)):
            ft = features[idx]
            ft = ft.detach().clone().to(self.device)
            class_ = classes[idx]
            loc = locations[idx]
            if loc == self.gallery_id:
                self.gallery[class_].append(ft)
            elif loc == self.probe_id:
                self.probe[class_].append(ft)
            else:
                raise ValueError(f'Invalid location: {loc}')

    def evaluate(self):
        y_true = []
        y_score = []
        for k2, probe_features in self.probe.items():
            k1_list = []
            k1_scores = []
            for k1, gallery_features in self.gallery.items():
                if k1 == k2:
                    gt = 1
                else:
                    gt = 0
                score = self.score_gallery_probe(gallery_features, probe_features)
                y_true.append(gt)
                y_score.append(score)
                k1_list.append(k1)
                k1_scores.append(score)
                # print(k1, k2, gt, score)
                # if gt == 0 and score > 0.8 or gt == 1 and score < 0.8:
                #     print(k1, k2, gt, score)
            if self.verbose:
                print('*' * 70)
                indices = np.argsort(k1_scores)[-10:]
                print(k2)
                print([k1_list[i] for i in indices[::-1]])
                print([round(k1_scores[i], 3) for i in indices[::-1]])

        result, fig = scores_to_metrics(y_true, y_score)

        return result, fig

    def score_gallery_probe(self, gallery: List[torch.Tensor], probe: List[torch.Tensor]):
        fusion = self.fusion
        if fusion == 'score-avg':
            scores = []
            for g in gallery:
                for p in probe:
                    score = self.score(g, p)
                    scores.append(score)
            final_score = np.mean(scores)
        elif fusion == 'feat-avg':
            g = torch.stack(gallery).mean(dim=0)
            p = torch.stack(probe).mean(dim=0)
            final_score = self.score(g, p)
        else:
            raise ValueError(f'Invalid fusion method: {fusion}')
        return final_score

    def score(self, x1: torch.Tensor, x2: torch.Tensor):
        with torch.no_grad():
            result = (1 + self.similarity_fn(x1, x2)) / 2
            result = result.item()
        return result

    @staticmethod
    def summarize(res):
        out = ""
        for k, v in res.items():
            out += f'{k[:25]:-<25s} : {v:.3f}\n'
        return out


class FingerprintEvaluatorWithLogits(FingerprintEvaluator):
    def __init__(self, model, dim=-1, store_device='cpu', compute_device='cuda', verbose=True):
        super().__init__(dim=dim, device=store_device, verbose=verbose)
        self.model = model
        self.model.eval()
        self.model.to(compute_device)
        self.compute_device = compute_device

    def score(self, x1: torch.Tensor, x2: torch.Tensor):
        with torch.no_grad():
            x1 = x1.unsqueeze(0).to(self.compute_device)
            x2 = x2.unsqueeze(0).to(self.compute_device)
            result = F.sigmoid(self.model.classify(x1, x2)['logits'])
        return result.item()
