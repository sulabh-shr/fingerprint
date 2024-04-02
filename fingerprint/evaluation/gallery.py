import torch
import numpy as np
from typing import List, Union, Dict
import torch.nn.functional as F
import torch.distributed as dist
from collections import defaultdict
from sklearn.decomposition import PCA

from .base import BaseEvaluator
from .utils import scores_to_metrics

__all__ = ['FingerprintEvaluator', 'FingerprintEvaluatorWithLogits']


class FingerprintEvaluator(BaseEvaluator):
    def __init__(self, dim=-1, fusion='score-avg', gallery_id='data1', probe_id='data2', device='cpu', verbose=True):
        super().__init__()
        self.FUSIONS = ('score-avg', 'feat-avg', 'cat-replicate', 'cat-pca')
        self.dim = dim
        assert fusion in self.FUSIONS, f'Invalid fusion: {fusion}. Must be one of {self.FUSIONS}'
        self.fusion = fusion

        self.gallery_id = gallery_id
        self.probe_id = probe_id
        self.device = device
        self.verbose = verbose
        self.similarity_fn = torch.nn.CosineSimilarity(dim=dim)
        self.gallery: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = defaultdict(list)
        self.probe: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = defaultdict(list)
        self.pca = None
        self.reset()

    def reset(self):
        self.gallery = defaultdict(list)
        self.probe = defaultdict(list)
        self.pca = None

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

    def process_contrastive(self, x1: List[torch.Tensor], x2: List[torch.Tensor], key: List[str]):
        self.process(x1, key, ['data1'] * len(x1))
        self.process(x2, key, ['data2'] * len(x1))

    def evaluate(self):
        self._preprocess()

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

    def score_gallery_probe(self,
                            gallery: Union[List[torch.Tensor], torch.Tensor],
                            probe: Union[List[torch.Tensor], torch.Tensor]
                            ) -> float:
        fusion = self.fusion
        if fusion == 'score-avg':
            scores = []
            for g in gallery:
                for p in probe:
                    score = self.score(g, p)
                    scores.append(score)
            final_score = np.mean(scores)
        elif fusion == 'feat-avg':
            final_score = self.score(gallery, probe)
        elif fusion in ('cat-pca', 'cat-replicate'):
            n_g = len(gallery)
            n_p = len(probe)
            num_tiles = n_g // n_p
            probe = probe.tile(num_tiles)
            if n_g % n_p != 0:
                probe = torch.cat([probe, probe[:n_g - len(probe)]])
            final_score = self.score(gallery, probe)
        else:
            raise ValueError(f'Invalid fusion method: {fusion}')
        return final_score

    def _preprocess(self):
        if self.fusion == 'cat-pca':
            print(f'Calculating PCA')
            all_gallery_features = []
            for k1, gallery_features in self.gallery.items():
                for f in gallery_features:
                    all_gallery_features.append(f.cpu().numpy())
            all_gallery_features = np.array(all_gallery_features)
            print(f'All Gallery Features: {all_gallery_features.shape}')
            pca = PCA(n_components=32)
            pca.fit(all_gallery_features)
            self.pca = pca
            for k1, gallery_features in self.gallery.items():
                self.gallery[k1] = torch.from_numpy(self.pca.transform(np.array(gallery_features))).reshape(-1)
            for k2, probe_features in self.probe.items():
                self.probe[k2] = torch.from_numpy(self.pca.transform(np.array(probe_features))).reshape(-1)
        elif self.fusion == 'cat-replicate':
            for k1, gallery_features in self.gallery.items():
                self.gallery[k1] = torch.cat(gallery_features)
            for k2, probe_features in self.probe.items():
                self.probe[k2] = torch.cat(probe_features)
        elif self.fusion == 'feat-avg':
            for k1, gallery_features in self.gallery.items():
                self.gallery[k1] = torch.stack(gallery_features).mean(dim=0)
            for k2, probe_features in self.probe.items():
                self.probe[k2] = torch.stack(probe_features).mean(dim=0)

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
