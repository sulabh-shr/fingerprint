import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from typing import List


class FingerprintEvaluator(object):
    def __init__(self, dim=-1, device='cpu', verbose=True):
        self.dim = dim
        self.device = device
        self.verbose = verbose
        self.similarity_fn = torch.nn.CosineSimilarity(dim=dim)
        self.gallery = {}
        self.probe = {}
        self._reset()

    def _reset(self):
        self.gallery = {}
        self.probe = {}

    def reset(self):
        self._reset()

    def process(self, gallery: List[torch.Tensor], probe: List[torch.Tensor], key: str):
        self.gallery[key] = [x.detach().clone().to(self.device) for x in gallery]
        self.probe[key] = [x.detach().clone().to(self.device) for x in probe]

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
                # if gt == 0 and score > 0.8 or gt == 1 and score < 0.8:
                #     print(k1, k2, gt, score)
            if self.verbose:
                print('*' * 70)
                indices = np.argsort(k1_scores)[-10:]
                print(k2)
                print([k1_list[i] for i in indices[::-1]])
                print([round(k1_scores[i], 3) for i in indices[::-1]])
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        fnr = 1 - tpr
        delta = np.absolute(fpr - fnr)

        eer_index = np.nanargmin(delta)
        eer_threshold = thresholds[eer_index]
        eer_fpr = fpr[eer_index]
        eer_fnr = fnr[eer_index]
        # eer_delta = delta[eer_index]
        roc_auc = metrics.auc(fpr, tpr)

        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        display_result = display.plot()
        fig = display_result.figure_
        # z = np.polyfit(fpr, tpr, deg=2)
        # f = np.poly1d(z)
        x = np.linspace(0.0, 1.0, 10)
        y = 1 - x
        # y = f(x)

        plt.plot(x, y, '--')
        # plt.savefig('auc-curve.png')
        # plt.show()

        result = {'AUC': round(roc_auc, 3), 'EER': round(min(eer_fnr, eer_fpr), 3),
                  'EER threshold': round(eer_threshold, 3)}
        return result, fig

    def score_gallery_probe(self, gallery: List[torch.Tensor], probe: List[torch.Tensor]):
        scores = []
        for g in gallery:
            for p in probe:
                score = self.score(g, p)
                scores.append(score)
        final_score = np.median(scores)
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
