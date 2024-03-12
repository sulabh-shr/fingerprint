import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn import metrics
import torch.distributed as dist


class ContrastiveEvaluator(object):
    def __init__(self, dim=-1, device='cpu'):
        self.dim = dim
        self.device = device
        self.similarity_fn = torch.nn.CosineSimilarity(dim=dim)
        self.data1 = {}
        self.data2 = {}
        self._reset()

    def _reset(self):
        self.data1 = {}
        self.data2 = {}

    def reset(self):
        self._reset()

    def process(self, x1: torch.Tensor, x2: torch.Tensor, key: str):
        self.data1[key] = x1.detach().clone().to(self.device)
        self.data2[key] = x2.detach().clone().to(self.device)

    def evaluate(self):
        y_true = []
        y_score = []
        for k1, v1 in self.data1.items():
            for k2, v2 in self.data2.items():
                if k1 == k2:
                    gt = 1
                else:
                    gt = 0
                score = self.score(v1, v2)
                y_true.append(gt)
                y_score.append(score)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        fnr = 1 - tpr
        delta = np.absolute(fpr - fnr)

        eer_index = np.nanargmin(delta)
        eer_threshold = thresholds[eer_index]
        eer_fpr = fpr[eer_index]
        eer_fnr = fnr[eer_index]
        # eer_delta = delta[eer_index]
        roc_auc = metrics.auc(fpr, tpr)

        # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        # fig = display.plot()
        # z = np.polyfit(fpr, tpr, deg=2)
        # f = np.poly1d(z)
        # x = np.linspace(0.05, 0.95, 10)
        # y = f(x)
        # plt.plot(fpr, tpr, 'o', x, y)
        # plt.savefig('auc-curve.png')
        # plt.show()

        result = {'AUC': round(roc_auc, 3), 'EER': round(min(eer_fnr, eer_fpr), 3),
                  'EER threshold': round(eer_threshold, 3)}
        return result

    def score(self, x1, x2):
        result = (1 + self.similarity_fn(x1, x2)) / 2
        return result

    @staticmethod
    def summarize(res):
        out = ""
        for k, v in res.items():
            out += f'{k[:25]:-<25s} : {v:.3f}\n'
        return out
