import torch
import numpy as np
from typing import List
from sklearn import metrics
import matplotlib.pyplot as plt
import torch.distributed as dist

from .base import BaseEvaluator
from .utils import scores_to_metrics

__all__ = ['ContrastiveEvaluator', 'ContrastiveEvaluatorMulti']


class ContrastiveEvaluator(BaseEvaluator):
    def __init__(self, dim=-1, device='cpu'):
        super().__init__()
        self.dim = dim
        self.device = device
        self.similarity_fn = torch.nn.CosineSimilarity(dim=dim)
        self.data1 = {}
        self.data2 = {}
        self.reset()

    def reset(self):
        self.data1 = {}
        self.data2 = {}

    def process_single(self, x1: torch.Tensor, x2: torch.Tensor, key: str):
        self.data1[key] = x1.detach().clone().to(self.device)
        self.data2[key] = x2.detach().clone().to(self.device)

    def process(self, x1: List[torch.Tensor], x2: List[torch.Tensor], key: List[str]):
        for idx in range(len(key)):
            self.process_single(x1[idx], x2[idx], key[idx])

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
        result, fig = scores_to_metrics(y_true, y_score)
        return result, fig

    def score(self, x1, x2):
        result = (1 + self.similarity_fn(x1, x2)) / 2
        return result

    @staticmethod
    def summarize(res):
        out = ""
        for k, v in res.items():
            out += f'{k[:25]:-<25s} : {v:.3f}\n'
        return out


class ContrastiveEvaluatorMulti(BaseEvaluator):
    def __init__(self, dim=-1, device='cpu'):
        super().__init__()
        self.dim = dim
        self.device = device
        self.similarity_fn = torch.nn.CosineSimilarity(dim=dim)
        self.data1 = []
        self.key1 = []
        self.data2 = []
        self.key2 = []
        self.reset()

    def reset(self):
        self.data1 = []
        self.key1 = []
        self.data2 = []
        self.key2 = []

    def process(self, x1: torch.Tensor = None, x2: torch.Tensor = None,
                key1: str = None, key2: str = None):
        assert x1 is not None or x2 is not None
        if x1 is not None:
            assert key1 is not None
            x1 = x1.detach().clone().to(self.device)
            self.data1.append(x1)
            self.key1.append(key1)
        if x2 is not None:
            assert key2 is not None
            x2 = x2.detach().clone().to(self.device)
            self.data2.append(x2)
            self.key2.append(key2)

    def evaluate(self):
        y_true = []
        y_score = []
        for k1, v1 in zip(self.key1, self.data1):
            for k2, v2 in zip(self.key2, self.data2):
                if k1 == k2:
                    gt = 1
                else:
                    gt = 0
                score = self.score(v1, v2)
                y_true.append(gt)
                y_score.append(score)

        result, fig = scores_to_metrics(y_true, y_score)

        return result, fig

    def score(self, x1, x2):
        result = (1 + self.similarity_fn(x1, x2)) / 2
        return result

    @staticmethod
    def summarize(res):
        out = ""
        for k, v in res.items():
            out += f'{k[:25]:-<25s} : {v:.3f}\n'
        return out
