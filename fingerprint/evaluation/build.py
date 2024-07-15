from .base import BaseEvaluator
from .gallery import (
    FingerprintEvaluator,
    FingerprintEvaluatorMultiScores
)
from .contrastive import ContrastiveEvaluator, ContrastiveEvaluatorMulti


def build_evaluator(cfg) -> BaseEvaluator:
    class_ = globals()[cfg.NAME]
    evaluator = class_(**cfg.KWARGS)
    return evaluator
