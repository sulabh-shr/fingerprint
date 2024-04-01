import abc


class BaseEvaluator(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass

    @abc.abstractmethod
    def process(self, *args, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def summarize(self) -> str:
        pass

