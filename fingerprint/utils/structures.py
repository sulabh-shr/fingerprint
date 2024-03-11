from typing import Any

__all__ = ['dummy_fn', 'DummyClass']


def dummy_fn(*args, **kwargs):
    return None


class DummyClass(object):
    def __getattr__(self, item):
        return dummy_fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass
