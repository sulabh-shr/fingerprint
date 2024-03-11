import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from .mmfv import MMFVContrastive


def build_dataset(cfg) -> Dataset:
    class_ = globals()[cfg.DATASET.NAME]
    dataset = class_(**cfg.DATASET.KWARGS)
    return dataset


def build_dataloader(cfg, dataset):
    if torch.distributed.is_initialized():
        shuffle = False
        dataloader = DataLoader(
            dataset,
            sampler=DistributedSampler(dataset),
            shuffle=shuffle,
            **cfg.DATALOADER.KWARGS
        )
    else:
        dataloader = DataLoader(dataset, **cfg.DATALOADER.KWARGS)

    if len(dataloader) == 0:
        raise FileNotFoundError(f'Dataset is empty!')

    return dataloader
