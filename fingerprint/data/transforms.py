import numpy as np
from PIL import ImageOps, ImageFilter
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.0 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def get_train_transforms(cfg, debug=False):
    h, w = cfg.IMAGE.IMG_SIZE
    mean = cfg.IMAGE.MEAN
    std = cfg.IMAGE.STD

    brightness = cfg.get('brightness', 0.4)
    contrast = cfg.get('contrast', 0.4)
    hue = cfg.get('hue', 0.1)
    saturation = cfg.get('saturation', 0.2)
    rotate = cfg.get('rotate', 5)

    first_list = [
        transforms.RandomApply(
            [transforms.RandomRotation(degrees=rotate, interpolation=InterpolationMode.BICUBIC)
             ], p=0.5),
        transforms.RandomResizedCrop((h, w), scale=(0.5, 1.0), ratio=(0.95, 1.10),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
             ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(p=0.8),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    second_list = [
        transforms.RandomApply(
            [transforms.RandomRotation(degrees=rotate, interpolation=InterpolationMode.BICUBIC)
             ], p=0.5),
        transforms.RandomResizedCrop((h, w), scale=(0.5, 1.0), ratio=(0.85, 1.15),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
             ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(p=0.1),
        Solarization(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    # Skip Tensor and Normalization for viewing
    if debug:
        first_list = first_list[:-2]
        second_list = second_list[:-2]

    result = {
        'transforms1': transforms.Compose(first_list),
        'transforms2': transforms.Compose(second_list)
    }
    return result


def get_test_transforms(cfg, debug=False):
    h, w = cfg.IMAGE.IMG_SIZE
    mean = cfg.IMAGE.MEAN
    std = cfg.IMAGE.STD

    transforms_list = [
        transforms.Resize(size=max(h, w), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    if debug:
        transforms_list = transforms_list[:-2]

    result = {
        'test': transforms.Compose(transforms_list)
    }
    return result
