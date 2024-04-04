import numpy as np
from PIL import ImageOps, ImageFilter
import torchvision.transforms as tr
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
    binary = cfg.IMAGE.get('BINARY', False)

    br = cfg.get('brightness', 0.4)
    cont = cfg.get('contrast', 0.4)
    hue = cfg.get('hue', 0.1)
    sat = cfg.get('saturation', 0.2)
    rotate = cfg.get('rotate', 5)

    spaces = [
        tr.RandomApply([tr.RandomRotation(degrees=rotate, interpolation=InterpolationMode.BICUBIC)], p=0.5),
        tr.RandomResizedCrop((h, w), scale=(0.5, 1.0), ratio=(0.95, 1.05), interpolation=InterpolationMode.BICUBIC),
    ]
    colors = [
        tr.RandomApply([tr.ColorJitter(brightness=br, contrast=cont, saturation=sat, hue=hue)], p=0.8),
        tr.RandomGrayscale(p=0.2),
    ]
    pixels = [GaussianBlur(p=0.2)]
    finals = [
        tr.ToTensor(),
        tr.Normalize(mean, std)
    ]

    second_only = [Solarization(p=0.2)]
    if binary:
        colors = []
        second_only = []

    first_list = spaces + colors + pixels + finals
    second_list = spaces + colors + pixels + second_only + finals

    # Skip Tensor and Normalization for viewing
    if debug:
        first_list = first_list[:-2]
        second_list = second_list[:-2]

    result = {
        'transforms1': tr.Compose(first_list),
        'transforms2': tr.Compose(second_list)
    }
    return result


def get_test_transforms(cfg, debug=False):
    h, w = cfg.IMAGE.IMG_SIZE
    mean = cfg.IMAGE.MEAN
    std = cfg.IMAGE.STD

    transforms_list = [
        tr.Resize(size=max(h, w), interpolation=InterpolationMode.BICUBIC),
        tr.CenterCrop(size=(h, w)),
        tr.ToTensor(),
        tr.Normalize(mean=mean, std=std)
    ]

    if debug:
        transforms_list = transforms_list[:-2]

    result = {
        'test': tr.Compose(transforms_list)
    }
    return result
