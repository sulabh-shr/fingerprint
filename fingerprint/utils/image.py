import cv2
import numpy as np


def equalize_clahe(img: np.ndarray, clip: float = 2.0, grid: int = 32, channels='RGB'):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))

    # gray image
    if img.ndim == 2:
        result = clahe.apply(img)
        return result

    # color image
    if channels == 'BGR':
        color_mode = cv2.COLOR_BGR2LAB
        color_mode_org = cv2.COLOR_LAB2BGR
    elif channels == 'RGB':
        color_mode = cv2.COLOR_RGB2LAB
        color_mode_org = cv2.COLOR_LAB2RGB
    else:
        raise ValueError(f'Invalid channel type: {channels}')
    img_lab = cv2.cvtColor(img, color_mode)
    img_l = img_lab[:, :, 0]
    img_l_enhanced = clahe.apply(img_l)
    img_lab_enhanced = img_lab.copy()
    img_lab_enhanced[:, :, 0] = img_l_enhanced
    result = cv2.cvtColor(img_lab_enhanced, color_mode_org)
    return result
