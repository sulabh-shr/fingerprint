import os
import cv2
import random
import numpy as np
import PIL.ImageQt
import torch
from PIL import Image
from itertools import product
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import Dataset


class MMFVContrastive(Dataset):
    def __init__(self,
                 root,
                 segment=True,
                 mode='train',
                 transforms1=None,
                 transforms2=None,
                 subjects='train.txt',
                 fingers=('f1', 'f2', 'f3', 'f4'),
                 movements=('Roll', 'Pitch', 'Yaw'),
                 ):
        self.root = root
        self.segment = segment
        self.mode = mode
        self.subjects = subjects
        self.fingers = fingers
        self.movements = movements
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.data1 = defaultdict(dict)  # gallery
        self.data2 = defaultdict(dict)  # probe
        self.movement_pairs = tuple(product(movements, movements))  # Track which movement pair to sample
        self.movement_index = {}
        self.keys = ()
        self._init_paths_and_keys()

    def _init_paths_and_keys(self):
        root = self.root
        subject_file = self.subjects

        with open(os.path.join(root, subject_file)) as f:
            content = f.readlines()
            subjects = [i.strip() for i in content]

        for idx, subject in enumerate(os.listdir(root)):
            if subject not in subjects:
                continue
            subject_path = os.path.join(root, subject)
            for sess in os.listdir(subject_path):
                sess_path = os.path.join(subject_path, sess)
                for finger in os.listdir(sess_path):
                    finger_path = os.path.join(sess_path, finger)
                    if finger not in self.fingers:
                        continue
                    for mov in os.listdir(finger_path):
                        if mov not in self.movements:
                            continue
                        mov_path = os.path.join(finger_path, mov)
                        frames1 = [i for i in os.listdir(mov_path) if i.startswith('1_')]
                        frames2 = [i for i in os.listdir(mov_path) if i.startswith('2_')]
                        all_frames = frames1 + frames2
                        paths = [os.path.join(mov_path, i) for i in all_frames]
                        key = f'{subject}-{finger}'
                        if sess == '1':
                            self.data1[key][mov] = paths
                        else:
                            self.data2[key][mov] = paths
        self.keys = tuple(self.data1.keys())
        self.data1 = dict(self.data1)
        self.data2 = dict(self.data2)

        for key in self.data1:
            self.movement_index[key] = 0

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        key = self.keys[idx]

        if self.mode == 'train':
            num_mov_pairs = len(self.movement_pairs)
            previously_sampled = self.movement_index[key] % num_mov_pairs
            mov_idx = previously_sampled % num_mov_pairs
            self.movement_index[key] = (previously_sampled + 1) % num_mov_pairs
            data1_mov, data2_mov = self.movement_pairs[mov_idx]
            path1 = random.choice(self.data1[key][data1_mov])
            path2 = random.choice(self.data2[key][data2_mov])
        else:  # Fixed sampling
            data1_mov, data2_mov = self.movement_pairs[0]
            path1 = self.data1[key][data1_mov][0]
            path2 = self.data2[key][data2_mov][0]

        img1 = self._get_image(path1)
        img2 = self._get_image(path2)
        if self.transforms1 is not None:
            img1 = self.transforms1(img1)
        if self.transforms2 is not None:
            img2 = self.transforms2(img2)

        result = {
            'image1': img1,
            'image2': img2,
            'path1': path1,
            'path2': path2,
            'key': key,
            'mov1': data1_mov,
            'mov2': data2_mov
        }
        return result

    def _get_image(self, path) -> PIL.Image.Image:
        img = Image.open(path)
        cropped_img = crop_fingerprint(np.array(img), segment=self.segment, channels='RGB')
        h, w, _ = cropped_img.shape
        # use original image if finger not found
        if h == 0 or w == 0:
            print(f'Finger contour not found for image: {path}')
            return img
        cropped_img = Image.fromarray(cropped_img)
        return cropped_img

    def __str__(self):
        result = (f'MMFV Contrastive\n Root: {self.root}\n Length: {len(self)}\n '
                  f'Fingers: {self.fingers}\n Movements: {self.movements}')
        return result


class MMFVEval(Dataset):
    def __init__(self,
                 root,
                 segment=True,
                 randomize=True,
                 transforms1=None,
                 transforms2=None,
                 subjects='test.txt',
                 fingers=('f1', 'f2', 'f3', 'f4'),
                 gallery_movements=('Roll', 'Pitch', 'Yaw'),
                 probe_movements=('Pitch',)
                 ):
        self.root = root
        self.segment = segment
        self.randomize = randomize
        self.subjects = subjects
        self.fingers = fingers
        self.gallery_movements = gallery_movements
        self.probe_movements = probe_movements
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.data1 = defaultdict(dict)  # gallery
        self.data2 = defaultdict(dict)  # probe
        self.keys = ()
        self._init_paths_and_keys()

    def _init_paths_and_keys(self):
        root = self.root
        subject_file = self.subjects

        with open(os.path.join(root, subject_file)) as f:
            content = f.readlines()
            subjects = [i.strip() for i in content]

        for idx, subject in enumerate(os.listdir(root)):
            if subject not in subjects:
                continue
            subject_path = os.path.join(root, subject)
            for sess in os.listdir(subject_path):
                sess_path = os.path.join(subject_path, sess)
                for finger in os.listdir(sess_path):
                    finger_path = os.path.join(sess_path, finger)
                    if finger not in self.fingers:
                        continue
                    for mov in os.listdir(finger_path):
                        mov_path = os.path.join(finger_path, mov)
                        frames1 = [i for i in os.listdir(mov_path) if i.startswith('1_')]
                        frames2 = [i for i in os.listdir(mov_path) if i.startswith('2_')]
                        # Sample k frames per video
                        if self.randomize:
                            frames1 = random.sample(frames1, k=1)
                            if len(frames2) > 2:
                                frames2 = random.sample(frames2, k=1)
                            all_frames = frames1 + frames2
                        else:
                            mid_idx = len(frames1) // 2
                            frames1 = [frames1[mid_idx]]
                            if len(frames2) > 2:
                                mid_idx = len(frames2) // 2
                                frames2 = [frames2[mid_idx]]
                            all_frames = frames1 + frames2
                        paths = [os.path.join(mov_path, i) for i in all_frames]
                        key = f'{subject}-{finger}'
                        if sess == '1':
                            if mov not in self.gallery_movements:
                                continue
                            self.data1[key][mov] = paths
                        else:
                            if mov not in self.probe_movements:
                                continue
                            self.data2[key][mov] = paths
        self.keys = tuple(self.data1.keys())
        self.data1 = dict(self.data1)
        self.data2 = dict(self.data2)

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        key = self.keys[idx]
        result = {
            'gallery': [],
            'gallery_mov': [],
            'probe': [],
            'probe_mov': [],
            'key': key
        }

        for data, transforms, gp_key in (
                (self.data1, self.transforms1, 'gallery'),
                (self.data2, self.transforms2, 'probe')):
            for mov, paths in data[key].items():
                for path in paths:
                    img = self._get_image(path)
                    result[gp_key].append(img)
                    result[f'{gp_key}_mov'].append(mov)

            if transforms is not None:
                result[gp_key] = [transforms(i) for i in result[gp_key]]
                if isinstance(result[gp_key][0], torch.Tensor):
                    result[gp_key] = torch.stack(result[gp_key])

        return result

    def _get_image(self, path) -> PIL.Image.Image:
        img = Image.open(path)
        cropped_img = crop_fingerprint(np.array(img), segment=self.segment, channels='RGB')
        h, w, _ = cropped_img.shape
        # use original image if finger not found
        if h == 0 or w == 0:
            print(f'Finger contour not found for image: {path}')
            return img
        cropped_img = Image.fromarray(cropped_img)
        return cropped_img

    def __str__(self):
        result = (f'MMFV Eval\n Root: {self.root}\n Length: {len(self)}\n '
                  f'Fingers: {self.fingers}\n '
                  f'Gallery Movements: {self.gallery_movements}\n '
                  f'Probe Movements: {self.probe_movements}')
        return result


def rotate_contour(contour, m):
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    x2 = m[0, 0] * x + m[0, 1] * y + m[0, 2]
    y2 = m[1, 0] * x + m[1, 1] * y + m[1, 2]
    return x2, y2


def crop_fingerprint(img: np.ndarray, segment, channels='RGB', verbose=False, viz=False):
    if channels == 'BGR':  # cv2.imread
        color_mode = cv2.COLOR_BGR2GRAY
    elif channels == 'RGB':  # Image.open
        color_mode = cv2.COLOR_RGB2GRAY
    else:
        raise ValueError(f'Invalid channel type: {channels}')

    img_gray = cv2.cvtColor(img, color_mode)

    # Get largest contour
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(~thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit ellipse on largest contour
    (x, y), (MA, ma), angle = cv2.fitEllipse(largest_contour)
    if angle > 90:
        angle = -(180 - angle)

    # Make none contour pixels zero
    if segment:
        contour_mask = np.zeros_like(img_gray)
        cv2.drawContours(contour_mask, [largest_contour], 0, color=(255, 255, 255), thickness=cv2.FILLED)
        img[contour_mask == 0, :] = (0, 0, 0)

    vertical_img, cropped_img, rot_contour_x, rot_contour_y = rotate_and_crop_with_contour(
        img, x, y, angle, largest_contour)

    # Fit rectangle if ellipse does not work
    if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
        if verbose:
            print('Using rectangle for contour rotation estimation')
        (x, y), (MA, ma), angle = cv2.minAreaRect(largest_contour)
        if angle > 90:
            angle = -(180 - angle)
        vertical_img, cropped_img, rot_contour_x, rot_contour_y = rotate_and_crop_with_contour(
            img, x, y, angle, largest_contour)

    if viz:
        o1 = img.copy()
        cv2.drawContours(o1, contours, -1, (0, 0, 255), 5)

        o2 = img.copy()
        cv2.drawContours(o2, [largest_contour], 0, (255, 0, 0), 10)

        o3 = vertical_img.copy()
        rot_largest_contour = np.zeros_like(largest_contour)
        rot_largest_contour[:, 0, 0] = rot_contour_x
        rot_largest_contour[:, 0, 1] = rot_contour_y
        cv2.drawContours(o3, [rot_largest_contour], 0, (255, 0, 0), 10)

        fig, ax = plt.subplots(1, 4, figsize=(20, 10))
        # Image.open is used to load image so channel inversion not required
        ax[0].imshow(o1)
        ax[1].imshow(o2)
        ax[2].imshow(o3)
        ax[3].imshow(cropped_img)

        plt.show()
        plt.close()

    return cropped_img


def rotate_and_crop_with_contour(img, x, y, angle, contour):
    # Rotate the image about ellipse so that it is vertical
    rot_mat = cv2.getRotationMatrix2D((x, y), angle, 1.0)
    vertical_image = cv2.warpAffine(img, rot_mat, img.shape[1::-1],
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Rotate contour separately but identical to the image
    rot_contour_x, rot_contour_y = rotate_contour(contour, rot_mat)

    # Find lowest point in rotated contour
    lowest_idx = np.argmax(rot_contour_y)
    lowest_x, lowest_y = rot_contour_x[lowest_idx], rot_contour_y[lowest_idx]
    # print(lowest_x, lowest_y)
    lowest_x, lowest_y = round(lowest_x), round(lowest_y)

    # Crop based on the lowest point and fixed offset
    dx = 300
    dy1 = 550
    dy2 = 50
    x1 = max(lowest_x - dx, 0)
    x2 = min(lowest_x + dx, img.shape[1])
    y1 = max(lowest_y - dy1, 0)
    y2 = min(lowest_y + dy2, img.shape[0])

    cropped_image = vertical_image[y1:y2, x1:x2, :]
    return vertical_image, cropped_image, rot_contour_x, rot_contour_y
