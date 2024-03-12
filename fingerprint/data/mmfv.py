import os
import cv2
import random
import numpy as np
import PIL.ImageQt
from PIL import Image
from itertools import product
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import Dataset


class MMFVContrastive(Dataset):
    def __init__(self,
                 root,
                 transforms1=None,
                 transforms2=None,
                 subjects='train.txt',
                 fingers=('f1', 'f2', 'f3', 'f4'),
                 movements=('Roll', 'Pitch', 'Yaw'),
                 ):
        self.root = root
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
        num_mov_pairs = len(self.movement_pairs)
        previously_sampled = self.movement_index[key] % num_mov_pairs
        mov_idx = previously_sampled % num_mov_pairs
        self.movement_index[key] = (previously_sampled + 1) % num_mov_pairs
        data1_mov, data2_mov = self.movement_pairs[mov_idx]
        path1 = random.choice(self.data1[key][data1_mov])
        path2 = random.choice(self.data2[key][data2_mov])

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

    @staticmethod
    def _get_image(path) -> PIL.Image.Image:
        img = Image.open(path)
        cropped_img = crop_fingerprint(np.array(img), channels='RGB')
        h, w, _ = cropped_img.shape
        # use original image if finger not found
        if h == 0 or w == 0:
            print(f'Finger contour not found for image: {path}')
            crop_fingerprint(np.array(img), channels='RGB', verbose=True, viz=True)
            cropped_img = img
        else:
            cropped_img = Image.fromarray(cropped_img)
        return cropped_img

    def __str__(self):
        result = (f'MMFV Contrastive\n Root: {self.root}\n Length: {len(self)}\n '
                  f'Movements: {self.movements}\n Fingers: {self.fingers}')
        return result


def rotate_contour(contour, m):
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    x2 = m[0, 0] * x + m[0, 1] * y + m[0, 2]
    y2 = m[1, 0] * x + m[1, 1] * y + m[1, 2]
    return x2, y2


def crop_fingerprint(img: np.ndarray, channels='RGB', verbose=False, viz=False):
    if channels == 'BGR':
        color_mode = cv2.COLOR_BGR2GRAY
    else:
        color_mode = cv2.COLOR_RGB2GRAY
    img_gray = cv2.cvtColor(img, color_mode)

    # Get largest contour
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(~thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit ellipse on largest contour
    (x, y), (MA, ma), angle = cv2.fitEllipse(largest_contour)
    if angle > 90:
        angle = -(180 - angle)

    vertical_image, result, rot_contour_x, rot_contour_y = rotate_and_crop_with_contour(
        img, x, y, angle, largest_contour)

    # Fit rectangle if ellipse does not work
    if result.shape[0] == 0 or result.shape[1] == 0:
        if verbose:
            print('Using rectangle for contour rotation estimation')
        (x, y), (MA, ma), angle = cv2.minAreaRect(largest_contour)
        if angle > 90:
            angle = -(180 - angle)
        vertical_image, result, rot_contour_x, rot_contour_y = rotate_and_crop_with_contour(
            img, x, y, angle, largest_contour)

    if viz:
        o1 = img.copy()
        cv2.drawContours(o1, contours, -1, (0, 0, 255), 5)

        o2 = img.copy()
        cv2.drawContours(o2, [largest_contour], 0, (255, 0, 0), 10)

        o3 = vertical_image.copy()
        rot_largest_contour = np.zeros_like(largest_contour)
        rot_largest_contour[:, 0, 0] = rot_contour_x
        rot_largest_contour[:, 0, 1] = rot_contour_y
        cv2.drawContours(o3, [rot_largest_contour], 0, (255, 0, 0), 10)

        fig, ax = plt.subplots(1, 4, figsize=(20, 10))
        ax[0].imshow(o1[:, :, ::-1])
        ax[1].imshow(o2[:, :, ::-1])
        ax[2].imshow(o3[:, :, ::-1])
        # ax[3].imshow(result[:, :, ::-1])

        plt.show()
        plt.close()

    return result


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
