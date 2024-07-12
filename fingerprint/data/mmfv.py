import os
import abc
import torch
import random
import numpy as np
import PIL.ImageQt
from PIL import Image
from itertools import product
from natsort import natsorted
from typing import List, Dict, Union, Callable
from collections import defaultdict
from torch.utils.data import Dataset

from fingerprint.utils import crop_fingerprint, equalize_clahe

__all__ = ['MMFVBase', 'MMFVPair', 'MMFVContrastive', 'MMFVSingle', 'MMFVContrastiveClass']


class MMFVBase(Dataset, abc.ABC):

    def __init__(
            self,
            root: str,
            crop: bool,
            segment: bool,
            hist: bool,
            randomize: bool,
            subjects: str,
            fingers: List[str],
            gallery_movements: List[str],
            probe_movements: List[str],
            every_nth_frame: int,
            frames_per_video: Union[None, int],
            transforms1: Union[None, Callable],
            transforms2: Union[None, Callable],
            mode=None  # future-proof

    ):
        """

        Args:
            root:
            segment:
            randomize:
            subjects:
            fingers:
            gallery_movements:
            probe_movements:
            every_nth_frame: sample every nth frame before other sampling
            frames_per_video:
            transforms1:
            transforms2:
            mode:

        Notes:
            Data is stored as a Dictionary in the format:
                {
                    Subject-Finger: {
                        Mov: [frame1, frame2, ...],
                    }
                }
        """
        self.mode = mode
        self.crop = crop
        self.root = root
        self.segment = segment
        self.hist = hist
        self.randomize = randomize
        self.fingers = fingers
        self.gallery_movements = gallery_movements
        self.probe_movements = probe_movements
        self.every_nth_frame = every_nth_frame
        self.frames_per_video = frames_per_video
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.data1: Dict[str, Dict[str, List[str]]] = defaultdict(dict)  # gallery
        self.data2: Dict[str, Dict[str, List[str]]] = defaultdict(dict)  # probe
        self.keys = ()

        # Load selected subjects
        if subjects is not None:
            subject_path = os.path.join(root, subjects)
            subjects = _load_subjects(subject_path)
        self.subjects = subjects

        self._init_paths_and_keys()

    def _init_paths_and_keys(self):
        root = self.root
        subjects = self.subjects

        num_data1_paths = []
        num_data2_paths = []

        for idx, subject in enumerate(natsorted(os.listdir(root))):
            if subjects is not None and subject not in subjects:
                continue
            subject_path = os.path.join(root, subject)
            for sess in os.listdir(subject_path):
                sess_path = os.path.join(subject_path, sess)
                for finger in os.listdir(sess_path):
                    finger_path = os.path.join(sess_path, finger)
                    if finger not in self.fingers:
                        continue
                    for mov in os.listdir(finger_path):
                        if mov not in self.gallery_movements and mov not in self.probe_movements:
                            continue
                        mov_path = os.path.join(finger_path, mov)
                        frames1 = natsorted([i for i in os.listdir(mov_path) if i.startswith('1_')])
                        frames2 = natsorted([i for i in os.listdir(mov_path) if i.startswith('2_')])
                        frames1 = self.sample_video_frame(frames1)[::self.every_nth_frame]
                        frames2 = self.sample_video_frame(frames2)[::self.every_nth_frame]
                        all_frames = frames1 + frames2
                        paths = [os.path.join(mov_path, i) for i in all_frames]
                        key = f'{subject}-{finger}'
                        if sess == '1' and mov in self.gallery_movements:
                            self.data1[key][mov] = paths
                            num_data1_paths.append(len(paths))
                        elif sess == '2' and mov in self.probe_movements:
                            self.data2[key][mov] = paths
                            num_data2_paths.append(len(paths))

        print(f'Average paths per class in DATA1: {np.mean(num_data1_paths):.1f}')
        print(f'Average paths per class in DATA2: {np.mean(num_data2_paths):.1f}')

        self.keys = tuple(self.data1.keys())
        self.data1 = dict(self.data1)
        self.data2 = dict(self.data2)

    def sample_video_frame(self, frames: List[str]) -> List[str]:
        """Sample k frames per video"""

        # randomize even if no sampling of frames

        k = self.frames_per_video
        if k is None:
            k = len(frames)

        if self.randomize:
            k = min(k, len(frames))
            sampled_frames = random.sample(frames, k=k)
        else:
            sampled_frames = frames[:self.frames_per_video]

        return sampled_frames

    def _get_image(self, path) -> PIL.Image.Image:
        img = Image.open(path)
        img_np = np.array(img)

        # default is original image if crop not set or not found
        cropped_img = img_np

        if self.crop:
            cropped_img = crop_fingerprint(img_np, segment=self.segment, channels='RGB')
            h, w, _ = cropped_img.shape
            if h == 0 or w == 0:
                print(f'Finger contour not found for image: {path}')
                cropped_img = img_np

        # equalize histogram
        if self.hist:
            cropped_img = equalize_clahe(cropped_img, channels='RGB')

        # back to pil image format
        cropped_img = Image.fromarray(cropped_img)

        return cropped_img

    def __len__(self):
        return len(self.data1)

    def __str__(self):
        result = (f'{self.__class__.__name__}\n Root: {self.root}\n Length: {len(self)}\n '
                  f'Crop: {self.crop}\n Segment: {self.segment}\n Randomize: {self.randomize}\n '
                  f'Fingers: {self.fingers}\n Gallery Movements: {self.gallery_movements}\n '
                  f'Probe Movements: {self.probe_movements}\n Frames per Video: {self.frames_per_video}')
        return result


class MMFVContrastive(MMFVBase):
    """
        Single pair of images per iteration.
        Movement from which image is selected changes in next epoch.
    """

    def __init__(self,
                 root,
                 mode=None,
                 crop=True,
                 segment=True,
                 hist=True,
                 randomize=True,
                 subjects='train.txt',
                 fingers=('f1', 'f2', 'f3', 'f4'),
                 gallery_movements=('Roll', 'Pitch', 'Yaw'),
                 probe_movements=('Roll', 'Pitch', 'Yaw'),
                 every_nth_frame=1,
                 frames_per_video=None,
                 transforms1=None,
                 transforms2=None,
                 ):
        super().__init__(
            root,
            crop,
            segment,
            hist,
            randomize,
            subjects,
            fingers,
            gallery_movements,
            probe_movements,
            every_nth_frame,
            frames_per_video,
            transforms1,
            transforms2,
            mode
        )
        # Information to vary movement pairs next epoch
        self.mov_pairs = tuple(product(gallery_movements, probe_movements))  # Track which movement pair to sample
        self.mov_pair_idx = {}
        for key in self.data1:
            self.mov_pair_idx[key] = 0

    def __getitem__(self, idx):
        key = self.keys[idx]
        num_mov_pairs = len(self.mov_pairs)
        mov_pair_idx = self.mov_pair_idx[key] % num_mov_pairs
        data1_mov, data2_mov = self.mov_pairs[mov_pair_idx]
        path1 = random.choice(self.data1[key][data1_mov])
        path2 = random.choice(self.data2[key][data2_mov])
        # Update movement pair index
        self.mov_pair_idx[key] = (mov_pair_idx + 1) % num_mov_pairs
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


class MMFVPair(MMFVBase):
    """
        Return all images belonging to a single class per iteration.
    """

    def __init__(self,
                 root,
                 crop=True,
                 segment=True,
                 hist=True,
                 randomize=True,
                 subjects='train.txt',
                 fingers=('f1', 'f2', 'f3', 'f4'),
                 gallery_movements=('Roll', 'Pitch', 'Yaw'),
                 probe_movements=('Roll', 'Pitch', 'Yaw'),
                 every_nth_frame=1,
                 frames_per_video=1,
                 transforms1=None,
                 transforms2=None,
                 mode=None
                 ):
        super().__init__(
            root,
            crop,
            segment,
            hist,
            randomize,
            subjects,
            fingers,
            gallery_movements,
            probe_movements,
            every_nth_frame,
            frames_per_video,
            transforms1,
            transforms2,
            mode
        )

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


class MMFVSingle(MMFVBase):
    """
        Single image per iteration.
    """

    def __init__(self,
                 root,
                 crop=False,
                 segment=True,
                 hist=True,
                 randomize=False,
                 subjects='train.txt',
                 fingers=('f1', 'f2', 'f3', 'f4'),
                 gallery_movements=('Roll', 'Pitch', 'Yaw'),
                 probe_movements=('Roll', 'Pitch', 'Yaw'),
                 every_nth_frame=1,
                 frames_per_video=1,
                 transforms1=None,
                 transforms2=None,
                 mode=None
                 ):
        super().__init__(
            root,
            crop,
            segment,
            hist,
            randomize,
            subjects,
            fingers,
            gallery_movements,
            probe_movements,
            every_nth_frame,
            frames_per_video,
            transforms1,
            transforms2,
            mode
        )
        self._flatten_data()

    def _flatten_data(self):
        """Convert subject-movement-path data dicts into a linear list of dicts."""
        data_list = []
        datas = [self.data1, self.data2]
        locations = ['data1', 'data2']

        for idx in range(2):  # Gallery/Probe
            data_dict = datas[idx]
            loc = locations[idx]
            for class_, s_dict in data_dict.items():
                for mov, paths in s_dict.items():
                    for path in paths:
                        data_list.append(
                            {
                                'path': path,
                                'location': loc,
                                'class': class_,
                                'movement': mov
                            }
                        )
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        path = data['path']
        img = self._get_image(path)
        if data['location'] == 'data1' and self.transforms1 is not None:
            img = self.transforms1(img)
        elif data['location'] == 'data2' and self.transforms2 is not None:
            img = self.transforms2(img)
        result = {}
        for k, v in data.items():
            result[k] = v
        result['image'] = img

        return result


class MMFVContrastiveClass(MMFVBase):
    """
        Single pair of images per iteration.
        All possible movement pairs are included in one batch.
    """

    def __init__(self,
                 root,
                 mode=None,
                 crop=True,
                 segment=True,
                 hist=True,
                 randomize=True,
                 subjects='train.txt',
                 fingers=('f1', 'f2', 'f3', 'f4'),
                 gallery_movements=('Roll', 'Pitch', 'Yaw'),
                 probe_movements=('Roll', 'Pitch', 'Yaw'),
                 every_nth_frame=1,
                 frames_per_video=None,
                 transforms1=None,
                 transforms2=None,
                 ):
        super().__init__(
            root,
            crop,
            segment,
            hist,
            randomize,
            subjects,
            fingers,
            gallery_movements,
            probe_movements,
            every_nth_frame,
            frames_per_video,
            transforms1,
            transforms2,
            mode
        )
        self._create_pairs_data()

    def _create_pairs_data(self):
        mov_pairs = tuple(product(self.gallery_movements, self.probe_movements))
        data = {}
        count = 0
        for label, key in enumerate(self.keys):
            for data1_mov, data2_mov in mov_pairs:
                data[count] = {
                    'key': key,
                    'mov1': data1_mov,
                    'mov2': data2_mov,
                    'label': label,
                }
                count += 1
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        key, label = data['key'], data['label']
        data1_mov, data2_mov = data['mov1'], data['mov2']
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
            'mov2': data2_mov,
            'label': label
        }
        return result

    def __str__(self):
        result = (f'{self.__class__.__name__}\n Root: {self.root}\n Length: {len(self)}\n '
                  f'Crop: {self.crop}\n Segment: {self.segment}\n Randomize: {self.randomize}\n '
                  f'Fingers: {self.fingers}\n Gallery Movements: {self.gallery_movements}\n '
                  f'Probe Movements: {self.probe_movements}\n Frames per Video: {self.frames_per_video}')
        return result


def _load_subjects(path):
    with open(os.path.join(path)) as f:
        content = f.readlines()
        subjects = [i.strip() for i in content]
    return subjects
