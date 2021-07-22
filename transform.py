import random
import os

import numpy as np
import cv2
import torch

from torchvision.transforms import functional as F

from utils import (
    generate_shiftscalerotate_matrix,
)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)

        return img, target

    def __repr__(self):
        format_str = self.__class__.__name__ + '('
        for t in self.transforms:
            format_str += '\n'
            format_str += f'    {t}'
        format_str += '\n)'

        return format_str

class Resize:
    def __init__(self, dst_width, dst_height, dst_K):
        self.dst_width = dst_width
        self.dst_height = dst_height
        self.dst_K = dst_K

    def __call__(self, img, target):
        M = np.matmul(self.dst_K, np.linalg.inv(target.K))
        # 
        img = cv2.warpAffine(img, M[:2], (self.dst_width, self.dst_height), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
        target = target.transform(M, self.dst_K, self.dst_width, self.dst_height)
        return img, target

class RandomShiftScaleRotate:
    def __init__(self, shift_limit, scale_limit, rotate_limit, dst_width, dst_height, dst_K):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        # 
        self.dst_width = dst_width
        self.dst_height = dst_height
        self.dst_K = dst_K

    def __call__(self, img, target):
        M = generate_shiftscalerotate_matrix(
                self.shift_limit, self.scale_limit, self.rotate_limit, 
                self.dst_width, self.dst_height
            )
        img = cv2.warpAffine(img, M[:2], (self.dst_width, self.dst_height), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
        target = target.transform(M, self.dst_K, self.dst_width, self.dst_height)
        return img, target

class RandomHSV:
    def __init__(self, h_ratio, s_ratio, v_ratio):
        self.h_ratio = h_ratio
        self.s_ratio = s_ratio
        self.v_ratio = v_ratio
    def __call__(self, img, target):
        img = distort_hsv(img, self.h_ratio, self.s_ratio, self.v_ratio)
        return img, target

class RandomNoise:
    def __init__(self, noise_ratio):
        self.noise_ratio = noise_ratio
    def __call__(self, img, target):
        img = distort_noise(img, self.noise_ratio)
        return img, target

class RandomSmooth:
    def __init__(self, smooth_ratio):
        self.smooth_ratio = smooth_ratio
    def __call__(self, img, target):
        img = distort_smooth(img, self.smooth_ratio)
        return img, target

class ToTensor:
    def __call__(self, img, target):
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        target = target.to_tensor()
        return img, target
    
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
        img = img - np.array(self.mean).reshape(1,1,3)
        img = img / np.array(self.std).reshape(1,1,3)
        return img, target
