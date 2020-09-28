import random

import cv2
import numpy as np
import torch


class Crop(object):
    """ Crop the image """

    def __init__(self, top: int = 0, bottom: int = 1, left: int = 0, right: int = 1):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __call__(self, sample):
        img, label = sample["img"], sample["label"]
        cropped_img = img[self.top:-self.bottom, self.left:-self.right]
        return {"img": cropped_img, "label": label}


class RandomCrop(object):
    """ Random crops image """

    def __init__(self, size_reduction_factor: int = 0.9):
        self.size = size_reduction_factor

    def __call__(self, sample):
        img, label = sample["img"], sample["label"]
        h = random.randint(0, int(img.shape[0]*(1-self.size))-1)
        w = random.randint(0, int(img.shape[1]*(1-self.size))-1)
        cropped_img = img[h:h+int(img.shape[0]*self.size), w:w+int(img.shape[1]*self.size)]

        return {"img": cropped_img, "label": label}


class Resize(object):
    """ Resize the image in a sample to a given size. """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        img = cv2.resize(img, (self.width, self.height))
        return {'img': img, 'label': label}


class Normalize(object):
    """ Normalize the image so that its values are in [0, 1] """

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        return {'img': img/255.0, 'label': label}


class VerticalFlip(object):
    """ Randomly flips the img around the x-axis """

    def __call__(self, sample):
        img, label = sample["img"], sample["label"]
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
        return {"img": img, "label": label}


class HorizontalFlip(object):
    """ Randomly flips the img around the y-axis """

    def __call__(self, sample):
        img, label = sample["img"], sample["label"]
        if random.random() > 0.5:
            img = cv2.flip(img, 1)

        return {"img": img, "label": label}


class Rotate180(object):
    """ Randomly rotate the image by 180 degrees """

    def __call__(self, sample):
        img, label = sample["img"], sample["label"]
        if random.random() > 0.5:
            img = cv2.rotate(img, cv2.ROTATE_180)
        return {"img": img, "label": label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img, label = sample['img'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        img = img.transpose((2, 0, 1))
        return {'img': torch.from_numpy(img),
                'label': torch.from_numpy(np.asarray(label))}


class Noise(object):
    """ Add random noise to the image """

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        noise_offset = (torch.rand(img.shape)-0.5)*0.05
        noise_scale = (torch.rand(img.shape) * 0.2) + 0.9

        img = img * noise_scale + noise_offset
        img = torch.clamp(img, 0, 1)

        return {'img': img, 'label': label}
