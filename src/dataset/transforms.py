import cv2
import numpy as np
import torch


class Resize(object):
    """ Resize the image in a sample to a given size. """

    def __init__(self, output_size: int):
        self.output_size = output_size

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        img = cv2.resize(img, (self.output_size, self.output_size))
        return {'img': img, 'label': label}


class Normalize(object):
    """ Normalize the image so that its values are in [0, 1] """

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        return {'img': img/255.0, 'label': label}


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
