import random
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def compose_transformations(transformations: list[Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]):
    """ Returns a function that applies all the given transformations"""
    def compose_transformations_fn(imgs: list[np.ndarray], labels: list[np.ndarray]):
        """ Apply transformations on a batch of data"""
        for fn in transformations:
            imgs, labels = fn(imgs, labels)
        return imgs, labels
    return compose_transformations_fn


def resize(img_size: tuple[int, int]) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """ Returns a function that resizes a batch of images.

    Args:
        img_size (int): The size to which the images should be resized

    Returns:
        callable: The function doing the resizing
    """
    def resize_fn(imgs: np.ndarray, labels: np.ndarray):
        # Hard-coded last dim, might need to be changed later
        resized_imgs = np.empty((imgs.shape[0], img_size[1], img_size[0], 3))
        for i, img in enumerate(imgs):
            resized_imgs[i] = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        return resized_imgs, labels
    return resize_fn


def crop(top: int = 0, bottom: int = 1, left: int = 0, right: int = 1):
    """ Returns a function that crops a batch of images.

    Args:
        top (int): The number of pixels to remove from the top of the images
        bottom (int): The number of pixels to remove from the bottom of the images
        left (int): The number of pixels to remove from the left of the images
        right (int): The number of pixels to remove from the right of the images

    Returns:
        callable: The function doing the cropping
    """
    def crop_fn(imgs: np.ndarray, labels: np.ndarray):
        imgs = imgs[:, top:-bottom, left:-right]
        return imgs, labels
    return crop_fn


def random_crop(reduction_factor: float = 0.9):
    """ Randomly crops image """
    def random_crop_fn(imgs: np.ndarray, labels: np.ndarray):
        """ Randomly crops a batch of data (the "same" patch is taken across all images) """
        h = random.randint(0, int(imgs.shape[1]*(1-reduction_factor))-1)
        w = random.randint(0, int(imgs.shape[2]*(1-reduction_factor))-1)
        cropped_imgs = imgs[:, h:h+int(imgs.shape[1]*reduction_factor), w:w+int(imgs.shape[2]*reduction_factor)]
        return cropped_imgs, labels
    return random_crop_fn


def vertical_flip(imgs: np.ndarray, labels: np.ndarray):
    """ Randomly flips images around the x-axis """
    for i in range(len(imgs)):
        if random.random() > 0.5:
            imgs[i] = cv2.flip(imgs[i], 0)
    return imgs, labels


def horizontal_flip(imgs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Randomly flips images around the y-axis """
    for i in range(len(imgs)):
        if random.random() > 0.5:
            imgs[i] = cv2.flip(imgs[i], 1)
    return imgs, labels


def rotate180(imgs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Randomly rotate images by 180 degrees """
    for i in range(len(imgs)):
        if random.random() > 0.5:
            imgs[i] = cv2.rotate(imgs[i], cv2.ROTATE_180)
    return imgs, labels


def to_tensor():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def to_tensor_fn(imgs: np.ndarray, labels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert ndarrays in sample to Tensors."""
        imgs = imgs.transpose((0, 3, 1, 2))
        return torch.from_numpy(imgs).to(device).float(), torch.from_numpy(labels).to(device)
    return to_tensor_fn


# TODO: Make utils that computes means and var and have those in the config
def normalize_fn(imgs: torch.Tensor, labels: torch.Tensor):
    """ Normalize a batch of images so that its values are in [0, 1] """
    return imgs/255.0, labels


def noise():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def noise_fn(imgs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Add random noise to the images """
        noise_offset = (torch.rand(imgs.shape, device=device)-0.5)*0.05
        noise_scale = (torch.rand(imgs.shape, device=device) * 0.2) + 0.9

        imgs = imgs * noise_scale + noise_offset
        imgs = torch.clamp(imgs, 0, 1)

        return imgs, labels
    return noise_fn


def padding(desired_size: tuple[int, int]):
    """ Returns a function that padds images to a the desired size

    Note: for this function to work, all the input images must have the same size.
    # TODO: do the non-gpu version with cv2.copyMakeBorder

    Args:
        desired_size (tuple): Desired width and height

    Returns
        (callable): Function that padds a batch of images to the desired size
    """
    def padding_fn(imgs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta_width = desired_size[0] - imgs.shape[3]
        delta_height = desired_size[1] - imgs.shape[2]
        padding_left, padding_right = delta_width // 2, delta_width - delta_width // 2
        padding_top, padding_bottom = delta_height // 2, delta_height - delta_height // 2
        padding = (padding_left, padding_right, padding_top, padding_bottom)
        imgs = F.pad(imgs, padding)
        return imgs, labels
    return padding_fn
