"""Data augmentation module using albumentations."""
from functools import singledispatch
from pathlib import Path
from typing import Callable

import albumentations
import numpy as np
import torch
from einops import rearrange

NumpyOrTensor = np.ndarray | torch.Tensor


def albumentation_wrapper(transform: albumentations.Compose) -> Callable[[np.ndarray, np.ndarray],
                                                                         tuple[np.ndarray, np.ndarray]]:
    """Returns a function that applies the albumentations transforms to a batch."""
    def albumentation_transform_fn(imgs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply transformations on a batch of data."""
        out_sizes = transform(image=imgs[0, ..., :3])["image"].shape[:2]
        out_imgs = np.empty((imgs.shape[0], *out_sizes, imgs.shape[-1]), dtype=np.float32)
        for i, img in enumerate(imgs):
            # The depth map is treated as a mask (kept for the resize operation).
            if img.shape[-1] == 4:
                transformed = transform(image=img[..., :3], mask=img[..., 3:])
                out_imgs[i] = np.concatenate((transformed["image"], transformed["mask"]), axis=-1)
            else:
                transformed = transform(image=img)
                out_imgs[i] = transformed["image"]
        return out_imgs, labels
    return albumentation_transform_fn


def compose_transformations(transformations: list[Callable[[NumpyOrTensor, NumpyOrTensor],
                                                           tuple[NumpyOrTensor, NumpyOrTensor]]]):
    """Returns a function that applies all the given transformations."""
    def compose_transformations_fn(imgs: NumpyOrTensor, labels: NumpyOrTensor):
        """Apply transformations on a batch of data."""
        for fn in transformations:
            imgs, labels = fn(imgs, labels)
        return imgs, labels
    return compose_transformations_fn


def to_tensor():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def to_tensor_fn(imgs: np.ndarray, labels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert ndarrays in sample to Tensors."""
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        imgs = imgs.transpose((0, 3, 1, 2))
        return torch.from_numpy(imgs).float().to(device), torch.from_numpy(labels).to(device)  # TODO: remove float
    return to_tensor_fn


def destandardize_img(img_mean: tuple[float, float, float],
                      img_std: tuple[float, float, float],
                      ) -> Callable[[NumpyOrTensor, NumpyOrTensor], np.ndarray]:
    """Create a function to undo the standardization process on a batch of images.

    Notes: The singe dispatch thing is because I was bored and mypy was complaining when using an if isinstance().

    Args:
        img_mean (tuple): The mean values that were used to standardize the image.
        img_std (tuple): The std values that were used to standardize the image.

    Returns:
        The function to destandardize a batch of images.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean_array = np.asarray(img_mean)
    std_array = np.asarray(img_std)
    mean_tensor = torch.Tensor(img_mean).to(device)
    std_tensor = torch.Tensor(img_std).to(device)

    @singledispatch
    def destandardize_fn(imgs):
        raise NotImplementedError(f"Wrong data type: {type(imgs)}")

    @destandardize_fn.register
    def denormalize_numpy(imgs: np.ndarray) -> np.ndarray:
        imgs = (imgs * std_array + mean_array) * 255
        imgs = imgs.astype(np.uint8)
        return imgs

    @destandardize_fn.register
    def denormalize_tensors(imgs: torch.Tensor) -> np.ndarray:
        # Destandardize the images
        imgs = rearrange(imgs, "b c w h -> b w h c")
        imgs = (imgs * std_tensor + mean_tensor) * 255
        imgs = imgs.cpu().detach().numpy().astype(np.uint8)
        return imgs

    return destandardize_fn
