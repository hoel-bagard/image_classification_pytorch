"""Data augmentation module using albumentations."""
from __future__ import annotations

import typing
from functools import singledispatch
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
from einops import rearrange

from classification.utils.type_aliases import (
    ImgArray,
    ImgArrayOrTensor,
    ImgRaw,
    ImgStandardized,
    LabelArray,
    LabelDtype,
    StandardizedImgDType,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

    import albumentations


def albumentation_img_wrapper(
    transform: albumentations.Compose | albumentations.ImageOnlyTransform,
) -> Callable[[ImgArray], ImgArray]:
    """Return a function that applies the albumentations transforms to an image.

    This is done to make the function more general, since albumentation is keyword argument only.
    """

    def albumentation_transform_fn(img: ImgArray) -> ImgArray:
        return transform(image=img)["image"]

    return albumentation_transform_fn


def albumentation_batch_wrapper(
    transform: albumentations.Compose | albumentations.ImageOnlyTransform,
) -> Callable[[ImgArray, LabelArray], tuple[ImgArray, LabelArray]]:
    """Return a function that applies the albumentations transforms to a batch."""

    def albumentation_transform_fn(imgs: ImgArray, labels: LabelArray) -> tuple[ImgArray, LabelArray]:
        """Apply transformations on a batch of data."""
        out_sizes = transform(image=imgs[0])["image"].shape
        out_imgs = np.empty((imgs.shape[0], *out_sizes), dtype=StandardizedImgDType)
        for i, img in enumerate(imgs):
            transformed = transform(image=img)
            out_imgs[i] = transformed["image"]
        return out_imgs, labels

    return albumentation_transform_fn


def compose_transformations(
    transformations: Iterable[Callable[[ImgArray, LabelArray], tuple[ImgArray, LabelArray]]],
) -> Callable[[ImgArray, LabelArray], tuple[ImgArray, LabelArray]]:
    """Return a function that applies all the given transformations."""

    def compose_transformations_fn(imgs: ImgArray, labels: LabelArray) -> tuple[ImgArray, LabelArray]:
        """Apply transformations on a batch of data."""
        for fn in transformations:
            imgs, labels = fn(imgs, labels)
        return imgs, labels

    return compose_transformations_fn


def to_tensor() -> Callable[[ImgArray, npt.NDArray[LabelDtype]], tuple[torch.Tensor, torch.Tensor]]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def to_tensor_fn(imgs: ImgArray, labels: npt.NDArray[LabelDtype]) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert ndarrays in sample to Tensors."""
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        imgs = imgs.transpose((0, 3, 1, 2))
        return torch.from_numpy(imgs).float().to(device), torch.from_numpy(labels).to(device)

    return to_tensor_fn


def destandardize_img(
    img_mean: tuple[float, float, float],
    img_std: tuple[float, float, float],
) -> Callable[[ImgArrayOrTensor, ImgArrayOrTensor], ImgRaw]:
    """Create a function to undo the standardization process on a batch of images.

    Notes: The single dispatch thing is because I was bored and mypy was complaining when using an if isinstance().

    Args:
        img_mean: The mean values that were used to standardize the image.
        img_std: The std values that were used to standardize the image.

    Returns:
        The function to destandardize a batch of images.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean_array = np.asarray(img_mean, dtype=StandardizedImgDType)
    std_array = np.asarray(img_std, dtype=StandardizedImgDType)
    mean_tensor = torch.Tensor(img_mean).to(device)
    std_tensor = torch.Tensor(img_std).to(device)

    @singledispatch
    def destandardize_fn(imgs) -> typing.NoReturn:  #   noqa: ANN001 pyright: ignore
        # TODO: 3.11 | Use typing.Never if switching to 3.11
        msg = f"Wrong data type: {type(imgs)}"
        raise NotImplementedError(msg)

    @destandardize_fn.register
    def destandardize_numpy(imgs: np.ndarray) -> ImgRaw:  # pyright: ignore[reportMissingTypeArgument]
        imgs = typing.cast(ImgStandardized, imgs)
        imgs = ((imgs * std_array + mean_array) * 255.0).astype(np.uint8)
        return imgs

    @destandardize_fn.register
    def destandardize_tensors(imgs: torch.Tensor) -> ImgRaw:
        # Destandardize the images
        imgs = rearrange(imgs, "b c w h -> b w h c")
        imgs = (imgs * std_tensor + mean_tensor) * 255
        destandardized_imgs = imgs.cpu().detach().numpy().astype(np.uint8)
        return destandardized_imgs

    return destandardize_fn
