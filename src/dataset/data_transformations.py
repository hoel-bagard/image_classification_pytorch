"""Data augmentation module using albumentations."""
from functools import singledispatch
from pathlib import Path
from typing import Callable, Optional

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


if __name__ == "__main__":
    def _test_fn():
        import argparse

        import cv2

        from src.dataset.default_loader import default_load_data, default_loader
        from src.torch_utils.utils.imgs_misc import show_img

        parser = argparse.ArgumentParser(description=("Script to test the augmentation pipeline. "
                                                      "Run with 'python -m src.dataset.data_transformations <path>'."),
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("data_path", type=Path, help="Path to a folder with images.")
        parser.add_argument("--means", "-m", type=float, default=[0.485, 0.456, 0.406], nargs=3,
                            help="Mean values to use for standardization. (defaults to the ImageNet ones)")
        parser.add_argument("--stds", "-s", type=float, default=[0.229, 0.224, 0.225], nargs=3,
                            help="Mean values to use for standardization. (defaults to the ImageNet ones)")
        args = parser.parse_args()

        data_path: Path = args.data_path
        mean: tuple[float, float, float] = tuple(args.means)
        std: tuple[float, float, float] = tuple(args.stds)

        label_map = {}
        with open(data_path.parent / "classes.names") as text_file:
            for key, line in enumerate(text_file):
                label_map[key] = line.strip()

        # Load only one img/label pair since it makes visualizing the augmentations easier.
        img_paths, labels = default_loader(data_path, label_map, limit=1)
        img = default_load_data(img_paths[0])
        height, width = img.shape[:2]
        batch_size = 10
        imgs_batch = [img.copy() for _ in range(batch_size)]
        labels_batch = [labels[0] for _ in range(batch_size)]

        # Data augmentation done on cpu.
        augmentation_pipeline = albumentation_wrapper(albumentations.Compose([
            albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            albumentations.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=45, val_shift_limit=30, p=0.5),
        ]))

        common_pipeline = albumentation_wrapper(albumentations.Compose([
            albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            albumentations.Resize(*(320, 320), interpolation=cv2.INTER_LINEAR)
        ]))
        pipeline = compose_transformations((augmentation_pipeline, common_pipeline))
        denormalize_img_fn = destandardize_img(mean, std)

        aug_imgs, aug_labels = pipeline(np.asarray(imgs_batch), np.asarray(labels_batch))
        aug_imgs = denormalize_img_fn(aug_imgs)
        for img, label in zip(aug_imgs, aug_labels):
            print(label)
            show_img(img)
    _test_fn()
