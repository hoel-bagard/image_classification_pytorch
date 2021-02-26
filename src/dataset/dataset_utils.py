from pathlib import Path
from typing import Union

import numpy as np
import cv2

from src.torch_utils.utils.misc import clean_print


def dogs_vs_cats(data_path: Path, label_map: dict[int, str]) -> np.ndarray:
    """
    dogs-vs-cats loading function
    Args:
        data_path: path to the folder containing the images
        label_map: dictionarry mapping an int to a class
    Return:
        numpy array containing the images' paths and the associated label
    """
    labels = []
    for key in range(len(label_map)):
        for image_path in data_path.glob(f"{label_map[key]}*.jpg"):
            clean_print(f"Loading data {image_path}")
            labels.append([image_path, key])
    labels = np.asarray(labels)
    return labels


def default_loader(data_path: Path, label_map: dict[int, str],
                   limit: int = None, load_images: bool = False) -> np.ndarray:
    """
    Args:
        data_path: Path to the root folder of the dataset.
                   This folder is expected to contain subfolders for each class, with the images inside.
        label_map: dictionarry mapping an int to a class
        limit (int, optional): If given then the number of elements for each class in the dataset
                               will be capped to this number
        load_images: If true then this function returns the images instead of their paths
    Return:
        numpy array containing the images' paths and the associated label
    """
    labels = []
    exts = ("*.png", "*.jpg", "*.bmp")
    for key in range(len(label_map)):
        class_dir_path = data_path / label_map[key]
        image_paths = list([path for path in class_dir_path.rglob('*') if path.suffix in exts])
        for i, image_path in enumerate(image_paths):
            clean_print(f"Loading data {image_path}    ({i}/{len(image_paths)})")
            if load_images:
                img = default_load_data(image_path)
                labels.append([img, key])
            else:
                labels.append([image_path, key])
            if limit and i >= limit:
                break

    return np.asarray(labels)


def default_load_data(data: Union[Path, list[Path]]) -> np.ndarray:
    """
    Function that loads image(s) from path(s)
    Args:
        data: either an image path or a batch of image paths, and return the loaded image(s)
    """
    if type(data) == Path:
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        imgs = []
        for image_path in data:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        return imgs
