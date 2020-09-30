import os
import glob
from typing import Dict

import numpy as np
import cv2


def dogs_vs_cats(data_path: str, label_map: Dict) -> np.ndarray:
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
        for image_path in glob.glob(os.path.join(data_path, f"{label_map[key]}*.jpg")):
            msg = f"Loading data {image_path}"
            print(msg + ' ' * (os.get_terminal_size()[0]-len(msg)), end="\r")
            labels.append([image_path, key])
    labels = np.asarray(labels)
    return labels


def default_loader(data_path: str, label_map: Dict, limit: int = None, load_images: bool = False) -> np.ndarray:
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
    img_types = ("*.jpg", "*.bmp")
    for key in range(len(label_map)):
        pathname = os.path.join(data_path, label_map[key], "**")
        image_paths = []
        [image_paths.extend(glob.glob(os.path.join(pathname, ext), recursive=True)) for ext in img_types]
        for i, image_path in enumerate(image_paths):
            msg = f"Loading data {image_path}"
            print(msg + ' ' * (os.get_terminal_size()[0] - len(msg)), end="\r")
            if load_images:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                labels.append([img, key])
            else:
                labels.append([image_path, key])
            if limit and i >= limit:
                break

    return np.asarray(labels)
