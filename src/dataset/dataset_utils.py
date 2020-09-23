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
            print(f"Loading data {image_path}   ", end="\r")
            labels.append([image_path, key])
    labels = np.asarray(labels)
    return labels


def default_loader(data_path: str, label_map: Dict, load_images: bool = False) -> np.ndarray:
    """
    chugai specific loading function
    Args:
        data_path: Path to the root folder of the dataset.
                   This folder is expected to contain subfolders for each class, with the images inside.
        label_map: dictionarry mapping an int to a class
        load_images: If true then this function returns the images instead of their paths
    Return:
        numpy array containing the images' paths and the associated label
    """
    labels = []
    for key in range(len(label_map)):
        img_types = ("*.jpg", "*.bmp")
        pathname = os.path.join(data_path, label_map[key], "**")
        image_paths = []
        [image_paths.extend(glob.glob(os.path.join(pathname, ext), recursive=True)) for ext in img_types]
        for image_path in image_paths:
            print(f"Loading data {image_path}   ", end="\r")
            if load_images:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                labels.append([img, key])
            else:
                labels.append([image_path, key])
    labels = np.asarray(labels)
    return labels