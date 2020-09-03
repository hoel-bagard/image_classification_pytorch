import os
import glob
from typing import Dict

import numpy as np


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


def chugai(data_path: str, label_map: Dict) -> np.ndarray:
    """
    chugai specific loading function
    Args:
        data_path: Path to the root folder of the dataset.
                   This folder is expected to contain subfolders for each class, with the images inside.
        label_map: dictionarry mapping an int to a class
    Return:
        numpy array containing the images' paths and the associated label
    """
    labels = []
    for key in range(len(label_map)):
        pathname = os.path.join(data_path, f"{label_map[key][0]}*", "**", "*.jpg")
        print(pathname)
        for image_path in glob.glob(pathname, recursive=True):
            print(f"Loading data {image_path}   ", end="\r")
            if "non_visible" in image_path:
                labels.append([image_path, 0])
            else:
                labels.append([image_path, key])
    labels = np.asarray(labels)
    return labels
