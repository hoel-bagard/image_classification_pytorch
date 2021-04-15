from pathlib import Path
from typing import (
    Callable,
    Union,
    Optional
)

import cv2
import numpy as np

from src.torch_utils.utils.misc import clean_print


def default_loader(data_path: Path,
                   label_map: dict[int, str],
                   limit: int = None,
                   load_data: bool = False,
                   data_preprocessing_fn: Optional[Callable[[Path], np.ndarray]] = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """ Default loading function for image classification.

    The data folder is expected to contain subfolders for each class, with the images inside.

    Args:
        data_path (Path): Path to the root folder of the dataset.
        label_map (dict): dictionarry mapping an int to a class
        limit (int, optional): If given then the number of elements for each class in the dataset
                               will be capped to this number
        load_data (bool): If true then this function returns the images already loaded instead of their paths.
                          The images are loaded using the preprocessing functions (they must be provided)
        data_preprocessing_fn (callable, optional): Function used to load data (imgs) from their paths.
    Return:
        numpy array containing the images' paths and the associated label or the loaded data
    """
    # TODO: benchmark if it's faster to use np.append than arrays
    labels, data = [], []
    exts = (".png", ".jpg", ".bmp")
    for key in range(len(label_map)):
        class_dir_path = data_path / label_map[key]
        image_paths = list([path for path in class_dir_path.rglob('*') if path.suffix in exts])
        for i, image_path in enumerate(image_paths, start=1):
            clean_print(f"Loading data {image_path}    ({i}/{len(image_paths)}) for class {label_map[key]}", end='\r')
            if load_data:
                data.append(data_preprocessing_fn(image_path))
            else:
                data.append(image_path)
            labels.append(key)
            if limit and i >= limit:
                break

    return np.asarray(data), np.asarray(labels)


def default_load_data(data: Union[Path, list[Path]]) -> np.ndarray:
    """Function that loads image(s) from path(s)

    Args:
        data (path): either an image path or a batch of image paths, and return the loaded image(s)
    Returns:
        Image or batch of image
    """
    if isinstance(data, Path):
        img = cv2.imread(str(data))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        imgs = []
        for image_path in data:
            imgs.append(default_load_data(image_path))
        return np.asarray(imgs)
