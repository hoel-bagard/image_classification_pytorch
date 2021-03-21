from pathlib import Path
from typing import (
    Callable,
    Optional
)

import numpy as np

from src.torch_utils.utils.misc import clean_print


def dog_vs_cat_loader(data_path: Path,
                      label_map: dict[int, str],
                      limit: int = None,
                      load_data: bool = False,
                      data_preprocessing_fn: Optional[Callable[[Path], np.ndarray]] = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """ Loading function for the dog vs cat dataset

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
    labels, data = [], []
    for key in range(len(label_map)):
        image_paths = list([path for path in data_path.glob(f"{label_map[key]}*.jpg")])
        for i, image_path in enumerate(image_paths, start=1):
            clean_print(f"Loading data {image_path}    ({i}/{len(image_paths)}) for class label_map[key]", end="\r")
            if load_data:
                data.append(data_preprocessing_fn(image_path))
            else:
                data.append(image_path)
            labels.append(key)
            if limit and i >= limit:
                break

    return np.asarray(data), np.asarray(labels)
