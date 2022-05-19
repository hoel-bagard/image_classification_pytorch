from pathlib import Path
from typing import (
    Callable,
    Optional,
    Union
)

import numpy as np

from src.torch_utils.utils.misc import clean_print


def name_loader(data_path: Path,
                label_map: dict[int, str],
                limit: int = None,
                load_data: bool = False,
                data_preprocessing_fn: Optional[Callable[[Path], np.ndarray]] = None,
                return_img_paths: bool = False,
                shuffle: bool = False,
                ) -> Union[tuple[np.ndarray, np.ndarray, list[Path]], tuple[np.ndarray, np.ndarray]]:
    """Loading function for datasets where the class is in the name of the file.

    Args:
        data_path (Path): Path to the root folder of the dataset.
        label_map (dict): dictionarry mapping an int to a class
        limit (int, optional): If given then the number of elements for each class in the dataset
                               will be capped to this number
        load_data (bool): If true then this function returns the images already loaded instead of their paths.
                          The images are loaded using the preprocessing functions (they must be provided)
        data_preprocessing_fn (callable, optional): Function used to load data (imgs) from their paths.
        return_img_paths: If true, then the image paths will also be returned.
        shuffle: If true then the data is shuffled once before being returned

    Return:
        numpy array containing the images' paths and the associated label or the loaded data
    """
    if return_img_paths:
        all_paths = []

    labels, data = [], []
    for key in range(len(label_map)):
        exts = [".jpg", ".png"]
        image_paths = list([p for p in data_path.rglob(f"{label_map[key]}*") if p.suffix in exts])
        if return_img_paths:
            all_paths.extend(image_paths if not limit else image_paths[:limit])

        for i, image_path in enumerate(image_paths, start=1):
            clean_print(f"Loading data {image_path}    ({i}/{len(image_paths)}) for class label_map[key]", end="\r")
            if load_data:
                data.append(data_preprocessing_fn(image_path))
            else:
                data.append(image_path)
            labels.append(key)
            if limit and i >= limit:
                break

    data, labels, image_paths = np.asarray(data), np.asarray(labels), np.asarray(image_paths, dtype=object)
    if shuffle:
        index_list = np.arange(len(labels))
        np.random.shuffle(index_list)
        data, labels, = data[index_list], labels[index_list]
        if return_img_paths:
            all_paths = all_paths[index_list]

    if return_img_paths:
        return data, labels, all_paths
    else:
        return data, labels
