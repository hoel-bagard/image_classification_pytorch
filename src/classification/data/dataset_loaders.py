from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from classification.torch_utils.utils.misc import clean_print

if TYPE_CHECKING:
    from pathlib import Path

    from classification.utils.type_aliases import ImgRaw


def name_loader(
    data_path: Path,
    label_map: dict[int, str],
    limit: int | None = None,
    *,
    data_preprocessing_fn: Callable[[Path], ImgRaw] | None = None,
    return_img_paths: bool = False,
    shuffle_rng: np.random.Generator | None = None,
) -> (tuple[npt.NDArray[np.uint8], npt.NDArray[np.object_], list[Path]]
      | tuple[npt.NDArray[np.uint8], npt.NDArray[np.object_]]):
    """Load datasets where the class is in the name of the file.

    Args:
        data_path: Path to the root folder of the dataset.
        label_map: dictionarry mapping an int to a class
        limit: If given then the number of elements for each class in the dataset
               will be capped to this number
        data_preprocessing_fn: If given, then this function returns the images already loaded instead of their paths.
                               The images are loaded using this preprocessing function.
        return_img_paths: If true, then the image paths will also be returned.
        shuffle_rng: If given, then the data is shuffled once using this generator before being returned.

    Return:
        numpy array containing the images' paths and the associated label or the loaded data
    """
    if return_img_paths:
        all_paths = []

    labels, data = [], []
    for key in range(len(label_map)):
        exts = [".jpg", ".png"]
        image_paths = [p for p in data_path.rglob(f"{label_map[key]}*") if p.suffix in exts]
        if return_img_paths:
            all_paths.extend(image_paths if not limit else image_paths[:limit])

        for i, image_path in enumerate(image_paths, start=1):
            clean_print(f"Loading data {image_path}    ({i}/{len(image_paths)}) for class label_map[key]", end="\r")
            if data_preprocessing_fn is not None:
                data.append(data_preprocessing_fn(image_path))
            else:
                data.append(image_path)
            labels.append(key)
            if limit and i >= limit:
                break

    data, labels, image_paths = np.asarray(data), np.asarray(labels), np.asarray(image_paths, dtype=object)
    if shuffle_rng is not None:
        index_list = np.arange(len(labels))
        shuffle_rng.shuffle(index_list)
        data, labels, = data[index_list], labels[index_list]
        if return_img_paths:
            all_paths = all_paths[index_list]

    if return_img_paths:
        return data, labels, all_paths
    else:
        return data, labels
