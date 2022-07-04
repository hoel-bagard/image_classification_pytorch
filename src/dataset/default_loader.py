from collections.abc import Iterable
from pathlib import Path
from typing import Optional


import cv2
import numpy as np
import numpy.typing as npt

from src.torch_utils.utils.misc import clean_print


def default_loader(data_path: Path,
                   label_map: dict[int, str],
                   limit: Optional[int] = None,
                   shuffle: bool = False,
                   verbose: bool = True
                   ) -> tuple[npt.NDArray[np.object_], npt.NDArray[np.int64]]:
    """Default loading function for image classification.

    The data folder is expected to contain subfolders for each class, with the images inside.

    Args:
        data_path (Path): Path to the root folder of the dataset.
        label_map (dict): dictionarry mapping an int to a class
        limit (int, optional): If given then the number of elements for each class in the dataset
                               will be capped to this number
        shuffle (bool): If true then the data is shuffled once before being returned
        verbose (bool): Verbose mode, print loading progress.

    Return:
        2 numpy arrays, one containing the images' paths and the other containing the labels.
    """
    labels: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
    data: npt.NDArray[np.object_] = np.empty(0, dtype=Path)
    exts = (".png", ".jpg", ".bmp")
    for key in range(len(label_map)):
        class_dir_path = data_path / label_map[key]
        img_paths: list[Path] = [path for path in class_dir_path.rglob('*') if path.suffix in exts]
        for i, img_path in enumerate(img_paths, start=1):
            if verbose:
                clean_print(f"Processing image {img_path.name}    ({i}/{len(img_paths)}) for class {label_map[key]}",
                            end="\r" if (i != len(img_paths) and i != limit) else "\n")
            data = np.append(data, img_path)  # type: ignore
            labels = np.append(labels, key)
            if limit and i >= limit:
                break

    data, labels = np.asarray(data), np.asarray(labels)
    if shuffle:
        index_list = np.arange(len(labels), dtype=np.int64)
        np.random.shuffle(index_list)
        data, labels, = data[index_list], labels[index_list]

    return data, labels


def default_load_data(data: Path | Iterable[Path]) -> npt.NDArray[np.uint8]:
    """Function that loads image(s) from path(s).

    Args:
        data (path): either an image path or a batch of image paths, and return the loaded image(s)

    Returns:
        Image or batch of image
    """
    if isinstance(data, Path):
        img = cv2.imread(str(data))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img  # type: ignore
    else:
        imgs = []
        for image_path in data:
            imgs.append(default_load_data(image_path))
        return np.asarray(imgs)


if __name__ == "__main__":
    def _test_fn():
        from argparse import ArgumentParser
        from src.torch_utils.utils.imgs_misc import show_img
        parser = ArgumentParser(description=("Script to test the loading function. "
                                             "Run with 'python -m src.dataset.default_loader <path>'"))
        parser.add_argument("data_path", type=Path, help="Path to a classification dataset (Train or Validation).")
        args = parser.parse_args()

        data_path: Path = args.data_path

        label_map = {}
        with open(data_path.parent / "classes.names") as text_file:
            for key, line in enumerate(text_file):
                label_map[key] = line.strip()

        data, labels = default_loader(data_path, label_map, limit=20)
        img1, _img2 = default_load_data(data[:2])
        print(labels[0])
        show_img(img1)

    _test_fn()
