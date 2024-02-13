from __future__ import annotations

import typing
from pathlib import Path
from typing import overload

import cv2
import numpy as np
import numpy.typing as npt

from classification.torch_utils.utils.misc import clean_print
from classification.utils.type_aliases import ImgArrayT, ImgRaw, LabelArray, LabelDtype

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def default_loader(
    data_path: Path,
    label_map: dict[int, str],
    limit: int | None = None,
    shuffle_rng: np.random.Generator | None = None,
    *,
    verbose: bool = True
) -> tuple[npt.NDArray[np.object_], LabelArray]:
    """Load datasets where the class is given by the parent folder.

    The data folder is expected to contain subfolders for each class, with the images inside.

    Args:
        data_path: Path to the root folder of the dataset.
        label_map: dictionarry mapping an int to a class
        limit: If given then the number of elements for each class in the dataset will be capped to this number.
        shuffle_rng: If given, then the data is shuffled once using this generator before being returned.
        verbose: Verbose mode, print loading progress.

    Return:
        2 numpy arrays, one containing the images' paths and the other containing the labels.
    """
    labels: LabelArray = np.empty(0, dtype=LabelDtype)
    data: npt.NDArray[np.object_] = np.empty(0, dtype=Path)
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    for key in range(len(label_map)):
        class_dir_path = data_path / label_map[key]
        img_paths: list[Path] = [path for path in class_dir_path.rglob("*") if path.suffix.lower() in exts]
        for i, img_path in enumerate(img_paths, start=1):
            if verbose:
                clean_print(
                    f"Processing image {img_path.name}    ({i}/{len(img_paths)}) for class {label_map[key]}",
                    end="\r" if (i != len(img_paths) and i != limit) else "\n",
                )
            data = np.append(data, img_path)  # type: ignore[reportGeneralTypeIssues]
            labels = np.append(labels, key)
            if limit and i >= limit:
                break

    data, labels = np.asarray(data), np.asarray(labels)
    if shuffle_rng is not None:
        index_list = np.arange(len(labels), dtype=np.int64)
        shuffle_rng.shuffle(index_list)
        data, labels, = data[index_list], labels[index_list]

    return data, labels


@overload
def default_load_data(data: Path | Iterable[Path], preprocessing_pipeline: None = None) -> ImgRaw:
    ...

@overload
def default_load_data(data: Path | Iterable[Path], preprocessing_pipeline: Callable[[ImgRaw], ImgArrayT]) -> ImgArrayT:
    ...

def default_load_data(
    data: Path | Iterable[Path],
    preprocessing_pipeline: Callable[[ImgRaw], ImgArrayT] | None = None,
) -> ImgRaw | ImgArrayT:
    """Load image(s) from path(s).

    Args:
        data: either an image path or a batch of image paths, and return the loaded image(s)
        preprocessing_pipeline: Function used to load/pre-process the images.

    Returns:
        Image or batch of image
    """
    if isinstance(data, Path):
        img = cv2.imread(str(data))
        img = typing.cast(ImgRaw, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if preprocessing_pipeline is not None:
            img = preprocessing_pipeline(img)
        return img
    else:
        imgs = np.asarray([
            default_load_data(image_path, preprocessing_pipeline)
            for image_path in data
        ])
        return imgs


if __name__ == "__main__":
    def _test_fn() -> None:
        from argparse import ArgumentParser

        from classification.torch_utils.utils.imgs_misc import show_img
        parser = ArgumentParser(description=("Script to test the loading function. "
                                             "Run with 'python -m classification.dataset.default_loader <path>'"))
        parser.add_argument("data_path", type=Path, help="Path to a classification dataset (Train or Validation).")
        args = parser.parse_args()

        data_path: Path = args.data_path

        label_map: dict[int, str] = {}
        with (data_path.parent / "classes.names").open("r") as text_file:
            for key, line in enumerate(text_file):
                label_map[key] = line.strip()

        data, labels = default_loader(data_path, label_map, limit=20)
        img1, _img2 = default_load_data(data[:2])
        print(labels[0])
        show_img(img1)

    _test_fn()
