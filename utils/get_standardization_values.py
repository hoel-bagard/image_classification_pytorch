import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt


def get_img_mean_std(img_path: Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the means and stds of the given images.

    Args:
    ----
        img_path (Path): Path to the image whose means and stds should be computed.

    Returns:
    -------
        The means and stds of the images.

    """
    # Get the mean and std for the RGB images
    img = cv2.imread(str(img_path))
    img_mean = np.mean(img, axis=(0, 1))
    img_std = np.std(img, axis=(0, 1))

    return img_mean, img_std


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description=("Script to get the means and stds for the images of a dataset. "
                                         "Run with 'python utils/get_norm_values <path>'."))
    parser.add_argument("data_path", type=Path, help="Path to the dataset.")
    args = parser.parse_args()

    data_path: Path = args.data_path

    np.set_printoptions(precision=3)

    exts = (".png", ".jpg", ".bmp")
    img_paths_list = [path for path in data_path.rglob("*") if path.suffix in exts]
    nb_imgs = len(img_paths_list)
    means, stds = np.zeros(3), np.zeros(3)
    with Pool(processes=int(os.cpu_count() * 0.8)) as pool:
        for i, (img_mean, img_std) in enumerate(pool.imap(get_img_mean_std, img_paths_list, chunksize=10), start=1):
            msg = f"Processing status: ({i}/{nb_imgs})"
            print(msg + " " * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)),
                  end="\r" if i != nb_imgs else "\n", flush=True)
            means += img_mean
            stds += img_std

    means = means[::-1] / (255*nb_imgs)  # [::-1] to convert to RGB
    stds = stds[::-1] / (255*nb_imgs)
    print(f"Result (RGB):\n\tMeans: {means}, \n\tStds: {stds}")
