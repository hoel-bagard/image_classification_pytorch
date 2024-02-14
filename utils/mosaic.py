from __future__ import annotations

import math
import os
import shutil
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


def mosaic_worker(args: tuple[Path, Path, tuple[int, int, int, int], bool]) -> Path:  # noqa: D417
    """Worker in charge of turning an image into a mosaic.

    This function only handle the case where the image's width is larger than its height (for now at least)

    Args:
        img_path (Path): Path to the image to process
        output_path (Path): Folder to where the new image will be saved
        crop (tuple, optional): (left, right, top, bottom), if not None then image will be cropped by the given values.
        padding (bool, optional): If true then the mosaic image will be a square will black padding at the bottom

    Return:
        output_file_path: Path of the saved image.

    """
    img_path, output_path, crop, use_padding = args
    output_file_path = output_path / img_path.relative_to(output_path.parent)

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

    if crop is not None:
        left, right, top, bottom = crop
        img = img[top:-bottom, left:-right]  # Doesn't work for top=bottom=0

    height, width, _ = img.shape
    if width % height != 0:
        msg = "The image cannot be cleanly cut into a mosaic, maybe try cropping it."
        raise ValueError(msg)

    ratio = width // height
    # if ratio != math.isqrt(ratio) ** 2:
    #     print("The image's ratio does not lead to a square number of mosaic tiles,"
    #           "black tiles will be used to complete the image.")

    nb_tiles_side = math.ceil(math.sqrt(width / height))  # Width of the new (square) image in number of tiles

    if use_padding:
        # zeros and not empty to have black tiles by default
        mosaic_img = np.zeros((nb_tiles_side * height, nb_tiles_side * height, 3))
    else:
        mosaic_img = np.empty((math.ceil(ratio / nb_tiles_side) * height, nb_tiles_side * height, 3))

    for tile_idx in range(ratio):
        i, j = (tile_idx // nb_tiles_side) * height, (tile_idx % nb_tiles_side) * height
        mosaic_img[i: i + height, j: j + height] = img[:, tile_idx * height: (tile_idx + 1) * height]

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file_path), mosaic_img)
    return output_file_path


def main() -> None:
    parser = ArgumentParser(
        "Converts a long, rectangular image into a square-ish image by turning it into a mosaic."
        "Saves the data in a mosaic_img folder, in the dataset's parent folder"
    )
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("--crop", "--c", type=int, nargs=4, help="If cropping the images, (left, right, top, bottom)")
    parser.add_argument("--use_padding", "--p", action="store_true", help="If true then pad the images to have squares")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = data_path.parent / "mosaic"
    output_path.mkdir(parents=True, exist_ok=True)

    # Get a list of all the images
    exts = [".jpg", ".png"]
    file_list = [p for p in data_path.rglob("*") if p.suffix in exts]
    nb_imgs = len(file_list)

    mp_args = [(img_path, output_path, args.crop, args.use_padding) for img_path in file_list]
    nb_images_processed = 0  # Use to count the number of good / bad samples
    with Pool(processes=int(nb_cpus * 0.8) if (nb_cpus := os.cpu_count()) is not None else 1) as pool:
        for _result in pool.imap(mosaic_worker, mp_args, chunksize=10):
            nb_images_processed += 1
            msg = f"Processing status: ({nb_images_processed}/{nb_imgs})"
            print(msg + " " * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r", flush=True)

    print(f"\nFinished processing dataset. Converted {nb_images_processed} images, and saved them in {output_path}")


if __name__ == "__main__":
    main()
