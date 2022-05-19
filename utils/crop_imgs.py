import os
import shutil
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import cv2


def worker(args: tuple[Path, Path, tuple[int, int, int, int]]):  # noqa D417
    """Worker in charge of cropping an image.  # noqa D417

    Args:
        img_path (Path): Path to the image to process
        output_path (Path): Folder to where the new image will be saved
        crop (tuple, optional): (left, right, top, bottom), if not None then image will be cropped by the given values.

    Return:
        output_file_path: Path of the saved image.
    """
    img_path, output_path, crop = args
    output_file_path = output_path / img_path.relative_to(output_path.parent)

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

    left, right, top, bottom = crop
    img = img[top:-bottom, left:-right]  # Doesn't work for top=bottom=0

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file_path), img)
    return output_file_path


def main():
    parser = ArgumentParser("Crops all the image in the given folder by the given values."
                            "Saves the data in a cropped_imgs folder, in the dataset's parent folder")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("crop", type=int, nargs=4, help="How much should be cropped, (left, right, top, bottom)")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = data_path.parent / "cropped_imgs"
    output_path.mkdir(parents=True, exist_ok=True)

    # Get a list of all the images
    exts = [".jpg", ".png"]
    file_list = list([p for p in data_path.rglob('*') if p.suffix in exts])
    nb_imgs = len(file_list)

    mp_args = list([(img_path, output_path, args.crop) for img_path in file_list])
    nb_images_processed = 0
    with Pool(processes=int(os.cpu_count() * 0.8)) as pool:
        for _result in pool.imap(worker, mp_args, chunksize=10):
            nb_images_processed += 1
            msg = f"Processing status: ({nb_images_processed}/{nb_imgs})"
            print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)

    print(f"\nFinished processing dataset. Converted {nb_images_processed} images, and saved them in {output_path}")


if __name__ == "__main__":
    main()
