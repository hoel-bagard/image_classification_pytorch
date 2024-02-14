import argparse
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


def worker(args: tuple) -> None:
    """Worker in charge of moving an image and possible doing some data augmentation.

    Args:
        args: entry (tuple): Entry to process (tuple with img path, is_train and is_defect_on_right_side)
              train_list (list): List with all the train entries
              train_path (Path): Path to the output train folder
              val_path (Path): Path to the output validation folder

    """
    entry, train_lists, data_path, train_path, val_path = args
    train_list_left_ok, train_list_right_ok = train_lists
    file_path, is_train, cls, side = entry
    rng = np.random.default_rng(42)

    # Do some data augmentation by mixing training images
    if is_train:
        data_aug_factor = 20  # The amount of data will be original amount times this factor for the train dataset

        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        # Expects the image to be already split in a vertical fashion (from the mosaic script)
        # Long live single use scripts!
        if side == "right":
            original_img_part = img[img.shape[0] // 2:, :]
        elif side == "left":
            original_img_part = img[: img.shape[0] // 2, :]

        for i in range(data_aug_factor):
            if side == "left":
                file_path_aug, _, _, _ = train_list_right_ok[rng.integers(len(train_list_right_ok))]
            else:
                file_path_aug, _, _, _ = train_list_left_ok[rng.integers(len(train_list_left_ok))]

            img = cv2.imread(str(file_path_aug), cv2.IMREAD_UNCHANGED)
            if side == "right":
                augmentation_img_part = img[: img.shape[0] // 2, :]
                final_img = cv2.vconcat((augmentation_img_part, original_img_part))
            else:
                augmentation_img_part = img[img.shape[0] // 2:, :]
                final_img = cv2.vconcat((original_img_part, augmentation_img_part))

            dest_path = (train_path / file_path.relative_to(data_path)).parent / (
                file_path.stem + f"_{i}" + file_path.suffix
            )
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest_path), final_img)
    # Just move the image to the validation folder
    else:
        dest_path = (val_path / file_path.relative_to(data_path)).parent
        dest_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_path, dest_path)


def main() -> None:
    parser = argparse.ArgumentParser("Validation/Train splitting that does some data augmentation")
    parser.add_argument(
        "data_path",
        type=Path,
        help=("Path to the dataset." " The Train and Validation directories will be placed there."),
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help=(
            "Path to the csv specifying how to split." "Expected format: (image_path,image_id,class,class,split,side)"
        ),
    )
    args = parser.parse_args()
    # Note: The way this is implemented can lead to generating the same ok image twice.
    #       Shouldn't matter in the grand scheme of things though.

    train_path: Path = args.data_path / "Train"
    train_path.mkdir(exist_ok=True)
    val_path: Path = args.data_path / "Validation"
    val_path.mkdir(exist_ok=True)

    with args.csv_path.open() as spec_file:
        spec_list = [
            (
                args.data_path / line.split(",")[0],  # Image path
                line.split(",")[3].strip() == "train",  # Val or train
                line.split(",")[2].strip(),  # Class
                line.split(",")[4].strip(),
            )  # Side
            for line in spec_file
        ]

    train_list = [entry for entry in spec_list if entry[1]]
    # Took elements at random at first but that was extremely inefficient, hence the lists
    train_list_left_ok = [entry for entry in train_list if entry[2] == "ok" and entry[3] == "left"]
    train_list_right_ok = [entry for entry in train_list if entry[2] == "ok" and entry[3] == "right"]
    train_lists = (train_list_left_ok, train_list_right_ok)

    mp_args = [(entry, train_lists, args.data_path, train_path, val_path) for entry in spec_list]
    nb_images_processed, nb_imgs = 0, len(spec_list)
    with Pool(processes=int(nb_cpus * 0.8) if (nb_cpus := os.cpu_count()) is not None else 1) as pool:
        for _result in pool.imap(worker, mp_args, chunksize=10):
            nb_images_processed += 1
            msg = f"Processing status: ({nb_images_processed}/{nb_imgs})"
            print(msg + " " * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r", flush=True)

    print("\nFinished processing dataset")


if __name__ == "__main__":
    main()
