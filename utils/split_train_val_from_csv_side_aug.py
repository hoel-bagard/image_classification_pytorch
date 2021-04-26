import argparse
from pathlib import Path
from multiprocessing import Pool
import os
import shutil

import cv2
import numpy as np


def worker(args: tuple[Path, Path, tuple[int, int, int, int]]):
    """ Worker in charge of moving an image and possible doing some data augmentation

    Args:
        entry (tuple): Entry to process (tuple with img path, is_train and is_defect_on_right_side)
        train_list (list): List with all the train entries
        train_path (Path): Path to the output train folder
        val_path (Path): Path to the output validation folder

    Return:
        output_file_path: Path of the saved image.
    """
    entry, train_list, data_path, train_path, val_path = args
    file_path, is_train, cls, side = entry

    # Do some data augmentation by mixing training images
    if is_train:
        data_aug_factor = 2  # The amount of data will be original amount times this factor for the train dataset
        ok_images_saved = ng_images_saved = 0

        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        # Expects the image to be already split in a vertical fashion (from the mosaic script)
        # Long live single use scripts!
        if side == "right":
            original_img_part = img[img.shape[0]//2:, :]
        elif side == "left":
            original_img_part = img[:img.shape[0]//2, :]

        while ok_images_saved < data_aug_factor or ng_images_saved < data_aug_factor:
            # Take a random element
            file_path_aug, _, cls2, side_aug = train_list[np.random.randint(len(train_list))]

            # Skip if same side or both defects
            if side == side_aug or (cls != "ok" and cls2 != "ok"):
                continue

            # Create only N augmentations of an image for both ng and ok cat
            if cls == cls2 == "ok" and ok_images_saved >= data_aug_factor:
                continue
            elif ng_images_saved >= data_aug_factor:
                continue

            img = cv2.imread(str(file_path_aug), cv2.IMREAD_UNCHANGED)
            if side_aug == "right":
                augmentation_img_part = img[img.shape[0]//2:, :]
                final_img = cv2.vconcat((original_img_part, augmentation_img_part))
            elif side_aug == "left":
                augmentation_img_part = img[:img.shape[0]//2, :]
                final_img = cv2.vconcat((augmentation_img_part, original_img_part))

            # If end class is ok or original class then keep the original path
            if cls == cls2 == "ok":
                dest_path = (train_path / file_path.relative_to(data_path)).parent \
                    / (file_path.stem + f"_{ok_images_saved}" + file_path.suffix)
                ok_images_saved += 1
            elif cls != "ok" and cls2 == "ok":
                dest_path = (train_path / file_path.relative_to(data_path)).parent \
                    / (file_path.stem + f"_{ng_images_saved}" + file_path.suffix)
                ng_images_saved += 1
            # If class changed, then use the second path
            elif cls == "ok" and cls2 != "ok":
                dest_path = (train_path / file_path_aug.relative_to(data_path)).parent \
                    / (file_path_aug.stem + f"_{ng_images_saved}" + file_path_aug.suffix)
                ng_images_saved += 1

            dest_path.mkdir(parents=True, exist_ok=True)
            print(f"Did augmentation, out:   {dest_path}")
            cv2.imwrite(str(dest_path), final_img)
    else:
        dest_path = (val_path / file_path.relative_to(data_path)).parent
        dest_path.mkdir(parents=True, exist_ok=True)
        print(file_path, dest_path)
        shutil.copy(file_path, dest_path)
    return dest_path


def main():
    parser = argparse.ArgumentParser("Validation/Train splitting that does some data augmentation")
    parser.add_argument("data_path", type=Path, help=("Path to the dataset."
                                                      " The Train and Validation directories will be placed there."))
    parser.add_argument("csv_path", type=Path, help=("Path to the csv specifying how to split."
                                                     "Expected format: (image_path,image_id,class,class,split,side)"))
    args = parser.parse_args()
    # Note: The way this is implemented can lead to generating the same image twice.
    #       Shouldn't matter in the grand scheme of things though.

    train_path: Path = args.data_path / "Train"
    train_path.mkdir(exist_ok=True)
    val_path: Path = args.data_path / "Validation"
    val_path.mkdir(exist_ok=True)

    with open(args.csv_path) as spec_file:
        spec_list = list([(args.data_path / line.split(",")[0],
                           line.split(",")[3].strip() == "train",
                           line.split(",")[2].strip(),
                           line.split(",")[4].strip()) for line in spec_file])

    train_list = [entry for entry in spec_list if entry[1]]

    mp_args = list([(entry, train_list, args.data_path, train_path, val_path) for entry in spec_list])
    nb_images_processed, nb_imgs = 0, len(spec_list)
    with Pool(processes=int(os.cpu_count() * 0.8)) as pool:
        for result in pool.imap(worker, mp_args, chunksize=10):
            nb_images_processed += 1
            msg = f"Processing status: ({nb_images_processed}/{nb_imgs})"
            print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)

    print("\nFinished processing dataset")


if __name__ == "__main__":
    main()
