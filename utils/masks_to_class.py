from argparse import ArgumentParser
from pathlib import Path
import shutil

import cv2
import numpy as np


def main():
    parser = ArgumentParser("Sorts image tiles into good and bad samples. Expects masks to be in red")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("output_path", type=Path, help="Output path")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = args.output_path
    good_output_path = output_path / "good"
    bad_output_path = output_path / "bad"

    good_output_path.mkdir(parents=True, exist_ok=True)
    bad_output_path.mkdir(parents=True, exist_ok=True)

    bad_img_count = 0
    good_img_count = 0

    exts = [".jpg", ".png"]
    file_list = list([p for p in data_path.rglob('*') if p.suffix in exts and "mask" not in str(p)])
    nb_imgs = len(file_list)
    for i, img_path in enumerate(file_list):
        msg = f"Processing image {img_path.name} ({i+1}/{nb_imgs})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')

        img_mask_name = "_".join(str(img_path.name).split("_")[:-1]) + "_mask_" + str(img_path.name).split("_")[-1]
        img_mask_path = img_path.parent / img_mask_name

        assert img_mask_path.exists(), f"\nMask for image {img_path} is missing"

        img_mask = cv2.imread(str(img_mask_path))
        height, width, _ = img_mask.shape

        # Check if there is a red pixel somewhere on the mask
        if any([np.array_equal(img_mask[i][j], [0, 0, 255]) for i in range(width) for j in range(height)]):
            bad_img_count += 1
            shutil.copy(img_mask_path, bad_output_path / img_mask_name)
            shutil.copy(img_path, bad_output_path / img_path.name)
        else:
            good_img_count += 1
            shutil.copy(img_mask_path, good_output_path / img_mask_name)
            shutil.copy(img_path, good_output_path / img_path.name)

    print("\nFinished processing dataset")
    print(f"Found {good_img_count} good samples and {bad_img_count} bad samples")


if __name__ == "__main__":
    main()
