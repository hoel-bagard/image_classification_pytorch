"""Script to slightly change the imagenette2 dataset's structure. The changes are done in place.

See https://github.com/fastai/imagenette for more information on imagenette.
"""
import argparse
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script to slightly change the imagenette dataset's structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data_path", type=Path, help="Path to the folder with the imagenette data.")
    args = parser.parse_args()

    data_path: Path = args.data_path

    class_map: dict[str, str] = {
        "n01440764": "tench",
        "n02102040": "English springer",
        "n02979186": "cassette player",
        "n03000684": "chain saw",
        "n03028079": "church",
        "n03394916": "French horn",
        "n03417042": "garbage truck",
        "n03425413": "gas pump",
        "n03445777": "golf ball",
        "n03888257": "parachute"
    }

    val_dataset = data_path / "val"
    train_dataset = data_path / "train"

    for path in (train_dataset, val_dataset):
        for original_name, class_name in class_map.items():
            print(f"Renaming {path / original_name} to {path / class_name}")
            shutil.move(path / original_name, path / class_name)

    print("Creating classes.names file.")
    with (data_path / "classes.names").open("w", encoding="utf-8") as classes_file:
        classes_file.write("\n".join(class_map.values()))

    print(f"Finished.")


if __name__ == "__main__":
    main()
