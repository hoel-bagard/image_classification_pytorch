"""Script to read the cifar dataset and convert it to an image format.

The instructions/explanations can be found here: https://www.cs.toronto.edu/~kriz/cifar.html
"""
import argparse
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Script to convert cifar-10 data to images.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", type=Path, help="Path to the folder with the cifar-10 data.")
    parser.add_argument("--output_path", "-o", type=Path, default=Path("data/cifar_10_images"), help="Output path.")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = args.output_path

    with open(data_path / "batches.meta", "rb") as pickle_file:
        label_map = pickle.load(pickle_file, encoding="bytes")[b"label_names"]

    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "classes.names", "w") as classes_file:
        for cls_name in label_map:
            classes_file.write(cls_name.decode("utf-8") + "\n")

    # Train data
    for batch_idx, pickle_batch_path in enumerate(data_path.glob("data_*"), start=1):
        print(f"Processing training batch ({batch_idx}/5)")
        with open(pickle_batch_path, "rb") as pickle_file:
            data: dict = pickle.load(pickle_file, encoding="bytes")

        for img_idx, (label, img, filename) in enumerate(zip(*tuple(data.values())[1:]), start=1):
            msg = f"Processing status: ({img_idx}/10000)"
            print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)),
                  end=('\r' if img_idx < 10000 else '\n'), flush=True)

            # Separate and assemble the color channels, and then reshape the 2D array (list of colors) into an image.
            img = np.asarray(img)
            img = np.stack((img[2048:], img[1024:2048], img[:1024]), axis=-1)  # BGR
            img = np.reshape(img, (32, 32, 3))
            # Save the image
            out_img_path = output_path / "Train" / label_map[label].decode("utf-8") / filename.decode("utf-8")
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img_path), img)

    # Validation data
    print("Processing the validation data")
    with open(data_path / "test_batch", "rb") as pickle_file:
        data = pickle.load(pickle_file, encoding="bytes")

    for img_idx, (label, img, filename) in enumerate(zip(*tuple(data.values())[1:]), start=1):
        msg = f"Processing status: ({img_idx}/10000)"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)),
              end=('\r' if img_idx < 10000 else '\n'), flush=True)

        # Separate and assemble the color channels, and then reshape the 2D array (list of colors) into an image.
        img = np.asarray(img)
        img = np.stack((img[2048:], img[1024:2048], img[:1024]), axis=-1)  # BGR
        img = np.reshape(img, (32, 32, 3))
        # Save the image
        out_img_path = output_path / "Validation" / label_map[label].decode("utf-8") / filename.decode("utf-8")
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img_path), img)

    print(f"Finished.\nImages saved to {output_path}")


if __name__ == "__main__":
    main()
