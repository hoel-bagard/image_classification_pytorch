import shutil
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotates an image by the given angle (in degrees) and returns it."""
    img_center = np.asarray(img.shape[1::-1]) / 2
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_img


def main():
    parser = ArgumentParser("Just shows the images of a dataset.")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("--crop", "--c", type=int, nargs=4, help="If cropping the images, (left, right, top, bottom)")
    parser.add_argument("--angle", "--a", type=float, help="Rotates the image by an angle before displaying it")
    args = parser.parse_args()

    exts = [".jpg", ".png"]
    file_list = list([p for p in args.data_path.rglob('*') if p.suffix in exts])
    nb_imgs = len(file_list)

    for i, img_path in enumerate(file_list):
        msg = f"Showing image: ({i}/{nb_imgs})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r', flush=True)

        img = cv2.imread(str(img_path))

        if args.angle:
            img = rotate_image(img, args.angle)

        if args.crop is not None:
            left, right, top, bottom = args.crop
            img = img[top:-bottom, left:-right]  # Doesn't work for top=bottom=0

        while True:
            cv2.imshow("Image", img)
            key = cv2.waitKey(10)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break

    print("\nReached end of dataset")


if __name__ == "__main__":
    main()
