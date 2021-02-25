import argparse
import shutil
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser("Tool to label images for classification")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("output_path", type=Path, help="Output path")
    parser.add_argument("--resize", nargs=2, default=[1080, 720], type=int, help="Resizes the images to given size")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = args.output_path

    good_output_path = output_path / "good"
    bad_output_path = output_path / "bad"
    unsure_output_path = output_path / "unsure"
    good_output_path.mkdir(parents=True, exist_ok=True)
    bad_output_path.mkdir(parents=True, exist_ok=True)
    unsure_output_path.mkdir(parents=True, exist_ok=True)

    file_list = sorted(list(data_path.rglob("*.png")))
    nb_imgs = len(file_list)
    for i, file_path in enumerate(file_list):
        msg = f"Processing image {file_path.name} ({i+1}/{nb_imgs})"
        print(msg + ' ' * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end='\r')
        img = cv2.imread(str(file_path))

        if args.resize:
            img = cv2.resize(img, tuple(args.resize))

        text = ("Press \"d\" if there is a defect, \"a\" if there are none, \"w\" if you are unsure"
                "and \"q\" to quit")
        img = cv2.copyMakeBorder(img, 70, 0, 0, 0, cv2.BORDER_CONSTANT, None, 0)
        img = cv2.putText(img, text, (20, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img, msg, (20, 45),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        while True:
            cv2.imshow("Image", img)
            key = cv2.waitKey(10)
            if key == ord("a"):  # No defect
                shutil.move(file_path, good_output_path / file_path.name)
                break
            elif key == ord("d"):  # Defect
                shutil.move(file_path, bad_output_path / file_path.name)
                break
            elif key == ord("w"):  # Gray zone
                shutil.move(file_path, unsure_output_path / file_path.name)
                break
            elif key == ord("q"):  # quit
                cv2.destroyAllWindows()
                return -1

    print("\nFinished labelling dataset")


if __name__ == "__main__":
    main()
