import argparse
import os
import glob
import shutil

import cv2


def main():
    parser = argparse.ArgumentParser("Tool to label images for classification")
    parser.add_argument('data_path', help='Path to the dataset')
    parser.add_argument('output_path', help='Output path')
    parser.add_argument("--resize", nargs=2, default=[1080, 720], type=int, help="Resizes the images to given size")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_path, "good"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "defect"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "unsure"), exist_ok=True)

    file_list = glob.glob(os.path.join(args.data_path, "**", "*.png"), recursive=True)
    nb_imgs = len(file_list)
    for i, file_path in enumerate(file_list):
        msg = f"Processing image {os.path.basename(file_path)} ({i+1}/{nb_imgs})"
        print(msg + ' ' * (os.get_terminal_size()[0] - len(msg)), end='\r')
        img = cv2.imread(file_path)

        # Image too small =(
        text = ("Press \"d\" if there is a defect, \"a\" if there are none, \"w\" if you are unsure"
                "and \"q\" to quit")
        img = cv2.copyMakeBorder(img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, None, 0)
        img = cv2.putText(img, text, (20, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        while True:
            # if any([size > 1080 for size in resize]):
            #     cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
            #     cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Image", img)
            key = cv2.waitKey(10)
            if key == ord("a"):  # No defect
                shutil.move(file_path, os.path.join(args.output_path, "good", os.path.basename(file_path)))
                break
            elif key == ord("d"):  # Defect
                shutil.move(file_path, os.path.join(args.output_path, "defect", os.path.basename(file_path)))
                break
            elif key == ord("w"):  # Gray zone
                shutil.move(file_path, os.path.join(args.output_path, "unsure", os.path.basename(file_path)))
                break
            elif key == ord("q"):  # quit
                cv2.destroyAllWindows()
                return -1

    print("\nFinished labelling dataset")


if __name__ == "__main__":
    main()
