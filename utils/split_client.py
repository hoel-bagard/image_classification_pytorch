import argparse
import os
import glob
import shutil
from random import shuffle


def main():
    parser = argparse.ArgumentParser("Validation/Train splitting")
    parser.add_argument('data_path', help='Path to the train dataset')
    args = parser.parse_args()

    val_path = os.path.join(os.path.dirname(args.data_path.strip("/")), "Validation")
    os.makedirs(val_path, exist_ok=True)

    folder_list = glob.glob(os.path.join(args.data_path, "*"), recursive=True)
    nb_folders = len(folder_list)
    shuffle(folder_list)  # Too lazy to split each class individually
    for i, folder_path in enumerate(folder_list):
        print(f"Processing folder {os.path.basename(folder_path)} ({i+1}/{nb_folders})", end='\r')
        if i >= 0.9*nb_folders:
            file_list = glob.glob(os.path.join(folder_path, "**", "*.bmp"), recursive=True)
            for file_path in file_list:
                subpath = os.path.join(*os.path.normpath(file_path).split(os.sep)[-4:])
                dest_path = os.path.join(val_path, subpath)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(file_path, dest_path)
    print("\nFinished splitting dataset")


if __name__ == "__main__":
    main()
