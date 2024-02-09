import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser("Validation/Train splitting")
    parser.add_argument("data_path", type=Path, help=("Path to the dataset."
                                                      " The Train and Validation directories will be placed there."))
    parser.add_argument("csv_path", type=Path, help=("Path to the csv specifying how to split."
                                                     "Expected format: (image_path,image_id,class,split)"))
    args = parser.parse_args()

    train_path: Path = args.data_path / "Train"
    train_path.mkdir(exist_ok=True)
    val_path: Path = args.data_path / "Validation"
    val_path.mkdir(exist_ok=True)

    with args.csv_path.open() as spec_file:
        spec_list = [(args.data_path / line.split(",")[0], line.split(",")[3].strip() == "train") for line in spec_file]

    nb_imgs = len(spec_list)
    for i, (file_path, is_train) in enumerate(spec_list):
        msg = f"Processing image {file_path.name} ({i+1}/{nb_imgs})"
        print(msg + " " * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r")
        if is_train:
            dest_path = (train_path / file_path.relative_to(args.data_path)).parent
        else:
            dest_path = (val_path / file_path.relative_to(args.data_path)).parent
        dest_path.mkdir(parents=True, exist_ok=True)
        shutil.move(file_path, dest_path)
    print("\nFinished splitting dataset")


if __name__ == "__main__":
    main()
