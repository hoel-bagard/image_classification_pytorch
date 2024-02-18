import argparse
import shutil
from pathlib import Path
from random import shuffle


def main() -> None:
    parser = argparse.ArgumentParser("Validation/Train splitting")
    parser.add_argument("data_path", type=Path, help="Path to the train dataset")
    parser.add_argument("--split_ratio", "--s", type=float, default=0.8, help="Fraction of the dataset used for train")
    args = parser.parse_args()

    val_path: Path = args.data_path.parent / "Validation"
    val_path.mkdir(parents=True, exist_ok=True)

    exts = [".jpg", ".png"]
    file_list = [p for p in args.data_path.rglob("*") if p.suffix in exts]

    nb_imgs = len(file_list)
    shuffle(file_list)  # Too lazy to split each class individually
    for i, file_path in enumerate(file_list):
        msg = f"Processing image {file_path.name} ({i + 1}/{nb_imgs})"
        print(msg + " " * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r")
        if i >= args.split_ratio * nb_imgs:
            dest_path = (val_path / file_path.relative_to(args.data_path)).parent
            dest_path.mkdir(parents=True, exist_ok=True)
            shutil.move(file_path, dest_path)
    print("\nFinished splitting dataset")


if __name__ == "__main__":
    main()
