import argparse
from pathlib import Path
from shutil import get_terminal_size

import cv2


def main() -> None:
    parser = argparse.ArgumentParser("Cuts images into small tiles")
    parser.add_argument("data_path", type=Path, help="Path to the dataset")
    parser.add_argument("output_path", type=Path, help="Output path")
    parser.add_argument("--tile_size", nargs=2, default=[256, 256], type=int, help="Size of the tiles (w, h)")
    args = parser.parse_args()

    data_path: Path = args.data_path
    output_path: Path = args.output_path

    output_path.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height = args.tile_size

    file_list = list(data_path.rglob("*.png"))
    nb_imgs = len(file_list)
    for i, file_path in enumerate(file_list):
        msg = f"Processing image {file_path.name} ({i+1}/{nb_imgs})"
        print(msg + " " * (get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r")
        img = cv2.imread(str(file_path))
        height, width, _ = img.shape

        tile_index = 0
        for x in range(0, width, tile_width):
            for y in range(0, height, tile_height):
                if y + tile_height > height:
                    y = height - tile_height
                if x + tile_width > width:
                    x = width - tile_width
                tile = img[y : y + tile_height, x : x + tile_width]

                new_tile_name = file_path.stem + "_" + str(tile_index).zfill(5) + file_path.suffix
                tile_path = output_path / new_tile_name
                cv2.imwrite(str(tile_path), tile)
                tile_index += 1

    print("\nFinished tiling dataset")


if __name__ == "__main__":
    main()
