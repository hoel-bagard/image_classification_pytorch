import argparse
import os
import glob

import cv2


def main():
    parser = argparse.ArgumentParser("Cuts images into small tiles")
    parser.add_argument('data_path', help='Path to the dataset')
    parser.add_argument('output_path', help='Output path')
    parser.add_argument("--tile_size", nargs=2, default=[256, 256], type=int, help="Size of the tiles (w, h)")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    tile_width, tile_height = args.tile_size

    file_list = glob.glob(os.path.join(args.data_path, "**", "*.png"), recursive=True)
    nb_imgs = len(file_list)
    for i, file_path in enumerate(file_list):
        msg = f"Processing image {os.path.basename(file_path)} ({i+1}/{nb_imgs})"
        print(msg + ' ' * (os.get_terminal_size()[0] - len(msg)), end='\r')
        img = cv2.imread(file_path)
        file_name = os.path.basename(file_path)
        height, width, _ = img.shape

        tile_index = 0
        for x in range(0, width, tile_width):
            for y in range(0, height, tile_height):
                if y+tile_height > height:
                    y = height - tile_height
                if x+tile_width > width:
                    x = width - tile_width
                tile = img[y:y+tile_height, x:x+tile_width]

                new_tile_name = os.path.splitext(file_name)[0] + '_' + str(tile_index).zfill(5) \
                    + os.path.splitext(file_name)[1]
                tile_path = os.path.join(args.output_path, new_tile_name)
                cv2.imwrite(tile_path, tile)
                tile_index += 1

    print("\nFinished tiling dataset")


if __name__ == "__main__":
    main()
