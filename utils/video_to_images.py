from __future__ import annotations

import argparse
import os
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


def imwrite(output_path: Path, frame: npt.NDArray[np.int8], resize_ratio: float = 1) -> None:
    """Save the given to disk."""
    frame = cv2.resize(frame, (int(frame.shape[1] * resize_ratio), int(frame.shape[0] * resize_ratio)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)


def worker(args: tuple[argparse.Namespace, Path]) -> None:
    """Worker in charge of processing one video."""
    cli_args, video_path = args

    data_path: Path = cli_args.data_path
    output_path: Path = cli_args.output_path
    img_format: str = cli_args.img_format
    resize_ratio: float = cli_args.resize_ratio
    best_frame_only: bool = cli_args.single_frame

    video = cv2.VideoCapture(str(video_path))
    video_output_path = output_path / video_path.relative_to(data_path).parent / video_path.stem
    frame_count = 0
    best_sharpness = 0.  # Used when keeping only the least blurry frame (the lower the more blurry)
    while True:
        success, frame = video.read()
        if not success:
            break

        if best_frame_only:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Not necessary
            if (sharpness := cv2.Laplacian(frame_gray, cv2.CV_64F).var()) > best_sharpness:
                best_sharpness, best_frame = sharpness, frame
        else:
            imwrite(video_output_path / (str(frame_count).zfill(3) + img_format), frame, resize_ratio)
        frame_count += 1
    if best_frame_only:
        imwrite(video_output_path.with_suffix(img_format), best_frame, resize_ratio)


def main() -> None:
    parser = argparse.ArgumentParser(description="Script to convert videos into images.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", type=Path, help="Path to the folder with the mp4 videos.")
    parser.add_argument("--output_path", "-o", type=Path, default=Path("../out_images"), help="Output path.")
    parser.add_argument("--img_format", "-f", type=str, choices=[".jpg", ".png"], default=".jpg", help="Image format.")
    parser.add_argument("--resize_ratio", "-r", type=float, default=1, help="Resize the images using that ratio.")
    parser.add_argument("--single_frame", "-s", action="store_true",
                        help="Keep only the least blury frame for each video.")
    args = parser.parse_args()

    data_path: Path = args.data_path

    video_paths = list(data_path.rglob("*.mp4"))
    nb_videos = len(video_paths)
    mp_args = [(args, video_path) for video_path in video_paths]
    with Pool(processes=int(os.cpu_count() * 0.8)) as pool:
        for nb_videos_processed, _result in enumerate(pool.imap(worker, mp_args, chunksize=10), start=1):
            msg = f"Processing status: ({nb_videos_processed}/{nb_videos})"
            print(msg + " " * (shutil.get_terminal_size(fallback=(156, 38)).columns - len(msg)), end="\r", flush=True)
    print("\nFinished")


if __name__ == "__main__":
    main()
