from __future__ import annotations

import random
import time
import typing
from argparse import ArgumentParser
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import albumentations
import cv2
import numpy as np
import torch
from hbtools import create_logger

import classification.data.data_transformations as transforms
from classification.configs import LOGGER_NAME, TrainConfig
from classification.data.default_loader import default_load_data
from classification.data.default_loader import default_loader as data_loader
from classification.networks.build_network import build_model
from classification.torch_utils.utils.draw import draw_pred_img
from classification.torch_utils.utils.misc import clean_print, get_dataclass_as_dict
from classification.utils.type_aliases import ImgRaw, ImgStandardized


def main() -> None:  # noqa: PLR0915
    parser = ArgumentParser()
    parser.add_argument("model_path", type=Path, help="Path to the checkpoint to use.")
    parser.add_argument("test_data_path", type=Path, help="Path to the test dataset.")
    parser.add_argument("--limit", "-l", default=None, type=int, help="Limits the number of apparition of each class.")
    parser.add_argument(
        "--classes_names_path", type=Path, default=None, help="Path to a file containing the classes names."
    )
    parser.add_argument(
        "--classes_names", type=str, default=None, nargs="*", help="Path to a file containing the classes names."
    )
    parser.add_argument(
        "--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str, help="Logger level."
    )
    parser.add_argument("--show", "--s", action="store_true", help="Show the images where the network failed.")
    args = parser.parse_args()

    model_path: Path = args.model_path
    test_data_path: Path = args.test_data_path
    limit: int = args.limit
    classes_names_path: Path | None = args.classes_names_path
    classes_names: list[str] | None = args.classes_names
    verbose_level: Literal["debug", "info", "error"] = args.verbose_level

    inference_start_time = time.perf_counter()

    # Set random
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)  # noqa: NPY002

    logger = create_logger(LOGGER_NAME, verbose_level=verbose_level)

    if classes_names_path is not None:
        train_config = TrainConfig.from_classes_path(classes_names_path)
    elif classes_names is not None:
        train_config = TrainConfig.from_classes_names(classes_names)
    else:
        msg = "Either --classes_names_path or --classes_names must be provided"
        raise ValueError(msg)

    # Creates and load the model
    model = build_model(
        train_config.MODEL,
        model_path=model_path,
        eval_mode=True,
        **dict(get_dataclass_as_dict(train_config)),
    )
    logger.info("Weights loaded")

    test_imgs_paths, test_labels = data_loader(test_data_path, train_config.LABEL_MAP, limit=limit)
    nb_imgs = len(test_imgs_paths)
    logger.info("Data loaded")

    resize_fn = typing.cast(
        Callable[[ImgRaw], ImgStandardized],
        partial(cv2.resize, dsize=train_config.IMAGE_SIZES, interpolation=cv2.INTER_LINEAR),
    )
    load_data_fn = partial(default_load_data, preprocessing_pipeline=resize_fn)
    standardize_fn = transforms.albumentation_batch_wrapper(
        albumentations.Normalize(mean=train_config.IMG_MEAN, std=train_config.IMG_STD, p=1.0),
    )  # TODO: Do the normalization on GPU.
    to_tensor_fn = transforms.to_tensor()

    results: list[int] = []  # Variable used to keep track of the classification results
    for i, (img_path, label) in enumerate(zip(test_imgs_paths, test_labels, strict=True), start=1):
        clean_print(f"Processing image {img_path}    ({i} / {nb_imgs})", end="\r" if i != nb_imgs else "\n")
        img = load_data_fn(img_path)
        standardized_img, label_batched = standardize_fn(np.expand_dims(img, 0), np.asarray([label]))
        img_tensor, label_tensor = to_tensor_fn(standardized_img, label_batched)
        with torch.no_grad():
            output = model(img_tensor)
            output = torch.nn.functional.softmax(output, dim=-1)

            prediction = torch.argmax(output)
            pred_correct = label == prediction
            if pred_correct:
                results.append(1)
            else:
                results.append(0)

            if args.show and not pred_correct:
                out_img = draw_pred_img(
                    img_tensor, output, label_tensor, train_config.LABEL_MAP, size=train_config.IMAGE_SIZES
                )
                out_img = cv2.cvtColor(out_img[0], cv2.COLOR_RGB2BGR)
                while True:
                    cv2.imshow("Image", out_img)
                    key = cv2.waitKey(10)
                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        break

    total_time = time.perf_counter() - inference_start_time
    logger.info("Finished running inference on the test dataset.")
    logger.info(
        f"Total inference time was {total_time:.3f}s, which averages to {total_time / len(results):.5f}s per image"
    )
    logger.info(f"Precision: {np.mean(results)}")


if __name__ == "__main__":
    main()
