from __future__ import annotations

import os
import random
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
from hbtools import create_logger, show_img, yes_no_prompt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import classification.data.data_transformations as transforms
from classification.configs import LOGGER_NAME, TrainConfig
from classification.data.default_loader import default_load_data
from classification.data.default_loader import default_loader as data_loader
from classification.networks.build_network import build_model, ModelHelper
from classification.torch_utils.utils.misc import clean_print, get_dataclass_as_dict
from classification.utils.type_aliases import ImgRaw


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("model_path", type=Path, help="Path to the checkpoint to use")
    parser.add_argument("data_path", type=Path, help="Path to the dataset.")
    parser.add_argument("--limit", "-l", default=None, type=int, help="Limits the number of apparition of each class.")
    parser.add_argument(
        "--classes_names_path", type=Path, default=None, help="Path to a file containing the classes names."
    )
    parser.add_argument(
        "--classes_names", type=str, default=None, nargs="*", help="Path to a file containing the classes names."
    )
    parser.add_argument("--show", "--s", action="store_true", help="Show the gradcam images.")
    parser.add_argument(
        "--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str, help="Logger level."
    )
    args = parser.parse_args()

    data_path: Path = args.data_path
    limit: int | None = args.limit
    classes_names_path: Path | None = args.classes_names_path
    classes_names: list[str] | None = args.classes_names
    show: bool = args.show
    verbose_level: Literal["debug", "info", "error"] = args.verbose_level

    if classes_names_path is not None:
        train_config = TrainConfig.from_classes_path(classes_names_path)
    elif classes_names is not None:
        train_config = TrainConfig.from_classes_names(classes_names)
    else:
        msg = "Either --classes_names_path or --classes_names must be provided"
        raise ValueError(msg)

    # Set random
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)  # noqa: NPY002

    logger = create_logger(LOGGER_NAME, verbose_level=verbose_level)

    # Creates and load the model
    model = build_model(
        train_config.MODEL,
        model_path=args.model_path,
        eval=True,
        **dict(get_dataclass_as_dict(train_config)),
    )
    logger.info("Weights loaded")

    match train_config.MODEL:
        case ModelHelper.ResNet34 | ModelHelper.ResNet50:
            target_layers = [model.layer4[-1]]
        case _:
            msg = "GradCAM is not implemented for this model"
            raise NotImplementedError(msg)
    cam = GradCAM(model=model, target_layers=target_layers)

    imgs_paths, labels = data_loader(data_path, train_config.LABEL_MAP, limit=limit)
    nb_imgs = len(imgs_paths)
    logger.info("Data loaded")

    resize_fn = typing.cast(
        Callable[[ImgRaw], ImgRaw],
        partial(cv2.resize, dsize=train_config.IMAGE_SIZES, interpolation=cv2.INTER_LINEAR),
    )
    standardize_fn = transforms.albumentation_batch_wrapper(
        albumentations.Normalize(mean=train_config.IMG_MEAN, std=train_config.IMG_STD, p=1.0),
    )
    to_tensor_fn = transforms.to_tensor()

    for i, (img_path, label) in enumerate(zip(imgs_paths, labels, strict=True)):
        clean_print(f"Processing image {img_path}    ({i} / {nb_imgs})", end="\r" if i != nb_imgs else "\n")
        img = default_load_data(img_path, preprocessing_pipeline=resize_fn)
        standardized_img, label_batched = standardize_fn(np.expand_dims(img, 0), np.asarray([label]))
        img_tensor, _label_tensor = to_tensor_fn(standardized_img, label_batched)

        # If targets is None, the highest scoring category will be used for every image in the batch.
        grayscale_cam = cam(input_tensor=img_tensor, targets=None, eigen_smooth=True)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img / 255, grayscale_cam, use_rgb=True)

        prediction = int(cam.outputs.argmax(dim=1).cpu().detach().numpy())

        if show:
            show_img(visualization, f"{img_path=}, {prediction=}")
            if "DISPLAY" not in os.environ:
                yes_no_prompt("Show next image ?")


if __name__ == "__main__":
    main()
