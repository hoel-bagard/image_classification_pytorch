from __future__ import annotations

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
from hbtools import create_logger

import classification.data.data_transformations as transforms
from classification.configs import LOGGER_NAME, TrainConfig
from classification.data.default_loader import default_load_data
from classification.data.default_loader import default_loader as data_loader
from classification.networks.build_network import build_model
from classification.torch_utils.utils.draw import draw_pred_img
from classification.torch_utils.utils.misc import clean_print, get_dataclass_as_dict
from classification.utils.type_aliases import ImgRaw, ImgStandardized


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
    parser.add_argument(
        "--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str, help="Logger level."
    )
    args = parser.parse_args()

    data_path: Path = args.data_path
    limit: int | None = args.limit
    classes_names_path: Path | None = args.classes_names_path
    classes_names: list[str] | None = args.classes_names
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
    np.random.seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = create_logger(LOGGER_NAME, verbose_level=verbose_level)

    # Creates and load the model
    model = build_model(
        train_config.MODEL,
        model_path=args.model_path,
        eval=True,
        **dict(get_dataclass_as_dict(train_config)),
    )
    logger.info("Weights loaded")

    imgs_paths, labels = data_loader(data_path, train_config.LABEL_MAP, limit=limit)
    nb_imgs = len(imgs_paths)
    logger.info("Data loaded")


    resize_fn = typing.cast(
        Callable[[ImgRaw], ImgStandardized],
        partial(cv2.resize, dsize=train_config.IMAGE_SIZES, interpolation=cv2.INTER_LINEAR)
    )
    standardize_fn = transforms.albumentation_batch_wrapper(
        albumentations.Normalize(mean=train_config.IMG_MEAN, std=train_config.IMG_STD, p=1.0),
    )  # TODO: Do the normalization on GPU.
    to_tensor_fn = transforms.to_tensor()

    for i, (img_path, label) in enumerate(zip(imgs_paths, labels, strict=True)):
        clean_print(f"Processing image {img_path}    ({i} / {nb_imgs})", end="\r" if i != nb_imgs else "\n")
        img = default_load_data(img_path, preprocessing_pipeline=resize_fn)
        standardized_img, label_batched = standardize_fn(np.expand_dims(img, 0), np.asarray([label]))
        img_tensor, _label_tensor = to_tensor_fn(standardized_img, label_batched)

        # Feed the image to the model
        output = model(img_tensor)
        output = torch.nn.functional.softmax(output, dim=-1)

        # Get top prediction and turn it into a one hot
        prediction = output.argmax(dim=1)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][prediction] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * output)

        # Get gradients and activations
        model.zero_grad()
        one_hot.backward(retain_graph=True)
        grads_val = model.get_gradients()[-1].cpu().data.numpy()

        activations = model.get_activations()
        activations = activations.cpu().data.numpy()[0, :]

        # Make gradcam mask
        weights = np.mean(grads_val, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, train_config.IMAGE_SIZES)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        # Draw prediction (logits) on the image
        img = draw_pred_img(img, output, label, train_config.LABEL_MAP, size=train_config.IMAGE_SIZES)

        # Fuse input image and gradcam mask
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap)
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)

        while True:
            cv2.imshow("Image", cam)
            key = cv2.waitKey(10)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    main()
