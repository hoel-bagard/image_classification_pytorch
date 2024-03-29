from __future__ import annotations

import argparse
import random
import sys
import time
import traceback
import typing
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import albumentations
import cv2
import numpy as np
import torch
from hbtools import create_logger
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

import classification.data.data_transformations as transforms
from classification.configs import LOGGER_NAME, RecordConfig, TrainConfig
from classification.data.default_loader import default_load_data
from classification.data.default_loader import default_loader as data_loader
from classification.networks.build_network import build_model
from classification.torch_utils.utils.batch_generator import BatchGenerator
from classification.torch_utils.utils.misc import get_dataclass_as_dict
from classification.torch_utils.utils.prepare_folders import prepare_folders
from classification.torch_utils.utils.resource_usage import resource_usage
from classification.torch_utils.utils.torch_summary import summary
from classification.torch_utils.utils.trainer import Trainer
from classification.utils.classification_metrics import ClassificationMetrics
from classification.utils.classification_tensorboard import ClassificationTensorBoard
from classification.utils.type_aliases import ImgRaw, ImgStandardized


def main() -> None:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser(
        description="Training script", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--limit", "-l", default=None, type=int, help="Limits the number of apparition of each class.")
    parser.add_argument("--train_data_path", type=Path, required=True, help="Path to the training dataset.")
    parser.add_argument("--val_data_path", type=Path, required=True, help="Path to the validation dataset.")
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

    limit: int = args.limit
    train_data_path: Path = args.train_data_path
    val_data_path: Path = args.val_data_path
    classes_names_path: Path | None = args.classes_names_path
    classes_names: list[str] | None = args.classes_names
    verbose_level: Literal["debug", "info", "error"] = args.verbose_level

    record_config = RecordConfig()
    if classes_names_path is not None:
        train_config = TrainConfig.from_classes_path(classes_names_path)
    elif classes_names is not None:
        train_config = TrainConfig.from_classes_names(classes_names)
    else:
        msg = "Either --classes_names_path or --classes_names must be provided"
        raise ValueError(msg)

    prepare_folders(
        record_config.TB_DIR if record_config.USE_TB else None,
        record_config.CHECKPOINTS_DIR if record_config.USE_CHECKPOINTS else None,
        repo_name="image-classification-pytorch",
        extra_files=[
            Path("src/classification/configs/record_config.py"),
            Path("src/classification/configs/train_config.py"),
        ],
    )
    log_dir = record_config.CHECKPOINTS_DIR / "print_logs" if record_config.USE_CHECKPOINTS else None
    logger = create_logger(LOGGER_NAME, log_dir=log_dir, verbose_level=verbose_level)
    logger.info("Finished preparing tensorboard and checkpoints folders.")

    # Makes training quite a bit faster
    torch.backends.cudnn.benchmark = True

    # Set random
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)  # noqa: NPY002
    numpy_rng = np.random.default_rng(42)

    train_data, train_labels = data_loader(train_data_path, train_config.LABEL_MAP, limit=limit)
    logger.info("Train data loaded")
    val_data, val_labels = data_loader(val_data_path, train_config.LABEL_MAP, limit=limit, shuffle_rng=numpy_rng)
    logger.info("Validation data loaded")

    # Data augmentation done on cpu.
    augmentation_pipeline = transforms.albumentation_batch_wrapper(
        albumentations.Compose(
            [
                albumentations.HorizontalFlip(p=0.5),
                # albumentations.VerticalFlip(p=0.5),
                # albumentations.RandomRotate90(),
                # albumentations.ShiftScaleRotate(),
                # albumentations.CLAHE(),
                # albumentations.AdvancedBlur(),
                albumentations.Sharpen(p=0.1),
                albumentations.ImageCompression(p=0.1),
                albumentations.GaussNoise(var_limit=(10, 50), p=0.1),
                albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                albumentations.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),
                albumentations.CoarseDropout(
                    max_holes=6,
                    max_height=10,
                    max_width=10,
                    min_holes=3,
                    min_height=2,
                    min_width=2,
                    p=0.4,
                ),
            ]
        )
    )

    resize_fn = typing.cast(
        Callable[[ImgRaw], ImgStandardized],
        partial(cv2.resize, dsize=train_config.IMAGE_SIZES, interpolation=cv2.INTER_LINEAR),
    )
    denormalize_imgs_fn = transforms.destandardize_img(train_config.IMG_MEAN, train_config.IMG_STD)
    load_data_fn = partial(default_load_data, preprocessing_pipeline=resize_fn)
    common_pipeline = transforms.albumentation_batch_wrapper(
        albumentations.Normalize(mean=train_config.IMG_MEAN, std=train_config.IMG_STD, p=1.0),
    )  # TODO: Do the normalization on GPU.
    train_pipeline = transforms.compose_transformations((augmentation_pipeline, common_pipeline))

    with BatchGenerator(
        train_data,
        train_labels,  # pyright: ignore[reportArgumentType]
        train_config.BATCH_SIZE,
        nb_workers=train_config.NB_WORKERS,
        data_preprocessing_fn=load_data_fn,
        cpu_pipeline=train_pipeline,  # pyright: ignore[reportArgumentType]
        gpu_pipeline=transforms.to_tensor(),  # pyright: ignore[reportArgumentType]
        shuffle=True,
    ) as train_dataloader, BatchGenerator(
        val_data,
        val_labels,  # pyright: ignore[reportArgumentType]
        train_config.BATCH_SIZE,
        nb_workers=train_config.NB_WORKERS,
        data_preprocessing_fn=load_data_fn,
        cpu_pipeline=common_pipeline,  # pyright: ignore[reportArgumentType]
        gpu_pipeline=transforms.to_tensor(),  # pyright: ignore[reportArgumentType]
        shuffle=False,
    ) as val_dataloader:
        print(f"\nLoaded {len(train_dataloader)} train data and", f"{len(val_dataloader)} validation data", flush=True)

        print("Building model. . .", end="\r")
        model = build_model(train_config.MODEL, **dict(get_dataclass_as_dict(train_config)))

        logger.info(f"{'-' * 24} Starting train {'-' * 24}")
        logger.info("From command : %s", " ".join(sys.argv))
        logger.info(f"Input shape: {train_dataloader.data_shape}")
        logger.info("")
        logger.info("Using model:")
        for line in summary(model, train_dataloader.data_shape):
            logger.info(line)
        logger.info("")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weights = torch.Tensor(train_config.LOSS_WEIGTHS).to(device) if train_config.LOSS_WEIGTHS else None
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        # loss_fn = SmoothCrossEntropyLoss(train_config.LABEL_SMOOTHING)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_config.START_LR, betas=(0.9, 0.95), weight_decay=train_config.WEIGHT_DECAY
        )
        trainer = Trainer(model, loss_fn, optimizer, train_dataloader, val_dataloader)
        scheduler = CosineAnnealingLR(optimizer, train_config.MAX_EPOCHS, eta_min=train_config.END_LR)
        # TODO: Try this https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/cosine_lr.py

        if record_config.USE_TB:
            metrics = ClassificationMetrics(
                model, train_dataloader, val_dataloader, train_config.LABEL_MAP, max_batches=None
            )
            tensorboard = ClassificationTensorBoard(
                model,
                record_config.TB_DIR,
                train_dataloader,
                val_dataloader,
                logger,
                train_config,
                record_config,
                metrics,
                denormalize_imgs_fn,
            )
        else:
            metrics, tensorboard = None, None

        best_loss = 1000
        last_checkpoint_epoch = 0
        train_start_time = time.time()

        epoch_loss, val_epoch_loss, train_acc, val_acc = 0, 0, 0, 0
        try:
            for epoch in range(train_config.MAX_EPOCHS):
                epoch_start_time = time.perf_counter()
                print()  # logger doesn't handle \n super well
                logger.info(f"Epoch {epoch}/{train_config.MAX_EPOCHS}")

                epoch_loss = typing.cast(float, trainer.train_epoch())

                if tensorboard is not None:
                    tensorboard.write_loss(epoch, epoch_loss)
                    tensorboard.write_lr(epoch, scheduler.get_last_lr()[0])

                if (
                    epoch_loss < best_loss
                    and record_config.USE_CHECKPOINTS
                    and epoch >= record_config.RECORD_START
                    and (epoch - last_checkpoint_epoch) >= record_config.CHECKPT_SAVE_FREQ
                ):
                    save_path = record_config.CHECKPOINTS_DIR / f"train_{epoch}.pt"
                    logger.info(
                        f"Loss improved from {best_loss:.5e} to {epoch_loss:.5e}," f"saving model to {save_path}"
                    )
                    best_loss, last_checkpoint_epoch = epoch_loss, epoch
                    torch.save(model.state_dict(), save_path)

                logger.info(f"Epoch loss: {epoch_loss:.5e}  -  Took {time.perf_counter() - epoch_start_time:.5f}s")

                # Validation and other metrics
                if epoch % record_config.VAL_FREQ == 0 and epoch >= record_config.RECORD_START:
                    if tensorboard is not None:
                        tensorboard.write_weights_grad(epoch)
                    with torch.no_grad():
                        validation_start_time = time.perf_counter()
                        model.eval()
                        val_epoch_loss = typing.cast(float, trainer.val_epoch())

                        if tensorboard is not None and metrics is not None:
                            print("Starting to compute TensorBoard metrics", end="\r", flush=True)
                            tensorboard.write_loss(epoch, val_epoch_loss, mode="Validation")

                            # Metrics for the Train dataset
                            tensorboard.write_images(epoch)
                            tensorboard.write_metrics(epoch)
                            train_acc = metrics.get_avg_acc()

                            # Metrics for the Validation dataset
                            tensorboard.write_images(epoch, mode="Validation")
                            tensorboard.write_metrics(epoch, mode="Validation")
                            val_acc = metrics.get_avg_acc()

                            logger.info(f"Train accuracy: {train_acc:.3f}  -  Validation accuracy: {val_acc:.3f}")
                        model.train()

                        logger.info(
                            f"Validation loss: {val_epoch_loss:.5e}  -  "
                            f"Took {time.perf_counter() - validation_start_time:.5f}s"
                        )
                scheduler.step()
        except KeyboardInterrupt:
            print("\n")
        except Exception:
            logger.exception("".join(traceback.format_exception(*sys.exc_info())))
            raise

    if tensorboard is not None:
        metrics = {
            "Z - Final Results/Train loss": float(epoch_loss),
            "Z - Final Results/Validation loss": float(val_epoch_loss),
            "Z - Final Results/Train accuracy": float(train_acc),
            "Z - Final Results/Validation accuracy": float(val_acc),
        }
        tensorboard.write_config(get_dataclass_as_dict(train_config), metrics)
        tensorboard.close_writers()

    train_stop_time = time.time()
    end_msg = f"Finished Training\n\tTraining time : {train_stop_time - train_start_time:.03f}s"
    mem_peak, gpu_memory = resource_usage()
    end_msg += f"\n\tRAM peak : {mem_peak // 1024 if mem_peak is not None else 'N/A'} MB\n\tVRAM usage : {gpu_memory}"
    logger.info(end_msg)


if __name__ == "__main__":
    main()
