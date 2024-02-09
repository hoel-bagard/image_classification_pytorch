from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from typing_extensions import Self

from classification.networks.build_network import ModelHelper

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainConfig:
    # Training parameters
    BATCH_SIZE: int = 32
    MAX_EPOCHS: int = 500
    START_LR: float = 1e-3
    END_LR: float = 5e-6
    WEIGHT_DECAY: float = 1e-2  # Weight decay for the optimizer

    # Data processing
    IMAGE_SIZES: tuple[int, int] = (224, 224)  # All images will be resized to this size
    # The mean and std used to normalize the dataset (the default values for the images are the ImageNet ones).
    IMG_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMG_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

    LABEL_SMOOTHING: float = 0.1  # Value to use for label smoothing if the loss supports it. 0 for no smoothing.
    LOSS_WEIGTHS: list[float] | None = None  # Weights the classes during the loss

    # Network part
    MODEL: type = ModelHelper.SmallDarknet
    # Parameters for standard CNN-like networks
    # CHANNELS: list[int] = field(default_factory=lambda: [3, 8, 16, 32, 32, 16])
    # SIZES: list[int | tuple[int, int]] = field(default_factory=lambda: [5, 3, 3, 3, 3])  # Kernel sizes
    # STRIDES: list[int | tuple[int, int]] = field(default_factory=lambda: [5, 3, 3, 2, 2])
    # PADDINGS: list[int | tuple[int, int]] = field(default_factory=lambda: [2, 1, 1, 1, 1])
    # BLOCKS: list[int] = field(default_factory=lambda: [1, 2, 2, 1, 1])

    # Cifar-10
    # CROP_IMAGE_SIZES: tuple[int, int] = (32, 32)  # Center crop
    # RESIZE_IMAGE_SIZES: tuple[int, int] = (32, 32)  # All images will be resized to this size
    CHANNELS: list[int] = field(default_factory=lambda: [3, 16, 32, 16])
    SIZES: list[int | tuple[int, int]] = field(default_factory=lambda: [3, 3, 3])   # Kernel sizes
    STRIDES: list[int | tuple[int, int]] = field(default_factory=lambda: [2, 2, 2])
    PADDINGS: list[int | tuple[int, int]] = field(default_factory=lambda: [1, 1, 1])
    BLOCKS: list[int] = field(default_factory=lambda: [1, 2, 1])


    # Number of workers to use for dataloading
    NB_WORKERS: int = field(
        default_factory=lambda: int(nb_cpus * 0.8) if (nb_cpus := os.cpu_count()) is not None else 1
    )

    LABEL_MAP: dict[int, str] = field(default_factory=dict)
    NB_CLASSES: int = 0


    @classmethod
    def from_classes_path(cls, classes_names_path: Path) -> Self:
        label_map: dict[int, str] = {}
        with classes_names_path.open("r", encoding="utf-8") as classes_file:
            for key, class_name in enumerate(classes_file):
                label_map[key] = class_name.strip()
        return cls(LABEL_MAP=label_map, NB_CLASSES=len(label_map))

    @classmethod
    def from_classes_names(cls, classes_names: Iterable[str]) -> Self:
        label_map: dict[int, str] = {key: class_name.strip() for key, class_name in enumerate(classes_names)}
        return cls(LABEL_MAP=label_map, NB_CLASSES=len(label_map))
