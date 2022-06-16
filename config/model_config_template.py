from dataclasses import dataclass, field
from typing import Optional

from src.networks.build_network import ModelHelper


@dataclass(frozen=True, slots=True)
class ModelConfig:
    # Training parameters
    BATCH_SIZE: int = 32
    MAX_EPOCHS: int = 500
    START_LR: float = 1e-3
    END_LR: float = 5e-6
    WEIGHT_DECAY: float = 1e-2   # Weight decay for the optimizer

    # Data processing
    IMAGE_SIZES: tuple[int, int] = (224, 224)  # All images will be resized to this size
    # The mean and std used to normalize the dataset (the default values for the images are the ImageNet ones).
    IMG_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMG_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

    LABEL_SMOOTHING: float = 0.1   # Value to use for label smoothing if the loss supports it. 0 for no smoothing.
    LOSS_WEIGTHS: Optional[list[float]] = None   # Weights the classes during the loss

    # Network part
    MODEL: type = ModelHelper.SmallDarknet
    # Parameters for standard CNN-like networks
    CHANNELS: list[int] = field(default_factory=lambda: [3, 8, 16, 32, 32, 16])
    SIZES: list[int | tuple[int, int]] = field(default_factory=lambda: [5, 3, 3, 3, 3])   # Kernel sizes
    STRIDES: list[int | tuple[int, int]] = field(default_factory=lambda: [5, 3, 3, 2, 2])
    PADDINGS: list[int | tuple[int, int]] = field(default_factory=lambda: [2, 1, 1, 1, 1])
    BLOCKS: list[int] = field(default_factory=lambda: [1, 2, 2, 1, 1])


def get_model_config() -> ModelConfig:
    return ModelConfig()
