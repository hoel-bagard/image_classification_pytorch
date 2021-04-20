from typing import Union, Optional

from src.networks.build_network import ModelHelper


class ModelConfig:
    # Training parameters
    BATCH_SIZE         = 16            # Batch size
    MAX_EPOCHS         = 2000          # Number of Epochs
    LR                 = 1e-3          # Learning Rate
    LR_DECAY           = 0.998
    DECAY_START        = 20
    REG_FACTOR         = 0.005       # Regularization factor (Used to be 0.005 for the fit mode)

    LABEL_SMOOTHING: float = 0.1   # Value to use for label smoothing if the loss supports it. 0 for no smoothing.
    LOSS_WEIGTHS: Optional[list[float]] = None   # Weights the classes during the loss

    # Data processing
    IMAGE_SIZES: tuple[int, int] = (256, 256)
    GRAYSCALE = False   # Not fully implemented for the True case

    # Network part
    MODEL = ModelHelper.SmallDarknet
    # Parameters for standard CNN-like networks
    CHANNELS: list[int] = [3, 8, 16, 32, 32, 16]
    SIZES: list[Union[int, tuple[int, int]]]  = [5, 3, 3, 3, 3]   # Kernel sizes
    STRIDES: list[Union[int, tuple[int, int]]]  = [5, 3, 3, 2, 2]
    PADDINGS: list[Union[int, tuple[int, int]]]  = [2, 1, 1, 1, 1]
    BLOCKS: list[int] = [1, 2, 2, 1, 1]
    # Parameters for lambda networks
    SMALL_INPUT = False    # If True, the first conv of the layer has a bigger kernel / stride / padding
    LAMBDA_LAYERS = [3, 4, 6, 3]
    LAMBDA_CHANNELS = [64, 128, 256, 512]
    LAMBDA_STRIDES = [1, 2, 2, 2]
