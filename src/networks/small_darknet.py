from collections import OrderedDict
from typing import Union, Callable

import torch
import torch.nn as nn

from src.networks.layers import DarknetBlock
from src.torch_utils.networks.network_utils import (
    layer_init,
    get_cnn_output_size
)
from config.model_config import ModelConfig


class SmallDarknet(nn.Module):
    def __init__(self,
                 channels: list[int],
                 sizes: list[Union[int, tuple[int, int, int]]],
                 strides: list[Union[int, tuple[int, int, int]]],
                 paddings: list[Union[int, tuple[int, int, int]]],
                 blocks: list[int],
                 output_classes: int,
                 layer_init: Callable[[nn.Module], None] = layer_init, **kwargs):
        """
        Feature extractor
        Args:
            channels: List with the number of channels for each convolution
            sizes: List with the kernel size for each convolution
            strides: List with the stride for each convolution
            paddings: List with the padding for each convolution
            blocks: List with the number of blocks for the darknet blocks
            output_class: Number of output classes
            layer_init: Function used to initialise the layers of the network
        """
        super().__init__()
        self.feature_extractor = nn.Sequential(*[DarknetBlock(channels[i-1], channels[i], ModelConfig.NB_BLOCKS[i-1])
                                                 for i in range(1, len(channels))])

        feature_extractor_output_shape: int = get_cnn_output_size(kwargs["images_sizes"], sizes, strides, paddings,
                                                                  output_channels=channels[-1], dense=True, **kwargs)

        self.dense = nn.Linear(feature_extractor_output_shape, output_classes)

        # Used for grad-cam
        self.gradients: torch.Tensor = None
        self.activations: torch.Tensor = None

        self.apply(layer_init)

    def forward(self, inputs):
        x = self.feature_extractor(inputs)

        # Used for grad-cam
        if self.train and x.requires_grad:
            x.register_hook(self.activations_hook)
            self.activations = x

        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        x = torch.sigmoid(x)  # TODO: Maybe not sigmoid ?
        return x

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def get_gradients(self):
        return self.gradients

    def get_activations(self):
        return self.activations

    def get_weight_and_grads(self):
        weight_grads = OrderedDict()
        for ind, block in enumerate(self.feature_extractor):
            weight_grads[f"block_{ind}"] = block.conv.weight, block.conv.weight.grad
        weight_grads["dense"] = self.dense.weight, self.dense.weight.grad
        return weight_grads
