from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np
import torch
from torch import nn

from classification.torch_utils.networks.cnn_feature_extractor import CNNFeatureExtractor
from classification.torch_utils.networks.network_utils import layer_init


class CNN(nn.Module):
    def __init__(
        self,
        channels: list[int],
        sizes: list[int],
        strides: list[int | tuple[int, int]],
        paddings: list[int | tuple[int, int]],
        nb_classes: int,
        image_sizes: tuple[int, int],
        layer_init: Callable[[nn.Module], None] = layer_init,
    ) -> None:
        """Feature extractor.

        Args:
            channels: List with the number of channels for each convolution
            sizes: List with the kernel size for each convolution
            strides: List with the stride for each convolution
            paddings: List with the padding for each convolution
            nb_classes: Number of output classes
            image_sizes: Size of the input images.
            layer_init: Function used to initialize the layers of the network

        """
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(channels, sizes, strides, paddings)

        fe_output = np.prod(self.feature_extractor(torch.zeros(1, 3, *image_sizes, device="cpu")).shape[1:])

        self.dense = nn.Linear(fe_output, nb_classes)

        # Used for grad-cam
        self.gradients: torch.Tensor = None
        self.activations: torch.Tensor = None

        self.apply(layer_init)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(inputs)

        # Used for grad-cam
        if self.train and x.requires_grad:
            x.register_hook(self.activations_hook)
            self.activations = x

        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        return x

    # hook for the gradients of the activations
    def activations_hook(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def get_gradients(self) -> torch.Tensor:
        return self.gradients

    def get_activations(self) -> torch.Tensor:
        return self.activations

    def get_weight_and_grads(self) -> OrderedDict[str, tuple[torch.Tensor, torch.Tensor]]:
        weight_grads: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        for ind, block in enumerate(self.feature_extractor):
            weight_grads[f"block_{ind}"] = block.conv.weight, block.conv.weight.grad
        weight_grads["dense"] = self.dense.weight, self.dense.weight.grad
        return weight_grads
