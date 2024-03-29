from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from classification.torch_utils.networks.layers import DarknetBlock
from classification.torch_utils.networks.network_utils import layer_init


class SmallDarknet(nn.Module):
    def __init__(
        self,
        channels: list[int],
        blocks: list[int],
        nb_classes: int,
        image_sizes: tuple[int, int],
        layer_init: Callable[[nn.Module], None] = layer_init,
        **_kwargs: Any,
    ) -> None:
        """Feature extractor.

        Args:
            channels: List with the number of channels for each convolution
            blocks: List with the number of blocks for the darknet blocks
            nb_classes: Number of output classes
            image_sizes: Size of the input images.
            layer_init: Function used to initialise the layers of the network
            kwargs: ignored
        """
        super().__init__()
        self.feature_extractor = nn.Sequential(
            *[DarknetBlock(channels[i - 1], channels[i], blocks[i - 1]) for i in range(1, len(channels))]
        )

        fe_output = np.prod(self.feature_extractor(torch.zeros(1, 3, *image_sizes, device="cpu")).shape[1:])

        self.dense = nn.Linear(fe_output, nb_classes)

        # Used for grad-cam
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

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
        if self.gradients is None:
            msg = "Gradients have not been recorded yet"
            raise ValueError(msg)
        return self.gradients

    def get_activations(self) -> torch.Tensor:
        if self.activations is None:
            msg = "Activations have not been recorded yet"
            raise ValueError(msg)
        return self.activations

    def get_weight_and_grads(self) -> OrderedDict[str, tuple[torch.Tensor, torch.Tensor]]:
        weight_grads: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        for ind, block in enumerate(self.feature_extractor):
            weight_grads[f"block_{ind}"] = block.conv.conv.weight, block.conv.conv.weight.grad

        if self.dense.weight.grad is None:
            msg = "No gradients available for the dense layer."
            raise ValueError(msg)
        weight_grads["dense"] = self.dense.weight, self.dense.weight.grad
        return weight_grads
