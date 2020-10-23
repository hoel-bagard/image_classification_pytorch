from collections import OrderedDict

import torch
import torch.nn as nn

from src.networks.layers import (
    DarknetConv,
    DarknetBlock
)
from .network_utils import layer_init
from config.model_config import ModelConfig


class SmallDarknet(nn.Module):
    def __init__(self):
        super(SmallDarknet, self).__init__()
        self.output_size = 2
        channels = ModelConfig.CHANNELS

        self.first_conv = DarknetConv(3, ModelConfig.CHANNELS[0], 3)
        self.blocks = nn.Sequential(*[DarknetBlock(channels[i-1], channels[i], ModelConfig.NB_BLOCKS[i-1])
                                      for i in range(1, len(channels))])
        self.last_conv = nn.Conv2d(ModelConfig.CHANNELS[-1], self.output_size, 6, bias=False)

        # Used for grad-cam
        self.gradients: torch.Tensor = None
        self.activations: torch.Tensor = None

        self.apply(layer_init)

    def forward(self, inputs):
        x = self.first_conv(inputs)
        for block in self.blocks:
            x = block(x)

        # Used for grad-cam
        if self.train and x.requires_grad:
            x.register_hook(self.activations_hook)
            self.activations = x

        x = self.last_conv(x)
        x = torch.flatten(x, start_dim=1)
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
        weight_grads["first_conv"] = self.first_conv.conv.weight, self.first_conv.conv.weight.grad
        for ind, block in enumerate(self.blocks):
            weight_grads[f"block_{ind}"] = block.conv.weight, block.conv.weight.grad
        weight_grads["dense"] = self.dense.weight, self.dense.weight.grad
        weight_grads["last_conv"] = self.last_conv.conv.weight, self.last_conv.conv.weight.grad
        return weight_grads
