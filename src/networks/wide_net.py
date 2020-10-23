import math
from collections import OrderedDict

import torch
import torch.nn as nn

from src.networks.layers import DarknetConv
from config.model_config import ModelConfig


def get_cnn_output_size():
    width, height = ModelConfig.IMAGE_SIZES
    for kernel_size, stride, padding in zip(ModelConfig.SIZES, ModelConfig.STRIDES, ModelConfig.PADDINGS):
        width = ((width - kernel_size + 2*padding) // stride) + 1

    for kernel_size, stride, padding in zip(ModelConfig.SIZES, ModelConfig.STRIDES, ModelConfig.PADDINGS):
        height = ((height - kernel_size + 2*padding) // stride) + 1

    return width*height*ModelConfig.CHANNELS[-1]


class WideNet(nn.Module):
    def __init__(self):
        super(WideNet, self).__init__()
        self.output_size = ModelConfig.OUTPUT_CLASSES
        channels = ModelConfig.CHANNELS
        sizes = ModelConfig.SIZES
        strides = ModelConfig.STRIDES
        cnn_output_size = get_cnn_output_size()

        self.first_conv = DarknetConv(3, channels[0], sizes[0], stride=strides[0])
        self.blocks = nn.Sequential(*[DarknetConv(channels[i-1], channels[i], sizes[i], stride=strides[i],
                                                  padding=ModelConfig.PADDINGS[i])
                                      for i in range(1, len(channels))])
        self.dense = nn.Linear(cnn_output_size, self.output_size)

        # Used for grad-cam
        self.gradients: torch.Tensor = None
        self.activations: torch.Tensor = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        x = self.first_conv(inputs)
        for block in self.blocks:
            # Used for grad-cam
            if self.train and x.requires_grad:
                x.register_hook(self.activations_hook)
                self.activations = x

            x = block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
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
        return weight_grads
