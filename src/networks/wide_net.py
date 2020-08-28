import math

import torch
import torch.nn as nn

from src.networks.layers import DarknetConv
from config.model_config import ModelConfig


def get_cnn_output_size():
    width, height = ModelConfig.IMAGE_SIZES
    for kernel_size, stride in zip(ModelConfig.SIZES, ModelConfig.STRIDES):
        width = ((width - kernel_size) // stride) + 1

    for kernel_size, stride in zip(ModelConfig.SIZES, ModelConfig.STRIDES):
        height = ((height - kernel_size) // stride) + 1

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
        self.blocks = nn.Sequential(*[DarknetConv(channels[i-1], channels[i], sizes[i], stride=strides[i])
                                      for i in range(1, len(channels))])
        self.dense = nn.Linear(cnn_output_size, self.output_size)

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
            x = block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        return x
