from collections import OrderedDict

import torch
import torch.nn as nn
from lambda_networks import λLayer


from src.networks.layers import DarknetBlock
from src.torch_utils.networks.network_utils import (
    layer_init,
    get_cnn_output_size
)


# r is the receptive field for relative positional encoding (23 x 23)
# self.layer = λLayer(dim=32, dim_out=32, r=23, dim_k=16, heads=4, dim_u=4)



# Code is from https://gist.github.com/PistonY/ad33ab9e3d5f9a6a38345eb184e68cb4

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = λLayer(planes, dim_k=16, r=15, heads=4, dim_u=1)
        self.pool = nn.AvgPool2d(3, 2, 1) if stride != 1 else nn.Identity()

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act = nn.ReLU(inplace=True)
        self.downsample = nn.Identity() if downsample is None else downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


class LambdaResnet(nn.Module):
    def __init__(self, small_input=False):
        super().__init__()
        num_classes = ModelConfig.OUTPUT_CLASSES
        layers = [1, 2, 2, 1]
        channels = [16, 32, 48, 64]

        self.inplanes = channels[0]  # Not sure what this is yet

        if small_input:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(channels[0], layers[0])
        self.layer2 = self._make_layer(channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(channels[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(channels[-1] * Bottleneck.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion)
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
