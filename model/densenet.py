#
# This file contains the densenet building blocks
# Components have taken from the pytorch implementation
# of densenet and minor modifications wherever necessary.
# Link to pytorch implementation:
#   https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
#

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from model.attention import MultiHeadAttention


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module(
            'conv1',
            nn.Conv2d(
                num_input_features,
                bn_size *
                growth_rate,
                kernel_size=1,
                stride=1,
                bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Module):
    def __init__(
            self,
            num_layers,
            num_input_features,
            bn_size,
            growth_rate,
            drop_rate,
            num_heads,
            total_key_filters,
            total_value_filters,
            output_filters):
        super(DenseBlock, self).__init__()
        self.denselayers = nn.ModuleDict()
        self.multi_attn = MultiHeadAttention(
            num_heads=num_heads,
            num_input_channels=num_input_features,
            total_key_filters=total_key_filters,
            total_value_filters=total_value_filters,
            output_filters=output_filters)

        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features +
                i *
                growth_rate,
                growth_rate,
                bn_size,
                drop_rate)
            self.denselayers.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        # attend to the global features
        attn = self.multi_attn(x)

        for layer in self.denselayers:
            x = self.denselayers[layer](x)

        # concatentate the attn features with denselayer output
        return torch.cat([x, attn], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
