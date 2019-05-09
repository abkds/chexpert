import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


def clones(module, N):
    "Return n identical modules"
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def compute_qkv(input, conv_qk, conv_v):
    """
    Computes the query, key and value vectors for input

    Args:
        input: A tensor of shape [batch, _h, _w, channels]
        conv_qk: A module list of conv layers for calculating
            query and keys, having total_key_filters as output channels
        conv_v: A conv layer for calculating values having total_value_filters
            as output channels

    Returns:
        A tuple (q, k, v) of tensors with the following shapes
        [batch, _h, _w, total_key_filters], [batch, _h, _w, total_key_filters], [batch, _h, _w, total_value_filters]

    """
    # linear transformation for query
    q = conv_qk[0](input)

    # linear transformation for key
    k = conv_qk[1](input)

    # linear transformation for value
    v = conv_v(input)

    return q, k, v


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 num_heads,
                 num_input_channels,
                 total_key_filters,
                 total_value_filters,
                 output_filters):
        "Store the number of filters and attention heads"
        super(MultiHeadAttention, self).__init__()
        assert total_key_filters % num_heads == 0
        # We assume total_key_filters and total_value_filters to be same
        self.num_heads = num_heads

        # conv layers for images are like the linear
        # layers for text embeddings, query and key
        # have same number of filters ie key_filters
        # hence declared 2 of the same convolutions
        self.conv_qk = clones(nn.Conv2d(num_input_channels,
                                        total_key_filters, 1, 1), 2)
        self.conv_v = nn.Conv2d(num_input_channels, total_value_filters, 1, 1)
        self.conv_out = nn.Conv2d(total_value_filters, output_filters, 1, 1)

    def forward(self, input):
        """
        Args:
            input: A tensor of shape [batch, _h, _w, channels]

        Returns:
            A tensor of shape [batch, _h, _w, output_filters]
        """
        # compute the query, key and values from input
        q, k, v = compute_qkv(input, self.conv_qk, self.conv_v)
