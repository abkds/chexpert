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


def flatten(x):
    """
    Flatten x
       _       ____
     /_/|     |    |
    | | |  -> |    |
    |_|/      |____|

    Args:
        x: A tensor of shape [batch, heads, _h, _w, channels]

    Return
                                        (_h x _w)
        A tensor of shape [batch, heads, length, channels]
    """
    batch, heads, _h, _w, channels = x.shape

    return x.reshape(batch, heads, _h * _w, channels)


def split_last_dimension(x, n):
    """
    Split the last dimension of a tensor

    Args:
        x: A tensor of shape [..., m]
        n: Number of dimensions to split x into.

    Returns:
        A tensor of shape [..., n, m//n]

    Raises:
        ValueError if m is not divisible by n
    """

    if m % n != 0:
        raise ValueError('%s is not divisible by %s' % (m, n))

    first, last = list(x.shape[:-1]), x.shape[-1]
    split_last = last // n
    first.extend([n, split_last])
    x.reshape(first)


def split_heads(x, num_heads):
    """
    Split the tensor into multiple heads

    Args:
        x: A tensor of shape [batch, _h, _w, channels]
        num_heads: Number of heads

    Returns:
        A tensor of shape [batch, head, _h, _w, channels/head]

    Raise:
        ValueError if channels is not divisible by num_heads
    """
    return torch.transpose(split_last_dimension(x, num_heads), 1, 3)


def bmm_(a, b):
    """
    Multiply two tensors of rank greater than 3

    Args:
        a: A tensor of shape [..., m, n]
        b: A tensor of shape [..., n, p]

    Returns:
        A tensor of shape [..., m, p]

    Raises:
        ValueError if remaining shape except for the last two dimensions
        are not same
    """

    if a.shape[:-2] != b.shape[:-2]:
        raise ValueError(
            """Remaining shape except the last two dimensions are not equal.
                            Shape of a : %s
                            Shape of b : %s
                            """ %
            (a.shape, b.shape))

    a_ = a.view(-1, a.shape[-2], a.shape[-1])
    b_ = b.view(-1, b.shape[-2], b.shape[-1])

    out = torch.bmm(a_, b_)
    out_shape = a.shape[:-2] + out.shape[-2:]
    return out.reshape(out_shape)


def attention(q, k, v, dropout_rate=0.1):
    """
    Self Attention mechanism

    Args:
        q: (query)  A tensor with shape [batch, heads, _h, _w, channels_k]
        k: (keys)   A tensor with shape [batch, heads, _h, _w, channels_k]
        v: (values) A tensor with shape [batch, heads, _h, _w, channels_v]

    Return:
        A tensor of shape [batch, heads, _h, _w, channels_v]
    """
    # store shape in which to return
    v_shape = torch.Size(q)[:-1] + torch.size(v)[-1]

    # flatten
    q_, k_, v_ = [flatten(x) for x in (q, k, v)]

    qk_t = bmm_(q_, k_.transpose(-2, -1))
    w = F.softmax(qk_t, dim=-1)

    dropout = nn.Dropout(dropout_rate)

    w = dropout(w)

    return bmm_(w, v_).reshape(v_shape)


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

        # split heads for q, k and v
        # after splitting shape is [batch, num_heads, _h, _w, channels /
        # num_heads]
        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)

        key_filters_per_head = self.total_key_filters / self.num_heads

        # divide by √d_k
        q *= key_filters_per_head ** -0.5

