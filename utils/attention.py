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
        input: A tensor of shape [batch, channels, _h, _w]
        conv_qk: A module list of conv layers for calculating
            query and keys, having total_key_filters as output channels
        conv_v: A conv layer for calculating values having total_value_filters
            as output channels

    Returns:
        A tuple (q, k, v) of tensors with the following shapes
        q: [batch, total_key_filters, _h, _w]
        k: [batch, total_key_filters, _h, _w]
        v: [batch, total_value_filters, _h, _w]

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
        x: A tensor of shape [batch, heads, channels, _h, _w]

    Return
                                        (_h x _w)
        A tensor of shape [batch, heads, channels, length]
    """
    batch, heads, channels, _h, _w = x.shape

    return x.reshape(batch, heads, channels, _h * _w)


def split_heads(x, num_heads):
    """
    Split the tensor into multiple heads

    Args:
        x: A tensor of shape [batch, channel, _h, _w]
        num_heads: Number of heads

    Returns:
        A tensor of shape [batch, head, channels/head, _h, _w]

    Raise:
        ValueError if channels is not divisible by num_heads
    """
    channels = x.shape[1]

    if channels % num_heads != 0:
        raise ValueError('%s is not divisible by %s' % (channels, num_heads))

    channels_per_head = channels // num_heads
    out_shape = x.shape[:1] + \
        torch.Size((num_heads, channels_per_head)) + x.shape[2:]

    return x.reshape(out_shape)


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
        q: (query)  A tensor with shape [batch, heads, channels_k, _h, _w]
        k: (keys)   A tensor with shape [batch, heads, channels_k, _h, _w]
        v: (values) A tensor with shape [batch, heads, channels_v, _h, _w]

    Return:
        A tensor of shape [batch, heads, _h, _w, channels_v]
    """
    # store shape in which to return
    v_shape = q.shape[:2] + v.shape[2:3] + q.shape[3:]

    # flatten
    q_, k_, v_ = [flatten(x) for x in (q, k, v)]

    qk_t = bmm_(q_.permute(0, 1, 3, 2), k_)
    w = F.softmax(qk_t, dim=-1)

    dropout = nn.Dropout(dropout_rate)

    w = dropout(w)

    # shape [batch, heads, length, channels]
    dot = bmm_(w, v_.permute(0, 1, 3, 2))

    return dot.permute(0, 1, 3, 2).reshape(v_shape)


def combine_heads(x):
    """
    Combine attention heads

    Args:
        x: A input tensor of shape [batch, heads, channels/heads, _h, _w]

    Reutrns:
        A tensor of shape [batch, channels, _h, _w]
    """

    out_shape = x.shape[:1] + \
        torch.Size((x.shape[1] * x.shape[2], )) + x.shape[3:]
    return x.reshape(out_shape)


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
        self.total_key_filters = total_key_filters
        self.total_value_filters = total_value_filters

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
        # split shape [batch, heads, channels/heads, _h, _w]
        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)

        key_filters_per_head = self.total_key_filters / self.num_heads

        # divide by √d_k
        q *= key_filters_per_head ** -0.5

        # apply attention
        attn = attention(q, k, v, 0.2)

        # attn shape [batch, _h, _w, total_value_filters]
        attn = combine_heads(attn)

        attn = self.conv_out(attn)
        return attn


if name == '__main__':
    multi_attn = MultiHeadAttention(
        num_heads=8,
        num_input_channels=3,
        total_key_filters=32,
        total_value_filters=32,
        output_filters=32)
    batch, h, w, channels = 10, 16, 16, 3
    x = torch.rand(batch, channels, h, w)
    out = multi_attn(x)
    print(out.shape)
