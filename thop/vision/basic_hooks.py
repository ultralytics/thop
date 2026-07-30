# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import logging

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from thop.vision.calc_func import (
    calculate_conv2d_flops,
    calculate_linear,
    calculate_norm,
    calculate_softmax,
    calculate_zero_ops,
    l_prod,
)

multiply_adds = 1


def zero_ops(m, x, y):
    """Incrementally add zero operations to the model's total operations count."""
    m.total_ops += calculate_zero_ops()


def count_embedding(m, x, y):
    """Charge nothing for an embedding gather, warning when max_norm makes that incomplete."""
    if m.max_norm is not None:  # it renormalizes only the rows the indices push over the limit, so data decides
        logging.getLogger(__name__).warning("max_norm renormalization is not counted")


def count_convNd(m: _ConvNd, x, y: torch.Tensor):
    """Calculate and add the number of convolutional operations (FLOPs) for a ConvNd layer to the model's total ops."""
    x = x[0]

    m.total_ops += calculate_conv2d_flops(
        input_size=list(x.shape),
        output_size=list(y.shape),
        kernel_size=list(m.weight.shape),
        groups=m.groups,
        bias=m.bias,
        transpose=False,
    )
    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    # m.total_ops += calculate_conv(
    #     bias_ops,
    #     torch.zeros(m.weight.size()[2:]).numel(),
    #     y.nelement(),
    #     m.in_channels,
    #     m.groups,
    # )


def count_convtNd(m: _ConvNd, x, y: torch.Tensor):
    """Calculate and add the number of convolutional operations (FLOPs) for a ConvNd layer to the model's total ops."""
    x = x[0]

    m.total_ops += calculate_conv2d_flops(
        input_size=list(x.shape),
        output_size=list(y.shape),
        kernel_size=list(m.weight.shape),
        groups=m.groups,
        bias=m.bias,
        transpose=True,
    )


def count_normalization(m: nn.modules.batchnorm._BatchNorm, x, y):
    """Calculate and add the FLOPs for a batch normalization layer, including elementwise and affine operations."""
    # https://github.com/Lyken17/pytorch-OpCounter/issues/124
    # y = (x - mean) / sqrt(eps + var) * weight + bias
    x = x[0]
    # bn is by default fused in inference
    flops = calculate_norm(x.numel())
    if getattr(m, "affine", False) or getattr(m, "elementwise_affine", False):
        flops *= 2
    m.total_ops += flops


# def count_layer_norm(m, x, y):
#     x = x[0]
#     m.total_ops += calculate_norm(x.numel())


# def count_instance_norm(m, x, y):
#     x = x[0]
#     m.total_ops += calculate_norm(x.numel())


def count_prelu(m, x, y):
    """Calculate and update the total operation counts for a PReLU layer using input element number."""
    x = x[0]

    if not m.training:
        m.total_ops += x.numel()


def count_softmax(m, x, y):
    """Calculate and update the total operation counts for a Softmax layer in a PyTorch model."""
    x = x[0]
    # Softmax2d always normalizes channels, with or without a batch dimension.
    dim = -3 if isinstance(m, nn.Softmax2d) else m.dim
    if dim is None:
        # Match torch's deprecated implicit dimension.
        dim = 0 if x.dim() in {0, 1, 3} else 1
    # a scalar normalizes over itself, which no shape entry can say: torch returns 1.0 for it, so the cost is
    # the one exponential and the one division that produce that, and indexing the empty shape only raises
    nfeatures = x.size()[dim] if x.dim() else 1
    batch_size = x.numel() // nfeatures if nfeatures else 0

    m.total_ops += calculate_softmax(batch_size, nfeatures)
    if isinstance(m, nn.Softmin):
        m.total_ops += x.numel()
    elif isinstance(m, nn.LogSoftmax):
        m.total_ops += batch_size


def count_avgpool(m, x, y):
    """Calculate and update the total operation count for an AvgPool layer."""
    kernel = m.kernel_size
    if isinstance(kernel, int):  # only AvgPool2d and AvgPool3d keep it as passed, AvgPool1d normalizes to a tuple
        kernel = (kernel,) * (3 if isinstance(m, nn.AvgPool3d) else 2)
    dims = len(kernel)
    stride = (m.stride,) * dims if isinstance(m.stride, int) else m.stride
    padding = (m.padding,) * dims if isinstance(m.padding, int) else m.padding
    windows = 1
    for size, output, k, s, p in zip(x[0].shape[-dims:], y.shape[-dims:], kernel, stride, padding):
        lower, upper = (-p, size + p) if m.count_include_pad else (0, size)
        windows *= sum(max(min(i * s - p + k, upper) - max(i * s - p, lower), 0) for i in range(output))
    m.total_ops += l_prod(x[0].shape[:-dims]) * windows + y.numel()  # one add per input, one divide per output


def count_adap_avgpool(m, x, y):
    """Calculate and update the total operation counts for an AdaptiveAvgPool layer from the windows it pools."""
    # the windows nn.AdaptiveAvgPool* actually slices, per dimension: output j spans
    # [j * I // O, ceil((j + 1) * I / O)), so their sizes are integral and unequal whenever O does not divide I.
    # The input-over-output ratio this replaced was a float, which made total_ops fractional, and it was also
    # smaller than the window it stood for.
    # Every axis is walked, not only the pooled ones: an axis the layer leaves alone has one input per output, so
    # it contributes its own length and the batch and channel counts fall out of the same product. That is what
    # the pooled-axes slice had to multiply back by hand, and it needs no batch axis to be there to slice off.
    m.total_ops += (
        l_prod(
            o if i == o else sum(-(-(j + 1) * i // o) - j * i // o for j in range(o))
            for i, o in zip(x[0].shape, y.shape)
        )
        + y.numel()
    )  # one add per pooled input, one divide per output


# TODO: verify the accuracy
def count_upsample(m, x, y):
    """Update total operations counter for upsampling layers based on the mode used."""
    ops_per_element = {
        "nearest": 0,
        "nearest-exact": 0,
        "linear": 5,
        "bilinear": 11,
        "bicubic": 259,
        "trilinear": 31,
    }.get(m.mode)
    if ops_per_element is None:  # one lookup owns both the cost and whether the mode has one at all
        logging.getLogger(__name__).warning(f"mode {m.mode} is not implemented yet, take it a zero op")
        ops_per_element = 0
    m.total_ops += ops_per_element * y.nelement()


# nn.Linear
def count_linear(m, x, y):
    """Counts total operations for nn.Linear layers using input and output element dimensions."""
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()

    m.total_ops += calculate_linear(total_mul, num_elements)


def count_bilinear(m, x, y):
    """Count both contractions of each bilinear output."""
    m.total_ops += calculate_linear((m.in1_features + 1) * m.in2_features, y.numel())


def count_multihead_attention(m: nn.MultiheadAttention, x, y):
    """Count projections, attention matrix products, and softmax.

    MultiheadAttention calls its projection weights functionally, so none of this work reaches a child-module hook.
    Returned weights reveal appended key positions; calls that suppress them fall back to the key input.
    """
    out, weights = y if isinstance(y, tuple) else (y, None)
    batch_first = getattr(m, "batch_first", False)  # added in torch 1.9; before it the layout is always (L, N, E)
    tgt_len = out.shape[-2] if batch_first or out.dim() == 2 else out.shape[0]
    batch_size = 1 if out.dim() == 2 else out.shape[0 if batch_first else 1]
    appended = (m.bias_k is not None) + m.add_zero_attn  # each adds one key position, after the projections have run
    if weights is not None:
        attended = weights.shape[-1]
    else:
        key = x[1]
        attended = (key.shape[-2] if batch_first or key.dim() == 2 else key.shape[0]) + appended

    # Heads sum to embed_dim across both attention matrix products, so num_heads only multiplies the softmax rows.
    projections = m.embed_dim * (2 * tgt_len * m.embed_dim + (attended - appended) * (m.kdim + m.vdim))
    attention = 2 * tgt_len * attended * m.embed_dim
    softmax = calculate_softmax(batch_size * m.num_heads * tgt_len, attended)
    m.total_ops += batch_size * (projections + attention) + softmax
