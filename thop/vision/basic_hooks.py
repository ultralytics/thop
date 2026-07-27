# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import logging

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from thop.vision.calc_func import (
    UPSAMPLE_OPS_PER_ELEMENT,
    calculate_avgpool,
    calculate_conv2d_flops,
    calculate_linear,
    calculate_norm,
    calculate_relu_flops,
    calculate_softmax,
    calculate_upsample,
    calculate_zero_ops,
    l_prod,
)

multiply_adds = 1


def zero_ops(m, x, y):
    """Incrementally add zero operations to the model's total operations count."""
    m.total_ops += calculate_zero_ops()


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


def count_relu(m, x, y):
    """Calculate and update the total operation counts for a ReLU layer."""
    x = x[0]
    m.total_ops += calculate_relu_flops(list(x.shape))


def count_softmax(m, x, y):
    """Calculate and update the total operation counts for a Softmax layer in a PyTorch model."""
    x = x[0]
    nfeatures = x.size()[m.dim]
    batch_size = x.numel() // nfeatures

    m.total_ops += calculate_softmax(batch_size, nfeatures)


def count_avgpool(m, x, y):
    """Calculate and update the total number of operations (FLOPs) for an AvgPool layer based on the output elements."""
    # total_div = 1
    # kernel_ops = total_add + total_div
    num_elements = y.numel()
    m.total_ops += calculate_avgpool(num_elements)


def count_adap_avgpool(m, x, y):
    """Calculate and update the total operation counts for an AdaptiveAvgPool layer from the windows it pools."""
    # the windows nn.AdaptiveAvgPool* actually slices, per dimension: output j spans
    # [j * I // O, ceil((j + 1) * I / O)), so their sizes are integral and unequal whenever O does not divide I.
    # The input-over-output ratio this replaced was a float, which made total_ops fractional, and it was also
    # smaller than the window it stood for
    windows = l_prod(
        sum(-(-(j + 1) * i // o) - j * i // o for j in range(o)) for i, o in zip(x[0].shape[2:], y.shape[2:])
    )
    num_elements = y.numel()
    m.total_ops += windows * (num_elements // l_prod(y.shape[2:])) + num_elements  # one add per input, one divide out


# TODO: verify the accuracy
def count_upsample(m, x, y):
    """Update total operations counter for upsampling layers based on the mode used."""
    if m.mode not in UPSAMPLE_OPS_PER_ELEMENT:
        logging.getLogger(__name__).warning(f"mode {m.mode} is not implemented yet, take it a zero op")
    m.total_ops += calculate_upsample(m.mode, y.nelement())


# nn.Linear
def count_linear(m, x, y):
    """Counts total operations for nn.Linear layers using input and output element dimensions."""
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()

    m.total_ops += calculate_linear(total_mul, num_elements)


def count_multihead_attention(m: nn.MultiheadAttention, x, y):
    """Counts the four projections and the two attention matrix products of an nn.MultiheadAttention layer."""
    # the target length comes from the attention output, which carries the query's shape, and the source length from
    # the key: value is free to arrive as a keyword, which a forward hook never sees, and it is required to have the
    # key's sequence length anyway. Unbatched input drops the batch dimension, and batch_first swaps the other two
    out, key = y[0], x[1]
    embed_dim = out.shape[-1]
    tgt_len = out.shape[-2] if m.batch_first or out.dim() == 2 else out.shape[0]
    src_len = key.shape[-2] if m.batch_first or key.dim() == 2 else key.shape[0]
    batch_size = out.numel() // (tgt_len * embed_dim)

    # the query and output projections are embed_dim wide, the key and value ones are as wide as what they read.
    # Scores against the keys and the weighted sum of the values are tgt_len x src_len x embed_dim each once the
    # heads are summed, so num_heads does not appear: every head contributes embed_dim // num_heads of it
    projections = embed_dim * (2 * tgt_len * embed_dim + src_len * (m.kdim + m.vdim))
    attention = 2 * tgt_len * src_len * embed_dim
    m.total_ops += batch_size * (projections + attention)
