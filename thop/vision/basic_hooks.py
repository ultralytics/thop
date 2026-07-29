# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import logging

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from thop.vision.calc_func import (
    calculate_avgpool,
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
    # nn.Softmax2d owns no dim at all: it always normalizes the channel axis, which is the third from the end
    # for both the batched and the unbatched input it accepts. Naming the one class it holds for keeps every
    # other dim-less module raising as it did, rather than being handed a channel axis it never asked for
    dim = -3 if isinstance(m, nn.Softmax2d) else m.dim
    if dim is None:
        # nn.Softmax(dim=None) is deprecated but still legal and still runs, resolving the dimension itself, so
        # the count follows that rule rather than raising: dimension 0 for 0-, 1- and 3-dimensional input, 1 otherwise
        dim = 0 if x.dim() in {0, 1, 3} else 1
    # a scalar normalizes over itself, which no shape entry can say: torch returns 1.0 for it, so the cost is
    # the one exponential and the one division that produce that, and indexing the empty shape only raises
    nfeatures = x.size()[dim] if x.dim() else 1
    # an empty normalized axis is no work at all, and it leaves no vector count to divide out of the element count
    batch_size = x.numel() // nfeatures if nfeatures else 0

    m.total_ops += calculate_softmax(batch_size, nfeatures)
    # what each variant costs on top of the plain softmax. Both terms are worth writing because calculate_softmax
    # charges nfeatures - 1 additions rather than nfeatures, so it is already exact to within one op per vector
    if isinstance(m, nn.Softmin):  # torch runs it as (-x).softmax(dim), one negation per element
        m.total_ops += x.numel()
    elif isinstance(m, nn.LogSoftmax):  # torch runs it as x - logsumexp(x), one logarithm per normalized vector
        m.total_ops += batch_size


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
