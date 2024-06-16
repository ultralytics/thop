import logging

import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from .calc_func import *

multiply_adds = 1


def count_parameters(m, x, y):
    """Calculate and update the total number of parameters in a given PyTorch model."""
    m.total_params[0] = calculate_parameters(m.parameters())


def zero_ops(m, x, y):
    """Incrementally add the number of zero operations to the model's total operations count."""
    m.total_ops += calculate_zero_ops()


def count_convNd(m: _ConvNd, x, y: torch.Tensor):
    """Calculate and add the number of convolutional operations (FLOPs) to the model's total operations count."""
    x = x[0]
    
    m.total_ops += calculate_conv2d_flops(
        input_size=list(x.shape),
        output_size=list(y.shape),
        kernel_size=list(m.weight.shape),
        groups=m.groups,
        bias=m.bias,
    )
    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    # m.total_ops += calculate_conv(
    #     bias_ops,
    #     torch.zeros(m.weight.size()[2:]).numel(),
    #     y.nelement(),
    #     m.in_channels,
    #     m.groups,
    # )


def count_convNd_ver2(m: _ConvNd, x, y: torch.Tensor):
    """Calculates the total operations for a convolutional layer and updates the layer's total_ops attribute."""
    x = x[0]

    # N x H x W (exclude Cout)
    output_size = torch.zeros((y.size()[:1] + y.size()[2:])).numel()
    # # Cout x Cin x Kw x Kh
    # kernel_ops = m.weight.nelement()
    # if m.bias is not None:
    #     # Cout x 1
    #     kernel_ops += + m.bias.nelement()
    # # x N x H x W x Cout x (Cin x Kw x Kh + bias)
    # m.total_ops += torch.DoubleTensor([int(output_size * kernel_ops)])
    m.total_ops += calculate_conv(m.bias.nelement(), m.weight.nelement(), output_size)


def count_normalization(m: nn.modules.batchnorm._BatchNorm, x, y):
    """Calculate and add the FLOPs for a batch normalization layer, considering elementwise operations and possible
    affine parameters.
    """
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
    """Calculate and update the total operation counts for a PReLU layer."""
    x = x[0]

    nelements = x.numel()
    if not m.training:
        m.total_ops += calculate_relu(nelements)


def count_relu(m, x, y):
    """Calculate and update the total operation counts for a ReLU layer."""
    x = x[0]
    m.total_ops += calculate_relu_flops(list(x.shape))


def count_softmax(m, x, y):
    """Calculate and update the total operation counts for a Softmax layer."""
    x = x[0]
    nfeatures = x.size()[m.dim]
    batch_size = x.numel() // nfeatures

    m.total_ops += calculate_softmax(batch_size, nfeatures)


def count_avgpool(m, x, y):
    """Calculate and update the total operation counts for an AvgPool layer."""
    # total_div = 1
    # kernel_ops = total_add + total_div
    num_elements = y.numel()
    m.total_ops += calculate_avgpool(num_elements)


def count_adap_avgpool(m, x, y):
    """Calculate and update the total operation counts for an AdaptiveAvgPool layer."""
    kernel = torch.div(torch.DoubleTensor([*(x[0].shape[2:])]), torch.DoubleTensor([*(y.shape[2:])]))
    total_add = torch.prod(kernel)
    num_elements = y.numel()
    m.total_ops += calculate_adaptive_avg(total_add, num_elements)


# TODO: verify the accuracy
def count_upsample(m, x, y):
    """Update the total operations counter in the given module for supported upsampling modes."""
    if m.mode not in (
        "nearest",
        "linear",
        "bilinear",
        "bicubic",
    ):  # "trilinear"
        logging.warning(f"mode {m.mode} is not implemented yet, take it a zero op")
        m.total_ops += 0
    else:
        x = x[0]
        m.total_ops += calculate_upsample(m.mode, y.nelement())


# nn.Linear
def count_linear(m, x, y):
    """Counts total operations for nn.Linear layers by calculating multiplications and additions based on input and
    output elements.
    """
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()

    m.total_ops += calculate_linear(total_mul, num_elements)
