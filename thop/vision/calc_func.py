import warnings

import numpy as np
import torch


def l_prod(in_list):
    """Calculate the product of all elements in a list."""
    res = 1
    for _ in in_list:
        res *= _
    return res


def l_sum(in_list):
    """Calculate the sum of all elements in a list."""
    return sum(in_list)


def calculate_parameters(param_list):
    """Calculate the total number of parameters in a list of tensors."""
    return sum(torch.DoubleTensor([p.nelement()]) for p in param_list)


def calculate_zero_ops():
    """Return a tensor initialized to zero."""
    return torch.DoubleTensor([0])


def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    """Calculate FLOPs for a Conv2D layer given input/output sizes, kernel size, groups, and bias flag."""
    # n, in_c, ih, iw = input_size
    # out_c, in_c, kh, kw = kernel_size
    in_c = input_size[1]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])


def calculate_conv(bias, kernel_size, output_size, in_channel, group):
    warnings.warn("This API is being deprecated.")
    """Inputs are all numbers!"""
    return torch.DoubleTensor([output_size * (in_channel / group * kernel_size + bias)])


def calculate_norm(input_size):
    """Input is a number not a array or tensor."""
    return torch.DoubleTensor([2 * input_size])


def calculate_relu_flops(input_size):
    """Calculates the FLOPs for a ReLU activation function based on the input size."""
    return 0


def calculate_relu(input_size: torch.Tensor):
    """Convert an input tensor to a DoubleTensor with the same value."""
    warnings.warn("This API is being deprecated")
    return torch.DoubleTensor([int(input_size)])


def calculate_softmax(batch_size, nfeatures):
    """Calculate the number of FLOPs required for a softmax activation function based on batch size and number of
    features.
    """
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    return torch.DoubleTensor([int(total_ops)])


def calculate_avgpool(input_size):
    """Calculate the average pooling size given the input size."""
    return torch.DoubleTensor([int(input_size)])


def calculate_adaptive_avg(kernel_size, output_size):
    """Calculate the number of operations for adaptive average pooling given kernel and output sizes."""
    total_div = 1
    kernel_op = kernel_size + total_div
    return torch.DoubleTensor([int(kernel_op * output_size)])


def calculate_upsample(mode: str, output_size):
    """Calculate the number of operations for upsample methods given the mode and output size."""
    total_ops = output_size
    if mode == "bicubic":
        total_ops *= 224 + 35
    elif mode == "bilinear":
        total_ops *= 11
    elif mode == "linear":
        total_ops *= 5
    elif mode == "trilinear":
        total_ops *= 13 * 2 + 5
    return torch.DoubleTensor([int(total_ops)])


def calculate_linear(in_feature, num_elements):
    """Calculate the linear operation count for an input feature and number of elements."""
    return torch.DoubleTensor([int(in_feature * num_elements)])


def counter_matmul(input_size, output_size):
    """Calculate the total number of operations for a matrix multiplication given input and output sizes."""
    input_size = np.array(input_size)
    output_size = np.array(output_size)
    return np.prod(input_size) * output_size[-1]


def counter_mul(input_size):
    """Calculate the total number of operations for a matrix multiplication given input and output sizes."""
    return input_size


def counter_pow(input_size):
    """Calculate the total number of scalar multiplications for a power operation given an input size."""
    return input_size


def counter_sqrt(input_size):
    """Calculate the total number of scalar operations for a square root operation given an input size."""
    return input_size


def counter_div(input_size):
    """Calculate the total number of scalar operations for a division operation given an input size."""
    return input_size
