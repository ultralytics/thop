import numpy as np
import torch
from onnx import numpy_helper

from thop.vision.basic_hooks import zero_ops

from .calc_func import (
    calculate_avgpool,
    calculate_conv,
    calculate_norm,
    calculate_softmax,
    calculate_zero_ops,
    counter_div,
    counter_matmul,
    counter_mul,
    counter_pow,
    counter_sqrt,
)


def onnx_counter_matmul(diction, node):
    """Calculates multiply-accumulate operations and output size for matrix multiplication in an ONNX model node."""
    input1 = node.input[0]
    input2 = node.input[1]
    input1_dim = diction[input1]
    input2_dim = diction[input2]
    out_size = np.append(input1_dim[0:-1], input2_dim[-1])
    output_name = node.output[0]
    macs = counter_matmul(input1_dim, out_size[-2:])
    return macs, out_size, output_name


def onnx_counter_add(diction, node):
    """Calculate multiply-accumulate operations (MACs), output size, and output name for ONNX addition nodes."""
    if np.array(diction[node.input[1]]).size >= np.array(diction[node.input[0]]).size:
        out_size = diction[node.input[1]]
    else:
        out_size = diction[node.input[0]]
    output_name = node.output[0]
    macs = calculate_zero_ops()
    # if '140' in diction:
    #     print(diction['140'],output_name)
    return macs, out_size, output_name


def onnx_counter_conv(diction, node):
    """Calculates MACs, output size, and name for an ONNX convolution node based on input tensor dimensions and node
    attributes.
    """
    # bias,kernelsize,outputsize
    dim_bias = 0
    input_count = 0
    for i in node.input:
        input_count += 1
    if input_count == 3:
        dim_bias = 1
        dim_weight = diction[node.input[1]]
    else:
        dim_weight = diction[node.input[1]]
    for attr in node.attribute:
        # print(attr)
        if attr.name == "kernel_shape":
            dim_kernel = attr.ints  # kw,kh
        if attr.name == "strides":
            dim_stride = attr.ints
        if attr.name == "pads":
            dim_pad = attr.ints
        if attr.name == "dilations":
            dim_dil = attr.ints
        if attr.name == "group":
            group = attr.i
            # print(dim_dil)
    dim_input = diction[node.input[0]]
    output_size = np.append(dim_input[0 : -np.array(dim_kernel).size - 1], dim_weight[0])
    hw = np.array(dim_input[-np.array(dim_kernel).size :])
    for i in range(hw.size):
        hw[i] = int((hw[i] + 2 * dim_pad[i] - dim_dil[i] * (dim_kernel[i] - 1) - 1) / dim_stride[i] + 1)
    output_size = np.append(output_size, hw)
    macs = calculate_conv(dim_bias, np.prod(dim_kernel), np.prod(output_size), dim_weight[1], group)
    output_name = node.output[0]

    # if '140' in diction:
    #     print("conv",diction['140'],output_name)
    return macs, output_size, output_name


def onnx_counter_constant(diction, node):
    """Calculate MACs, output size, and output name for a constant operation in an ONNX model."""
    macs = calculate_zero_ops()
    output_name = node.output[0]
    output_size = [1]
    # print(macs, output_size, output_name)
    return macs, output_size, output_name


def onnx_counter_mul(diction, node):
    """Calculate MACs, output size, and output name for a multiplication operation in an ONNX model."""
    if np.array(diction[node.input[1]]).size >= np.array(diction[node.input[0]]).size:
        input_size = diction[node.input[1]]
    else:
        input_size = diction[node.input[0]]
    macs = counter_mul(np.prod(input_size))
    output_size = diction[node.input[0]]
    output_name = node.output[0]
    return macs, output_size, output_name


def onnx_counter_bn(diction, node):
    """Calculates MACs, output size, and output name for batch normalization layers in an ONNX model."""
    input_size = diction[node.input[0]]
    macs = calculate_norm(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    return macs, output_size, output_name


def onnx_counter_relu(diction, node):
    """Calculates MACs, output size, and output name for ReLU layers in an ONNX model."""
    input_size = diction[node.input[0]]
    macs = calculate_zero_ops()
    output_name = node.output[0]
    output_size = input_size
    # print(macs, output_size, output_name)
    # if '140' in diction:
    #     print("relu",diction['140'],output_name)
    return macs, output_size, output_name


def onnx_counter_reducemean(diction, node):
    """Compute MACs, output size, and name for the ReduceMean ONNX node, adjusting dimensions based on the 'axes' and
    'keepdims' attributes.
    """
    keep_dim = 0
    for attr in node.attribute:
        if "axes" in attr.name:
            dim_axis = np.array(attr.ints)
        elif "keepdims" in attr.name:
            keep_dim = attr.i

    input_size = diction[node.input[0]]
    macs = calculate_zero_ops()
    output_name = node.output[0]
    if keep_dim == 1:
        output_size = input_size
    else:
        output_size = np.delete(input_size, dim_axis)
    # output_size = input_size
    return macs, output_size, output_name


def onnx_counter_sub(diction, node):
    """Computes MACs, output size, and output name for a given ONNX node with specified input size."""
    input_size = diction[node.input[0]]
    macs = calculate_zero_ops()
    output_name = node.output[0]
    output_size = input_size
    return macs, output_size, output_name


def onnx_counter_pow(diction, node):
    """Calculates MACs, output size, and output name for a given ONNX 'Pow' node with specified input size."""
    if np.array(diction[node.input[1]]).size >= np.array(diction[node.input[0]]).size:
        input_size = diction[node.input[1]]
    else:
        input_size = diction[node.input[0]]
    macs = counter_pow(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    return macs, output_size, output_name


def onnx_counter_sqrt(diction, node):
    """Calculate MACs and output information for the SQRT operation in an ONNX node."""
    input_size = diction[node.input[0]]
    macs = counter_sqrt(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    return macs, output_size, output_name


def onnx_counter_div(diction, node):
    """Calculate MACs and output information for the DIV operation in an ONNX node."""
    if np.array(diction[node.input[1]]).size >= np.array(diction[node.input[0]]).size:
        input_size = diction[node.input[1]]
    else:
        input_size = diction[node.input[0]]
    macs = counter_div(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    return macs, output_size, output_name


def onnx_counter_instance(diction, node):
    """Calculate MACs, output size, and name for an ONNX node instance."""
    input_size = diction[node.input[0]]
    macs = calculate_norm(np.prod(input_size))
    output_name = node.output[0]
    output_size = input_size
    return macs, output_size, output_name


def onnx_counter_softmax(diction, node):
    """Calculate MACs, output size, and name for an ONNX softmax node instance."""
    input_size = diction[node.input[0]]
    dim = node.attribute[0].i
    nfeatures = input_size[dim]
    batch_size = np.prod(input_size) / nfeatures
    macs = calculate_softmax(nfeatures, batch_size)
    output_name = node.output[0]
    output_size = input_size
    return macs, output_size, output_name


def onnx_counter_pad(diction, node):
    """Compute memory access cost (MACs), output size, and output name for ONNX pad operation."""
    # if
    # if (np.array(diction[node.input[1]]).size >= np.array(diction[node.input[0]]).size):
    #     input_size = diction[node.input[1]]
    # else:
    #     input_size = diction[node.input[0]]
    input_size = diction[node.input[0]]
    macs = calculate_zero_ops()
    output_name = node.output[0]
    output_size = input_size
    return macs, output_size, output_name


def onnx_counter_averagepool(diction, node):
    """Calculate MACs and output size for an AveragePool ONNX operation based on input dimensions and attributes."""
    macs = calculate_avgpool(np.prod(diction[node.input[0]]))
    output_name = node.output[0]
    dim_pad = None
    for attr in node.attribute:
        # print(attr)
        if attr.name == "kernel_shape":
            dim_kernel = attr.ints  # kw,kh
        elif attr.name == "strides":
            dim_stride = attr.ints
        elif attr.name == "pads":
            dim_pad = attr.ints
        elif attr.name == "dilations":
            dim_dil = attr.ints
            # print(dim_dil)
    dim_input = diction[node.input[0]]
    hw = dim_input[-np.array(dim_kernel).size :]
    if dim_pad is not None:
        for i in range(hw.size):
            hw[i] = int((hw[i] + 2 * dim_pad[i] - dim_kernel[i]) / dim_stride[i] + 1)
        output_size = np.append(dim_input[0 : -np.array(dim_kernel).size], hw)
    else:
        for i in range(hw.size):
            hw[i] = int((hw[i] - dim_kernel[i]) / dim_stride[i] + 1)
        output_size = np.append(dim_input[0 : -np.array(dim_kernel).size], hw)
    # print(macs, output_size, output_name)
    return macs, output_size, output_name


def onnx_counter_flatten(diction, node):
    """Returns MACs, output size, and output name for an ONNX Flatten node."""
    macs = calculate_zero_ops()
    output_name = node.output[0]
    axis = node.attribute[0].i
    input_size = diction[node.input[0]]
    output_size = np.append(input_size[axis - 1], np.prod(input_size[axis:]))
    # print("flatten",output_size)
    return macs, output_size, output_name


def onnx_counter_gemm(diction, node):
    """Calculate multiply–accumulate operations (MACs), output size, and name for ONNX Gemm node."""
    # Compute Y = alpha * A' * B' + beta * C
    input_size = diction[node.input[0]]
    dim_weight = diction[node.input[1]]
    # print(input_size,dim_weight)
    macs = np.prod(input_size) * dim_weight[1] + dim_weight[0]
    output_size = np.append(input_size[0:-1], dim_weight[0])
    output_name = node.output[0]
    return macs, output_size, output_name
    pass


def onnx_counter_maxpool(diction, node):
    """Calculate MACs and output size for ONNX MaxPool operation based on input node attributes and dimensions."""
    # print(node)
    macs = calculate_zero_ops()
    output_name = node.output[0]
    dim_pad = None
    for attr in node.attribute:
        # print(attr)
        if attr.name == "kernel_shape":
            dim_kernel = attr.ints  # kw,kh
        elif attr.name == "strides":
            dim_stride = attr.ints
        elif attr.name == "pads":
            dim_pad = attr.ints
        elif attr.name == "dilations":
            dim_dil = attr.ints
            # print(dim_dil)
    dim_input = diction[node.input[0]]
    hw = dim_input[-np.array(dim_kernel).size :]
    if dim_pad is not None:
        for i in range(hw.size):
            hw[i] = int((hw[i] + 2 * dim_pad[i] - dim_kernel[i]) / dim_stride[i] + 1)
        output_size = np.append(dim_input[0 : -np.array(dim_kernel).size], hw)
    else:
        for i in range(hw.size):
            hw[i] = int((hw[i] - dim_kernel[i]) / dim_stride[i] + 1)
        output_size = np.append(dim_input[0 : -np.array(dim_kernel).size], hw)
    # print(macs, output_size, output_name)
    return macs, output_size, output_name


def onnx_counter_globalaveragepool(diction, node):
    """Counts MACs and computes output size for a global average pooling layer in an ONNX model."""
    macs = calculate_zero_ops()
    output_name = node.output[0]
    input_size = diction[node.input[0]]
    output_size = input_size
    return macs, output_size, output_name


def onnx_counter_concat(diction, node):
    """Counts MACs and computes output size for a concatenation layer along a specified axis in an ONNX model."""
    # print(diction[node.input[0]])
    axis = node.attribute[0].i
    input_size = diction[node.input[0]]
    for i in node.input:
        dim_concat = diction[i][axis]
    output_size = input_size
    output_size[axis] = dim_concat
    output_name = node.output[0]
    macs = calculate_zero_ops()
    return macs, output_size, output_name


def onnx_counter_clip(diction, node):
    """Calculate MACs, output size, and output name for an ONNX node clip operation using provided dimensions and input
    size.
    """
    macs = calculate_zero_ops()
    output_name = node.output[0]
    input_size = diction[node.input[0]]
    output_size = input_size
    return macs, output_size, output_name


onnx_operators = {
    "MatMul": onnx_counter_matmul,
    "Add": onnx_counter_add,
    "Conv": onnx_counter_conv,
    "Mul": onnx_counter_mul,
    "Constant": onnx_counter_constant,
    "BatchNormalization": onnx_counter_bn,
    "Relu": onnx_counter_relu,
    "ReduceMean": onnx_counter_reducemean,
    "Sub": onnx_counter_sub,
    "Pow": onnx_counter_pow,
    "Sqrt": onnx_counter_sqrt,
    "Div": onnx_counter_div,
    "InstanceNormalization": onnx_counter_instance,
    "Softmax": onnx_counter_softmax,
    "Pad": onnx_counter_pad,
    "AveragePool": onnx_counter_averagepool,
    "MaxPool": onnx_counter_maxpool,
    "Flatten": onnx_counter_flatten,
    "Gemm": onnx_counter_gemm,
    "GlobalAveragePool": onnx_counter_globalaveragepool,
    "Concat": onnx_counter_concat,
    "Clip": onnx_counter_clip,
    None: None,
}
