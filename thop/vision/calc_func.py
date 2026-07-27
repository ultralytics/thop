# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


def l_prod(in_list):
    """Compute the product of all elements in the input list."""
    res = 1
    for _ in in_list:
        res *= _
    return res


def calculate_parameters(param_list):
    """Calculate the total number of parameters in a list of tensors using the product of their shapes."""
    return sum(p.nelement() for p in param_list)


def calculate_zero_ops():
    """Return the zero operation count."""
    return 0


def calculate_conv2d_flops(
    input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False, transpose: bool = False
):
    """Calculate FLOPs for a Conv2D layer using input/output sizes, kernel size, groups, and the bias flag."""
    # n, in_c, ih, iw = input_size
    # out_c, in_c, kh, kw = kernel_size
    if transpose:
        out_c = output_size[1]
        return l_prod(input_size) * (out_c // groups) * l_prod(kernel_size[2:])
    else:
        in_c = input_size[1]
        return l_prod(output_size) * (in_c // groups) * l_prod(kernel_size[2:])


def calculate_norm(input_size):
    """Compute the L2 norm of a tensor or array based on its input size."""
    return 2 * input_size


def calculate_relu_flops(input_size):
    """Calculates the FLOPs for a ReLU activation function based on the input tensor's dimensions."""
    return 0


def calculate_softmax(batch_size, nfeatures):
    """Compute FLOPs for a softmax activation given batch size and feature count."""
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    return int(total_ops)


def calculate_avgpool(input_size):
    """Calculate the average pooling size for a given input tensor."""
    return int(input_size)


# the interpolation cost of one output element, per upsampling mode. The nearest-neighbour modes gather an input
# element and do no arithmetic at all, so they cost nothing, as nn.MaxPool2d and nn.ZeroPad2d already do here
UPSAMPLE_OPS_PER_ELEMENT = {
    "nearest": 0,
    "nearest-exact": 0,
    "linear": 5,
    "bilinear": 11,
    "bicubic": 224 + 35,
    "trilinear": 13 * 2 + 5,
}


def calculate_linear(in_feature, num_elements):
    """Calculate the linear operation count for given input feature and number of elements."""
    return int(in_feature * num_elements)
