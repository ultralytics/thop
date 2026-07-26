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


def calculate_adaptive_avg(kernel_size, output_size):
    """Calculate FLOPs for adaptive average pooling given kernel size and output size."""
    total_div = 1
    kernel_op = kernel_size + total_div
    return int(kernel_op * output_size)


def calculate_upsample(mode: str, output_size):
    """Calculate the operations required for various upsample methods based on mode and output size."""
    total_ops = output_size
    if mode == "bicubic":
        total_ops *= 224 + 35
    elif mode == "bilinear":
        total_ops *= 11
    elif mode == "linear":
        total_ops *= 5
    elif mode == "trilinear":
        total_ops *= 13 * 2 + 5
    return int(total_ops)


def calculate_linear(in_feature, num_elements):
    """Calculate the linear operation count for given input feature and number of elements."""
    return int(in_feature * num_elements)
