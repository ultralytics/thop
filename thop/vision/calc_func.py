# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from torch.nn.parameter import UninitializedParameter


def l_prod(in_list):
    """Compute the product of all elements in the input list."""
    res = 1
    for _ in in_list:
        res *= _
    return res


def calculate_parameters(param_list):
    """Sum the element counts of an iterable of parameters, skipping any a lazy module has not created yet."""
    # a lazy module's parameters hold no elements until its first forward pass, and asking one for its
    # element count raises rather than answering 0, so they are skipped instead of being counted.
    # isinstance rather than torch.nn.parameter.is_lazy: that helper only exists from torch 1.9 on
    return sum(p.nelement() for p in param_list if not isinstance(p, UninitializedParameter))


def calculate_zero_ops():
    """Return the zero operation count."""
    return 0


def calculate_conv2d_flops(
    input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False, transpose: bool = False
):
    """Calculate FLOPs for a ConvNd layer using input/output sizes, kernel size, groups, and the bias flag."""
    # A transpose scatters rather than gathers, so it walks the input and takes its channel from the output.
    volume, channels = (input_size, output_size) if transpose else (output_size, input_size)
    # The channel axis is counted from the trailing end, so unbatched input lands on it too.
    return l_prod(volume) * (channels[-(len(kernel_size) - 1)] // groups) * l_prod(kernel_size[2:])


def calculate_norm(input_size):
    """Compute the L2 norm of a tensor or array based on its input size."""
    return 2 * input_size


def calculate_softmax(batch_size, nfeatures):
    """Compute FLOPs for a softmax activation given batch size and feature count."""
    total_exp = nfeatures
    total_add = max(nfeatures - 1, 0)  # normalizing an empty axis costs nothing, rather than one negative addition
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    return int(total_ops)


def calculate_linear(in_feature, num_elements):
    """Calculate the linear operation count for given input feature and number of elements."""
    return int(in_feature * num_elements)
