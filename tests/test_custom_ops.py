# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
from torch import nn

from thop import profile


class CustomModule(nn.Module):
    """A third-party style module with no built-in counting rule, wrapping a single convolution."""

    def __init__(self):
        """Initialize the example convolution layer."""
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)

    def forward(self, x):
        """Apply the wrapped convolution to the input tensor."""
        return self.conv(x)


def count_with_tensor(module, x, y):
    """Accumulate MACs with a tensor, the recipe documented in the README for custom_ops."""
    k_h, k_w = module.conv.kernel_size
    _, _, h, w = y.shape
    macs = (k_h * k_w * module.conv.in_channels * module.conv.out_channels * h * w) / module.conv.groups
    module.total_ops += torch.DoubleTensor([macs])


def count_with_int(module, x, y):
    """Accumulate MACs with a plain Python number."""
    k_h, k_w = module.conv.kernel_size
    _, _, h, w = y.shape
    module.total_ops += k_h * k_w * module.conv.in_channels * module.conv.out_channels * h * w // module.conv.groups


class TestUtils:
    """Tests that custom_ops rules are honored and always yield plain Python numbers."""

    def test_custom_ops_tensor_accumulator(self):
        """A rule accumulating a DoubleTensor must still return plain floats, per the documented README recipe."""
        net = CustomModule()
        macs, params = profile(
            net, inputs=(torch.randn(1, 3, 224, 224),), custom_ops={CustomModule: count_with_tensor}, verbose=False
        )
        assert type(macs) is float, f"expected float, got {type(macs).__name__}"
        assert type(params) is float, f"expected float, got {type(params).__name__}"
        assert macs == 173408256.0, f"{macs} v.s. 173408256.0"
        assert params == 1792.0, f"{params} v.s. 1792.0"

    def test_custom_ops_int_accumulator(self):
        """A rule accumulating a plain int must agree exactly with the tensor-accumulating rule."""
        net = CustomModule()
        macs, params = profile(
            net, inputs=(torch.randn(1, 3, 224, 224),), custom_ops={CustomModule: count_with_int}, verbose=False
        )
        assert type(macs) is float, f"expected float, got {type(macs).__name__}"
        assert macs == 173408256.0, f"{macs} v.s. 173408256.0"
        assert params == 1792.0, f"{params} v.s. 1792.0"

    def test_builtin_rules_return_plain_floats(self):
        """Built-in rules must return plain Python floats, not tensors or ints."""
        net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        macs, params = profile(net, inputs=(torch.randn(1, 3, 32, 32),), verbose=False)
        assert type(macs) is float, f"expected float, got {type(macs).__name__}"
        assert type(params) is float, f"expected float, got {type(params).__name__}"
