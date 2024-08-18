import torch
import torch.nn as nn

from thop import profile


class TestUtils:
    """Utility class for testing neural network components and profiling their performance metrics using THOP."""

    def test_relu(self):
        """Tests ReLU activation ensuring zero FLOPs and displays parameter count using THOP profiling."""
        n, in_c, _out_c = 1, 100, 200
        net = nn.ReLU()
        flops, params = profile(net, inputs=(torch.randn(n, in_c),))
        print(flops, params)
        assert flops == 0
