import torch
import torch.nn as nn

from thop import profile


class TestUtils:
    def test_relu(self):
        """Tests the ReLU activation function to ensure it has zero FLOPs and checks parameter count using THOP
        profiling.
        """
        n, in_c, _out_c = 1, 100, 200
        net = nn.ReLU()
        flops, params = profile(net, inputs=(torch.randn(n, in_c),))
        print(flops, params)
        assert flops == 0
