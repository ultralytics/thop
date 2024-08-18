import torch
import torch.nn as nn

from thop import profile


class TestUtils:
    """Utility functions for testing and profiling the efficiency of PyTorch neural network layers."""

    def test_matmul_case2(self):
        """Test matrix multiplication case by profiling FLOPs and parameters of a PyTorch nn.Linear layer."""
        n, in_c, out_c = 1, 100, 200
        net = nn.Linear(in_c, out_c)
        flops, params = profile(net, inputs=(torch.randn(n, in_c),))
        print(flops, params)
        assert flops == n * in_c * out_c

    def test_matmul_case2(self):
        """Tests matrix multiplication to profile FLOPs and parameters of nn.Linear layer using random dimensions."""
        for _ in range(10):
            n, in_c, out_c = torch.randint(1, 500, (3,)).tolist()
            net = nn.Linear(in_c, out_c)
            flops, params = profile(net, inputs=(torch.randn(n, in_c),))
            print(flops, params)
            assert flops == n * in_c * out_c

    def test_conv2d(self):
        """Tests FLOPs and parameters for a nn.Linear layer with random dimensions using torch.profiler."""
        n, in_c, out_c = torch.randint(1, 500, (3,)).tolist()
        net = nn.Linear(in_c, out_c)
        flops, params = profile(net, inputs=(torch.randn(n, in_c),))
        print(flops, params)
        assert flops == n * in_c * out_c
