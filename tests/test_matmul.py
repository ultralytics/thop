import torch
import torch.nn as nn

from thop import profile


class TestUtils:
    def test_matmul_case2(self):
        """Test matrix multiplication case asserting the FLOPs and parameters of a nn.Linear layer."""
        n, in_c, out_c = 1, 100, 200
        net = nn.Linear(in_c, out_c)
        flops, params = profile(net, inputs=(torch.randn(n, in_c),))
        print(flops, params)
        assert flops == n * in_c * out_c

    def test_matmul_case2(self):
        """Tests matrix multiplication to assert FLOPs and parameters of nn.Linear layer using random dimensions."""
        for _ in range(10):
            n, in_c, out_c = torch.randint(1, 500, (3,)).tolist()
            net = nn.Linear(in_c, out_c)
            flops, params = profile(net, inputs=(torch.randn(n, in_c),))
            print(flops, params)
            assert flops == n * in_c * out_c

    def test_conv2d(self):
        """Tests the number of FLOPs and parameters for a randomly initialized nn.Linear layer using torch.profiler."""
        n, in_c, out_c = torch.randint(1, 500, (3,)).tolist()
        net = nn.Linear(in_c, out_c)
        flops, params = profile(net, inputs=(torch.randn(n, in_c),))
        print(flops, params)
        assert flops == n * in_c * out_c
