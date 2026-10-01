# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch
from torch import nn

from thop import profile
from thop.profile import _COUNTS_FUNCTIONS


class TestUtils:
    """Utility functions for testing and profiling the efficiency of PyTorch neural network layers."""

    def test_matmul_case2(self):
        """Test matrix multiplication case by profiling FLOPs and parameters of a PyTorch nn.Linear layer."""
        n, in_c, out_c = 1, 100, 200
        net = nn.Linear(in_c, out_c)
        flops, params = profile(net, inputs=(torch.randn(n, in_c),))
        print(flops, params)
        assert flops == n * in_c * out_c

    def test_matmul_case3(self):  # Note renamed to case3 by Glenn Jocher as duplicated above function name
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

    @pytest.mark.skipif(not _COUNTS_FUNCTIONS, reason="functional products are counted on torch>=1.13")
    def test_functional_products(self):
        """Functional products are counted, except inside a module whose own rule already accounts for them."""

        class Gram(nn.Module):
            """Multiply the input by its own transpose, which no module hook observes."""

            def forward(self, x):
                """Return the batched Gram matrix."""
                return x @ x.transpose(-2, -1)

        x = torch.randn(2, 8, 16)
        assert profile(Gram(), inputs=(x,), verbose=False)[0] == 2 * 8 * 8 * 16
        assert profile(Gram(), inputs=(x,), custom_ops={Gram: lambda m, x, y: None}, verbose=False)[0] == 0

        class Similarity(nn.Module):
            """Score each image cell against each text embedding, as an open-vocabulary head does."""

            def forward(self, x, t):
                """Return region-text similarities."""
                return torch.einsum("bchw,bkc->bkhw", x, t)

        x, t = torch.randn(2, 16, 4, 5), torch.randn(2, 3, 16)
        assert profile(Similarity(), inputs=(x, t), verbose=False)[0] == 2 * 3 * 4 * 5 * 16
        if hasattr(nn.functional, "scaled_dot_product_attention"):  # torch>=2.0

            class SDPA(nn.Module):
                """Attend from the input to itself."""

                def forward(self, q):
                    """Return scaled dot-product self-attention."""
                    return nn.functional.scaled_dot_product_attention(q, q, q)

            q = torch.randn(2, 4, 8, 16)
            assert profile(SDPA(), inputs=(q,), verbose=False)[0] == 2 * 4 * 8 * 8 * (16 + 16)
