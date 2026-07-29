# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from thop import profile

try:
    from torch.utils._python_dispatch import TorchDispatchMode  # noqa: F401

    HAS_DISPATCH_MODE = True
except ImportError:
    HAS_DISPATCH_MODE = False


class BatchedMatMul(nn.Module):
    """Multiply two batches of matrices functionally."""

    def forward(self, left, right):
        """Return the batched matrix product."""
        return left @ right


class ViTAttention(nn.Module):
    """Minimal ViT attention using functional scaled dot-product attention."""

    def __init__(self, dim, heads):
        """Initialize fused QKV and output projections."""
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        """Project tokens, apply functional attention, and project its output."""
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, channels // self.heads)
        query, key, value = (tensor.transpose(1, 2) for tensor in qkv.unbind(2))
        output = F.scaled_dot_product_attention(query, key, value)
        return self.proj(output.transpose(1, 2).reshape(batch, tokens, channels))


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

    @pytest.mark.skipif(not HAS_DISPATCH_MODE, reason="functional operation counting requires torch dispatch modes")
    def test_batched_matmul(self):
        """Count a functional batched matrix product."""
        batch, rows, reduction, columns = 2, 3, 5, 7
        left = torch.randn(batch, rows, reduction)
        right = torch.randn(batch, reduction, columns)
        macs, params = profile(BatchedMatMul(), inputs=(left, right), verbose=False)
        assert macs == batch * rows * reduction * columns
        assert params == 0

    @pytest.mark.skipif(not HAS_DISPATCH_MODE, reason="functional operation counting requires torch dispatch modes")
    def test_custom_rule_owns_batched_matmul(self):
        """Do not count a functional product twice when a custom rule owns its module."""
        batch, rows, reduction, columns = 2, 3, 5, 7
        left = torch.randn(batch, rows, reduction)
        right = torch.randn(batch, reduction, columns)

        def count_product(module, x, y):
            """Count the module's batched matrix product."""
            module.total_ops += batch * rows * reduction * columns

        macs, _ = profile(
            BatchedMatMul(),
            inputs=(left, right),
            custom_ops={BatchedMatMul: count_product},
            verbose=False,
        )
        assert macs == batch * rows * reduction * columns

    @pytest.mark.skipif(
        not HAS_DISPATCH_MODE or not hasattr(F, "scaled_dot_product_attention"),
        reason="functional attention requires torch>=2.0",
    )
    def test_vit_scaled_dot_product_attention(self):
        """Count ViT projections and both functional attention products."""
        batch, tokens, channels, heads = 2, 7, 16, 4
        model = ViTAttention(channels, heads)
        macs, params = profile(model, inputs=(torch.randn(batch, tokens, channels),), verbose=False)
        projections = batch * tokens * channels * (3 * channels + channels)
        attention = 2 * batch * tokens * tokens * channels
        assert macs == projections + attention
        assert params == 4 * channels * channels

    def test_multihead_attention_is_not_double_counted(self):
        """Keep the module rule authoritative when MultiheadAttention dispatches functionally."""
        batch, tokens, channels, heads = 2, 7, 16, 4
        model = nn.MultiheadAttention(channels, heads)
        inputs = torch.randn(tokens, batch, channels)
        macs, _ = profile(model, inputs=(inputs, inputs, inputs, None, False), verbose=False)
        projections = channels * (4 * tokens * channels)
        attention = 2 * tokens * tokens * channels
        softmax = batch * heads * tokens * (3 * tokens - 1)
        assert macs == batch * (projections + attention) + softmax
