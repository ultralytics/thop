import torch
import torch.nn as nn

from thop import profile


class TestUtils:
    def test_conv2d_no_bias(self):
        """Tests a 2D convolutional layer without bias using THOP profiling with predefined input dimensions and
        convolution parameters.
        """
        n, in_c, ih, iw = 1, 3, 32, 32  # torch.randint(1, 10, (4,)).tolist()
        out_c, kh, kw = 12, 5, 5
        s, p, d, g = 1, 1, 1, 1

        net = nn.Conv2d(in_c, out_c, kernel_size=(kh, kw), stride=s, padding=p, dilation=d, groups=g, bias=False)
        data = torch.randn(n, in_c, ih, iw)
        out = net(data)

        _, _, oh, ow = out.shape

        flops, params = profile(net, inputs=(data,))
        assert flops == 810000, f"{flops} v.s. 810000"

    def test_conv2d(self):
        """Tests a Conv2D layer with specific input dimensions, kernel size, stride, padding, dilation, and groups."""
        n, in_c, ih, iw = 1, 3, 32, 32  # torch.randint(1, 10, (4,)).tolist()
        out_c, kh, kw = 12, 5, 5
        s, p, d, g = 1, 1, 1, 1

        net = nn.Conv2d(in_c, out_c, kernel_size=(kh, kw), stride=s, padding=p, dilation=d, groups=g, bias=True)
        data = torch.randn(n, in_c, ih, iw)
        out = net(data)

        _, _, oh, ow = out.shape

        flops, params = profile(net, inputs=(data,))
        assert flops == 810000, f"{flops} v.s. 810000"

    def test_conv2d_random(self):
        """Test Conv2D layer with random parameters and validate the computed FLOPs and parameters using 'profile'."""
        for _ in range(10):
            out_c, kh, kw = torch.randint(1, 20, (3,)).tolist()
            n, in_c, ih, iw = torch.randint(1, 20, (4,)).tolist()  # torch.randint(1, 10, (4,)).tolist()
            ih += kh
            iw += kw
            s, p, d, g = 1, 1, 1, 1

            net = nn.Conv2d(in_c, out_c, kernel_size=(kh, kw), stride=s, padding=p, dilation=d, groups=g, bias=False)
            data = torch.randn(n, in_c, ih, iw)
            out = net(data)

            _, _, oh, ow = out.shape

            flops, params = profile(net, inputs=(data,))
            print(flops, params)
            assert (
                flops == n * out_c * oh * ow // g * in_c * kh * kw
            ), f"{flops} v.s. {n * out_c * oh * ow // g * in_c * kh * kw}"
