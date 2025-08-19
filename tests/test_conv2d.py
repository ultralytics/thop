# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn

from thop import profile


class TestUtils:
    """Utility class for testing Conv2D layers with and without bias, profiling MACs and parameters using THOP."""

    def test_conv2d_no_bias(self):
        """Tests a 2D Conv layer without bias using THOP profiling on predefined input dimensions and parameters."""
        n, in_c, ih, iw = 1, 3, 32, 32  # torch.randint(1, 10, (4,)).tolist()
        out_c, kh, kw = 12, 5, 5
        s, p, d, g = 1, 1, 1, 1

        net = nn.Conv2d(in_c, out_c, kernel_size=(kh, kw), stride=s, padding=p, dilation=d, groups=g, bias=False)
        data = torch.randn(n, in_c, ih, iw)
        out = net(data)

        _, _, oh, ow = out.shape
        print(f"Conv2d: in={ih}x{iw}, kernel={kh}x{kw}, stride={s}, padding={p}, out={oh}x{ow}")
        macs, params = profile(net, inputs=(data,))
        assert macs == 810000, f"{macs} v.s. 810000"

    def test_conv2d(self):
        """Tests Conv2D layer with bias, profiling MACs and params for specific input dimensions and layer configs."""
        n, in_c, ih, iw = 1, 3, 32, 32  # torch.randint(1, 10, (4,)).tolist()
        out_c, kh, kw = 12, 5, 5
        s, p, d, g = 1, 1, 1, 1

        net = nn.Conv2d(in_c, out_c, kernel_size=(kh, kw), stride=s, padding=p, dilation=d, groups=g, bias=True)
        data = torch.randn(n, in_c, ih, iw)
        out = net(data)

        _, _, oh, ow = out.shape
        print(f"Conv2d: in={ih}x{iw}, kernel={kh}x{kw}, stride={s}, padding={p}, out={oh}x{ow}")
        macs, params = profile(net, inputs=(data,))
        assert macs == 810000, f"{macs} v.s. 810000"

    def test_conv2d_random(self):
        """Test Conv2D layer with random parameters and validate the computed MACs and parameters using 'profile'."""
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
            print(f"Conv2d: in={ih}x{iw}, kernel={kh}x{kw}, stride={s}, padding={p}, out={oh}x{ow}")
            macs, params = profile(net, inputs=(data,))
            assert macs == n * out_c * oh * ow // g * in_c * kh * kw, (
                f"{macs} v.s. {n * out_c * oh * ow // g * in_c * kh * kw}"
            )

    def test_convtranspose2d_no_bias(self):
        """Tests a 2D ConvTranspose layer without bias using THOP profiling on predefined input dimensions and
        parameters.
        """
        n, in_c, ih, iw = 1, 3, 2, 2
        out_c, kh, kw = 1, 2, 2
        s, p, d, g = 2, 0, 1, 1

        net = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=(kh, kw), stride=s, padding=p, dilation=d, groups=g, bias=False
        )
        data = torch.randn(n, in_c, ih, iw)
        out = net(data)

        _, _, oh, ow = out.shape

        profile_result = profile(net, inputs=(data,))
        macs = profile_result[0]
        profile_result[1]
        # For ConvTranspose: MACs = input_size * (output_channels / groups) * kernel_size
        print(f"ConvTranspose2d: in={ih}x{iw}, kernel={kh}x{kw}, stride={s}, padding={p}, out={oh}x{ow}")
        expected_macs = n * in_c * ih * iw * (out_c // g) * kh * kw
        assert macs == expected_macs, f"{macs} v.s. {expected_macs}"

    def test_convtranspose2d(self):
        """Tests ConvTranspose2D layer with bias, profiling MACs and params for specific input dimensions and layer
        configs.
        """
        n, in_c, ih, iw = 1, 3, 2, 2
        out_c, kh, kw = 1, 2, 2
        s, p, d, g = 2, 0, 1, 1

        net = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=(kh, kw), stride=s, padding=p, dilation=d, groups=g, bias=True
        )
        data = torch.randn(n, in_c, ih, iw)
        out = net(data)

        _, _, oh, ow = out.shape

        profile_result = profile(net, inputs=(data,))
        macs = profile_result[0]
        profile_result[1]
        # For ConvTranspose: MACs = input_size * (output_channels / groups) * kernel_size
        print(f"ConvTranspose2d: in={ih}x{iw}, kernel={kh}x{kw}, stride={s}, padding={p}, out={oh}x{ow}")
        expected_macs = n * in_c * ih * iw * (out_c // g) * kh * kw
        assert macs == expected_macs, f"{macs} v.s. {expected_macs}"

    def test_convtranspose2d_groups(self):
        """Tests ConvTranspose2D layer with groups, validating MAC calculation for grouped transposed convolution."""
        n, in_c, ih, iw = 1, 8, 4, 4
        out_c, kh, kw = 4, 3, 3
        s, p, d, g = 1, 1, 1, 2

        net = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=(kh, kw), stride=s, padding=p, dilation=d, groups=g, bias=False
        )
        data = torch.randn(n, in_c, ih, iw)
        out = net(data)

        _, _, oh, ow = out.shape

        profile_result = profile(net, inputs=(data,))
        macs = profile_result[0]
        profile_result[1]
        # For ConvTranspose with groups: MACs = input_size * (output_channels / groups) * kernel_size
        print(f"ConvTranspose2d: in={ih}x{iw}, kernel={kh}x{kw}, stride={s}, padding={p}, out={oh}x{ow}")
        expected_macs = n * in_c * ih * iw * (out_c // g) * kh * kw
        assert macs == expected_macs, f"{macs} v.s. {expected_macs}"

    def test_convtranspose2d_random(self):
        """Test ConvTranspose2D layer with random parameters and validate the computed MACs and parameters using
        'profile'.
        """
        for _ in range(10):
            # Generate random parameters ensuring valid ConvTranspose configurations
            out_c, kh, kw = torch.randint(1, 10, (3,)).tolist()
            n, in_c = torch.randint(1, 5, (2,)).tolist()
            stride = int(torch.randint(1, 3, (1,)).item())  # stride
            padding = int(torch.randint(0, 2, (1,)).item())  # padding
            dilation, groups = 1, 1  # Keep dilation=1 and groups=1 for simplicity

            # Ensure input size is large enough to produce valid output
            # ConvTranspose output size formula: (input_size - 1) * stride - 2 * padding + kernel_size
            # To ensure positive output, we need: input_size >= (2 * padding + 1) / stride + 1
            min_input_size = max(3, (2 * padding + kh) // stride + 1, (2 * padding + kw) // stride + 1)
            ih, iw = torch.randint(min_input_size, min_input_size + 10, (2,)).tolist()

            net = nn.ConvTranspose2d(
                in_c,
                out_c,
                kernel_size=(kh, kw),
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            data = torch.randn(n, in_c, ih, iw)
            out = net(data)

            _, _, oh, ow = out.shape

            profile_result = profile(net, inputs=(data,))
            macs = profile_result[0]
            profile_result[1]
            # For ConvTranspose: MACs = input_size * (output_channels / groups) * kernel_size
            expected_macs = n * in_c * ih * iw * (out_c // groups) * kh * kw
            print(
                f"ConvTranspose2d: in={ih}x{iw}, kernel={kh}x{kw}, stride={stride}, padding={padding}, out={oh}x{ow}, macs={macs}"
            )
            assert macs == expected_macs, f"ConvTranspose2d MACs: {macs} v.s. {expected_macs}"

    def test_conv_vs_convtranspose_symmetry(self):
        """
        Test that Conv2d and ConvTranspose2d with symmetric configurations have equal MAC counts.

        Test case: Conv2d downsamples 4x4 -> 2x2, ConvTranspose2d upsamples 2x2 -> 4x4.
        """
        # Conv2d: 4x4 -> 2x2
        conv_net = nn.Conv2d(1, 3, kernel_size=2, stride=2, bias=False)
        conv_data = torch.randn(1, 1, 4, 4)
        conv_out = conv_net(conv_data)
        conv_profile_result = profile(conv_net, inputs=(conv_data,))
        conv_macs = conv_profile_result[0]
        conv_params = conv_profile_result[1]

        # ConvTranspose2d: 2x2 -> 4x4 (symmetric operation)
        convt_net = nn.ConvTranspose2d(3, 1, kernel_size=2, stride=2, bias=False)
        convt_data = torch.randn(1, 3, 2, 2)
        convt_out = convt_net(convt_data)
        convt_profile_result = profile(convt_net, inputs=(convt_data,))
        convt_macs = convt_profile_result[0]
        convt_params = convt_profile_result[1]

        # Verify symmetric operations have equal MAC counts
        assert conv_macs == convt_macs, f"Symmetric operations should have equal MACs: {conv_macs} != {convt_macs}"

        # Manual verification
        # Conv: output_size * (input_channels / groups) * kernel_size
        conv_expected = (
            conv_out.numel()
            * (conv_net.in_channels // conv_net.groups)
            * (conv_net.kernel_size[0] * conv_net.kernel_size[1])
        )
        # ConvTranspose: input_size * (output_channels / groups) * kernel_size
        convt_expected = (
            convt_data.numel()
            * (convt_net.out_channels // convt_net.groups)
            * (convt_net.kernel_size[0] * convt_net.kernel_size[1])
        )
        print(f"Conv2d: {conv_data.shape} -> {conv_out.shape}, MACs: {conv_macs}, Params: {conv_params}")
        print(f"ConvTranspose2d: {convt_data.shape} -> {convt_out.shape}, MACs: {convt_macs}, Params: {convt_params}")
        print(f"Conv2d expected MACs: {conv_expected}, ConvTranspose2d expected MACs: {convt_expected}")
        assert conv_macs == conv_expected, f"Conv2d MAC calculation incorrect: {conv_macs} != {conv_expected}"
        assert convt_macs == convt_expected, (
            f"ConvTranspose2d MAC calculation incorrect: {convt_macs} != {convt_expected}"
        )
