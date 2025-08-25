<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🚀 THOP: PyTorch-OpCounter

Welcome to the [THOP](https://github.com/ultralytics/thop) repository, your comprehensive solution for profiling [PyTorch](https://pytorch.org/) models by computing the number of Multiply-Accumulate Operations (MACs) and parameters. Developed by Ultralytics, this tool is essential for [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) practitioners aiming to evaluate model efficiency and performance, crucial aspects discussed in our [model training tips guide](https://docs.ultralytics.com/guides/model-training-tips/).

[![Ultralytics Actions](https://github.com/ultralytics/thop/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/thop/actions/workflows/format.yml)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

## 📄 Description

THOP offers an intuitive API designed to profile PyTorch models by calculating the total number of MACs and parameters. This functionality is vital for assessing the computational efficiency and memory footprint of deep learning models, helping developers optimize performance for deployment, especially on [edge devices](https://www.ultralytics.com/glossary/edge-ai). Understanding these metrics is key to selecting the right model architecture, a topic explored in our [model comparison pages](https://docs.ultralytics.com/compare/).

## 📦 Installation

Get started with THOP quickly by installing it via pip:

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics-thop?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics-thop/) [![Downloads](https://static.pepy.tech/badge/ultralytics-thop)](https://clickpy.clickhouse.com/dashboard/ultralytics-thop) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics-thop?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics-thop/)

```bash
pip install ultralytics-thop
```

Alternatively, for the latest features and updates, install directly from the GitHub repository:

```bash
pip install --upgrade git+https://github.com/ultralytics/thop.git
```

This ensures you have the most recent version, incorporating the latest improvements and bug fixes.

## 🛠️ How to Use

### Basic Usage

Profiling a standard PyTorch model like [ResNet50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) is straightforward. Import the necessary libraries, load your model and a sample input tensor, then use the `profile` function:

```python
import torch
from torchvision.models import resnet50  # Example model

from thop import profile  # Import the profile function from THOP

# Load a pre-trained model (e.g., ResNet50)
model = resnet50()

# Create a dummy input tensor matching the model's expected input shape
dummy_input = torch.randn(1, 3, 224, 224)

# Profile the model
macs, params = profile(model, inputs=(dummy_input,))

print(f"MACs: {macs}, Parameters: {params}")
# Expected output: MACs: 4139975680.0, Parameters: 25557032.0
```

### Define Custom Rules for Third-Party Modules

If your model includes custom or third-party modules not natively supported by THOP, you can define custom profiling rules using the `custom_ops` argument. This allows for accurate profiling even with complex or non-standard architectures, which is useful when working with models like those found in the [Ultralytics models section](https://docs.ultralytics.com/models/).

```python
import torch
import torch.nn as nn

from thop import profile


# Define your custom module
class YourCustomModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers, e.g., a convolution
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)


# Define a custom counting function for your module
# This function should calculate and return the MACs for the module's operations
def count_your_custom_module(module, x, y):
    # Example: Calculate MACs for the conv layer
    # Note: This is a simplified example. Real calculations depend on the module's specifics.
    # MACs = output_height * output_width * kernel_height * kernel_width * in_channels * out_channels
    # For simplicity, we'll just assign a placeholder value or use a helper if available
    # In a real scenario, you'd implement the precise MAC calculation here.
    # For nn.Conv2d, THOP usually handles it, but this demonstrates the concept.
    macs = 0  # Placeholder: Implement actual MAC calculation based on module logic
    # You might need access to module properties like kernel_size, stride, padding, channels etc.
    # Example for a Conv2d layer (simplified):
    if isinstance(module, nn.Conv2d):
        _, _, H, W = y.shape  # Output shape
        k_h, k_w = module.kernel_size
        in_c = module.in_channels
        out_c = module.out_channels
        groups = module.groups
        macs = (k_h * k_w * in_c * out_c * H * W) / groups
    module.total_ops += torch.DoubleTensor([macs])  # Accumulate MACs


# Instantiate a model containing your custom module
model = YourCustomModule()  # Or a larger model incorporating this module

# Create a dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Profile the model, providing the custom operation mapping
macs, params = profile(model, inputs=(dummy_input,), custom_ops={YourCustomModule: count_your_custom_module})

print(f"Custom MACs: {macs}, Parameters: {params}")
# Expected output: Custom MACs: 87457792.0, Parameters: 1792.0
```

### Improve Output Readability

For clearer and more interpretable results, use the `thop.clever_format` function. This formats the raw MACs and parameter counts into human-readable strings (e.g., GigaMACs, MegaParams). This formatting helps in quickly understanding the scale of computational resources required, similar to the metrics provided in our [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).

```python
import torch
from torchvision.models import resnet50

from thop import clever_format, profile

model = resnet50()
dummy_input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(dummy_input,))

# Format the numbers into a readable format (e.g., 4.14 GMac, 25.56 MParams)
macs_readable, params_readable = clever_format([macs, params], "%.3f")

print(f"Formatted MACs: {macs_readable}, Formatted Parameters: {params_readable}")
# Expected output: Formatted MACs: 4.140G, Formatted Parameters: 25.557M
```

## 📊 Results of Recent Models

The table below showcases the parameters and MACs for several popular [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models, profiled using THOP. These benchmarks provide a comparative overview of model complexity and computational cost. You can reproduce these results by running the script located at `benchmark/evaluate_famous_models.py` in this repository. Comparing these metrics is essential for tasks like selecting models for [object detection](https://www.ultralytics.com/glossary/object-detection) or [image classification](https://www.ultralytics.com/glossary/image-classification). For more comparisons, see our [model comparison section](https://docs.ultralytics.com/compare/).

<table align="center">
<tr>
<td>

| Model            | Params(M) | MACs(G) |
| ---------------- | --------- | ------- |
| alexnet          | 61.10     | 0.77    |
| vgg11            | 132.86    | 7.74    |
| vgg11_bn         | 132.87    | 7.77    |
| vgg13            | 133.05    | 11.44   |
| vgg13_bn         | 133.05    | 11.49   |
| vgg16            | 138.36    | 15.61   |
| vgg16_bn         | 138.37    | 15.66   |
| vgg19            | 143.67    | 19.77   |
| vgg19_bn         | 143.68    | 19.83   |
| resnet18         | 11.69     | 1.82    |
| resnet34         | 21.80     | 3.68    |
| resnet50         | 25.56     | 4.14    |
| resnet101        | 44.55     | 7.87    |
| resnet152        | 60.19     | 11.61   |
| wide_resnet101_2 | 126.89    | 22.84   |
| wide_resnet50_2  | 68.88     | 11.46   |

</td>
<td>

| Model              | Params(M) | MACs(G) |
| ------------------ | --------- | ------- |
| resnext50_32x4d    | 25.03     | 4.29    |
| resnext101_32x8d   | 88.79     | 16.54   |
| densenet121        | 7.98      | 2.90    |
| densenet161        | 28.68     | 7.85    |
| densenet169        | 14.15     | 3.44    |
| densenet201        | 20.01     | 4.39    |
| squeezenet1_0      | 1.25      | 0.82    |
| squeezenet1_1      | 1.24      | 0.35    |
| mnasnet0_5         | 2.22      | 0.14    |
| mnasnet0_75        | 3.17      | 0.24    |
| mnasnet1_0         | 4.38      | 0.34    |
| mnasnet1_3         | 6.28      | 0.53    |
| mobilenet_v2       | 3.50      | 0.33    |
| shufflenet_v2_x0_5 | 1.37      | 0.05    |
| shufflenet_v2_x1_0 | 2.28      | 0.15    |
| shufflenet_v2_x1_5 | 3.50      | 0.31    |
| shufflenet_v2_x2_0 | 7.39      | 0.60    |
| inception_v3       | 27.16     | 5.75    |

</td>
</tr>
</table>

## 🙌 Contribute

We actively welcome and encourage community contributions to make THOP even better! Whether it's adding support for new [PyTorch layers](https://docs.pytorch.org/docs/stable/nn.html), improving existing calculations, enhancing documentation, or fixing bugs, your input is valuable. Please see our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for detailed instructions on how to participate. Together, we can ensure THOP remains a state-of-the-art tool for the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) community. Don't hesitate to share your feedback and suggestions!

## 📜 License

THOP is distributed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0.en.html). This license promotes open collaboration and sharing of improvements. For complete details, please refer to the [LICENSE](https://github.com/ultralytics/thop/blob/main/LICENSE) file included in the repository. Understanding the license is important before integrating THOP into your projects, especially for commercial applications which may require an [Enterprise License](https://www.ultralytics.com/license).

## 📧 Contact

Encountered a bug or have a feature request? Please submit an issue through our [GitHub Issues](https://github.com/ultralytics/thop/issues) page. For general discussions, questions, and community support, join the vibrant Ultralytics community on our [Discord server](https://discord.com/invite/ultralytics). We look forward to hearing from you and collaborating!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
