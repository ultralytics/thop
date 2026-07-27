<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

[English](README.md) | [简体中文](README.zh-CN.md)

# 🚀 THOP: PyTorch-OpCounter

[THOP](https://github.com/ultralytics/thop) profiles [PyTorch](https://pytorch.org/) models by counting Multiply-Accumulate Operations (MACs) and parameters. It is lightweight, easy to extend, and maintained by [Ultralytics](https://www.ultralytics.com/).

[![Ultralytics Actions](https://github.com/ultralytics/thop/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/thop/actions/workflows/format.yml)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

## 📄 Description

THOP measures a model with one forward pass, making it useful for comparing architecture complexity before training or deployment. It includes counting rules for common convolutional, normalization, pooling, activation, linear, and recurrent layers, with support for custom rules.

## 📦 Installation

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics-thop?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics-thop/) [![Downloads](https://static.pepy.tech/badge/ultralytics-thop)](https://clickpy.clickhouse.com/dashboard/ultralytics-thop) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics-thop?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics-thop/)

```bash
pip install ultralytics-thop
```

To install the latest development version:

```bash
pip install --upgrade git+https://github.com/ultralytics/thop.git
```

## 🛠️ How to Use

### Basic Usage

Pass the model and a tuple of example inputs to `profile()`:

```python
import torch
from torchvision.models import resnet50

from thop import profile

model = resnet50()
inputs = (torch.randn(1, 3, 224, 224),)
macs, params = profile(model, inputs=inputs)

print(f"MACs: {macs}, Parameters: {params}")
# Expected output: MACs: 4133742592.0, Parameters: 25557032.0
```

### Define Custom Rules for Third-Party Modules

Map an unsupported module type to a forward-hook function. The hook receives the module, its inputs, and its output, then adds the operation count to `module.total_ops`.

```python
import torch
from torch import nn

from thop import profile


def count_silu(module, inputs, output):
    """Count one operation per output element as a simple example."""
    module.total_ops += output.numel()


model = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.SiLU())
inputs = (torch.randn(1, 3, 224, 224),)
macs, params = profile(model, inputs=inputs, custom_ops={nn.SiLU: count_silu})

print(f"Custom MACs: {macs}, Parameters: {params}")
# Expected output: Custom MACs: 89915392.0, Parameters: 1792.0
```

### Improve Output Readability

Use `clever_format()` to convert raw counts into human-readable values:

```python
import torch
from torchvision.models import resnet50

from thop import clever_format, profile

model = resnet50()
inputs = (torch.randn(1, 3, 224, 224),)
macs, params = profile(model, inputs=inputs)
macs_readable, params_readable = clever_format([macs, params], "%.3f")

print(f"Formatted MACs: {macs_readable}, Formatted Parameters: {params_readable}")
# Expected output: Formatted MACs: 4.134G, Formatted Parameters: 25.557M
```

## 📊 Results of Recent Models

The following detection models were profiled at 640 × 640 from their fused architecture definitions using `ultralytics==8.4.106`. Install that version, then run `python benchmark/evaluate_famous_models.py` to reproduce the table without downloading model weights. FLOPs are often approximated as twice the MAC count.

| Model                                                                  | size<br><sup>(pixels)</sup> | params<br><sup>(M)</sup> | MACs<br><sup>(B)</sup> |
| ---------------------------------------------------------------------- | --------------------------- | ------------------------ | ---------------------- |
| [YOLOv8n](https://platform.ultralytics.com/ultralytics/yolov8/yolov8n) | 640                         | 3.15                     | 4.37                   |
| [YOLOv8s](https://platform.ultralytics.com/ultralytics/yolov8/yolov8s) | 640                         | 11.16                    | 14.30                  |
| [YOLOv8m](https://platform.ultralytics.com/ultralytics/yolov8/yolov8m) | 640                         | 25.89                    | 39.47                  |
| [YOLOv8l](https://platform.ultralytics.com/ultralytics/yolov8/yolov8l) | 640                         | 43.67                    | 82.57                  |
| [YOLOv8x](https://platform.ultralytics.com/ultralytics/yolov8/yolov8x) | 640                         | 68.20                    | 128.90                 |
| [YOLO11n](https://platform.ultralytics.com/ultralytics/yolo11/yolo11n) | 640                         | 2.62                     | 3.24                   |
| [YOLO11s](https://platform.ultralytics.com/ultralytics/yolo11/yolo11s) | 640                         | 9.44                     | 10.73                  |
| [YOLO11m](https://platform.ultralytics.com/ultralytics/yolo11/yolo11m) | 640                         | 20.09                    | 33.99                  |
| [YOLO11l](https://platform.ultralytics.com/ultralytics/yolo11/yolo11l) | 640                         | 25.34                    | 43.46                  |
| [YOLO11x](https://platform.ultralytics.com/ultralytics/yolo11/yolo11x) | 640                         | 56.92                    | 97.45                  |
| [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n) | 640                         | 2.41                     | 2.68                   |
| [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s) | 640                         | 9.50                     | 10.34                  |
| [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m) | 640                         | 20.41                    | 34.09                  |
| [YOLO26l](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l) | 640                         | 24.81                    | 43.21                  |
| [YOLO26x](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x) | 640                         | 55.73                    | 96.93                  |

## 🤝 Contribute

We thrive on community collaboration! THOP wouldn't be the tool it is without contributions from developers like you. Please see our [Contributing Guide](https://docs.ultralytics.com/help/contributing) to get started. We also welcome your feedback—share your experience by completing our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A huge **Thank You** 🙏 to everyone who contributes!

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=1280 -->

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/thop/graphs/contributors)

We look forward to your contributions to help make the Ultralytics ecosystem even better!

## 📜 License

Ultralytics offers two licensing options to suit different needs:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license/agpl-3.0) open-source license is perfect for students, researchers, and enthusiasts. It encourages open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/thop/blob/main/LICENSE) file for full details.
- **Ultralytics Enterprise License**: For development and production use, this license enables seamless integration of Ultralytics software and AI models into business products and services, including internal tools, automated workflows, and production deployments, bypassing the open-source requirements of AGPL-3.0. To get started, please contact us via [Ultralytics Licensing](https://www.ultralytics.com/license).

## 📞 Contact

For bug reports and feature requests related to THOP, please visit [GitHub Issues](https://github.com/ultralytics/thop/issues). For questions, discussions, and community support, join our active communities on [Discord](https://discord.com/invite/ultralytics), [Reddit](https://www.reddit.com/r/ultralytics/), and the [Ultralytics Community Forums](https://community.ultralytics.com/). We're here to help with all things Ultralytics!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
