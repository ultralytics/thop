<br>
<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🚀 THOP: PyTorch-OpCounter

Welcome to the [THOP](https://github.com/ultralytics/thop) repository, your comprehensive solution for profiling PyTorch models by computing the number of Multiply-Accumulate Operations (MACs) and parameters. This tool is essential for deep learning practitioners to evaluate model efficiency and performance.

[![GitHub Actions](https://github.com/ultralytics/thop/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/thop/actions/workflows/main.yml) <a href="https://discord.com/invite/ultralytics"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a> <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a> <a href="https://reddit.com/r/ultralytics"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>

## 📄 Description

THOP offers an intuitive API to profile PyTorch models by calculating the number of MACs and parameters. This functionality is crucial for assessing the computational efficiency and memory footprint of deep learning models.

## 📦 Installation

You can install THOP via pip:

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics-thop?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics-thop/) [![Downloads](https://static.pepy.tech/badge/ultralytics-thop)](https://pepy.tech/project/ultralytics-thop) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics-thop?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics-thop/)

```bash
pip install ultralytics-thop
```

Alternatively, install the latest version directly from GitHub:

```bash
pip install --upgrade git+https://github.com/ultralytics/thop.git
```

## 🛠 How to Use

### Basic Usage

To profile a model, you can use the following example:

```python
import torch
from torchvision.models import resnet50

from thop import profile

model = resnet50()
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input,))
```

### Define Custom Rules for Third-Party Modules

You can define custom rules for unsupported modules:

```python
import torch.nn as nn


class YourModule(nn.Module):
    # your definition
    pass


def count_your_model(model, x, y):
    # your rule here
    pass


input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input,), custom_ops={YourModule: count_your_model})
```

### Improve Output Readability

Use `thop.clever_format` for a more readable output:

```python
from thop import clever_format

macs, params = clever_format([macs, params], "%.3f")
```

## 📊 Results of Recent Models

The following table presents the parameters and MACs for popular models. These results can be reproduced using the script `benchmark/evaluate_famous_models.py`.

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

## 💡 Contribute

We welcome community contributions to enhance THOP. Please check our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for more details. Your feedback and suggestions are highly appreciated!

## 📄 License

THOP is licensed under the AGPL-3.0 License. For more information, see the [LICENSE](https://github.com/ultralytics/thop/blob/main/LICENSE) file.

## 📮 Contact

For bugs or feature requests, please open an issue on [GitHub Issues](https://github.com/ultralytics/thop/pulls). Join our community on [Discord](https://discord.com/invite/ultralytics) for discussions and support.

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
