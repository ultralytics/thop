<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

[English](README.md) | [简体中文](README.zh-CN.md)

# 🚀 THOP：PyTorch 运算量分析工具

[THOP](https://github.com/ultralytics/thop) 用于分析 [PyTorch](https://pytorch.org/) 模型的乘加运算次数（MACs）和参数量。它轻量、易扩展，并由 [Ultralytics](https://www.ultralytics.com/) 维护。

[![Ultralytics Actions](https://github.com/ultralytics/thop/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/thop/actions/workflows/format.yml)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

## 📄 简介

THOP 通过一次前向传播测量模型，适合在训练或部署前比较不同架构的复杂度。它为常见的卷积、归一化、池化、激活、线性和循环层提供计数规则，也支持自定义规则。

## 📦 安装

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics-thop?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics-thop/) [![Downloads](https://static.pepy.tech/badge/ultralytics-thop)](https://clickpy.clickhouse.com/dashboard/ultralytics-thop) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics-thop?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics-thop/)

```bash
pip install ultralytics-thop
```

安装最新开发版本：

```bash
pip install --upgrade git+https://github.com/ultralytics/thop.git
```

## 🛠️ 使用方法

### 基本用法

将模型和示例输入元组传给 `profile()`：

```python
import torch
from torchvision.models import resnet50

from thop import profile

model = resnet50()
inputs = (torch.randn(1, 3, 224, 224),)
macs, params = profile(model, inputs=inputs)

print(f"MACs: {macs}, Parameters: {params}")
# 预期输出：MACs: 4133742592.0, Parameters: 25557032.0
```

### 为第三方模块定义自定义规则

将不受支持的模块类型映射到前向钩子函数。钩子函数接收模块、输入和输出，并将运算量累加到 `module.total_ops`。

```python
import torch
from torch import nn

from thop import profile


def count_silu(module, inputs, output):
    """作为简单示例，将每个输出元素计为一次运算。."""
    module.total_ops += output.numel()


model = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.SiLU())
inputs = (torch.randn(1, 3, 224, 224),)
macs, params = profile(model, inputs=inputs, custom_ops={nn.SiLU: count_silu})

print(f"Custom MACs: {macs}, Parameters: {params}")
# 预期输出：Custom MACs: 89915392.0, Parameters: 1792.0
```

### 提升输出可读性

使用 `clever_format()` 将原始计数转换为易读格式：

```python
import torch
from torchvision.models import resnet50

from thop import clever_format, profile

model = resnet50()
inputs = (torch.randn(1, 3, 224, 224),)
macs, params = profile(model, inputs=inputs)
macs_readable, params_readable = clever_format([macs, params], "%.3f")

print(f"Formatted MACs: {macs_readable}, Formatted Parameters: {params_readable}")
# 预期输出：Formatted MACs: 4.134G, Formatted Parameters: 25.557M
```

## 📊 近期模型结果

下列目标检测模型使用融合后的架构定义和 640 × 640 输入进行分析。安装 `ultralytics` 后，运行 `python benchmark/evaluate_famous_models.py` 即可复现此表，无需下载模型权重。FLOPs 通常可近似为 MACs 的两倍。

| 模型                                                   | 参数量 (M) | MACs (G) |
| ------------------------------------------------------ | ---------- | -------- |
| [YOLOv8n](https://docs.ultralytics.com/models/yolov8/) | 3.15       | 4.37     |
| YOLOv8s                                                | 11.16      | 14.30    |
| YOLOv8m                                                | 25.89      | 39.47    |
| YOLOv8l                                                | 43.67      | 82.58    |
| YOLOv8x                                                | 68.20      | 128.91   |
| [YOLO11n](https://docs.ultralytics.com/models/yolo11/) | 2.62       | 3.24     |
| YOLO11s                                                | 9.44       | 10.74    |
| YOLO11m                                                | 20.09      | 33.99    |
| YOLO11l                                                | 25.34      | 43.46    |
| YOLO11x                                                | 56.92      | 97.46    |
| [YOLO26n](https://docs.ultralytics.com/models/yolo26/) | 2.41       | 2.68     |
| YOLO26s                                                | 9.50       | 10.35    |
| YOLO26m                                                | 20.41      | 34.09    |
| YOLO26l                                                | 24.81      | 43.22    |
| YOLO26x                                                | 55.73      | 96.94    |

## 🙌 贡献

欢迎贡献新的计数规则、精度改进、测试和文档。请参阅 [Ultralytics 贡献指南](https://docs.ultralytics.com/help/contributing/)开始参与。

## 📜 许可证

THOP 采用 [AGPL-3.0 许可证](LICENSE)发布。如需商业使用，请参阅 [Ultralytics 企业许可证](https://www.ultralytics.com/license)。

## 📧 联系方式

请通过 [GitHub Issues](https://github.com/ultralytics/thop/issues) 报告错误或提出功能建议。如需提问和社区支持，请加入 [Ultralytics Discord](https://discord.com/invite/ultralytics)。
