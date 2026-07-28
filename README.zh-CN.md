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

将不受支持的模块类型映射到前向钩子函数。钩子函数接收模块、输入和输出，并将运算量累加到 `module.total_ops`。参数量无需钩子处理——它直接从模块树读取。规则同样适用于其注册类型的子类，由最近的已注册祖先胜出，因此为 `nn.Conv2d` 注册的规则也会统计 `nn.Conv2d` 的子类。若子类的 forward 计算内容不同，则需为它单独注册规则，否则它会按基类计数。

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

下列目标检测模型使用 `ultralytics==8.4.106` 中融合后的架构定义和 640 × 640 输入进行分析。安装该版本后，运行 `python benchmark/evaluate_famous_models.py` 即可复现此表，无需下载模型权重。FLOPs 通常可近似为 MACs 的两倍。

| 模型                                                                   | 尺寸<br><sup>(像素)</sup> | 参数<br><sup>(百万)</sup> | MACs<br><sup>(十亿)</sup> |
| ---------------------------------------------------------------------- | ------------------------- | ------------------------- | ------------------------- |
| [YOLOv8n](https://platform.ultralytics.com/ultralytics/yolov8/yolov8n) | 640                       | 3.15                      | 4.37                      |
| [YOLOv8s](https://platform.ultralytics.com/ultralytics/yolov8/yolov8s) | 640                       | 11.16                     | 14.30                     |
| [YOLOv8m](https://platform.ultralytics.com/ultralytics/yolov8/yolov8m) | 640                       | 25.89                     | 39.47                     |
| [YOLOv8l](https://platform.ultralytics.com/ultralytics/yolov8/yolov8l) | 640                       | 43.67                     | 82.58                     |
| [YOLOv8x](https://platform.ultralytics.com/ultralytics/yolov8/yolov8x) | 640                       | 68.20                     | 128.91                    |
| [YOLO11n](https://platform.ultralytics.com/ultralytics/yolo11/yolo11n) | 640                       | 2.62                      | 3.24                      |
| [YOLO11s](https://platform.ultralytics.com/ultralytics/yolo11/yolo11s) | 640                       | 9.44                      | 10.74                     |
| [YOLO11m](https://platform.ultralytics.com/ultralytics/yolo11/yolo11m) | 640                       | 20.09                     | 33.99                     |
| [YOLO11l](https://platform.ultralytics.com/ultralytics/yolo11/yolo11l) | 640                       | 25.34                     | 43.46                     |
| [YOLO11x](https://platform.ultralytics.com/ultralytics/yolo11/yolo11x) | 640                       | 56.92                     | 97.46                     |
| [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n) | 640                       | 2.41                      | 2.68                      |
| [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s) | 640                       | 9.50                      | 10.35                     |
| [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26/yolo26m) | 640                       | 20.41                     | 34.09                     |
| [YOLO26l](https://platform.ultralytics.com/ultralytics/yolo26/yolo26l) | 640                       | 24.81                     | 43.22                     |
| [YOLO26x](https://platform.ultralytics.com/ultralytics/yolo26/yolo26x) | 640                       | 55.73                     | 96.94                     |

## 🤝 贡献

我们依靠社区协作蓬勃发展！没有像您这样的开发者的贡献，THOP 就不会成为如今优秀的工具。请参阅我们的[贡献指南](https://docs.ultralytics.com/help/contributing)开始贡献。我们也欢迎您的反馈——通过完成我们的[调查问卷](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)分享您的体验。非常**感谢** 🙏 每一位贡献者！

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=1280 -->

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/thop/graphs/contributors)

我们期待您的贡献，帮助 Ultralytics 生态系统变得更好！

## 📜 许可证

Ultralytics 提供两种许可选项以满足不同需求：

- **AGPL-3.0 许可证**：这种经 [OSI 批准](https://opensource.org/license/agpl-3.0)的开源许可证非常适合学生、研究人员和爱好者。它鼓励开放协作和知识共享。有关完整详细信息，请参阅 [LICENSE](https://github.com/ultralytics/thop/blob/main/LICENSE) 文件。
- **Ultralytics 企业许可证**：适用于开发和生产用途，此许可证允许将 Ultralytics 软件和 AI 模型无缝集成到业务产品和服务中，包括内部工具、自动化工作流和生产部署，绕过 AGPL-3.0 的开源要求。如需开始使用，请通过 [Ultralytics 授权许可](https://www.ultralytics.com/license)与我们联系。

## 📞 联系方式

有关 THOP 的错误报告和功能请求，请访问 [GitHub Issues](https://github.com/ultralytics/thop/issues)。如有疑问、讨论和社区支持，请加入我们在 [Discord](https://discord.com/invite/ultralytics)、[Reddit](https://www.reddit.com/r/ultralytics/) 和 [Ultralytics 社区论坛](https://community.ultralytics.com/)上的活跃社区。我们随时为您提供有关 Ultralytics 的所有帮助！

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
