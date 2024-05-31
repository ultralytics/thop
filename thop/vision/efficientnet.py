import argparse
import logging

import torch
import torch.nn as nn
from efficientnet_pytorch.utils import Conv2dDynamicSamePadding, Conv2dStaticSamePadding
from torch.nn.modules.conv import _ConvNd

register_hooks = {}
