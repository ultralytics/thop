# from .onnx_profile import OnnxProfile
import torch

from .profile import profile, profile_origin
from .utils import clever_format

default_dtype = torch.float64

__version__ = "0.2.2"
