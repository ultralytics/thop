# from .onnx_profile import OnnxProfile
import torch

from .profile import profile, profile_origin
from .utils import clever_format

default_dtype = torch.float64
from .__version__ import __version__
