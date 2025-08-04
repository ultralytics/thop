# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "2.0.15"


import torch

from .profile import profile, profile_origin
from .utils import clever_format

default_dtype = torch.float64
