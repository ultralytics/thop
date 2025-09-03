# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "2.0.17"

import torch

from .profile import profile, profile_origin
from .utils import clever_format

default_dtype = torch.float64

__all__ = ["profile", "profile_origin", "clever_format", "default_dtype"]
