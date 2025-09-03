# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "2.0.17"

from .profile import profile, profile_origin
from .utils import clever_format

import torch

default_dtype = torch.float64

__all__ = ["profile", "profile_origin", "clever_format", "default_dtype"]
