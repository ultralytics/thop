__version__ = "2.0.7"

import torch

from .profile import profile, profile_origin
from .utils import clever_format

default_dtype = torch.float64
