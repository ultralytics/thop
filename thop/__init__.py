__version__ = "0.2.3"

import torch

from .profile import profile, profile_origin
from .utils import clever_format

default_dtype = torch.float64

