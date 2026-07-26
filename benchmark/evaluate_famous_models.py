# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
from ultralytics import YOLO, __version__

from thop import profile

ULTRALYTICS_VERSION = "8.4.106"
if __version__ != ULTRALYTICS_VERSION:
    raise RuntimeError(f"Install ultralytics=={ULTRALYTICS_VERSION} to reproduce these benchmarks.")

model_names = [f"{family}{size}" for family in ("yolov8", "yolo11", "yolo26") for size in "nsmlx"]

print("Model | Params(M) | MACs(G)")
print("---|---|---")

device = "cuda" if torch.cuda.is_available() else "cpu"
for name in model_names:
    model = YOLO(f"{name}.yaml").model.fuse(verbose=False).to(device)
    inputs = torch.zeros(1, 3, 640, 640, device=device)
    total_ops, total_params = profile(model, (inputs,), verbose=False)
    print(f"{name} | {total_params / 1e6:.2f} | {total_ops / 1e9:.2f}")
