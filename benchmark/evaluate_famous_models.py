import torch
from thop.profile import profile
from torchvision import models

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower()
    and not name.startswith("__")  # and "inception" in name
    and callable(models.__dict__[name])
)

print("Model | Params(M) | FLOPs(G)")
print("---|---|---")

device = "cuda" if torch.cuda.is_available() else "cpu"
for name in model_names:
    try:
        model = models.__dict__[name]().to(device)
        dsize = (1, 3, 224, 224)
        if "inception" in name:
            dsize = (1, 3, 299, 299)
        inputs = torch.randn(dsize).to(device)
        total_ops, total_params = profile(model, (inputs,), verbose=False)
        print("%s | %.2f | %.2f" % (name, total_params / (1000**2), total_ops / (1000**3)))
    except Exception as e:
        print(f"Warning: failed to process {e}")
