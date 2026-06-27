import torch, os
print("TORCH_HOME:", os.getenv("TORCH_HOME"))
print("XDG_CACHE_HOME:", os.getenv("XDG_CACHE_HOME"))
print("torch hub dir:", torch.hub.get_dir())