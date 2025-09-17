from __future__ import annotations
import os, random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)
