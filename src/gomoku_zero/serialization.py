from __future__ import annotations
import os, json, torch
from typing import Any, Dict
from .utils import ensure_dir

def save_checkpoint(path: str, net, meta: Dict[str, Any]):
    ensure_dir(os.path.dirname(path) or ".")
    torch.save({"state_dict": net.state_dict(), "meta": meta}, path)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_checkpoint(path: str, net):
    obj = torch.load(path, map_location="cpu")
    net.load_state_dict(obj["state_dict"])
    return obj.get("meta", {})
