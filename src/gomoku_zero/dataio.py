"""Self-play 数据落盘 / 加载（V2a：打通 selfplay -> train 的真实数据流）。

设计要点：
- 每局存一个 ``game_{idx:06d}.npz``，含 ``s`` (T,3,N,N) / ``p`` (T,N*N) / ``z`` (T,)，
  z 已按每步“落子方”视角回填（与 AlphaZero 原义一致）。
- 存的是 **D4 增强前**的数据：加载进 ReplayBuffer 时再乘 8 倍增强，磁盘占用省 8 倍。
- 索引号由目录内文件自动推导，支持增量追加。
"""
from __future__ import annotations

import glob
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np

PREFIX = "game_"
SUFFIX = ".npz"


def _next_index(out_dir: str) -> int:
    idx = 0
    for f in list_playouts(out_dir):
        base = os.path.basename(f)
        try:
            idx = max(idx, int(base[len(PREFIX):-len(SUFFIX)]))
        except ValueError:
            continue
    return idx + 1


def save_game(
    out_dir: str,
    states: np.ndarray,
    pis: np.ndarray,
    zs: np.ndarray,
    index: Optional[int] = None,
) -> str:
    """把一局的 (s, p, z) 写为压缩 npz，返回写入路径。

    states: (T,3,N,N) float; pis: (T,N*N) float; zs: (T,)
    """
    os.makedirs(out_dir, exist_ok=True)
    if index is None:
        index = _next_index(out_dir)
    path = os.path.join(out_dir, f"{PREFIX}{index:06d}{SUFFIX}")
    np.savez_compressed(
        path,
        s=np.asarray(states, dtype=np.float32),
        p=np.asarray(pis, dtype=np.float32),
        z=np.asarray(zs, dtype=np.float32),
    )
    return path


def list_playouts(out_dir: str) -> List[str]:
    if not os.path.isdir(out_dir):
        return []
    return sorted(glob.glob(os.path.join(out_dir, f"{PREFIX}*{SUFFIX}")))


def count_steps(out_dir: str) -> int:
    total = 0
    for f in list_playouts(out_dir):
        with np.load(f) as d:
            total += int(d["z"].shape[0])
    return total


def load_playouts(
    out_dir: str, max_games: Optional[int] = None
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """按索引顺序加载已存的 (s, p, z)；可选限制局数。"""
    files = list_playouts(out_dir)
    if max_games is not None:
        files = files[:max_games]
    for f in files:
        with np.load(f) as d:
            yield d["s"], d["p"], d["z"]