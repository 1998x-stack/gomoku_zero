from __future__ import annotations
import numpy as np
from collections import deque
from typing import Tuple

def d4_transforms(x: np.ndarray) -> list[np.ndarray]:
    # x: (C,N,N) 或 (N,N)
    xs = []
    for k in range(4):
        r = np.rot90(x, k, axes=(-2, -1))
        xs.append(r)
        xs.append(np.flip(r, axis=-1))
    return xs

class ReplayBuffer:
    """
    存 (state_planes, pi, z)，其中：
    - state_planes: (3,N,N) float32
    - pi: (N*N,) float32
    - z: float scalar
    """
    def __init__(self, capacity_steps: int, board_size: int):
        self.capacity = capacity_steps
        self.n = board_size
        self.states = deque()
        self.pis = deque()
        self.zs = deque()
        self.size_steps = 0

    def add_game(self, states: list[np.ndarray], pis: list[np.ndarray], z: float):
        # 为整局每一步添加 (s, π, z_t)；z_t 需按该步落子方视角已处理
        for s, p in zip(states, pis):
            # D4 增强
            for ss, pp in zip(d4_transforms(s), d4_transforms(p.reshape(self.n, self.n))):
                self.states.append(ss.astype(np.float32))
                self.pis.append(pp.reshape(-1).astype(np.float32))
                self.zs.append(np.float32(z))
                self.size_steps += 1
                while self.size_steps > self.capacity:
                    self.states.popleft(); self.pis.popleft(); self.zs.popleft()
                    self.size_steps -= 1

    def sample(self, batch: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.randint(0, len(self.states), size=batch)
        s = np.stack([self.states[i] for i in idx], axis=0)
        p = np.stack([self.pis[i] for i in idx], axis=0)
        z = np.stack([self.zs[i] for i in idx], axis=0)
        return s, p, z
