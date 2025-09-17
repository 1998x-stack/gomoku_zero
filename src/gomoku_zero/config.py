# src/gomoku_zero/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import os
import yaml

@dataclass
class EnvCfg:
    id: str = "GomokuExact5-v0"
    board_size: int = 15
    seed: int = 42

@dataclass
class NetCfg:
    channels: int = 128
    blocks: int = 12

@dataclass
class MctsCfg:
    sims: int = 800
    c_puct: float = 2.0
    root_dirichlet_eps: float = 0.25
    root_dirichlet_alpha: float = 0.05
    temperature_moves: int = 12
    temperature_eval: float = 0.0
    virtual_loss: float = 1.0

@dataclass
class ReplayCfg:
    capacity_steps: int = 800_000
    batch_size: int = 256

@dataclass
class OptCfg:
    lr: float = 2e-3
    weight_decay: float = 1e-4

@dataclass
class TrainCfg:
    steps: int = 100_000
    save_every: int = 5_000
    logdir: str = "runs/gomoku_zero"

@dataclass
class EvalCfg:
    games: int = 400
    threshold: float = 0.55

@dataclass
class Cfg:
    # 用 default_factory 避免“mutable default”错误
    env: EnvCfg = field(default_factory=EnvCfg)
    net: NetCfg = field(default_factory=NetCfg)
    mcts: MctsCfg = field(default_factory=MctsCfg)
    replay: ReplayCfg = field(default_factory=ReplayCfg)
    opt: OptCfg = field(default_factory=OptCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    @staticmethod
    def load(path: str) -> "Cfg":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        def build(dc, cls):
            dc = dc or {}
            # 只取 dataclass 已声明字段，避免陌生键报错
            allowed = set(cls.__annotations__.keys())
            return cls(**{k: v for k, v in dc.items() if k in allowed})

        return Cfg(
            env=build(data.get("env"), EnvCfg),
            net=build(data.get("net"), NetCfg),
            mcts=build(data.get("mcts"), MctsCfg),
            replay=build(data.get("replay"), ReplayCfg),
            opt=build(data.get("opt"), OptCfg),
            train=build(data.get("train"), TrainCfg),
            eval=build(data.get("eval"), EvalCfg),
        )

def default_cfg_path() -> str:
    # 包内默认配置文件路径（兼容可编辑安装）
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "configs", "default.yaml"))
