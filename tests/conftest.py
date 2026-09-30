"""Shared fixtures for integration / regression / e2e layers.

Keep models and MCTS tiny so the suite stays fast on CPU.
"""
from __future__ import annotations

import pytest
import torch

from gomoku_zero.config import MctsCfg
from gomoku_zero.net import PolicyValueNet


@pytest.fixture
def cpu() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def tiny_cfg() -> MctsCfg:
    return MctsCfg(
        sims=20,
        c_puct=1.5,
        root_dirichlet_eps=0.25,
        root_dirichlet_alpha=0.03,
        temperature_moves=6,
        temperature_eval=0.0,
        virtual_loss=1.0,
    )


@pytest.fixture
def tiny_net() -> PolicyValueNet:
    return PolicyValueNet(board_size=5, ch=8, blocks=1)