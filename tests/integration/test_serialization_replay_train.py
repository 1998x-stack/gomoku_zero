"""Integration: 序列化往返（save/load）与增量训练过拟合能力。"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from gomoku_zero.config import Cfg
from gomoku_zero.net import PolicyValueNet
from gomoku_zero.replay import ReplayBuffer
from gomoku_zero.serialization import save_checkpoint, load_checkpoint
from gomoku_zero.train import train_loop, LossFn


def test_checkpoint_roundtrip(tmp_path, cpu):
    """save -> load 应无损还原权重，元数据落到 JSON 边车。"""
    net = PolicyValueNet(5, ch=8, blocks=1)
    meta = {"steps": 123, "win_rate": 0.62}

    path = str(tmp_path / "nested" / "best.pt")
    save_checkpoint(path, net, meta)

    assert os.path.exists(path)
    with open(path + ".json") as f:
        assert json.load(f) == meta

    net2 = PolicyValueNet(5, ch=8, blocks=1)
    loaded_meta = load_checkpoint(path, net2)
    assert loaded_meta["steps"] == 123

    # 权重与输出都应一致
    net.eval(); net2.eval()
    x = torch.randn(4, 3, 5, 5)
    with torch.no_grad():
        a = net(x); b = net2(x)
    assert torch.allclose(a[0], b[0])
    assert torch.allclose(a[1], b[1])


def _build_single_sample_buffer(board=5):
    """只含一种确定性样本（one-hot 策略 + z=+1），测“能真正学到”。

    注意不用均匀表为目标：CE loss 对均匀 softmax target 的下限是 log(N)，
    无法收敛到 0，测不出过拟合能力。
    """
    rb = ReplayBuffer(capacity_steps=100_000, board_size=board)
    s = np.zeros((3, board, board), dtype=np.float32)
    s[2] = 1.0  # 黑方回合
    pi = np.zeros(board * board, dtype=np.float32)
    pi[12] = 1.0  # one-hot：可收敛到 0 loss
    rb.add_game([s], [pi], 1.0)
    return rb


def test_training_can_overfit_fixed_samples(tmp_path, cpu):
    """固定少量样本上训练应能逼近 0 loss——证明梯度通路 & 优化器有效。"""
    rb = _build_single_sample_buffer()
    cfg = Cfg()
    cfg.env.board_size = 5
    cfg.opt.lr = 1e-2
    cfg.replay.batch_size = 16

    net = PolicyValueNet(5, ch=8, blocks=1).to(cpu)
    loss_fn = LossFn()

    s, p, z = rb.sample(16)
    xs = torch.from_numpy(s).to(cpu)
    ps = torch.from_numpy(p).to(cpu)
    zs = torch.from_numpy(z).to(cpu)

    net.train()
    before, _, _ = loss_fn(*net(xs), ps, zs)

    train_loop(cfg, rb, board_size=5, net=net, steps=300, logdir=str(tmp_path / "logs"))

    net.train()
    after, _, _ = loss_fn(*net(xs), ps, zs)
    assert after.item() < before.item() / 2.0, (
        f"过拟合训练未显著下降: {before.item():.4f} -> {after.item():.4f}"
    )