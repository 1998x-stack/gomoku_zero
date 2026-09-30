"""Integration: self-play 产出 -> ReplayBuffer -> train 的真实协作链。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gomoku_zero.config import Cfg
from gomoku_zero.replay import ReplayBuffer
from gomoku_zero.selfplay import play_one_game
from gomoku_zero.train import train_loop, LossFn


def _fill_stream(tiny_net, tiny_cfg, cpu, board=5, games=2):
    """真实 self-play 若干局并逐手写入经验池（与 cmd_selfplay 相同的接入方式）。"""
    rb = ReplayBuffer(capacity_steps=50_000, board_size=board)
    winners = []
    for _ in range(games):
        states, pis, zs, w = play_one_game(tiny_net, tiny_cfg, board, cpu)
        winners.append(w)
        for t in range(len(states)):
            rb.add_game([states[t]], [pis[t]], zs[t])
    return rb, winners


def test_selfplay_feeds_replay_correctly(tiny_net, tiny_cfg, cpu):
    rb, winners = _fill_stream(tiny_net, tiny_cfg, cpu)
    assert rb.size_steps > 0
    assert all(w in (-1, 1, 2) for w in winners)

    s, p, z = rb.sample(64)
    # 形状：D4 增强后仍是 (B,3,N,N) / (B,N*N) / (B,)
    assert s.shape == (64, 3, 5, 5)
    assert p.shape == (64, 25)
    assert z.shape == (64,)
    assert s.dtype == np.float32 and p.dtype == np.float32
    # 分布合法：pi 每行和为 1（增强保持归一），z 限定 {-1,0,1}
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-4)
    assert set(np.unique(z)).issubset({-1.0, 0.0, 1.0})


def test_train_loop_decreases_loss_fixed_batch(tiny_net, tiny_cfg, cpu, tmp_path):
    """在同一固定 mini-batch 上，train_loop 必须让总 loss 下降、且权重发生变化。"""
    rb, _ = _fill_stream(tiny_net, tiny_cfg, cpu, games=3)

    cfg = Cfg()
    cfg.env.board_size = 5
    cfg.mcts = tiny_cfg
    cfg.opt.lr = 1e-2
    cfg.replay.batch_size = 16

    net = type(tiny_net)(board_size=5, ch=8, blocks=1).to(cpu)
    loss_fn = LossFn()

    s, p, z = rb.sample(16)
    xs = torch.from_numpy(s).to(cpu)
    ps = torch.from_numpy(p).to(cpu)
    zs = torch.from_numpy(z).to(cpu)

    net.train()
    before, _, _ = loss_fn(*net(xs), ps, zs)
    wt_before = [p.clone() for p in net.parameters()]

    train_loop(cfg, rb, board_size=5, net=net, steps=60, logdir=str(tmp_path / "logs"))

    net.train()
    after, _, _ = loss_fn(*net(xs), ps, zs)
    assert after.item() < before.item(), f"loss 未下降: {before.item():.4f} -> {after.item():.4f}"
    assert any(
        not torch.allclose(a, b) for a, b in zip(net.parameters(), wt_before)
    ), "训练后权重未变化"