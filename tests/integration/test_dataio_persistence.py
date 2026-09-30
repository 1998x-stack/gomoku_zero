"""Integration: self-play 数据落盘 -> 训练真实恢复（打通 CLI 数据流）。"""
from __future__ import annotations

import numpy as np
import pytest

from gomoku_zero.dataio import (
    save_game, list_playouts, load_playouts, count_steps,
)
from gomoku_zero.replay import ReplayBuffer
from gomoku_zero.selfplay import play_one_game


def test_save_load_roundtrip(tmp_path):
    """落盘 -> 加载应精确还原数组（float32）。"""
    rng = np.random.default_rng(0)
    T, C, N = 7, 3, 5
    s = rng.random((T, C, N, N)).astype(np.float32)
    p = rng.random((T, N * N)).astype(np.float32)
    z = np.array([1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float32)

    path = save_game(str(tmp_path), s, p, z)
    assert path.endswith("game_000001.npz")

    files = list_playouts(str(tmp_path))
    assert files == [path]
    assert count_steps(str(tmp_path)) == T

    loaded = list(load_playouts(str(tmp_path)))
    assert len(loaded) == 1
    s2, p2, z2 = loaded[0]
    assert s2.dtype == np.float32 and np.allclose(s2, s)
    assert p2.dtype == np.float32 and np.allclose(p2, p)
    assert np.allclose(z2, z)


def test_incremental_names_and_ordering(tmp_path):
    for _ in range(3):
        save_game(str(tmp_path), np.zeros((2, 3, 5, 5), np.float32),
                  np.ones((2, 25), np.float32), np.full((2,), 1.0, np.float32))
    files = list_playouts(str(tmp_path))
    names = [f.split("/")[-1][: len("game_000001")] for f in files]
    assert names == ["game_000001", "game_000002", "game_000003"]
    # 排序即索引序；加载按序返回
    assert len(list(load_playouts(str(tmp_path)))) == 3


def test_empty_dir(tmp_path):
    assert list_playouts(str(tmp_path)) == []
    assert count_steps(str(tmp_path)) == 0
    assert list(load_playouts(str(tmp_path))) == []


def test_real_selfplay_save_then_buffer_load(tiny_net, tiny_cfg, cpu, tmp_path):
    """真实路径：play_one_game -> 落盘 -> 加载进 ReplayBuffer -> 采样合法。"""
    states, pis, zs, winner = play_one_game(tiny_net, tiny_cfg, 5, cpu)
    save_game(str(tmp_path), states, pis, zs)

    rb = ReplayBuffer(50_000, 5)
    for s, p, z in load_playouts(str(tmp_path)):
        rb.add_games(s, p, z)

    assert rb.size_steps == len(states) * 8  # D4 增强后 ×8
    samp_s, samp_p, samp_z = rb.sample(32)
    assert samp_s.shape == (32, 3, 5, 5)
    assert samp_p.shape == (32, 25)
    assert np.allclose(samp_p.sum(axis=1), 1.0, atol=1e-4)
    assert set(np.unique(samp_z)).issubset({-1.0, 0.0, 1.0})
    assert winner in (-1, 1, 2)


def test_train_uses_real_persisted_data(tiny_cfg, tmp_path, cpu):
    """训练必须由落盘数据驱动：真实自对弈数据比随机热身更有结构。
    这里只验证“能加载——训练不抛错 + 权重更新”，真实度过拟合由
    test_training_can_overfit_fixed_samples 覆盖。"""
    from gomoku_zero.net import PolicyValueNet
    from gomoku_zero.config import Cfg
    from gomoku_zero.train import train_loop

    net = PolicyValueNet(5, ch=8, blocks=1)
    states, pis, zs, _ = play_one_game(net, tiny_cfg, 5, cpu)
    save_game(str(tmp_path), states, pis, zs)

    rb = ReplayBuffer(50_000, 5)
    for s, p, z in load_playouts(str(tmp_path)):
        rb.add_games(s, p, z)

    cfg = Cfg()
    cfg.env.board_size = 5
    cfg.mcts = tiny_cfg
    cfg.opt.lr = 1e-2
    cfg.replay.batch_size = 16

    net2 = PolicyValueNet(5, ch=8, blocks=1)
    before = [p_.clone() for p_ in net2.parameters()]
    train_loop(cfg, rb, 5, net2, steps=20, logdir=str(tmp_path / "logs"))
    assert any(not np.allclose(a.detach().numpy(), b.detach().numpy())
               for a, b in zip(net2.parameters(), before))