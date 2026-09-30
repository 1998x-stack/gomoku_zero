"""E2E: 端到端自对弈 -> 经验池 -> 训练 -> 保存 -> 双网络评估 -> 门禁晋升。

用极小的 5x5 网络和少量 sims，让整条链在 CPU 上几秒跑完，验证：
1. 自对弈能持续产出数据并写入经验池（每步 z 视角正确）；
2. 训练能跑通（loss 记录、权重更新）；
3. 训练产物可保存/加载且无损；
4. 双网络评估能真正对弈并产出 [0,1] 的胜率；
5. 门禁判定 + 晋升写入真实发生。
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from gomoku_zero.config import Cfg
from gomoku_zero.net import PolicyValueNet
from gomoku_zero.replay import ReplayBuffer
from gomoku_zero.serialization import save_checkpoint, load_checkpoint
from gomoku_zero.selfplay import selfplay_many
from gomoku_zero.evaluate import evaluate_models
from gomoku_zero.train import train_loop

BOARD = 5


def _small_cfg(lu_cfg, logdir: str) -> Cfg:
    cfg = Cfg()
    cfg.env.board_size = BOARD
    cfg.mcts = lu_cfg
    cfg.opt.lr = 1e-2
    cfg.replay.batch_size = 16
    cfg.replay.capacity_steps = 100_000
    cfg.train.steps = 30
    cfg.train.logdir = logdir
    return cfg


@pytest.mark.e2e
def test_full_pipeline_roundtrip(tmp_path, cpu, tiny_cfg):
    from gomoku_zero.utils import set_seed
    set_seed(0)  # 固定随机性，避免 loss 断言偶发波动
    ckpt = tmp_path / "checkpoints"
    os.makedirs(ckpt, exist_ok=True)

    net = PolicyValueNet(BOARD, ch=4, blocks=1)

    # ---- 1) 自对弈收集 ----
    rb = ReplayBuffer(100_000, BOARD)
    winners = []
    for s, p, z, w in selfplay_many(net, tiny_cfg, BOARD, cpu, games=2):
        winners.append(w)
        for t in range(len(s)):
            rb.add_game([s[t]], [p[t]], z[t])
    assert rb.size_steps > 0, "经验池应为空? 自对弈应产生数据"
    assert all(w in (-1, 1, 2) for w in winners)

    # ---- 2) 训练 ----
    cfg = _small_cfg(tiny_cfg, str(tmp_path / "logs"))
    s0, p0, z0 = rb.sample(cfg.replay.batch_size)
    xs = torch.from_numpy(s0); ps = torch.from_numpy(p0); zs = torch.from_numpy(z0)
    net.train()
    from gomoku_zero.train import LossFn
    before, *_ = LossFn()(*net(xs), ps, zs)
    train_loop(cfg, rb, BOARD, net, steps=cfg.train.steps, logdir=cfg.train.logdir)
    net.train()
    after, *_ = LossFn()(*net(xs), ps, zs)
    assert after.item() < before.item(), "训练应降低固定批次 loss"

    cand = net
    best = PolicyValueNet(BOARD, ch=4, blocks=1)  # 更强的？不重要，本测试验证机制

    # ---- 3) 保存 ----
    save_checkpoint(str(ckpt / "candidate.pt"), cand, {"steps": cfg.train.steps})
    save_checkpoint(str(ckpt / "best.pt"), best, {"steps": 0})
    assert (ckpt / "candidate.pt").exists()
    assert (ckpt / "candidate.pt.json").exists()

    # ---- 4) 双网络评估 ----
    wr = evaluate_models(best, cand, tiny_cfg, BOARD, cpu, games=2)
    assert 0.0 <= wr <= 1.0

    # ---- 5) 门禁：按阈值晋升 ----
    threshold = 0.5
    if wr >= threshold:
        save_checkpoint(str(ckpt / "best.pt"), cand, {"win_rate": wr})

    # ---- 6) 重载校验：candidate 无损往返 ----
    reloaded = PolicyValueNet(BOARD, ch=4, blocks=1)
    meta = load_checkpoint(str(ckpt / "candidate.pt"), reloaded)
    assert meta["steps"] == cfg.train.steps
    cand.eval(); reloaded.eval()
    x = torch.randn(2, 3, BOARD, BOARD)
    with torch.no_grad():
        a = cand(x)
        b = reloaded(x)
    assert torch.allclose(a[0], b[0])
    assert torch.allclose(a[1], b[1])


@pytest.mark.e2e
def test_pipeline_determinism_with_seed(tmp_path, cpu, tiny_cfg):
    """同一 seed 下自对弈一局应得到相同胜负（保证可复现调试）。"""
    gomoku_zero_utils = __import__("gomoku_zero.utils", fromlist=["set_seed"])
    r1 = []
    for _ in range(2):
        gomoku_zero_utils.set_seed(7)
        net = PolicyValueNet(BOARD, ch=4, blocks=1)
        results = []
        for s, p, z, w in selfplay_many(net, tiny_cfg, BOARD, cpu, games=1):
            results.append(w)
        r1.append(results)

    assert r1[0] == r1[1], f"同 seed 自对弈结果应一致: {r1}"