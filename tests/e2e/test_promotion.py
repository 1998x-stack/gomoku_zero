"""E2E: “更强候选必然晋升”的确定性场景（经真实双网络 MCTS 评估 + 门禁写盘）。

设计背景（本测试为一个必然成立、零 flake 的晋升场景）：
- **确定性来源**：评估走 τ=0、无 Dirichlet 噪声，且网络 forward 纯确定——
  因此给定 (best, cand, sims, games)，评估结果是一个**常数**，与运行次数无关。
  只要固定配置下候选胜率稳定高于门禁阈值，晋升就“必然”发生。
- **为什么用脚本弱手当 best**：5x5 Exact-5 有个反直觉的性质——只要双方会防守
  （哪怕很弱），大多打成和棋（≥6 连还不算胜，更利于和棋）。实测手写启发式
  面对“会正常下但平庸”的对手几乎全是和棋，赢不了。因此“更强”必须体现在
  “能让对手想守也守不住”：
  * ``CompactFillerBest`` —— 只填前两行：它**永远无法取胜**（行 0/1 的 4 连被
    棋盘边封死，exact-5 补不了第 5 子），也**干扰不了下方区域**。确实极弱。
  * ``BuilderCandidate`` —— 活威胁计数构建者：即时取胜=+1、对手一步即胜→强制
    堵、平时倾向“一手能造出取胜格（活四）”并带中心偏好。本质是强一个量级。
- **经验校准**：在 sims ∈ {10..100} 扫描中，sims ≤ 60 时候选胜率恒为 1.0
  （宁浅勿深：估值瑕疵在过深搜索下会被放大，80+ 时降到 0.5）。这里冻结
  sims=40 / games=10，实测多轮恒为 1.0，留有远大于 0.55 阈值的安全边际。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from gomoku_exact5.rules import is_exact5_after_move
from gomoku_zero.config import MctsCfg
from gomoku_zero.evaluate import evaluate_models, gate_and_promote
from gomoku_zero.serialization import save_checkpoint

BOARD = 5


class CompactFillerBest(nn.Module):
    """极弱 best：永远只填前两行(row 0/1)的第一个空位。

    自身永远无法取胜（被棋盘边封死），且不参与其它区域，驻留给对手。
    """

    def forward(self, x):
        B, _, H, W = x.shape
        logits = torch.zeros(B, H * W)
        vals = torch.zeros(B)
        for bi in range(B):
            board = x[bi, 0].numpy() + 2 * x[bi, 1].numpy()
            flat = board.reshape(-1)
            a = None
            for idx in range(0, 2 * W):
                if flat[idx] == 0:
                    a = idx
                    break
            if a is None:
                a = int(np.argmax(flat == 0))
            logits[bi, a] = 80.0
        return logits, vals


class BuilderCandidate(nn.Module):
    """强的构建型候选：即时取胜 + 强制堵漏 + 活威胁(取胜格)中心构建。"""

    def __init__(self, board_size: int = BOARD):
        super().__init__()
        self.n = board_size

    def _wins(self, b, color):
        n = self.n
        ws = []
        for i in range(n):
            for j in range(n):
                if b[i, j] != 0:
                    continue
                b[i, j] = color
                if is_exact5_after_move(b, i, j, color):
                    ws.append((i, j))
                b[i, j] = 0
        return ws

    def forward(self, x):
        B, _, H, W = x.shape
        if H != self.n:
            self.n = H
        logits = torch.zeros(B, H * W)
        vals = torch.zeros(B)
        for bi in range(B):
            board = (x[bi, 0].numpy() + 2 * x[bi, 1].numpy()).astype(np.int8)
            empty = [(i, j) for i in range(H) for j in range(W) if board[i, j] == 0]
            if not empty:
                continue
            my = self._wins(board, 1)
            opp = self._wins(board, 2)
            if my:                                   # 一手取胜
                logits[bi, my[0][0] * W + my[0][1]] = 80.0
                vals[bi] = 1.0
                continue
            if opp:                                  # 强制堵漏
                logits[bi, opp[0][0] * W + opp[0][1]] = 80.0
                vals[bi] = 0.0
                continue
            cx, cy = (H - 1) / 2, (W - 1) / 2        # 构建：造活四 + 中心
            best, bs = None, -1e18
            for i, j in empty:
                board[i, j] = 1
                myn = len(self._wins(board, 1))
                oppn = len(self._wins(board, 2))
                board[i, j] = 0
                sc = myn * 20.0 - oppn * 8.0 - 4.0 * (abs(i - cx) + abs(j - cy))
                if sc > bs:
                    best, bs = (i * W + j), sc
            logits[bi, best] = 5.0
        return logits, vals


def _eval_cfg(sims: int = 40) -> MctsCfg:
    return MctsCfg(
        sims=sims,
        c_puct=2.0,
        root_dirichlet_eps=0.0,   # 评估禁噪声 -> 全确定性
        root_dirichlet_alpha=0.03,
        temperature_moves=100,    # 评估全程 τ=0（由 is_eval 决定，这里无害）
        temperature_eval=0.0,
        virtual_loss=1.0,
    )


GAMES = 10
THRESHOLD = 0.55


@pytest.mark.e2e
def test_stronger_candidate_necessarily_promotes(tmp_path, cpu):
    best = CompactFillerBest()
    cand = BuilderCandidate()

    # 真实双网络评估（候选打黑白两色，交替执棋）
    wr = evaluate_models(best, cand, _eval_cfg(), BOARD, cpu, games=GAMES)

    # 安全边际断言：必须强于阈值且有余量（实测恒为 1.0）
    assert wr >= 0.8, f"更强候选应有的稳定胜率，实际 w={wr:.2f}"

    # 门禁决策 + 真实写盘晋升
    ckpt = tmp_path / "checkpoints"
    os.makedirs(ckpt, exist_ok=True)
    save_checkpoint(str(ckpt / "candidate.pt"), cand, {"steps": 0})
    promoted = gate_and_promote(
        str(ckpt / "best.pt"), str(ckpt / "candidate.pt"), cand, wr, THRESHOLD
    )

    assert promoted, f"wr={wr:.2f} >= {THRESHOLD} 应晋升"
    best_meta = json.loads((ckpt / "best.pt.json").read_text())
    assert best_meta["win_rate"] == wr
    assert (ckpt / "candidate.pt").exists()


@pytest.mark.e2e
def test_weak_candidate_does_not_promote(tmp_path, cpu):
    """反向契约：把“弱的一方”当候选应被拒绝（确保门禁真的有判别力）。"""
    best = BuilderCandidate()
    cand = CompactFillerBest()

    wr = evaluate_models(best, cand, _eval_cfg(), BOARD, cpu, games=GAMES)
    assert wr < THRESHOLD, f"弱候选应被拒绝，实际 w={wr:.2f}"

    ckpt = tmp_path / "checkpoints"
    os.makedirs(ckpt, exist_ok=True)
    save_checkpoint(str(ckpt / "candidate.pt"), cand, {"steps": 0})
    promoted = gate_and_promote(
        str(ckpt / "best.pt"), str(ckpt / "candidate.pt"), cand, wr, THRESHOLD
    )
    assert not promoted