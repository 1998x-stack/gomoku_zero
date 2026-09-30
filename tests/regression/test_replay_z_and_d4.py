"""Regression: pi 向量必须始终完整长度 + ReplayBuffer z 语义 + D4 对齐不变量。"""
from __future__ import annotations

import numpy as np
import pytest

from gomoku_zero.game import encode_state
from gomoku_zero.replay import ReplayBuffer, d4_transforms
from gomoku_zero.selfplay import softmax_temperature


# ---------- 1) pi 全长（曾修的“截断 bug”） ----------

def test_softmax_temperature_always_full_length():
    """访问集最大索引很小时，pi 也必须是 25 全长，非法/未访问处为 0。"""
    visits = {0: 3, 1: 7}  # 模拟棋局后期只余低位动作
    pi = softmax_temperature(visits, tau=1.0, n_actions=25)
    assert pi.shape == (25,)
    assert np.allclose(pi[2:], 0.0)
    assert np.isclose(pi.sum(), 1.0)


def test_softmax_temperature_backward_compat_default_size():
    """不传 n_actions 时保持旧行为（max key + 1），单测 & 外部调用不受影响。"""
    pi = softmax_temperature({0: 3, 1: 7}, tau=1.0)
    assert pi.shape == (2,)
    assert np.isclose(pi.sum(), 1.0)


# ---------- 2) z 按每步视角（AlphaZero 原义）回填且能被 replay 忠实保存 ----------

def test_replay_z_aligned_with_to_play():
    """一局黑胜：黑方回合的样本 z=+1，白方回合 z=-1。
    D4 增强不改 to_play 平面（全 1/全 0 对称），故采样后仍严格对应。"""
    n = 5
    rb = ReplayBuffer(100_000, n)

    def mk(to_play):
        s = np.zeros((3, n, n), dtype=np.float32)
        s[2] = 1.0 if to_play == 1 else 0.0
        return s

    pi = np.ones(n * n, dtype=np.float32) / (n * n)
    players = [1, 2, 1, 2, 1]  # 黑方第 5 手获胜
    zs = [1.0 if p == 1 else -1.0 for p in players]

    for t in range(len(players)):
        rb.add_game([mk(players[t])], [pi], zs[t])

    s, _, z = rb.sample(256)
    for i in range(256):
        # ch2 全 1（黑方回合）<-> z=+1；全 0（白方回合）<-> z=-1
        if np.all(s[i, 2] == 1.0):
            assert z[i] == 1.0
        else:
            assert z[i] == -1.0


# ---------- 3) D4 增强保持 (棋盘, 策略) 对齐 ----------

def test_d4_preserves_policy_legal_mask_alignment():
    """D4 必须把棋盘和策略同步变换：变换后 pi 仍在“空位”上均匀、占位处为 0。"""
    n = 5
    board = np.zeros((n, n), dtype=np.int8)
    board[1, 2] = 1
    board[3, 3] = 2
    s = encode_state(board, 1)

    empties = n * n - 2
    pi = np.ones((n, n), dtype=np.float32) / empties
    pi[1, 2] = 0.0
    pi[3, 3] = 0.0

    for ss, pp in zip(d4_transforms(s), d4_transforms(pi)):
        occ = (ss[0] + ss[1]) > 0  # 我方或对方有子的位置
        pp2 = pp.reshape(n, n)
        assert np.allclose(pp2[occ], 0.0), "策略落在了被占格上"
        assert np.isclose(pp2.sum(), 1.0), "增强后策略未归一"


def test_d4_matches_identity_first_transform():
    """增强集第一个元素必须是恒等变换（0 旋转、不翻转）。"""
    x = np.random.default_rng(0).random((3, 5, 5))
    assert np.allclose(d4_transforms(x)[0], x)