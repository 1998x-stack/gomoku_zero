"""Regression: 评估/门禁必须是“候选 vs 最强”的双网络对弈（曾修的 P1 bug）。

旧 evaluate_models 只传了 candidate 一个网络（play_one_game 自对弈），
导致胜率恒 ~50%、门禁永不晋升。修复后 play_one_game 支持 net1/net2、
evaluate_models 让双方各用自己的模型、黑白交替。这些 spy 测试钉住接线，
不依赖对局结果，因此完全确定、不 flaky。
"""
from __future__ import annotations

import numpy as np

import gomoku_zero.evaluate as evaluate_mod
from gomoku_zero.net import PolicyValueNet


def test_eval_alternates_two_distinct_nets(monkeypatch, tiny_cfg, cpu):
    """游戏必须真实使用 best 与 cand 两个不同的模型，且黑白交替。"""
    calls = []

    def spy(net, mcts_cfg, board_size, device, is_eval=False, net2=None):
        calls.append((net, net2))
        # 模拟和棋 -> 不计胜
        return [np.zeros((1, 1, 5, 5), dtype=np.float32)], [np.zeros((25,))], [], -1

    monkeypatch.setattr(evaluate_mod, "play_one_game", spy)

    best = PolicyValueNet(5, ch=8, blocks=1)
    cand = PolicyValueNet(5, ch=8, blocks=1)
    wr = evaluate_mod.evaluate_models(best, cand, tiny_cfg, board_size=5, device=cpu, games=4)

    assert wr == 0.0  # 全是和棋 -> 0 胜
    assert len(calls) == 4
    # 黑白交替：偶数局候选执黑、奇数局候选执白
    assert calls[0] == (cand, best)
    assert calls[1] == (best, cand)
    assert calls[2] == (cand, best)
    assert calls[3] == (best, cand)
    # 关键回归点：best 必须真的上场（旧实现只有 cand 自己下自己）-> best 缺席即失败
    used = {id(n) for pair in calls for n in pair}
    assert id(best) in used and id(cand) in used
    assert id(best) != id(cand)


def test_eval_win_counting_respects_side(monkeypatch, tiny_cfg, cpu):
    """候选执黑胜算 1 胜，候选执白胜也算 1 胜（按候选视角统计）。"""
    # 偶=黑(cand)、奇=白(cand)：候选在偶数局执黑、奇数局执白
    # 候选胜：偶局需 w==1，奇局需 w==2 -> [1,2,?,?] 前两手候选胜
    # 候选负：偶局 w==2，奇局 w==1 -> 后两手候选负
    answered_winners = [1, 2, 2, 1]

    state = {"i": 0}

    def spy(net, mcts_cfg, board_size, device, is_eval=False, net2=None):
        w = answered_winners[state["i"]]
        state["i"] += 1
        return [np.zeros((1, 1, 5, 5), dtype=np.float32)], [np.zeros((25,))], [], w

    monkeypatch.setattr(evaluate_mod, "play_one_game", spy)

    best = PolicyValueNet(5, ch=8, blocks=1)
    cand = PolicyValueNet(5, ch=8, blocks=1)
    wr = evaluate_mod.evaluate_models(best, cand, tiny_cfg, board_size=5, device=cpu, games=4)
    # 候选 2 胜 2 负 -> 0.5
    assert wr == 0.5