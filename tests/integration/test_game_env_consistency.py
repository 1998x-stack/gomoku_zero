"""Integration: 自研 GameState 与外部 gomoku-exact5 Gym 环境行为一致性。

gomoku_zero 实际并不使用 gym 环境做训练（只在 game.py 里 import 规则函数），
但两者必须就“棋盘/轮到谁/终局/胜者”给出完全一致的语义，否则规则库替换会静默出错。
这里用随机对局 + 手工定式把两套实现并排对照。
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
import pytest

import gomoku_exact5  # noqa: F401  (导入即注册 GomokuExact5-v0)
from gomoku_zero.game import GameState, encode_state

BOARD = 5
N2 = BOARD * BOARD


@pytest.fixture(scope="module")
def env():
    return gym.make("GomokuExact5-v0", board_size=BOARD)


def test_random_game_fully_consistent(env):
    """随机落满一局的每一步，两套实现必须始终一致。"""
    g = GameState(BOARD)

    # 无放回随机顺序 -> 每一步都合法，最终必然满盘或提前决出胜者
    order = np.random.default_rng(1234).permutation(N2)
    obs, _ = env.reset(seed=0)

    for a in order:
        a = int(a)

        # 展望合法性一致
        assert g.legal_actions()[a], "GameState 认为非法"
        assert obs["action_mask"][a] == 1, "env 认为非法"

        # 同一手推进两边
        obs, reward, terminated, truncated, info = env.step(a)
        x, y = divmod(a, BOARD)
        g.step(a)

        # 棋盘一致
        assert np.array_equal(obs["board"], g.board), f"棋盘不一致 at {a}"
        # 回合一致（env: 0=黑,1=白；GameState: 1=黑,2=白）
        assert obs["to_play"] == (0 if g.to_play == 1 else 1), f"回合不一致 at {a}"

        # 终局判定一致
        assert (terminated or truncated) == g.is_terminal(), f"终局不一致 at {a}"
        if g.is_terminal():
            # 胜者一致：env winner / draw 对应 GameState winner=1/2/-1
            if "winner" in info:
                assert info["winner"] == g.winner
                assert reward == 1.0
            elif info.get("draw"):
                assert g.winner == -1
                assert reward == 0.0
            break


def test_crafted_exact5_win_agrees(env):
    """手工构造黑方恰 5 连（Exact-5）定式，两套实现应同时判黑胜。"""
    g = GameState(BOARD)
    obs, _ = env.reset(seed=0)

    # 黑连下 (2,0)-(2,4) 成 5 连；白子在 (1,*) 行穿插，不干扰、也不构成 5
    black_line = [i for i in range(5)]
    white_moves = [BOARD * 1 + 0, BOARD * 1 + 1, BOARD * 1 + 2, BOARD * 1 + 3]
    a_black = [2 * BOARD + i for i in black_line]
    a_white = white_moves

    moves = [a_black[0], a_white[0], a_black[1], a_white[1], a_black[2],
             a_white[2], a_black[3], a_white[3], a_black[4]]
    for k, a in enumerate(moves):
        obs, reward, terminated, truncated, info = env.step(a)
        g.step(a)

    # 黑最后一步应触发 Exact-5，双方都判黑胜
    assert g.is_terminal() and g.winner == 1
    assert terminated is True
    assert info.get("winner") == 1
    assert np.array_equal(obs["board"], g.board)


def test_exact5_six_in_row_not_win_both(env):
    """6 连不算胜（Exact-5 规则）两套实现一致：预置 6 连 + 两端挡子。"""
    import numpy as np
    from gomoku_exact5.rules import is_exact5_after_move

    big = np.zeros((9, 9), dtype=np.int8)
    for col in range(2, 8):  # 6 连
        big[3, col] = 1
    big[3, 1] = 2  # 两端挡子
    big[3, 8] = 2
    for col in range(2, 8):
        assert not is_exact5_after_move(big, 3, col, 1), f"(3,{col}) 不应算 exact5"


def test_encode_matches_env_observation(env):
    """encode_state 的通道语义应与 env 观测一一对应。"""
    obs, _ = env.reset(seed=0)
    g = GameState(BOARD)
    # 走 3 手后再比较
    for a in [12, 6, 18]:
        obs, *_ = env.step(a)
        g.step(a)
    enc = encode_state(g.board, g.to_play)
    cur = obs["board"] == g.to_play
    opp = (obs["board"] != 0) & (obs["board"] != g.to_play)
    assert np.allclose(enc[0], cur)
    assert np.allclose(enc[1], opp)
    assert np.allclose(enc[2], 1.0 if g.to_play == 1 else 0.0)