"""Regression: MCTS 终局回传的 value 视角（用“结果视角”契约钉死，防被误改）。

关键背景（实测验证，勿按直觉手推）：
- ``_simulate`` 里是先 ``node = _select_child(node)`` 再 ``path.append``，
  因此 path 里存的是**被选中的子节点**，不是父节点；
- 配合 ``value = result_from(path[0][0].to_play)``（根/落子方视角），恰好让最
  深子节点累积到“其父节点视角”的正确值；
- 实测：黑一步即胜的局，获胜子 q=+1 且被 PUCT 满额访问（400/400）；
  黑走后白胜的局，失败分支 q=-1，PUCT 正确避开。
- 若改写成“叶子待走方视角”反而会让符号反转（这个坑踩过）。

下面的 oracle 用“near-one-hot 策略”构造完全确定的终局路径，断言精确数值。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from gomoku_zero.game import GameState
from gomoku_zero.mcts import MCTS, Node


class _OneHotNet(nn.Module):
    """近乎 one-hot 的策略：黑固定 black_a、白固定 white_a，value=0。"""

    def __init__(self, black_a: int, white_a: int):
        super().__init__()
        self.black_a = black_a
        self.white_a = white_a

    def forward(self, x):
        b = x.shape[0]
        logits = torch.full((b, 25), -1000.0)
        black_turn = x[:, 2].mean(dim=(1, 2)) > 0.5
        for i in range(b):
            if black_turn[i]:
                logits[i, self.black_a] = 1000.0
            else:
                logits[i, self.white_a] = 1000.0
        return logits, torch.zeros(b)


def _run_searched(mcts, state, sims):
    """按 run() 的展开逻辑执行 sims 次搜索并返回 root（Dirichlet 关闭，确定）。"""
    root = Node(1.0, state.to_play)
    priors, _ = mcts._policy_value(state)
    legal = state.legal_actions()
    priors = mcts._mask_and_norm(priors, legal)
    for a, p in enumerate(priors):
        if p > 0.0:
            root.children[a] = Node(p, player=mcts._opponent(state.to_play))
    for _ in range(sims):
        mcts._simulate(state, root)
    return root


def test_one_move_win_q_is_plus_one(cpu, tiny_cfg):
    """黑方一手即胜（奇数深度终局）：获胜子节点 q 必须是 +1。"""
    g = GameState(5)
    g.board[2, 0] = 1
    g.board[2, 1] = 1
    g.board[2, 2] = 1
    g.board[2, 3] = 1
    g.moves = 8
    g.to_play = 1

    mcts = MCTS(_OneHotNet(black_a=14, white_a=21), tiny_cfg, cpu)
    root = _run_searched(mcts, g, sims=50)

    win_child = root.children[14]
    assert win_child.visit == 50, "获胜手应被全部搜索访问"
    assert win_child.q == 1.0, f"获胜手 q 应为 +1（黑视角胜），实为 {win_child.q}"


def test_opponent_win_after_black_move_q_is_minus_one(cpu, tiny_cfg):
    """黑走 (0,0) 后白方 exact5 取胜（偶数深度终局）：该分支 q 必须是 -1。"""
    g = GameState(5)
    # 白竖直四连 (0,1)..(3,1)，白落 (4,1) 即成 exact5
    g.board[0, 1] = 2
    g.board[1, 1] = 2
    g.board[2, 1] = 2
    g.board[3, 1] = 2
    g.board[4, 4] = 1
    g.moves = 6
    g.to_play = 1

    mcts = MCTS(_OneHotNet(black_a=0, white_a=21), tiny_cfg, cpu)
    root = _run_searched(mcts, g, sims=1)  # 先完成内层节点展开（首次访问非终局）
    # 重置目标节点计数，让后续搜索全是“精确终局”步
    target = root.children[0]
    target.visit = 0
    target.value_sum = 0.0
    for _ in range(50):
        mcts._simulate(g, root)

    losing = root.children[0]  # 黑想走 (0,0) 会被白反杀
    assert losing.visit == 50
    assert losing.q == -1.0, f"黑走 (0,0) 导致白胜，该分支 q 应为 -1（黑视角败），实为 {losing.q}"


def test_root_children_carry_opponent_side():
    """回归：节点 player 字段标注的必须是“该子节点对应的待走方”（root 的对手）。"""
    mcts = MCTS(_OneHotNet(14, 21), None, None)

    g_black = GameState(5)
    g_black.to_play = 1
    root = Node(1.0, g_black.to_play)
    mcts._simulate(g_black, root)
    assert root.children and all(n.player == 2 for n in root.children.values())

    g_white = GameState(5)
    g_white.to_play = 2
    root2 = Node(1.0, g_white.to_play)
    mcts._simulate(g_white, root2)
    assert all(n.player == 1 for n in root2.children.values())


# ---------- 双网络选择（门禁的地基） ----------

class _BlackNet(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 25), torch.ones(x.shape[0])


class _WhiteNet(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 25), torch.full((x.shape[0],), -1.0)


def test_ann_selects_net_by_side(cpu, tiny_cfg):
    """_ann 应让黑方用 net、白方用 net2（net2=None 时共用 net=自对弈）。"""
    black, white = _BlackNet(), _WhiteNet()
    mcts = MCTS(black, tiny_cfg, cpu, net2=white)

    s_black = GameState(5)
    s_black.to_play = 1
    s_white = GameState(5)
    s_white.to_play = 2

    assert mcts._ann(s_black) is black
    assert mcts._ann(s_white) is white

    mcts2 = MCTS(black, tiny_cfg, cpu)
    assert mcts2._ann(s_white) is black