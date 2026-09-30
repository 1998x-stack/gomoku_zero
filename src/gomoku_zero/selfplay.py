from __future__ import annotations
import numpy as np, torch, math
from typing import Tuple
from loguru import logger
from .game import GameState
from .mcts import MCTS

def softmax_temperature(visits: dict[int, int], tau: float, n_actions: int | None = None) -> np.ndarray:
    """
    由访问计数生成动作分布。

    gamma 输出固定为 ``n_actions`` 长度：未出现/非法的动作概率为 0。
    不传 ``n_actions`` 时回退为 ``max(keys)+1``（兼容直接调用的场景）。
    传入完整 ``board_size**2`` 保证 pi 始终是全长向量，避免 ReplayBuffer
    reshape 失败。
    """
    keys = list(visits.keys())
    counts = np.array([visits[k] for k in keys], dtype=np.float32)
    if n_actions is None:
        n_actions = (max(keys) + 1) if keys else 0
    pi = np.zeros((n_actions,), dtype=np.float32)
    if tau <= 1e-6:
        pi[keys[np.argmax(counts)]] = 1.0
        return pi
    log_c = np.log(counts + 1e-8) / tau
    w = np.exp(log_c - log_c.max())
    pi[keys] = w / w.sum()
    return pi

def play_one_game(
    net,
    mcts_cfg,
    board_size: int,
    device: torch.device,
    is_eval: bool = False,
    net2=None,
):
    """
    一局对弈。

    - ``net`` 执黑（to_play=1），``net2`` 执白（to_play=2）。
    - ``net2=None`` 时双方共用 ``net``（即自对弈）。
    - 评估（``is_eval=True``）时 τ=0、关闭 Dirichlet 噪声。
    """
    g = GameState(board_size)
    mcts = MCTS(net, mcts_cfg, device, net2=net2)
    states, pis, players = [], [], []

    while not g.is_terminal():
        add_dir = (not is_eval)
        visits = mcts.run(g, add_dirichlet=add_dir)
        move_idx = g.moves
        tau = 0.0 if (is_eval or move_idx >= mcts_cfg.temperature_moves) else 1.0
        pi = softmax_temperature(visits, tau, n_actions=board_size * board_size)
        s = g.encode()
        states.append(s); pis.append(pi); players.append(g.to_play)

        # 采样或贪心
        if tau > 0.5:
            a = np.random.choice(np.arange(pi.size), p=pi/np.sum(pi))
        else:
            a = int(np.argmax(pi))
        g.step(a)

    # 回填 z（按每步落子方的视角）
    if g.winner == -1:
        zs = [0.0 for _ in states]
    else:
        zs = [1.0 if p == g.winner else -1.0 for p in players]

    return states, pis, zs, g.winner

def selfplay_many(net, mcts_cfg, board_size: int, device: torch.device, games: int):
    results = []
    for i in range(games):
        s, p, z, w = play_one_game(net, mcts_cfg, board_size, device, is_eval=False)
        results.append(w)
        yield s, p, z, w
        if (i+1) % 10 == 0:
            w_rate = (np.array(results) == 1).mean()
            logger.info(f"[SelfPlay] {i+1}/{games}, X-win-rate={w_rate:.3f}")