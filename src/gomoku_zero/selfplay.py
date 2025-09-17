from __future__ import annotations
import numpy as np, torch, math
from typing import Tuple
from loguru import logger
from .game import GameState
from .mcts import MCTS

def softmax_temperature(visits: dict[int, int], tau: float) -> np.ndarray:
    keys = list(visits.keys())
    counts = np.array([visits[k] for k in keys], dtype=np.float32)
    if tau <= 1e-6:
        pi = np.zeros((max(keys)+1,), dtype=np.float32)
        pi[keys[np.argmax(counts)]] = 1.0
        return pi
    log_c = np.log(counts + 1e-8) / tau
    w = np.exp(log_c - log_c.max())
    pi = np.zeros((max(keys)+1,), dtype=np.float32)
    pi[keys] = w / w.sum()
    return pi

def play_one_game(net, mcts_cfg, board_size: int, device: torch.device, is_eval: bool=False):
    g = GameState(board_size)
    mcts = MCTS(net, mcts_cfg, device)
    states, pis, players = [], [], []

    while not g.is_terminal():
        add_dir = (not is_eval)
        visits = mcts.run(g, add_dirichlet=add_dir)
        move_idx = g.moves
        tau = 0.0 if (is_eval or move_idx >= mcts_cfg.temperature_moves) else 1.0
        pi = softmax_temperature(visits, tau)
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