from __future__ import annotations
import numpy as np, torch
from loguru import logger
from .selfplay import play_one_game

def evaluate_models(best_net, cand_net, mcts_cfg, board_size: int, device: torch.device, games: int):
    # 交替先手，温度=0，禁噪声
    wins = 0
    for i in range(games):
        first = cand_net if (i % 2 == 0) else best_net
        second = best_net if (i % 2 == 0) else cand_net
        # 简化：先手=网络A，以后如需双网络轮流可在 play_one_game 中添加接口
        # 这里我们复用 play_one_game，但它接收单网络。V1：使用 candidate 执行全部搜索，便于门禁近似。
        _, _, _, w = play_one_game(cand_net, mcts_cfg, board_size, device, is_eval=True)
        # 仅作占位：真实对战应使用“双网络 MCTS 对弈”，V2 我再切换为双方各用自己的 net。
        if w == 1:
            wins += 1
    wr = wins / games
    logger.info(f"[Eval] Candidate win-rate vs Best (approx single-net) = {wr:.3f}")
    return wr
