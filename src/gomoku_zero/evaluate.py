from __future__ import annotations
import torch
from loguru import logger
from .selfplay import play_one_game
from .serialization import save_checkpoint


def evaluate_models(
    best_net,
    cand_net,
    mcts_cfg,
    board_size: int,
    device: torch.device,
    games: int,
):
    """
    候选 vs 最强 的真正对弈（V2 门禁）：

    - 黑白交替：偶数局候选执黑、奇数局候选执白；
    - τ=0、禁 Dirichlet 噪声（is_eval=True）；
    - ``play_one_game`` 双网络模式：黑方用 black 的模型、白方用 white 的模型。

    返回候选的胜率（赢 = 候选获胜，和棋不计胜）。旧版本此处写死的
    “candidate 只和自己下”导致胜率恒 ~50%、门禁永不晋升，已修复。
    """
    wins = 0
    for i in range(games):
        black_net = cand_net if i % 2 == 0 else best_net
        white_net = best_net if i % 2 == 0 else cand_net
        _, _, _, w = play_one_game(
            black_net, mcts_cfg, board_size, device, is_eval=True, net2=white_net
        )
        # w: 1=黑胜, 2=白胜, -1=和棋
        if (i % 2 == 0 and w == 1) or (i % 2 == 1 and w == 2):
            wins += 1
    wr = wins / games
    logger.info(f"[Eval] Candidate win-rate vs Best (two-net, no noise) = {wr:.3f}")
    return wr


def gate_and_promote(best_path: str, cand_path: str, cand_net, wr: float, threshold: float) -> bool:
    """门禁决策：wr >= threshold 则用候选覆盖 best，返回是否晋升（真实写盘）。"""
    if wr >= threshold:
        save_checkpoint(best_path, cand_net, dict(win_rate=float(wr)))
        logger.info(f"[Gate] Candidate promoted to best (wr={wr:.3f} >= {threshold:.2f})")
        return True
    logger.info(f"[Gate] Candidate rejected (wr={wr:.3f} < {threshold:.2f})")
    return False