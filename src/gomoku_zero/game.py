from __future__ import annotations
import numpy as np
from typing import Tuple, List
from gomoku_exact5.rules import is_exact5_after_move, action_to_xy, xy_to_action

# 1=黑, 2=白
Player = int

def encode_state(board: np.ndarray, to_play: Player) -> np.ndarray:
    """
    输出 (C=3, N, N) float32:
    ch0: 当前玩家的子, ch1: 对手子, ch2: to_play 全1或全0（黑=1, 白=0）
    """
    assert board.ndim == 2
    cur = (board == to_play).astype(np.float32)
    opp = (board != 0).astype(np.float32) - cur
    tag = np.full_like(board, 1.0 if to_play == 1 else 0.0, dtype=np.float32)
    return np.stack([cur, opp, tag], axis=0)

class GameState:
    def __init__(self, board_size: int = 15) -> None:
        self.n = board_size
        self.board = np.zeros((self.n, self.n), dtype=np.int8)
        self.to_play: Player = 1
        self.moves = 0
        self.winner: int = 0  # 0=未终局, 1/2 胜者, -1=和

    def copy(self) -> "GameState":
        g = GameState(self.n)
        g.board = self.board.copy()
        g.to_play = self.to_play
        g.moves = self.moves
        g.winner = self.winner
        return g

    def legal_actions(self) -> np.ndarray:
        return (self.board.reshape(-1) == 0)

    def is_terminal(self) -> bool:
        return self.winner != 0

    def step(self, a: int) -> "GameState":
        x, y = divmod(a, self.n)
        assert self.board[x, y] == 0
        color = self.to_play
        self.board[x, y] = color
        self.moves += 1

        if is_exact5_after_move(self.board, x, y, color):
            self.winner = color
        elif self.moves >= self.n * self.n:
            self.winner = -1  # draw
        else:
            self.to_play = 2 if self.to_play == 1 else 1
        return self

    def result_from(self, player: Player) -> float:
        if self.winner == -1: return 0.0
        return 1.0 if self.winner == player else -1.0

    def encode(self) -> np.ndarray:
        return encode_state(self.board, self.to_play)
