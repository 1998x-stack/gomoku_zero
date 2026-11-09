"""P2 exact-five cross-project terminal contract checks."""
import gymnasium as gym
import pytest
import gomoku_exact5
from gomoku_zero.game import GameState


def test_core_and_environment_guard_terminal_in_same_episode():
    env=gym.make("GomokuExact5-v0",board_size=7)
    g=GameState(board_size=7)
    env.reset(seed=1)
    for a in [0,42,1,43,2,44,3,45,4]:
        env.step(a)
        g.step(a)
    assert g.winner==1
    with pytest.raises(RuntimeError):
        env.step(5)
    with pytest.raises(RuntimeError):
        g.step(5)
    env.close()


def test_invalid_and_occupied_actions_fail_before_mutation():
    g=GameState(board_size=5)
    with pytest.raises(ValueError):
        g.step(-1)
    with pytest.raises(ValueError):
        g.step(25)
    g.step(0)
    before=g.board.copy()
    with pytest.raises(ValueError):
        g.step(0)
    assert (g.board == before).all()
    assert g.moves==1
