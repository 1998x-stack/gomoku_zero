"""Focused exact-five contract gate without PyTorch or pytest conftest."""
import unittest
import numpy as np
import gymnasium as gym
import gomoku_exact5
from gomoku_zero.game import GameState


class ExactFiveContract(unittest.TestCase):
    def test_random_episode_observations_agree(self):
        g=GameState(5)
        env=gym.make("GomokuExact5-v0",board_size=5)
        obs,_=env.reset(seed=0)
        try:
            for a in np.random.default_rng(1234).permutation(25):
                a=int(a)
                self.assertTrue(g.legal_actions()[a])
                self.assertEqual(obs["action_mask"][a],1)
                obs,reward,terminated,truncated,info=env.step(a)
                g.step(a)
                self.assertTrue(np.array_equal(obs["board"],g.board))
                self.assertEqual(obs["to_play"],0 if g.to_play==1 else 1)
                self.assertEqual(terminated or truncated,g.is_terminal())
                if terminated or truncated:
                    if g.winner==-1:
                        self.assertTrue(info.get("draw"))
                    else:
                        self.assertEqual(info["winner"],g.winner)
                        self.assertEqual(reward,1.0)
                    break
        finally:
            env.close()

    def test_exact_five_and_terminal_guard_agree(self):
        env=gym.make("GomokuExact5-v0",board_size=7)
        g=GameState(7)
        env.reset(seed=1)
        try:
            for a in [0,42,1,43,2,44,3,45,4]:
                obs,reward,terminated,truncated,info=env.step(a)
                g.step(a)
            self.assertEqual(g.winner,1)
            self.assertTrue(terminated)
            self.assertEqual(reward,1.0)
            self.assertEqual(info.get("winner"),1)
            with self.assertRaises(RuntimeError): env.step(5)
            with self.assertRaises(RuntimeError): g.step(5)
        finally:
            env.close()

    def test_invalid_action_is_nonmutating(self):
        g=GameState(5)
        with self.assertRaises(ValueError): g.step(-1)
        with self.assertRaises(ValueError): g.step(25)
        g.step(0)
        previous=g.board.copy()
        with self.assertRaises(ValueError): g.step(0)
        self.assertTrue(np.array_equal(previous,g.board))
        self.assertEqual(g.moves,1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
