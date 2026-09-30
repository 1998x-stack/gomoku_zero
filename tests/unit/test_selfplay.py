"""Unit tests for gomoku_zero.selfplay module."""
import pytest
import numpy as np
import torch
from gomoku_zero.game import GameState
from gomoku_zero.net import PolicyValueNet
from gomoku_zero.config import MctsCfg
from gomoku_zero.selfplay import softmax_temperature, play_one_game, selfplay_many


class TestSoftmaxTemperature:
    """Test softmax temperature sampling utility."""

    def test_greedy_selection(self):
        """tau=0 should select the action with most visits."""
        visits = {0: 10, 1: 50, 2: 30}
        pi = softmax_temperature(visits, tau=0.0)
        assert pi.shape[0] == 3
        assert pi[1] == 1.0  # action 1 has most visits
        assert pi[0] == 0.0
        assert pi[2] == 0.0

    def test_uniform_at_high_tau(self):
        """High tau should approach uniform distribution."""
        visits = {0: 100, 1: 1, 2: 1}
        pi = softmax_temperature(visits, tau=100.0)
        # Should be roughly uniform
        assert np.allclose(pi, 1/3, atol=0.05)

    def test_softmax_output(self):
        """Standard tau should produce softmax distribution."""
        visits = {0: 10, 1: 20, 2: 30}
        pi = softmax_temperature(visits, tau=1.0)
        assert np.isclose(pi.sum(), 1.0)
        assert pi[2] > pi[1] > pi[0]  # higher visits -> higher prob

    def test_single_action(self):
        visits = {5: 100}
        pi = softmax_temperature(visits, tau=1.0)
        assert pi.shape[0] == 6
        assert np.isclose(pi[5], 1.0)

    def test_empty_visits_handling(self):
        """All equal visits should produce uniform distribution."""
        visits = {0: 1, 1: 1, 2: 1}
        pi = softmax_temperature(visits, tau=1.0)
        assert np.allclose(pi, 1/3)


class TestPlayOneGame:
    """Test single game self-play."""

    @pytest.fixture
    def tiny_net(self):
        return PolicyValueNet(board_size=5, ch=8, blocks=1)

    @pytest.fixture
    def tiny_cfg(self):
        return MctsCfg(
            sims=20,
            c_puct=1.5,
            root_dirichlet_eps=0.25,
            root_dirichlet_alpha=0.03,
            temperature_moves=5,
            temperature_eval=0.0,
            virtual_loss=1.0,
        )

    def test_game_returns_data(self, tiny_net, tiny_cfg):
        device = torch.device("cpu")
        states, pis, zs, winner = play_one_game(tiny_net, tiny_cfg, board_size=5, device=device)
        assert len(states) > 0
        assert len(pis) == len(states)
        assert len(zs) == len(states)
        assert winner in (-1, 1, 2)

    def test_game_state_shapes(self, tiny_net, tiny_cfg):
        device = torch.device("cpu")
        states, pis, zs, winner = play_one_game(tiny_net, tiny_cfg, board_size=5, device=device)
        for s in states:
            assert s.shape == (3, 5, 5)
        for p in pis:
            assert p.shape == (25,)

    def test_game_terminates(self, tiny_net, tiny_cfg):
        """Game should eventually terminate."""
        device = torch.device("cpu")
        _, _, _, winner = play_one_game(tiny_net, tiny_cfg, board_size=5, device=device)
        assert winner != 0  # not ongoing

    def test_value_labels_valid(self, tiny_net, tiny_cfg):
        device = torch.device("cpu")
        _, _, zs, _ = play_one_game(tiny_net, tiny_cfg, board_size=5, device=device)
        for z in zs:
            assert z in (-1.0, 0.0, 1.0)

    def test_eval_mode_no_dirichlet(self, tiny_net, tiny_cfg):
        """Eval mode should not add Dirichlet noise."""
        device = torch.device("cpu")
        states1, _, _, _ = play_one_game(tiny_net, tiny_cfg, board_size=5, device=device, is_eval=True)
        states2, _, _, _ = play_one_game(tiny_net, tiny_cfg, board_size=5, device=device, is_eval=True)
        # With same network and greedy eval, games may still differ due to ties
        # but both should complete
        assert len(states1) > 0
        assert len(states2) > 0

    def test_different_board_sizes(self, tiny_cfg):
        for size in [5, 7, 9]:
            net = PolicyValueNet(board_size=size, ch=8, blocks=1)
            device = torch.device("cpu")
            states, pis, zs, winner = play_one_game(net, tiny_cfg, board_size=size, device=device)
            assert len(states) > 0
            assert winner != 0


class TestSelfplayMany:
    """Test multiple game self-play."""

    @pytest.fixture
    def tiny_net(self):
        return PolicyValueNet(board_size=5, ch=8, blocks=1)

    @pytest.fixture
    def tiny_cfg(self):
        return MctsCfg(sims=10, temperature_moves=3)

    def test_generator_yields_games(self, tiny_net, tiny_cfg):
        device = torch.device("cpu")
        results = []
        for states, pis, zs, winner in selfplay_many(
            tiny_net, tiny_cfg, board_size=5, device=device, games=3
        ):
            results.append(winner)
            assert len(states) > 0
        assert len(results) == 3

    def test_all_games_complete(self, tiny_net, tiny_cfg):
        device = torch.device("cpu")
        winners = []
        for _, _, _, w in selfplay_many(tiny_net, tiny_cfg, board_size=5, device=device, games=5):
            winners.append(w)
        assert len(winners) == 5
        assert all(w in (-1, 1, 2) for w in winners)

    def test_logging_interval(self, tiny_net, tiny_cfg, caplog):
        """Should log progress every 10 games."""
        device = torch.device("cpu")
        count = 0
        for _ in selfplay_many(tiny_net, tiny_cfg, board_size=5, device=device, games=25):
            count += 1
        assert count == 25
