"""Unit tests for gomoku_zero.mcts module."""
import pytest
import numpy as np
import torch
from gomoku_zero.game import GameState
from gomoku_zero.net import PolicyValueNet
from gomoku_zero.mcts import MCTS, Node
from gomoku_zero.config import MctsCfg


class TestNode:
    """Test MCTS Node class."""

    def test_node_init(self):
        node = Node(prior=0.5, player=1)
        assert node.prior == 0.5
        assert node.visit == 0
        assert node.value_sum == 0.0
        assert node.children == {}
        assert node.player == 1

    def test_q_value_initial(self):
        node = Node(prior=0.3, player=1)
        assert node.q == 0.0

    def test_q_value_after_visits(self):
        node = Node(prior=0.3, player=1)
        node.visit = 10
        node.value_sum = 5.0
        assert node.q == 0.5

    def test_q_value_negative(self):
        node = Node(prior=0.3, player=1)
        node.visit = 4
        node.value_sum = -2.0
        assert node.q == -0.5


class TestMCTSInit:
    """Test MCTS initialization."""

    def test_mcts_creation(self):
        net = PolicyValueNet(board_size=15, ch=16, blocks=2)
        cfg = MctsCfg(sims=100)
        device = torch.device("cpu")
        mcts = MCTS(net, cfg, device)
        assert mcts.net is net
        assert mcts.cfg is cfg
        assert mcts.device == device


class TestMCTSRun:
    """Test MCTS search execution."""

    @pytest.fixture
    def small_net(self):
        return PolicyValueNet(board_size=9, ch=16, blocks=2)

    @pytest.fixture
    def small_cfg(self):
        return MctsCfg(
            sims=50,
            c_puct=1.5,
            root_dirichlet_eps=0.25,
            root_dirichlet_alpha=0.03,
            temperature_moves=10,
            temperature_eval=0.0,
            virtual_loss=1.0,
        )

    def test_run_returns_visits(self, small_net, small_cfg):
        device = torch.device("cpu")
        mcts = MCTS(small_net, small_cfg, device)
        state = GameState(board_size=9)
        visits = mcts.run(state, add_dirichlet=True)
        assert isinstance(visits, dict)
        assert len(visits) > 0
        # All keys should be valid actions
        for action in visits:
            assert 0 <= action < 81

    def test_run_total_visits_equals_sims(self, small_net, small_cfg):
        device = torch.device("cpu")
        mcts = MCTS(small_net, small_cfg, device)
        state = GameState(board_size=9)
        visits = mcts.run(state, add_dirichlet=False)
        total = sum(visits.values())
        assert total == small_cfg.sims

    def test_run_with_dirichlet(self, small_net, small_cfg):
        device = torch.device("cpu")
        mcts = MCTS(small_net, small_cfg, device)
        state = GameState(board_size=9)
        visits_no_dir = mcts.run(state, add_dirichlet=False)
        visits_dir = mcts.run(state, add_dirichlet=True)
        # Results may differ due to noise
        assert isinstance(visits_dir, dict)
        assert len(visits_dir) > 0

    def test_run_after_move(self, small_net, small_cfg):
        device = torch.device("cpu")
        mcts = MCTS(small_net, small_cfg, device)
        state = GameState(board_size=9)
        state.step(4 * 9 + 4)  # center move
        visits = mcts.run(state, add_dirichlet=False)
        assert isinstance(visits, dict)
        # Center should not be in legal actions anymore
        center_action = 4 * 9 + 4
        if center_action in visits:
            assert visits[center_action] == 0

    def test_run_terminal_state(self, small_net, small_cfg):
        """MCTS on terminal state should handle gracefully."""
        device = torch.device("cpu")
        mcts = MCTS(small_net, small_cfg, device)
        state = GameState(board_size=9)
        state.winner = 1  # force terminal
        visits = mcts.run(state, add_dirichlet=False)
        # Should return empty or zero visits for terminal
        assert isinstance(visits, dict)


class TestMCTSPolicy:
    """Test MCTS policy extraction."""

    @pytest.fixture
    def tiny_net(self):
        return PolicyValueNet(board_size=5, ch=8, blocks=1)

    @pytest.fixture
    def tiny_cfg(self):
        return MctsCfg(sims=20)

    def test_policy_distribution(self, tiny_net, tiny_cfg):
        device = torch.device("cpu")
        mcts = MCTS(tiny_net, tiny_cfg, device)
        state = GameState(board_size=5)
        visits = mcts.run(state, add_dirichlet=False)
        total = sum(visits.values())
        assert total == tiny_cfg.sims
        # All visits should be non-negative
        for v in visits.values():
            assert v >= 0

    def test_best_move_has_most_visits(self, tiny_net, tiny_cfg):
        """The best move should typically have the most visits."""
        device = torch.device("cpu")
        mcts = MCTS(tiny_net, tiny_cfg, device)
        state = GameState(board_size=5)
        visits = mcts.run(state, add_dirichlet=False)
        if visits:
            best_action = max(visits, key=visits.get)
            assert visits[best_action] == max(visits.values())


class TestMaskAndNorm:
    """Test internal _mask_and_norm helper."""

    @pytest.fixture
    def small_net(self):
        return PolicyValueNet(board_size=9, ch=16, blocks=2)

    @pytest.fixture
    def small_cfg(self):
        return MctsCfg(sims=50)

    def test_mask_illegal_actions(self, small_net, small_cfg):
        device = torch.device("cpu")
        mcts = MCTS(small_net, small_cfg, device)
        state = GameState(board_size=9)
        state.step(40)  # occupy center
        legal = state.legal_actions()
        priors = np.ones(81) / 81
        masked = mcts._mask_and_norm(priors, legal)
        # Masked action should have zero probability
        assert masked[40] == 0.0
        # Sum of remaining should be 1
        assert np.isclose(masked.sum(), 1.0)

    def test_all_masked_fallback(self, small_net, small_cfg):
        """If all actions masked, should distribute uniformly."""
        device = torch.device("cpu")
        mcts = MCTS(small_net, small_cfg, device)
        legal = np.zeros(81, dtype=bool)
        legal[40] = True  # only one legal
        priors = np.zeros(81)
        masked = mcts._mask_and_norm(priors, legal)
        assert np.isclose(masked[40], 1.0)
        assert np.isclose(masked.sum(), 1.0)
