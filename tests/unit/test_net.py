"""Unit tests for gomoku_zero.net module."""
import pytest
import torch
from gomoku_zero.net import PolicyValueNet, ResidualBlock, numerical_params


class TestResidualBlock:
    """Test ResidualBlock component."""

    def test_forward_shape(self):
        ch = 64
        block = ResidualBlock(ch)
        x = torch.randn(1, ch, 15, 15)
        y = block(x)
        assert y.shape == x.shape

    def test_residual_connection(self):
        """Verify residual connection preserves shape."""
        ch = 32
        block = ResidualBlock(ch)
        x = torch.randn(2, ch, 9, 9)
        y = block(x)
        assert y.shape == (2, ch, 9, 9)

    def test_output_not_zero(self):
        """Ensure block produces non-zero output."""
        ch = 16
        block = ResidualBlock(ch)
        x = torch.randn(1, ch, 5, 5)
        y = block(x)
        assert not torch.allclose(y, torch.zeros_like(y))


class TestPolicyValueNet:
    """Test PolicyValueNet architecture."""

    def test_init_default(self):
        net = PolicyValueNet()
        assert net.board_size == 15

    def test_init_custom(self):
        net = PolicyValueNet(board_size=13, in_ch=3, ch=64, blocks=8)
        assert net.board_size == 13

    def test_forward_policy_shape(self):
        net = PolicyValueNet(board_size=15, ch=32, blocks=4)
        x = torch.randn(1, 3, 15, 15)
        logits, value = net(x)
        assert logits.shape == (1, 225)  # 15*15

    def test_forward_value_shape(self):
        net = PolicyValueNet(board_size=15, ch=32, blocks=4)
        x = torch.randn(1, 3, 15, 15)
        logits, value = net(x)
        assert value.shape == (1,)

    def test_forward_batch(self):
        net = PolicyValueNet(board_size=13, ch=32, blocks=4)
        batch_size = 4
        x = torch.randn(batch_size, 3, 13, 13)
        logits, value = net(x)
        assert logits.shape == (batch_size, 169)  # 13*13
        assert value.shape == (batch_size,)

    def test_value_range(self):
        """Value head output should be in [-1, 1] due to Tanh."""
        net = PolicyValueNet(board_size=15, ch=32, blocks=4)
        x = torch.randn(10, 3, 15, 15)
        _, value = net(x)
        assert value.min() >= -1.0
        assert value.max() <= 1.0

    def test_initial_infer_no_grad(self):
        """initial_infer should work without gradient tracking."""
        net = PolicyValueNet(board_size=15, ch=32, blocks=4)
        x = torch.randn(1, 3, 15, 15)
        with torch.no_grad():
            logits, value = net.initial_infer(x)
        assert logits.shape == (1, 225)
        assert value.shape == (1,)

    def test_different_board_sizes(self):
        for size in [9, 13, 15, 19]:
            net = PolicyValueNet(board_size=size, ch=16, blocks=2)
            x = torch.randn(1, 3, size, size)
            logits, value = net(x)
            assert logits.shape == (1, size * size)
            assert value.shape == (1,)


class TestNumericalParams:
    """Test parameter counting utility."""

    def test_param_count_positive(self):
        net = PolicyValueNet(board_size=15, ch=32, blocks=4)
        count = numerical_params(net)
        assert count > 0

    def test_larger_network_more_params(self):
        net_small = PolicyValueNet(board_size=15, ch=32, blocks=4)
        net_large = PolicyValueNet(board_size=15, ch=128, blocks=12)
        assert numerical_params(net_large) > numerical_params(net_small)

    def test_board_size_affects_params(self):
        net_13 = PolicyValueNet(board_size=13, ch=64, blocks=8)
        net_15 = PolicyValueNet(board_size=15, ch=64, blocks=8)
        # Larger board = more FC params in policy/value heads
        assert numerical_params(net_15) > numerical_params(net_13)

    def test_expected_param_scale(self):
        """Rough sanity check on parameter count."""
        net = PolicyValueNet(board_size=15, ch=128, blocks=12)
        count = numerical_params(net)
        # AlphaZero-style networks are typically in the millions
        assert count > 1_000_000
        assert count < 100_000_000


class TestNetworkDeterminism:
    """Test network determinism with same seed."""

    def test_same_input_same_output(self):
        net = PolicyValueNet(board_size=15, ch=32, blocks=4)
        net.eval()
        x = torch.randn(1, 3, 15, 15)
        with torch.no_grad():
            logits1, value1 = net(x)
            logits2, value2 = net(x)
        assert torch.allclose(logits1, logits2)
        assert torch.allclose(value1, value2)
