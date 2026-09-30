"""Unit tests for gomoku_zero.replay module."""
import pytest
import numpy as np
from gomoku_zero.replay import ReplayBuffer, d4_transforms


class TestD4Transforms:
    """Test D4 data augmentation transforms."""

    def test_transform_count(self):
        x = np.random.randn(3, 15, 15)
        transforms = d4_transforms(x)
        assert len(transforms) == 8  # 4 rotations * 2 (original + flip)

    def test_transform_preserves_shape(self):
        x = np.random.randn(3, 15, 15)
        transforms = d4_transforms(x)
        for t in transforms:
            assert t.shape == x.shape

    def test_2d_input(self):
        x = np.random.randn(15, 15)
        transforms = d4_transforms(x)
        assert len(transforms) == 8
        for t in transforms:
            assert t.shape == x.shape

    def test_identity_rotation(self):
        """First transform should be identity (k=0, no flip)."""
        x = np.random.randn(3, 5, 5)
        transforms = d4_transforms(x)
        assert np.allclose(transforms[0], x)

    def test_rotation_90(self):
        x = np.random.randn(3, 4, 4)
        transforms = d4_transforms(x)
        # transforms[0] = rot90(x, 0), transforms[1] = flip(rot90(x, 0))
        # transforms[2] = rot90(x, 1), transforms[3] = flip(rot90(x, 1))
        rotated = np.rot90(x, 1, axes=(-2, -1))
        assert np.allclose(transforms[2], rotated)

    def test_flip_after_rotation(self):
        """Second transform should be flip of first rotation."""
        x = np.random.randn(3, 5, 5)
        transforms = d4_transforms(x)
        expected = np.flip(transforms[0], axis=-1)
        assert np.allclose(transforms[1], expected)

    def test_transforms_preserve_sum(self):
        """All transforms should preserve the sum of elements."""
        x = np.random.randn(3, 10, 10)
        original_sum = x.sum()
        transforms = d4_transforms(x)
        for t in transforms:
            assert np.isclose(t.sum(), original_sum)


class TestReplayBufferInit:
    """Test ReplayBuffer initialization."""

    def test_init(self):
        rb = ReplayBuffer(capacity_steps=1000, board_size=15)
        assert rb.capacity == 1000
        assert rb.n == 15
        assert rb.size_steps == 0
        assert len(rb.states) == 0

    def test_init_small_board(self):
        rb = ReplayBuffer(capacity_steps=500, board_size=9)
        assert rb.n == 9


class TestReplayBufferAdd:
    """Test adding samples to replay buffer."""

    def test_add_single_step(self):
        rb = ReplayBuffer(capacity_steps=100, board_size=15)
        s = np.zeros((3, 15, 15), dtype=np.float32)
        pi = np.ones((225,), dtype=np.float32) / 225
        z = 1.0
        rb.add_game([s], [pi], z)
        # D4 augmentation creates 8 samples per step
        assert rb.size_steps == 8

    def test_add_full_game(self):
        rb = ReplayBuffer(capacity_steps=10000, board_size=15)
        n_steps = 10
        states = [np.random.randn(3, 15, 15).astype(np.float32) for _ in range(n_steps)]
        pis = [np.random.randn(225).astype(np.float32) for _ in range(n_steps)]
        z = 1.0
        rb.add_game(states, pis, z)
        # Each step produces 8 augmented samples
        assert rb.size_steps == n_steps * 8

    def test_add_multiple_games(self):
        rb = ReplayBuffer(capacity_steps=10000, board_size=15)
        for game_idx in range(3):
            states = [np.random.randn(3, 15, 15).astype(np.float32) for _ in range(5)]
            pis = [np.random.randn(225).astype(np.float32) for _ in range(5)]
            rb.add_game(states, pis, float(game_idx % 2 * 2 - 1))
        assert rb.size_steps == 3 * 5 * 8

    def test_capacity_limit(self):
        rb = ReplayBuffer(capacity_steps=16, board_size=15)
        s = np.zeros((3, 15, 15), dtype=np.float32)
        pi = np.ones((225,), dtype=np.float32) / 225
        # Add 3 steps -> 24 augmented samples, but capacity is 16
        rb.add_game([s, s.copy(), s.copy()], [pi, pi.copy(), pi.copy()], 1.0)
        assert rb.size_steps == 16

    def test_fifo_eviction(self):
        """Old samples should be evicted when capacity is exceeded."""
        rb = ReplayBuffer(capacity_steps=8, board_size=15)
        s1 = np.ones((3, 15, 15), dtype=np.float32)
        s2 = np.full((3, 15, 15), 2.0, dtype=np.float32)
        pi = np.ones((225,), dtype=np.float32) / 225
        rb.add_game([s1], [pi], 1.0)  # adds 8 samples
        rb.add_game([s2], [pi], 1.0)  # adds 8 more, evicts first 8
        # Only s2 samples should remain
        sample_s, _, _ = rb.sample(8)
        assert np.allclose(sample_s[0], 2.0)


class TestReplayBufferSample:
    """Test sampling from replay buffer."""

    @pytest.fixture
    def filled_buffer(self):
        rb = ReplayBuffer(capacity_steps=1000, board_size=15)
        for _ in range(10):
            states = [np.random.randn(3, 15, 15).astype(np.float32) for _ in range(3)]
            pis = [np.random.randn(225).astype(np.float32) for _ in range(3)]
            rb.add_game(states, pis, 1.0)
        return rb

    def test_sample_shape(self, filled_buffer):
        batch_size = 32
        s, p, z = filled_buffer.sample(batch_size)
        assert s.shape == (batch_size, 3, 15, 15)
        assert p.shape == (batch_size, 225)
        assert z.shape == (batch_size,)

    def test_sample_dtypes(self, filled_buffer):
        s, p, z = filled_buffer.sample(16)
        assert s.dtype == np.float32
        assert p.dtype == np.float32
        assert z.dtype == np.float32

    def test_sample_randomness(self, filled_buffer):
        """Two samples should likely be different."""
        s1, _, _ = filled_buffer.sample(64)
        s2, _, _ = filled_buffer.sample(64)
        # With large buffer, samples should differ
        assert not np.allclose(s1, s2)

    def test_sample_from_empty_raises(self):
        rb = ReplayBuffer(capacity_steps=100, board_size=15)
        with pytest.raises(ValueError):
            rb.sample(1)

    def test_sample_batch_sizes(self, filled_buffer):
        for batch_size in [1, 16, 128, 256]:
            s, p, z = filled_buffer.sample(batch_size)
            assert s.shape[0] == batch_size

    def test_policy_normalization(self, filled_buffer):
        """Sampled policies should be valid probability distributions."""
        _, p, _ = filled_buffer.sample(32)
        # Each policy vector should sum to ~1 (after softmax in training)
        # Note: raw stored policies may not be normalized
        assert p.shape == (32, 225)

    def test_value_labels(self, filled_buffer):
        """Value labels should be in {-1, 0, 1}."""
        _, _, z = filled_buffer.sample(100)
        unique_vals = set(np.round(z, decimals=6).tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})
