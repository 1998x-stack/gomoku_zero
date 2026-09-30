"""Unit tests for gomoku_zero.game module."""
import pytest
import numpy as np
from gomoku_zero.game import GameState, encode_state


class TestEncodeState:
    """Test state encoding functionality."""

    def test_encode_empty_board_black(self):
        board = np.zeros((15, 15), dtype=np.int8)
        encoded = encode_state(board, 1)
        assert encoded.shape == (3, 15, 15)
        assert encoded.dtype == np.float32
        # Black to play: ch0 all zeros, ch1 all zeros, ch2 all ones
        assert np.allclose(encoded[0], 0)
        assert np.allclose(encoded[1], 0)
        assert np.allclose(encoded[2], 1)

    def test_encode_empty_board_white(self):
        board = np.zeros((15, 15), dtype=np.int8)
        encoded = encode_state(board, 2)
        assert encoded.shape == (3, 15, 15)
        # White to play: ch0 all zeros, ch1 all zeros, ch2 all zeros
        assert np.allclose(encoded[0], 0)
        assert np.allclose(encoded[1], 0)
        assert np.allclose(encoded[2], 0)

    def test_encode_with_pieces(self):
        board = np.zeros((15, 15), dtype=np.int8)
        board[7, 7] = 1  # black piece
        board[7, 8] = 2  # white piece
        encoded = encode_state(board, 1)
        # Black to play
        assert encoded[0, 7, 7] == 1.0  # current player's piece
        assert encoded[1, 7, 8] == 1.0  # opponent's piece

    def test_encode_different_players(self):
        board = np.zeros((15, 15), dtype=np.int8)
        board[5, 5] = 1
        board[6, 6] = 2
        enc_black = encode_state(board, 1)
        enc_white = encode_state(board, 2)
        # Channels should be swapped
        assert np.allclose(enc_black[0], enc_white[1])
        assert np.allclose(enc_black[1], enc_white[0])


class TestGameStateInit:
    """Test GameState initialization."""

    def test_default_init(self):
        g = GameState()
        assert g.n == 15
        assert g.board.shape == (15, 15)
        assert np.all(g.board == 0)
        assert g.to_play == 1
        assert g.moves == 0
        assert g.winner == 0

    def test_custom_size_init(self):
        g = GameState(board_size=13)
        assert g.n == 13
        assert g.board.shape == (13, 13)

    def test_copy(self):
        g = GameState()
        g.board[7, 7] = 1
        g.to_play = 2
        g.moves = 1
        g_copy = g.copy()
        assert np.array_equal(g.board, g_copy.board)
        assert g.to_play == g_copy.to_play
        assert g.moves == g_copy.moves
        assert g.winner == g_copy.winner
        # Verify deep copy
        g_copy.board[0, 0] = 1
        assert g.board[0, 0] == 0


class TestLegalActions:
    """Test legal action generation."""

    def test_all_legal_at_start(self):
        g = GameState()
        legal = g.legal_actions()
        assert legal.shape == (225,)
        assert legal.sum() == 225

    def test_legal_after_move(self):
        g = GameState()
        g.step(7 * 15 + 7)  # center
        legal = g.legal_actions()
        assert legal.sum() == 224
        assert not legal[7 * 15 + 7]

    def test_no_legal_when_full(self):
        g = GameState(board_size=3)
        for i in range(9):
            g.step(i)
        legal = g.legal_actions()
        assert legal.sum() == 0


class TestStepAndTerminal:
    """Test game progression and terminal states."""

    def test_step_changes_player(self):
        g = GameState()
        assert g.to_play == 1
        g.step(7 * 15 + 7)
        assert g.to_play == 2
        g.step(7 * 15 + 8)
        assert g.to_play == 1

    def test_step_increments_moves(self):
        g = GameState()
        assert g.moves == 0
        g.step(100)
        assert g.moves == 1
        g.step(101)
        assert g.moves == 2

    def test_not_terminal_at_start(self):
        g = GameState()
        assert not g.is_terminal()
        assert g.winner == 0

    def test_draw_on_full_board(self):
        g = GameState(board_size=3)
        for i in range(9):
            g.step(i)
        assert g.is_terminal()
        assert g.winner == -1

    def test_result_from_winner(self):
        g = GameState()
        g.winner = 1
        assert g.result_from(1) == 1.0
        assert g.result_from(2) == -1.0

    def test_result_from_draw(self):
        g = GameState()
        g.winner = -1
        assert g.result_from(1) == 0.0
        assert g.result_from(2) == 0.0


class TestExactFiveWin:
    """Test exact five-in-a-row win detection (requires gomoku_exact5)."""

    def test_horizontal_five(self):
        """Test horizontal five consecutive pieces."""
        g = GameState()
        # Place 5 black pieces horizontally
        for i in range(5):
            g.step(7 * 15 + i)
            if i < 4:
                g.step(8 * 15 + i)  # white plays in between
        assert g.winner == 1
        assert g.is_terminal()

    def test_vertical_five(self):
        """Test vertical five consecutive pieces."""
        g = GameState()
        for i in range(5):
            g.step((7 + i) * 15 + 7)
            if i < 4:
                g.step((7 + i) * 15 + 8)
        assert g.winner == 1

    def test_diagonal_five(self):
        """Test diagonal five consecutive pieces."""
        g = GameState()
        for i in range(5):
            g.step((7 + i) * 15 + (7 + i))
            if i < 4:
                g.step((7 + i) * 15 + (8 + i))
        assert g.winner == 1

    def test_six_is_not_win(self):
        """Test that six in a row is NOT a win (exact five rule).
        
        Note: This test is tricky because with alternating turns,
        getting exactly 6 in a row without forming 5 first is difficult.
        The exact5 rule checks that the run is exactly 5 with no extensions.
        Here we verify that 6 consecutive pieces does NOT win.
        """
        from gomoku_exact5.rules import is_exact5_after_move
        import numpy as np
        
        # Directly test the rule function with a board that has 6 in a row
        board = np.zeros((15, 15), dtype=np.int8)
        # Place 6 black pieces at row 7, cols 2-7
        for col in range(2, 8):
            board[7, col] = 1
        # White pieces block both ends
        board[7, 1] = 2
        board[7, 8] = 2
        
        # Check each of the 6 black positions - none should trigger exact5
        for col in range(2, 8):
            result = is_exact5_after_move(board, 7, col, 1)
            assert not result, f"Position (7,{col}) should not be exact5 win in a 6-run"

    def test_four_is_not_win(self):
        """Test that four in a row is not a win."""
        g = GameState()
        for i in range(4):
            g.step(7 * 15 + i)
            if i < 3:
                g.step(8 * 15 + i)
        assert g.winner == 0
        assert not g.is_terminal()


class TestEncodeMethod:
    """Test GameState.encode method."""

    def test_encode_returns_correct_shape(self):
        g = GameState()
        enc = g.encode()
        assert enc.shape == (3, 15, 15)
        assert enc.dtype == np.float32

    def test_encode_reflects_board_state(self):
        g = GameState()
        g.step(100)
        enc = g.encode()
        # After one move, there should be one piece
        assert enc[0].sum() + enc[1].sum() == 1
