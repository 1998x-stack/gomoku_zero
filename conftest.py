"""Root pytest configuration.

Makes the test-suite runnable **without** requiring `pip install -e .`:

1. Puts the local ``gomoku_zero`` package (``src/`` layout) on ``sys.path``.
2. If the sibling ``gomoku-exact5-env`` package is not installed, adds its
   ``src/`` directory (``../gomoku-exact5-env/src``) to ``sys.path`` so
   ``import gomoku_exact5`` works for the whole suite.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def _prepend(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


_prepend(os.path.join(ROOT, "src"))

try:
    import gomoku_exact5  # noqa: F401
except ImportError:  # pragma: no cover - only when the sibling is not installed
    env_src = os.path.join(ROOT, "..", "gomoku-exact5-env", "src")
    if os.path.isdir(env_src):
        _prepend(env_src)