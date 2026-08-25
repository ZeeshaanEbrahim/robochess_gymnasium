"""Algebraic square naming for the 78 mm board (decision D1 in DECISIONS.md).

Square centres are derived algebraically from the board's own regular 8x8
grid rather than read from the model: only the 32 dark squares exist as
named geoms in ``assets/scene.xml`` (the light squares are the slab itself),
so there is no ``sq_e4``-style geom for every square to look up.
"""

from __future__ import annotations

import numpy as np

from robochess.envs.robochess_env import SQUARE_PITCH

FILES = "abcdefgh"
RANKS = "12345678"

# All 64 squares, algebraic notation. Confirmed IK-reachable for a top-down
# grasp by the D2 sweep in DECISIONS.md -- see t1_pick_and_place.py's
# REACHABLE_SQUARES for the caveat on what "confirmed" means here.
ALL_SQUARES: tuple[str, ...] = tuple(f + r for f in FILES for r in RANKS)


def square_to_local_xy(square: str) -> np.ndarray:
    """Board-local (x, y) offset of a square's centre, in metres.

    Board-local means relative to the board body's own origin -- add the
    board body's world ``xpos[:2]`` to get a world-frame position.
    """
    if len(square) != 2 or square[0] not in FILES or square[1] not in RANKS:
        raise ValueError(f"Not a valid square: {square!r}")
    file_idx = FILES.index(square[0])
    rank_idx = RANKS.index(square[1])
    x = (file_idx - 3.5) * SQUARE_PITCH
    y = (rank_idx - 3.5) * SQUARE_PITCH
    return np.array([x, y], dtype=np.float64)
