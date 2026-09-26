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


def validate_excluded_squares(excluded: object) -> tuple[str, ...]:
    """The one place ``excluded_squares`` is validated (D26, same idea as F11).

    Called by the task wrappers' constructors and by the config layer. Must be
    a list/tuple (not a bare string) of distinct, valid square names; returns
    them as a tuple. Empty means "exclude nothing".
    """
    if not isinstance(excluded, (list, tuple)):
        raise ValueError(
            "excluded_squares must be a sequence (list or tuple) of square names, "
            f"got {type(excluded).__name__}: {excluded!r}."
        )
    squares = tuple(excluded)
    for sq in squares:
        if not isinstance(sq, str) or sq not in ALL_SQUARES:
            raise ValueError(f"excluded_squares: not a valid square: {sq!r}")
    if len(set(squares)) != len(squares):
        raise ValueError("excluded_squares must be distinct.")
    return squares


def reject_excluded(role: str, squares: object, excluded: tuple[str, ...]) -> None:
    """Raise ``ValueError`` if ``squares`` (one name or an iterable) hits ``excluded``.

    One rule for every role (start, target, obstacle): an excluded square is
    not used by the task in any role.
    """
    names = (squares,) if isinstance(squares, str) else tuple(squares)
    bad = [sq for sq in names if sq in excluded]
    if bad:
        raise ValueError(f"{role} {bad[0]!r} is an excluded square (excluded_squares={list(excluded)}).")
