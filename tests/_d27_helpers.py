"""Shared test helpers for D27 (carry-and-low-release success requirement)."""
from __future__ import annotations


def prime_carried_release(env, xy_error: float = 0.0, height: float = 0.0) -> None:
    """Stand in for a real carry-and-release on `env` (any wrapper stack).

    Stands in for a real carry, the same way these tests teleport the piece to
    stand in for a real place. D27 itself is tested with real scenarios in
    test_d27_release_latch.py.

    Sets the base env's `_ever_held` and `_release_record` (xy error and base
    height above the board, in metres) via ``env.unwrapped``. Call it AFTER
    ``reset()`` (which clears both) and before the step whose success is
    being tested.
    """
    base = env.unwrapped
    base._ever_held = True
    base._release_record = (float(xy_error), float(height))
