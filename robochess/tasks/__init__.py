"""Task-layer wrappers around RoboChessEnv (T1-T5)."""

from robochess.tasks.t1_pick_and_place import (
    REACHABLE_SQUARES,
    ShapedPickAndPlaceReward,
    SparsePickAndPlaceReward,
    T1PickAndPlaceTask,
)

__all__ = [
    "REACHABLE_SQUARES",
    "ShapedPickAndPlaceReward",
    "SparsePickAndPlaceReward",
    "T1PickAndPlaceTask",
]
