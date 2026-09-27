"""Task-layer wrappers around RoboChessEnv (T1-T5)."""

from robochess.tasks.t1_pick_and_place import (
    REACHABLE_SQUARES,
    PickupObservation,
    PickupReward,
    PlaceObservation,
    PlaceReward,
    ShapedPickAndPlaceReward,
    SparsePickAndPlaceReward,
    T1PickAndPlaceTask,
)
from robochess.tasks.t2_obstacle_pick_and_place import (
    N_OBSTACLE_SLOTS,
    OBSTACLE_BODIES,
    OBSTACLE_DISPLACEMENT_TOL,
    PARKED_XY,
    T2_SCENE_PATH,
    T2ObstacleTask,
    file_adjacent_squares,
    make_t2_env,
)

from robochess.tasks.t3_capture import T3_SCENE_PATH, T3CaptureTask, make_t3_env

__all__ = [
    "T3_SCENE_PATH",
    "T3CaptureTask",
    "make_t3_env",
    "REACHABLE_SQUARES",
    "PickupObservation",
    "PickupReward",
    "PlaceObservation",
    "PlaceReward",
    "ShapedPickAndPlaceReward",
    "SparsePickAndPlaceReward",
    "T1PickAndPlaceTask",
    "N_OBSTACLE_SLOTS",
    "OBSTACLE_BODIES",
    "OBSTACLE_DISPLACEMENT_TOL",
    "PARKED_XY",
    "T2_SCENE_PATH",
    "T2ObstacleTask",
    "file_adjacent_squares",
    "make_t2_env",
]
