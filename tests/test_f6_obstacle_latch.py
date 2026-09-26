"""F6 (Codex, 21 Sep): T2 forgets obstacle disturbances.

`_obstacle_status()` is recomputed from the live state on every call, with no
episode memory, so an obstacle moved 10 mm and put back reads "undisturbed"
again, and the episode can finish as a success. Measured on a real scripted
`b2->e5` episode with the obstacle displaced 10 mm over steps 11-13 and
returned: `obstacles_undisturbed` False at 11-13, True at 14, and the final
step `is_success=True`, `terminated_reason="success"`.

Fix under test: a per-episode latch. The instantaneous keys keep their meaning;
new keys `obstacle_peak_displacements_mm`, `max_obstacle_peak_displacement_mm`
and `obstacles_ever_disturbed` carry the memory, and `is_success` (with the
`success_obstacles_disturbed` relabel) is gated on `not obstacles_ever_disturbed`.

The "return" in these tests is a teleport back to the reset position, the same
technique test_t2_obstacle_pick_and_place.py uses to disturb an obstacle.
"""

from __future__ import annotations

import mujoco
import numpy as np
from _d27_helpers import prime_carried_release
import pytest

from robochess.tasks import (
    OBSTACLE_DISPLACEMENT_TOL,
    SparsePickAndPlaceReward,
    T2ObstacleTask,
    make_t2_env,
)


@pytest.fixture
def task():
    env = make_t2_env()
    t = T2ObstacleTask(env, start_square="a1", target_square="h8", n_obstacles=2,
                       obstacle_squares=("d4", "e5"))
    t.reset(seed=0)
    yield t
    env.close()


def _obstacle_pos(t, slot=0):
    return t._base.data.qpos[t._obstacle_qadrs[slot] : t._obstacle_qadrs[slot] + 7]


def _displace(t, dx, slot=0):
    _obstacle_pos(t, slot)[0] += dx
    mujoco.mj_forward(t._base.model, t._base.data)


def _restore(t, slot=0):
    p = _obstacle_pos(t, slot)
    p[0:2] = t._placed_xy[slot]
    p[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(t._base.model, t._base.data)


def _step(t):
    return t.step(t._base.home_action())[4]


def test_displaced_and_returned_obstacle_stays_flagged(task):
    """10 mm is twice the 5 mm tolerance. After it is put back, the instantaneous
    flag reads clean again (unchanged), but the latch must remember."""
    _displace(task, 0.010)
    assert _step(task)["obstacles_undisturbed"] is False
    _restore(task)
    info = _step(task)
    assert info["obstacles_undisturbed"] is True  # instantaneous key: unchanged meaning
    assert info["max_obstacle_displacement_mm"] < OBSTACLE_DISPLACEMENT_TOL * 1000
    assert info["obstacles_ever_disturbed"] is True


def test_peak_displacement_is_recorded_and_survives_the_return(task):
    _displace(task, 0.010)
    _step(task)
    _restore(task)
    info = _step(task)
    assert info["obstacle_peak_displacements_mm"][0] == pytest.approx(10.0, abs=0.5)
    assert info["obstacle_peak_displacements_mm"][1] < 1.0
    assert len(info["obstacle_peak_displacements_mm"]) == 2  # placed slots only
    assert info["max_obstacle_peak_displacement_mm"] == pytest.approx(10.0, abs=0.5)
    # the instantaneous displacement has returned to ~0
    assert info["max_obstacle_displacement_mm"] < 1.0


def test_tilted_and_righted_obstacle_stays_flagged(task):
    """The latch covers the tilt criterion, not just XY displacement."""
    half = np.pi / 4  # 90 degrees about x
    _obstacle_pos(task)[3:7] = [np.cos(half), np.sin(half), 0.0, 0.0]
    mujoco.mj_forward(task._base.model, task._base.data)
    assert _step(task)["obstacles_undisturbed"] is False
    _restore(task)
    info = _step(task)
    assert info["obstacles_undisturbed"] is True
    assert info["obstacles_ever_disturbed"] is True


def test_off_board_and_returned_obstacle_stays_flagged(task):
    _obstacle_pos(task)[0:2] = (1.5, 1.5)
    mujoco.mj_forward(task._base.model, task._base.data)
    assert _step(task)["obstacles_undisturbed"] is False
    _restore(task)
    info = _step(task)
    assert info["obstacles_undisturbed"] is True
    assert info["obstacles_ever_disturbed"] is True


def test_disturbance_within_tolerance_does_not_latch(task):
    _displace(task, 0.5 * OBSTACLE_DISPLACEMENT_TOL)
    info = _step(task)
    assert info["obstacles_ever_disturbed"] is False
    assert 2.0 < info["max_obstacle_peak_displacement_mm"] < 3.0


def test_latch_is_cleared_by_reset(task):
    _displace(task, 0.010)
    assert _step(task)["obstacles_ever_disturbed"] is True
    _, info = task.reset(seed=0)
    assert info["obstacles_ever_disturbed"] is False
    assert info["max_obstacle_peak_displacement_mm"] == 0.0
    assert _step(task)["obstacles_ever_disturbed"] is False


@pytest.mark.parametrize("n", [0, 1])
def test_latch_keys_exist_with_fewer_obstacles(n):
    env = make_t2_env()
    try:
        t = T2ObstacleTask(env, start_square="a1", target_square="h8", n_obstacles=n)
        _, info = t.reset(seed=0)
        for key in ("obstacles_ever_disturbed", "obstacle_peak_displacements_mm",
                    "max_obstacle_peak_displacement_mm"):
            assert key in info
        assert info["obstacles_ever_disturbed"] is False
        assert len(info["obstacle_peak_displacements_mm"]) == n
        assert info["max_obstacle_peak_displacement_mm"] == 0.0
    finally:
        env.close()


def _park_piece_on_goal(t):
    b = t._base
    q = b._piece_qadr
    b.data.qpos[q : q + 2] = b._desired_goal[:2]
    b.data.qpos[q + 2] = b._board_surface_z + 0.0005
    b.data.qpos[q + 3 : q + 7] = [1.0, 0.0, 0.0, 0.0]
    b.data.qvel[b._piece_vadr : b._piece_vadr + 6] = 0.0
    mujoco.mj_forward(b.model, b.data)


def _episode_with_optional_disturbance(disturb: bool):
    """Piece placed on its goal so the base criterion is met at the 3rd step.
    Disturbance (if any) is at step 1 and undone before step 2, so at the
    success step the obstacle is back in place."""
    env = make_t2_env()
    t = T2ObstacleTask(env, start_square="a1", target_square="h8", n_obstacles=1,
                       obstacle_squares=("d4",))
    s = SparsePickAndPlaceReward(t)
    s.reset(seed=0)
    _park_piece_on_goal(t)
    prime_carried_release(t)  # D27
    if disturb:
        _displace(t, 0.010)
    out = [s.step(t._base.home_action())]
    if disturb:
        _restore(t)
    for _ in range(4):
        out.append(s.step(t._base.home_action()))
        if out[-1][2]:
            break
    env.close()
    return out


def test_control_undisturbed_episode_is_still_a_success():
    out = _episode_with_optional_disturbance(disturb=False)
    _, reward, terminated, _, info = out[-1]
    assert terminated is True
    assert info["is_success"] is True and reward == 1.0
    assert info["terminated_reason"] == "success"
    assert info["obstacles_ever_disturbed"] is False


def test_success_is_denied_after_a_disturbance_that_was_undone():
    """The F6 bug end to end: obstacle moved 10 mm then returned before the
    success step. The old instantaneous gate scored this a success."""
    out = _episode_with_optional_disturbance(disturb=True)
    _, reward, terminated, _, info = out[-1]
    assert terminated is True  # base env still ends on its own criterion
    assert info["obstacles_undisturbed"] is True  # instantaneous: back in place
    # behavioural assertions first: on the unfixed code these are the failure
    assert info["is_success"] is False, "success granted after a 10 mm disturbance"
    assert reward == 0.0
    assert info["terminated_reason"] == "success_obstacles_disturbed"
    assert info["obstacles_ever_disturbed"] is True
