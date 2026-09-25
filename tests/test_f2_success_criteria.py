"""F2 (referee review 2026-09-21): the placement success criterion was too loose.

`_success_terms()` accepted a piece that was 10 mm off the board (5 mm is the
approved tolerance), that was still translating (only ANGULAR speed was
checked), and that was tilted well off vertical (no orientation check at all).

Approved thresholds: height < 5 mm (`PLACEMENT_REST_TOLERANCE`, separate from
`rest_tolerance`, which stays 10 mm for T1's `lifted` term and D18's grasp
check); linear speed < 0.01 m/s and tilt < 10 degrees, each sustained for 3
consecutive steps like the existing angular-speed streak.

Fixture method. Each case is a real env at seed 0 with the tracked piece put at
its goal square by writing its freejoint qpos/qvel, then stepped with the home
action so the real referee code runs on real MuJoCo state. The piece's
weight is cancelled by an equal and opposite `xfrc_applied` force at its centre
of mass (m*g; `body_gravcomp` is inert here because the model compiles with no
gravcomp bodies), so a pose that cannot physically be held (floating 7.5 mm up,
tilted 15 degrees on its rim, sliding without friction losses) stays exactly as
posed. Nothing else about the model changes. Every
test first asserts the fixture's measured height / speeds / tilt, so the case
cannot be failing on an unrelated term.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robochess.envs import RoboChessEnv

N_STEPS = 8
BASE_RADIUS = 0.015  # piece_a_base cylinder radius (m)
CLEARANCE = 0.0002  # keep the lowest rim point off the board so no contact fires


@pytest.fixture
def env():
    e = RoboChessEnv()
    e.reset(seed=0)
    yield e
    e.close()


def _tilt_deg(env) -> float:
    zz = float(env.data.xmat[env._piece_bid][8])
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def _lin_speed(env) -> float:
    return float(np.linalg.norm(env.data.qvel[env._piece_vadr : env._piece_vadr + 3]))


def _ang_speed(env) -> float:
    return float(np.linalg.norm(env.data.qvel[env._piece_vadr + 3 : env._piece_vadr + 6]))


def _pose(env, height_m, tilt_deg=0.0, lin_vel=(0.0, 0.0, 0.0)):
    """Put the piece at its goal square, `height_m` above the board surface."""
    weight = float(env.model.body_mass[env._piece_bid]) * -float(env.model.opt.gravity[2])
    env.data.xfrc_applied[env._piece_bid, 2] = weight
    q, v = env._piece_qadr, env._piece_vadr
    goal = env._desired_goal
    env.data.qpos[q : q + 3] = [goal[0], goal[1], env._board_surface_z + height_m]
    half = np.radians(tilt_deg) / 2
    env.data.qpos[q + 3 : q + 7] = [np.cos(half), np.sin(half), 0.0, 0.0]
    env.data.qvel[v : v + 3] = lin_vel
    env.data.qvel[v + 3 : v + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def _run(env):
    """Step with the home action; return (any is_success, last info, per-step infos)."""
    infos = []
    for _ in range(N_STEPS):
        _, _, terminated, truncated, info = env.step(env.home_action())
        infos.append(info)
        if terminated or truncated:
            break
    return any(i["is_success"] for i in infos), infos[-1], infos


def test_control_upright_still_piece_on_the_board_succeeds(env):
    """Sanity: the fixture method itself does not defeat success."""
    _pose(env, height_m=0.002)
    success, info, _ = _run(env)
    assert abs(_tilt_deg(env)) < 0.5 and _lin_speed(env) < 1e-3 and _ang_speed(env) < 1e-3
    assert success, info


def test_piece_resting_7_5mm_above_board_is_not_success(env):
    """Between the old 10 mm and new 5 mm tolerance: must now fail `resting`."""
    _pose(env, height_m=0.0075)
    success, info, _ = _run(env)
    # isolate: on square, released, upright, at rest -- only height is off.
    assert info["xy_distance_to_goal"] < env.goal_tolerance
    assert info["gripper_released"]
    assert _tilt_deg(env) < 0.5 and _lin_speed(env) < 1e-3 and _ang_speed(env) < 1e-3
    assert 0.0074 < info["piece_height_above_board"] < 0.0076
    assert not success, f"success at {1000 * info['piece_height_above_board']:.2f} mm"


def test_piece_sliding_translationally_is_not_success(env):
    """0.05 m/s sideways, zero angular velocity: the settled term passes, so only
    a linear-speed check can catch it."""
    _pose(env, height_m=CLEARANCE, lin_vel=(0.0, 0.05, 0.0))
    success, info, infos = _run(env)
    assert info["piece_settled"], "angular streak must be satisfied to isolate the linear term"
    assert _ang_speed(env) < 1e-6
    assert _lin_speed(env) > 0.04  # 5x the 0.01 m/s threshold
    assert info["xy_distance_to_goal"] < env.goal_tolerance
    assert info["piece_resting"] and info["gripper_released"]
    assert _tilt_deg(env) < 0.5
    assert not success, f"success while sliding at {_lin_speed(env):.3f} m/s"


def test_piece_tilted_15_degrees_is_not_success(env):
    """Tilted 15 deg on its rim, at rest, on square: only a tilt check can catch it."""
    tilt = 15.0
    _pose(env, height_m=BASE_RADIUS * np.sin(np.radians(tilt)) + CLEARANCE, tilt_deg=tilt)
    success, info, _ = _run(env)
    assert 14.5 < _tilt_deg(env) < 15.5
    assert _lin_speed(env) < 1e-3 and _ang_speed(env) < 1e-3
    assert info["xy_distance_to_goal"] < env.goal_tolerance
    assert info["piece_resting"], "height must be inside the new 5 mm tolerance too"
    assert info["piece_height_above_board"] < 0.005
    assert info["gripper_released"] and info["piece_settled"]
    assert not success, f"success while tilted {_tilt_deg(env):.1f} deg"
