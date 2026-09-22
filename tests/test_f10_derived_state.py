"""F10 (referee review 2026-09-21): derived simulator quantities must describe
the state env.step() returns, not the instant before its last substep.

mj_step integrates qpos/qvel and leaves xpos / site_xpos / geom_xpos and the
contact list computed from the state BEFORE the final integration, so after
env.step() every derived position lags the integrated state by one physics
substep of motion. The review measured up to 0.791 mm after five delta steps
by calling mj_forward and diffing; the round-1 probe reproduced 0.98 mm at the
pinch site. The referee reads these quantities against millimetre tolerances.

Method (round-1 probe_f10.py): five steps in delta mode from seed 0, snapshot
the derived positions, call mj_forward (which does not advance time), diff.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robochess.envs import RoboChessEnv

N_STEPS = 5
# mj_forward on an unchanged qpos is deterministic, so a refreshed env agrees
# with a second refresh to float precision. 1e-6 mm is "zero" with margin.
TOL_M = 1e-9


def _derived(env):
    d = env.data
    return {"xpos": d.xpos.copy(), "site_xpos": d.site_xpos.copy(), "geom_xpos": d.geom_xpos.copy()}


def _max_drift_mm(env):
    """Largest displacement (mm) any body/site/geom moves when mj_forward is
    called on the state env.step() just returned. Also asserts the refresh
    itself is inert: same qpos, same simulated time."""
    before = _derived(env)
    qpos, t = env.data.qpos.copy(), float(env.data.time)
    mujoco.mj_forward(env.model, env.data)
    assert np.array_equal(qpos, env.data.qpos)
    assert float(env.data.time) == t
    worst = {k: float(np.linalg.norm(getattr(env.data, k) - v, axis=1).max()) * 1000
             for k, v in before.items()}
    return worst


def _delta_action(env, arm):
    a = env.home_action().copy()
    a[:6] = arm
    return a


SEQUENCES = {
    # round-1 primary sequence: +-0.5 alternating signs, gripper held
    "const_0.5": lambda env: [_delta_action(env, 0.5 * np.array([1, -1, 1, -1, 1, -1.0]))] * N_STEPS,
    "const_1.0": lambda env: [_delta_action(env, 1.0 * np.array([1, -1, 1, -1, 1, -1.0]))] * N_STEPS,
    # a fresh vector each step from one generator, as the round-1 probe drew them
    "uniform_rng0": lambda env: (
        lambda rng: [_delta_action(env, rng.uniform(-1, 1, 6)) for _ in range(N_STEPS)]
    )(np.random.default_rng(0)),
    # holding still still drifts: the arm is settling under gravity compensation
    "hold": lambda env: [env.home_action()] * N_STEPS,
}


@pytest.mark.parametrize("name", list(SEQUENCES))
def test_derived_positions_match_the_integrated_state_after_delta_steps(name):
    env = RoboChessEnv(action_mode="delta")
    try:
        env.reset(seed=0)
        for a in SEQUENCES[name](env):
            env.step(a)
        worst = _max_drift_mm(env)
        assert max(worst.values()) < TOL_M * 1000, (
            f"derived positions lag the state by up to {max(worst.values()):.4f} mm "
            f"(xpos {worst['xpos']:.4f}, site_xpos {worst['site_xpos']:.4f}, "
            f"geom_xpos {worst['geom_xpos']:.4f})"
        )
    finally:
        env.close()


def test_pinch_site_specifically_is_current():
    """The site the review measured and the one the expert's IK and the pick-up
    reward read."""
    env = RoboChessEnv(action_mode="delta")
    try:
        env.reset(seed=0)
        for a in SEQUENCES["const_0.5"](env):
            env.step(a)
        sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "ur_gripper_pinch")
        before = env.data.site_xpos[sid].copy()
        mujoco.mj_forward(env.model, env.data)
        assert np.linalg.norm(env.data.site_xpos[sid] - before) < TOL_M
    finally:
        env.close()


def test_derived_positions_are_current_in_absolute_mode_too():
    env = RoboChessEnv()  # absolute joint targets, the scripted expert's mode
    try:
        env.reset(seed=0)
        a = env.home_action().copy()
        a[:6] = 0.3 * np.array([1, -1, 1, -1, 1, -1.0])
        for _ in range(N_STEPS):
            env.step(a)
        assert max(_max_drift_mm(env).values()) < TOL_M * 1000
    finally:
        env.close()


def test_a_falling_piece_reports_its_current_height():
    """The referee's own quantity: piece_height_above_board comes from the
    piece's xpos. A piece in free fall moves ~0.8 mm per substep, so a lagging
    xpos misreports its height by that much."""
    env = RoboChessEnv(action_mode="delta")
    try:
        env.reset(seed=0)
        env.data.qpos[env._piece_qadr + 2] += 0.05
        mujoco.mj_forward(env.model, env.data)
        _, _, _, _, info = env.step(env.home_action())

        reported = info["piece_height_above_board"]
        mujoco.mj_forward(env.model, env.data)
        current = float(env.data.xpos[env._piece_bid][2] - env._board_surface_z)
        # `info` reports the height through _achieved_goal(), which is float32
        # (~5e-8 m at this magnitude), so compare to 1 um: the lag being caught
        # is 785 um.
        assert abs(reported - current) < 1e-6, (
            f"reported height {reported * 1000:.4f} mm vs integrated state {current * 1000:.4f} mm"
        )
        assert env.data.qvel[env._piece_vadr + 2] < -0.05, "the piece should be falling"
    finally:
        env.close()


def test_refreshing_derived_state_does_not_advance_time():
    env = RoboChessEnv(action_mode="delta")
    try:
        env.reset(seed=0)
        for a in SEQUENCES["const_0.5"](env):
            env.step(a)
        expected = N_STEPS * env.frame_skip * env.model.opt.timestep
        assert float(env.data.time) == pytest.approx(expected, abs=1e-9)
    finally:
        env.close()
