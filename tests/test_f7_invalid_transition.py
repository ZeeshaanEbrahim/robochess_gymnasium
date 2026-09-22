"""F7 (referee review 2026-09-21): the simulator-failure fallback must return a
consistent transition, and a corrupted step must be identifiable downstream.

Reproduces both halves of the review's F7 case (h8->a1, forced first-step
instability):

  * the fallback observation carried the BASE env's default e4->e5 goal and
    piece instead of the task's h8->a1 reset state, because the D14 snapshot
    was seeded by RoboChessEnv.reset() before the task teleported the piece and
    set its goal;
  * a NaN piece velocity did not trip the guard (every comparison against NaN
    is False), and a NaN entering env.step() is silently swallowed by MuJoCo,
    which resets its own data and only increments a warning counter -- measured:
    sim time 0.12 -> 0.04, state back at the home pose, ordinary-looking step.

Like test_solver_instability_guard.py these inject the bad state directly
rather than waiting for physics to misbehave.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robochess.envs import RoboChessEnv
from robochess.tasks import (
    ShapedPickAndPlaceReward,
    SparsePickAndPlaceReward,
    T1PickAndPlaceTask,
)
from robochess.tasks.t1_pick_and_place import PickupObservation, PickupReward

TASK_STACKS = ["task", "shaped", "staged", "sparse", "pickup"]
ALL_STACKS = ["base"] + TASK_STACKS


def _build(kind):
    base = RoboChessEnv(action_mode="delta", max_episode_steps=200)
    if kind == "base":
        return base
    # h8 -> a1 is the review's case: the task goal (a1) differs from the base
    # default goal (e5) and the start square (h8) from the base spawn (e4).
    task = T1PickAndPlaceTask(base, start_square="h8", target_square="a1")
    if kind == "task":
        return task
    if kind == "shaped":
        return ShapedPickAndPlaceReward(task)
    if kind == "staged":
        return ShapedPickAndPlaceReward(task, staged=True)
    if kind == "sparse":
        return SparsePickAndPlaceReward(task)
    if kind == "pickup":
        return PickupReward(PickupObservation(task))
    raise AssertionError(kind)


@pytest.fixture(params=ALL_STACKS)
def stack(request):
    env = _build(request.param)
    yield env
    env.close()


@pytest.fixture(params=TASK_STACKS)
def task_stack(request):
    env = _build(request.param)
    yield env
    env.close()


def _inject_spike(base):
    """Implausible state that survives the macro-step (see the D14 tests)."""
    v = base.data.qvel
    v[base._piece_vadr : base._piece_vadr + 3] = (0.0, 0.0, 60.0)
    v[base._piece_vadr + 3 : base._piece_vadr + 6] = (0.0, 0.0, 4000.0)


def _hold(env):
    return env.unwrapped.home_action()


def _all_finite(obs):
    return all(bool(np.all(np.isfinite(v))) for v in obs.values())


def _same_obs(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


# ------------------------------------------------ part 1: wrong desired goal
def test_first_step_fallback_returns_the_tasks_own_goal_and_state(task_stack):
    obs0, _ = task_stack.reset(seed=0)
    base = task_stack.unwrapped
    requested_goal = base._desired_goal.copy()

    _inject_spike(base)
    obs, _, _, truncated, info = task_stack.step(_hold(task_stack))

    assert truncated is True and info["terminated_reason"] == "solver_instability"
    assert np.array_equal(obs["desired_goal"], requested_goal), (
        f"fallback goal {obs['desired_goal']} != the task's goal {requested_goal}"
    )
    # The whole fallback is the state reset() handed out -- goal, piece and (for
    # the pick-up stack) the appended features -- not a mix of instants.
    assert _same_obs(obs, obs0)


def test_reset_options_squares_are_snapshotted_too():
    """The snapshot follows a per-episode square override, not just the
    constructor's squares."""
    base = RoboChessEnv(action_mode="delta", max_episode_steps=200)
    env = T1PickAndPlaceTask(base)
    try:
        obs0, _ = env.reset(seed=3, options={"start_square": "b2", "target_square": "g7"})
        goal = base._desired_goal.copy()
        _inject_spike(base)
        obs, _, _, truncated, info = env.step(base.home_action())
        assert truncated is True and info["solver_instability"] is True
        assert np.array_equal(obs["desired_goal"], goal)
        assert np.array_equal(obs["achieved_goal"], obs0["achieved_goal"])
    finally:
        env.close()


# ------------------------------------------------ part 2: NaN is not detected
@pytest.mark.parametrize("where", ["piece_qvel", "arm_qvel", "piece_qpos", "gripper_qvel"])
def test_non_finite_state_trips_the_guard(where):
    env = RoboChessEnv(action_mode="delta")
    try:
        env.reset(seed=0)
        assert env._solver_instability() is False
        d = env.data
        if where == "piece_qvel":
            d.qvel[env._piece_vadr] = np.nan
        elif where == "arm_qvel":
            d.qvel[0] = np.inf
        elif where == "piece_qpos":
            d.qpos[env._piece_qadr] = np.nan
        else:
            d.qvel[env._grip_vadr] = np.nan
        assert env._solver_instability() is True
    finally:
        env.close()


def test_nan_entering_a_real_step_is_flagged_not_swallowed(stack):
    """MuJoCo resets its own data when a NaN reaches mj_step and only bumps a
    warning counter; the env used to return that as an ordinary transition."""
    obs0, _ = stack.reset(seed=0)
    base = stack.unwrapped
    stack.step(_hold(stack))  # one ordinary step so a snapshot exists past reset
    good, _, _, _, _ = stack.step(_hold(stack))
    good = {k: v.copy() for k, v in good.items()}

    base.data.qvel[base._piece_vadr] = np.nan
    obs, reward, terminated, truncated, info = stack.step(_hold(stack))

    assert info["solver_instability"] is True
    assert info["invalid_transition"] is True
    assert info["terminated_reason"] == "solver_instability"
    assert truncated is True and terminated is False
    assert _all_finite(obs) and np.isfinite(reward)
    assert _same_obs(obs, good), "fallback must be the last uncorrupted observation"


def test_nan_produced_inside_the_step_is_flagged(stack, monkeypatch):
    """A non-finite value created by the step itself (not present at its start)
    is only visible in the post-step state."""
    real_step = mujoco.mj_step

    def poisoned(model, data, *args, **kwargs):
        real_step(model, data, *args, **kwargs)
        data.qvel[stack.unwrapped._piece_vadr] = np.nan

    stack.reset(seed=0)
    good, _, _, _, _ = stack.step(_hold(stack))
    good = {k: v.copy() for k, v in good.items()}

    monkeypatch.setattr(mujoco, "mj_step", poisoned)
    obs, reward, terminated, truncated, info = stack.step(_hold(stack))

    assert info["solver_instability"] is True and info["invalid_transition"] is True
    assert truncated is True and terminated is False
    assert _all_finite(obs), "a non-finite value reached the returned observation"
    assert np.isfinite(reward), "a non-finite reward would poison VecNormalize"
    assert _same_obs(obs, good)


# --------------------------------------------- part 3: the downstream flag
def test_invalid_transition_flag_is_always_present_and_false_when_normal(stack):
    _, info = stack.reset(seed=0)
    assert info["invalid_transition"] is False
    for _ in range(3):
        _, _, _, _, info = stack.step(_hold(stack))
        assert info["invalid_transition"] is False
        assert info["solver_instability"] is False


def test_spike_sets_the_flag_through_every_wrapper(stack):
    stack.reset(seed=0)
    stack.step(_hold(stack))
    _inject_spike(stack.unwrapped)
    _, _, _, truncated, info = stack.step(_hold(stack))
    assert truncated is True
    assert info["invalid_transition"] is True
    assert info["solver_instability"] is True


def test_flag_does_not_leak_into_the_next_episode(stack):
    stack.reset(seed=0)
    stack.step(_hold(stack))
    _inject_spike(stack.unwrapped)
    stack.step(_hold(stack))

    _, info = stack.reset(seed=1)
    assert info["invalid_transition"] is False
    _, _, _, truncated, info = stack.step(_hold(stack))
    assert info["invalid_transition"] is False and truncated is False


def test_corrupted_step_never_reports_success(monkeypatch):
    """step() already excludes success when unstable; info['is_success'] (which
    the sparse reward reads) must agree instead of paying 1.0 on a bad step."""
    env = _build("sparse")
    try:
        env.reset(seed=0)
        base = env.unwrapped
        env.step(_hold(env))
        monkeypatch.setattr(base, "_success_terms", lambda: (True, True, True, True, 0.0))
        _inject_spike(base)
        _, reward, terminated, truncated, info = env.step(_hold(env))
        assert info["invalid_transition"] is True
        assert info["is_success"] is False
        assert reward == 0.0
        assert truncated is True and terminated is False
    finally:
        env.close()


def test_pickup_state_is_not_latched_from_a_corrupted_step(monkeypatch):
    """PickupReward must not advance its episode-latched state (first grasp,
    hold streak, pickup_success) from a state the guard has declared invalid."""
    env = _build("pickup")
    try:
        env.reset(seed=0)
        env.step(_hold(env))
        # A corrupted state that would otherwise read as a held, lifted piece:
        # both pads on the piece, grasp_success true, piece metres off the board.
        task = env.env.env  # PickupReward > PickupObservation > T1PickAndPlaceTask
        assert isinstance(task, T1PickAndPlaceTask)
        monkeypatch.setattr(env, "contact_state", lambda: (True, True))
        monkeypatch.setattr(task, "_grasp_success", lambda: True)
        _inject_spike(env.unwrapped)
        _, reward, _, truncated, info = env.step(_hold(env))
        assert truncated is True and info["invalid_transition"] is True
        assert np.isfinite(reward)
        assert env._held_once is False and env._hold_streak == 0
        assert info["pickup_success"] is False
    finally:
        env.close()
