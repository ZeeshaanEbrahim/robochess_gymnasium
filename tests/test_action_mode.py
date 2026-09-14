"""RoboChessEnv action modes: absolute (default) and per-step joint-target delta.

The absolute path is what the 16-cell pilot matrix trained on, so its tests pin
it to the exact pre-delta formula rather than to "still works": a change in
float arithmetic there would silently break byte-for-byte reproducibility of
every pilot run. (Byte-for-byte equivalence of full T1/T2 trajectories against
the pre-change code was verified out of suite; see
results/delta_calibration/equiv/trace.py. It cannot live here because the old
code no longer exists to compare against.)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from robochess.envs import RoboChessEnv
from robochess.envs.robochess_env import DEFAULT_MAX_JOINT_DELTA, N_ARM_ACTUATORS
from robochess.experiment import ConfigError, resolve_config

REPO = Path(__file__).resolve().parents[1]


def _random_actions(n, seed=1234):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1.3, 1.3, 7).astype(np.float32) for _ in range(n)]  # includes out-of-range values


@pytest.fixture
def delta_env():
    e = RoboChessEnv(action_mode="delta")
    yield e
    e.close()


# ------------------------------------------------------------ construction
def test_default_action_mode_is_absolute():
    e = RoboChessEnv()
    assert e.action_mode == "absolute"
    e.close()


def test_invalid_action_mode_is_rejected():
    with pytest.raises(ValueError, match="action_mode"):
        RoboChessEnv(action_mode="velocity")


def test_delta_mode_requires_normalized_actions():
    with pytest.raises(ValueError, match="normalize_action"):
        RoboChessEnv(action_mode="delta", normalize_action=False)


def test_max_joint_delta_must_be_positive():
    with pytest.raises(ValueError, match="max_joint_delta"):
        RoboChessEnv(action_mode="delta", max_joint_delta=0.0)


def test_action_space_is_unchanged_by_mode(delta_env):
    absolute = RoboChessEnv()
    assert delta_env.action_space == absolute.action_space
    absolute.close()


# ------------------------------------------------- flag-off equivalence
def test_absolute_mode_matches_the_pre_delta_formula_exactly():
    """_scale_action in absolute mode must stay bit-for-bit the original mapping."""
    e = RoboChessEnv()
    e.reset(seed=0)
    lo, hi = e._ctrl_low, e._ctrl_high
    for a in _random_actions(200):
        clipped = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        expected = lo + (clipped + 1.0) * 0.5 * (hi - lo)
        assert e._scale_action(a).tobytes() == expected.tobytes()
    e.close()


def test_default_and_explicit_absolute_produce_identical_trajectories():
    envs = [RoboChessEnv(), RoboChessEnv(action_mode="absolute")]
    traces = []
    for e in envs:
        e.reset(seed=3)
        rows = []
        for a in _random_actions(150):
            obs, r, term, trunc, _ = e.step(a)
            rows.append(np.concatenate([obs["observation"], [r, term, trunc], e.data.ctrl]))
            if term or trunc:
                e.reset(seed=3)
        traces.append(np.array(rows))
        e.close()
    assert traces[0].tobytes() == traces[1].tobytes()


def test_absolute_mode_never_touches_the_delta_target():
    e = RoboChessEnv()
    e.reset(seed=0)
    before = e._arm_target.copy()
    for a in _random_actions(20):
        e.step(a)
    np.testing.assert_array_equal(e._arm_target, before)
    e.close()


# ---------------------------------------------------------- delta clipping
def test_delta_starts_from_the_home_target(delta_env):
    delta_env.reset(seed=0)
    home = delta_env.model.key_ctrl[delta_env._home_key][:N_ARM_ACTUATORS]
    np.testing.assert_array_equal(delta_env._arm_target, home)


def test_full_action_moves_the_target_by_exactly_the_cap(delta_env):
    delta_env.reset(seed=0)
    start = delta_env._arm_target.copy()
    delta_env.step(np.array([1, -1, 1, -1, 1, -1, -1], dtype=np.float32))
    expected = start + np.array([1, -1, 1, -1, 1, -1]) * DEFAULT_MAX_JOINT_DELTA
    np.testing.assert_allclose(delta_env.data.ctrl[:N_ARM_ACTUATORS], expected, atol=1e-12)


def test_out_of_range_action_is_clipped_to_the_cap(delta_env):
    delta_env.reset(seed=0)
    start = delta_env._arm_target.copy()
    delta_env.step(np.array([5.0, -5.0, 3.0, -3.0, 2.0, -2.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(np.abs(delta_env.data.ctrl[:N_ARM_ACTUATORS] - start),
                               DEFAULT_MAX_JOINT_DELTA, atol=1e-12)


def test_partial_action_scales_linearly(delta_env):
    delta_env.reset(seed=0)
    start = delta_env._arm_target.copy()
    delta_env.step(np.array([0.25, -0.5, 0.0, 0.75, -0.1, 0.3, -1.0], dtype=np.float32))
    expected = start + np.array([0.25, -0.5, 0.0, 0.75, -0.1, 0.3], dtype=np.float32) * DEFAULT_MAX_JOINT_DELTA
    np.testing.assert_allclose(delta_env.data.ctrl[:N_ARM_ACTUATORS], expected, atol=1e-7)


def test_custom_cap_is_respected():
    e = RoboChessEnv(action_mode="delta", max_joint_delta=0.02)
    e.reset(seed=0)
    start = e._arm_target.copy()
    e.step(np.ones(7, dtype=np.float32))
    np.testing.assert_allclose(e.data.ctrl[:N_ARM_ACTUATORS] - start, 0.02, atol=1e-12)
    e.close()


def test_random_policy_never_exceeds_the_cap_or_the_joint_range(delta_env):
    delta_env.reset(seed=0)
    prev = delta_env.data.ctrl[:N_ARM_ACTUATORS].copy()
    for a in _random_actions(400, seed=7):
        _, _, term, trunc, _ = delta_env.step(a)
        target = delta_env.data.ctrl[:N_ARM_ACTUATORS].copy()
        assert np.all(np.abs(target - prev) <= DEFAULT_MAX_JOINT_DELTA + 1e-12)
        assert np.all(target >= delta_env._arm_target_low - 1e-12)
        assert np.all(target <= delta_env._arm_target_high + 1e-12)
        prev = target
        if term or trunc:
            delta_env.reset(seed=0)
            prev = delta_env.data.ctrl[:N_ARM_ACTUATORS].copy()


# ---------------------------------------------------------- delta clamping
def test_target_is_clamped_at_the_upper_joint_bound(delta_env):
    delta_env.reset(seed=0)
    high = delta_env._arm_target_high
    delta_env._arm_target = high - 0.05          # one step short of the bound
    delta_env.step(np.array([1] * 6 + [-1], dtype=np.float32))
    np.testing.assert_array_equal(delta_env.data.ctrl[:N_ARM_ACTUATORS], high)
    delta_env.step(np.array([1] * 6 + [-1], dtype=np.float32))
    np.testing.assert_array_equal(delta_env.data.ctrl[:N_ARM_ACTUATORS], high)


def test_target_is_clamped_at_the_lower_joint_bound(delta_env):
    delta_env.reset(seed=0)
    low = delta_env._arm_target_low
    delta_env._arm_target = low + 0.05
    delta_env.step(np.array([-1] * 6 + [-1], dtype=np.float32))
    np.testing.assert_array_equal(delta_env.data.ctrl[:N_ARM_ACTUATORS], low)


def test_clamp_bounds_are_the_joint_ranges_within_ctrlrange(delta_env):
    m = delta_env.model
    joints = m.actuator_trnid[:N_ARM_ACTUATORS, 0]
    np.testing.assert_array_equal(delta_env._arm_target_low,
                                  np.maximum(m.jnt_range[joints, 0], m.actuator_ctrlrange[:N_ARM_ACTUATORS, 0]))
    np.testing.assert_array_equal(delta_env._arm_target_high,
                                  np.minimum(m.jnt_range[joints, 1], m.actuator_ctrlrange[:N_ARM_ACTUATORS, 1]))
    # elbow is the one joint limited to +-pi
    assert delta_env._arm_target_high[2] == pytest.approx(np.pi, abs=1e-3)


def test_reset_discards_the_previous_episodes_target(delta_env):
    delta_env.reset(seed=0)
    for _ in range(10):
        delta_env.step(np.array([1] * 6 + [-1], dtype=np.float32))
    delta_env.reset(seed=1)
    home = delta_env.model.key_ctrl[delta_env._home_key][:N_ARM_ACTUATORS]
    np.testing.assert_array_equal(delta_env._arm_target, home)


# ------------------------------------------------------ gripper and no-op
def test_gripper_stays_absolute_in_delta_mode(delta_env):
    delta_env.reset(seed=0)
    for g, expected in ((-1.0, 0.0), (1.0, 255.0), (0.0, 127.5)):
        delta_env.step(np.array([0] * 6 + [g], dtype=np.float32))
        assert delta_env.data.ctrl[6] == pytest.approx(expected)


def test_home_action_holds_the_target_in_delta_mode(delta_env):
    delta_env.reset(seed=0)
    for _ in range(3):
        delta_env.step(np.array([0.7] * 6 + [-1], dtype=np.float32))
    held = delta_env._arm_target.copy()
    delta_env.step(delta_env.home_action())
    np.testing.assert_array_equal(delta_env.data.ctrl[:N_ARM_ACTUATORS], held)
    assert delta_env.data.ctrl[6] == pytest.approx(0.0)   # home gripper is open


# --------------------------------------------------------- config wiring
_BASE_CFG = dict(task="t1", algorithm="ppo", reward_type="shaped", observation="state",
                 seed=0, total_timesteps=1000)


def test_config_defaults_to_absolute():
    assert resolve_config(dict(_BASE_CFG))["action_mode"] == "absolute"


def test_config_rejects_unknown_action_mode():
    with pytest.raises(ConfigError, match="action_mode"):
        resolve_config(dict(_BASE_CFG, action_mode="torque"))


def _load_train():
    spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    return train


@pytest.mark.parametrize("task", ["t1", "t2"])
def test_train_build_env_passes_action_mode_through(task):
    train = _load_train()
    for mode in ("absolute", "delta"):
        cfg = resolve_config(dict(_BASE_CFG, task=task, action_mode=mode,
                                  n_obstacles=2 if task == "t2" else 0))
        env = train.build_env(cfg)
        assert env.unwrapped.action_mode == mode
        env.close()


def test_train_build_env_treats_a_pre_action_mode_manifest_as_absolute():
    """Manifests written before action_mode existed have no such key."""
    train = _load_train()
    cfg = resolve_config(dict(_BASE_CFG))
    del cfg["action_mode"]
    env = train.build_env(cfg)
    assert env.unwrapped.action_mode == "absolute"
    env.close()
