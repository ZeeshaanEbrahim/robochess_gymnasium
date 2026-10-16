"""D32: the D29 stack (place_mode, pickup_obs, place_obs) on T2, and the fixed obstacle squares.

What this pins:
  * config: place_mode / place_obs / pickup_obs resolve on t2; pickup_mode and train_pairs stay T1 only;
    `obstacle_squares` is validated (count, distinct, not on start/target/excluded, t2 only, needs a
    fixed start and target) and defaults to None, so every older config resolves as before.
  * layout: 21 base, 8 obstacle (21:29), 15 pick-up (29:44), 3 place (44:47) = 47, built by the same
    train.build_env that training and evaluation both call; flag-off T2 stays 29, the T1 D config 39.
  * PlaceReward over T2 reads the T2-overridden is_success: the 1020 lump is paid with the obstacles
    untouched and withheld (reason "success_obstacles_disturbed") when one was displaced.
  * zero-action full-stack smoke, the monitor keyword for the disturbance latch, and frozen traces of
    the paths the change must not alter (T1 D config, flag-off T2). Re-derive and re-freeze the traces
    only if a change is MEANT to alter those paths.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robochess.experiment import ConfigError, load_config, resolve_config
from robochess.tasks import (
    PickupObservation,
    ShapedPickAndPlaceReward,
    T2ObstacleTask,
    make_t2_env,
)
from robochess.tasks.t1_pick_and_place import pickup_features
from robochess.tasks.t2_obstacle_pick_and_place import PARKED_XY

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tests"))
from _d27_helpers import prime_carried_release  # noqa: E402

_spec = importlib.util.spec_from_file_location("train", _ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

BASE = dict(task="t1", algorithm="ppo", reward_type="shaped", observation="state", seed=0,
            total_timesteps=1000, max_episode_steps=300, action_mode="cartesian",
            start_square="e4", target_square="e5")
T2_D29 = dict(BASE, task="t2", n_obstacles=2, obstacle_squares=["d4", "f4"], pickup_obs=True,
              place_obs=True, place_mode=True, place_posture_gate=True, place_carry_potential=True,
              ppo_mean_squash=True, ppo_ent_coef=0.01, ppo_lr_schedule="linear", n_envs=4,
              vecnormalize_clip_reward=1000)
T1_D29 = {k: v for k, v in T2_D29.items() if k not in ("task", "n_obstacles", "obstacle_squares")}
T1_D29["task"] = "t1"


def cfg(**over):
    return resolve_config(dict(T2_D29, **over), config_path="t.yaml")


def stack(env):
    out, n = [], env
    while n is not None:
        out.append(type(n).__name__)
        n = getattr(n, "env", None)
    return out


def find_task(env):
    n = env
    while n is not None:
        if isinstance(n, T2ObstacleTask):
            return n
        n = getattr(n, "env", None)
    raise AssertionError("no T2ObstacleTask in the stack")


# ------------------------------------------------------------------ config
def test_the_d29_stack_resolves_on_t2():
    c = cfg()
    assert c["task"] == "t2" and c["place_mode"] and c["place_obs"] and c["pickup_obs"]
    assert c["obstacle_squares"] == ["d4", "f4"] and c["n_obstacles"] == 2


def test_pickup_mode_and_train_pairs_stay_t1_only():
    with pytest.raises(ConfigError, match="T1 only"):
        resolve_config(dict(BASE, task="t2", n_obstacles=1, pickup_mode=True), config_path="t.yaml")
    with pytest.raises(ConfigError, match="T1 only"):
        resolve_config(dict(BASE, task="t2", n_obstacles=1, start_square=None, target_square=None,
                            train_pairs=["e4:e5"]), config_path="t.yaml")


def test_place_obs_still_needs_pickup_obs_on_t2():
    with pytest.raises(ConfigError, match="needs pickup_obs"):
        cfg(pickup_obs=False)


@pytest.mark.parametrize("over, match", [
    (dict(obstacle_squares=["d4"]), "has 1 entries but n_obstacles is 2"),
    (dict(obstacle_squares=["d4", "d4"]), "distinct"),
    (dict(obstacle_squares=["d4", "z9"]), "Not a valid square"),
    (dict(obstacle_squares=["e4", "f4"]), "start square"),
    (dict(obstacle_squares=["d4", "e5"]), "target square"),
    (dict(obstacle_squares="d4f4"), "sequence"),
    (dict(excluded_squares=["d1", "e1"], obstacle_squares=["d1", "f4"]), "excluded"),
    (dict(start_square=None, target_square=None), "needs start_square and target_square"),
    (dict(start_square=None), "needs start_square and target_square"),
])
def test_obstacle_squares_are_validated(over, match):
    with pytest.raises(ConfigError, match=match):
        cfg(**over)


def test_obstacle_squares_is_rejected_on_t1_and_defaults_to_none():
    with pytest.raises(ConfigError, match="obstacles only exist on t2"):
        resolve_config(dict(T1_D29, obstacle_squares=["d4", "f4"]), config_path="t.yaml")
    assert resolve_config(dict(BASE, task="t2", n_obstacles=2), config_path="t.yaml")["obstacle_squares"] is None
    assert resolve_config(dict(BASE), config_path="t.yaml")["obstacle_squares"] is None


# ------------------------------------------------------------------ layout
def test_layout_47_obstacles_at_21_29_pickup_29_44_place_44_47():
    env = train.build_env(cfg())
    try:
        assert stack(env) == ["PlaceReward", "PlaceObservation", "PickupObservation",
                              "T2ObstacleTask", "RoboChessEnv"]
        obs, info = env.reset(seed=0)
        o = obs["observation"]
        assert o.shape == (47,) == env.observation_space["observation"].shape
        t2 = find_task(env)
        assert t2.current_obstacle_squares == ("d4", "f4")
        expected = np.concatenate([
            t2._board_world_xy("d4"), t2._board_world_xy("f4"), np.asarray(PARKED_XY[2]), np.asarray(PARKED_XY[3])])
        np.testing.assert_allclose(o[21:29], expected, atol=2e-3)       # settled obstacles, parked slots exact
        np.testing.assert_allclose(o[21:29], t2._obstacle_xy().ravel(), atol=1e-5)
        pickup = find_pickup(env)
        np.testing.assert_allclose(o[29:44], pickup_features(pickup.geometry()), atol=1e-5)
        np.testing.assert_allclose(o[44:47], obs["desired_goal"] - obs["achieved_goal"], atol=1e-5)
        assert info["obstacle_squares"] == ("d4", "f4")
    finally:
        env.close()


def find_pickup(env):
    n = env
    while n is not None:
        if isinstance(n, PickupObservation):
            return n
        n = getattr(n, "env", None)
    raise AssertionError("no PickupObservation in the stack")


def test_flag_off_t2_is_still_29_and_the_t1_d_config_is_39():
    old = resolve_config(dict(BASE, task="t2", n_obstacles=2), config_path="t.yaml")
    env = train.build_env(old)
    try:
        assert stack(env) == ["ShapedPickAndPlaceReward", "T2ObstacleTask", "RoboChessEnv"]
        assert env.reset(seed=0)[0]["observation"].shape == (29,)
    finally:
        env.close()
    env = train.build_env(resolve_config(T1_D29, config_path="t.yaml"))
    try:
        assert env.reset(seed=0)[0]["observation"].shape == (39,)
    finally:
        env.close()


def test_train_and_eval_construct_the_same_layout():
    """scripts/eval_probe.py and eval_final.py both call train.build_env(cfg); two builds agree."""
    a, b = train.build_env(cfg()), train.build_env(cfg())
    try:
        assert a.observation_space["observation"].shape == b.observation_space["observation"].shape == (47,)
        np.testing.assert_array_equal(a.observation_space["observation"].low, b.observation_space["observation"].low)
        np.testing.assert_array_equal(a.observation_space["observation"].high, b.observation_space["observation"].high)
        oa, ob = a.reset(seed=7)[0]["observation"], b.reset(seed=7)[0]["observation"]
        np.testing.assert_array_equal(oa, ob)
        # fixed squares: the same on every reset, whatever the seed
        for seed in (1, 2, 3):
            a.reset(seed=seed)
            assert find_task(a).current_obstacle_squares == ("d4", "f4")
    finally:
        a.close()
        b.close()


# ------------------------------------------------------------------ reward
def _place_on_target(env, displace: bool):
    """Section 6.4's method: prime a carry-and-release, teleport the piece onto the target, step."""
    t2 = find_task(env)
    base = t2.unwrapped
    env.reset(seed=0)
    prime_carried_release(env, xy_error=0.0, height=0.0)
    t2._teleport_piece("e5")
    if displace:
        qa = t2._obstacle_qadrs[0]
        # a direct qpos write: _set_obstacle_pose would re-baseline the obstacle and hide the move
        base.data.qpos[qa:qa + 2] = t2._placed_xy[0] + np.array([0.02, 0.0])
    mujoco.mj_forward(base.model, base.data)
    base._refresh_contact_flags()
    for _ in range(40):
        _, r, term, trunc, info = env.step(base.home_action())
        if term or trunc:
            return r, info
    raise AssertionError("the episode did not end")


def test_place_reward_over_t2_pays_the_lump_untouched_and_withholds_it_when_displaced():
    env = train.build_env(cfg())
    try:
        r_ok, info_ok = _place_on_target(env, displace=False)
        assert info_ok["terminated_reason"] == "success" and info_ok["is_success"] is True
        assert info_ok["place_terms"]["success_lump"] == pytest.approx(1020.0)
        assert info_ok["obstacles_ever_disturbed"] is False

        r_bad, info_bad = _place_on_target(env, displace=True)
        assert info_bad["obstacles_ever_disturbed"] is True
        assert info_bad["terminated_reason"] == "success_obstacles_disturbed"
        assert info_bad["is_success"] is False                    # the T2 override, read by PlaceReward
        assert info_bad["place_terms"]["success_lump"] == 0.0
        assert info_bad["place_terms"]["failure_penalty"] == 0.0   # no penalty, only the lump is withheld
        assert r_ok - r_bad == pytest.approx(1020.0)
    finally:
        env.close()


# ------------------------------------------------------------------ smoke, monitor
def test_zero_action_full_stack_smoke_is_finite_with_no_invalid_transition():
    env = train.build_env(cfg())
    try:
        env.reset(seed=0)
        for _ in range(5):
            obs, r, term, trunc, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            assert np.isfinite(r) and np.all(np.isfinite(obs["observation"]))
            assert not info.get("invalid_transition")
            assert info["place_state"] in ("S0", "S1", "S2", "S3")
    finally:
        env.close()


def test_monitor_keywords_carry_the_disturbance_latch_on_t2_only():
    assert "obstacles_ever_disturbed" in train.monitor_info_keywords(cfg())
    assert "obstacles_ever_disturbed" in train.monitor_info_keywords(
        resolve_config(dict(BASE, task="t2", n_obstacles=2), config_path="t.yaml"))
    assert "obstacles_ever_disturbed" not in train.monitor_info_keywords(resolve_config(T1_D29, config_path="t.yaml"))
    assert train.monitor_info_keywords(resolve_config(T1_D29, config_path="t.yaml"))[-3:] == (
        "place_state", "start_square", "target_square")


def test_monitor_reaches_the_end_of_a_t2_episode_with_the_latch_key(tmp_path):
    c = cfg(n_envs=1, max_episode_steps=20)
    venv = train.build_vec_env(c, monitor_dir=tmp_path)
    try:
        venv.reset()
        done = False
        for _ in range(25):
            _, _, d, infos = venv.step(np.zeros((1,) + venv.action_space.shape, dtype=np.float32))
            done = done or bool(d[0])
        assert done
    finally:
        venv.close()
    header = next(l for l in (tmp_path / "monitor.monitor.csv").read_text().splitlines() if not l.startswith("#"))
    assert "obstacles_ever_disturbed" in header.split(",")


# ------------------------------------------------------------------ frozen traces (must not move)
def _trace(env, n: int, kind: str) -> str:
    h = hashlib.sha256()
    rng = np.random.default_rng(123)
    obs, _ = env.reset(seed=0)
    for k in sorted(obs):
        h.update(np.ascontiguousarray(np.asarray(obs[k], dtype=np.float64)).tobytes())
    for _ in range(n):
        a = (rng.uniform(-1, 1, env.action_space.shape) if kind == "random"
             else np.zeros(env.action_space.shape)).astype(np.float32)
        obs, r, te, tr, inf = env.step(a)
        for k in sorted(obs):
            h.update(np.ascontiguousarray(np.asarray(obs[k], dtype=np.float64)).tobytes())
        h.update(np.float64(r).tobytes())
        h.update(bytes([int(te), int(tr)]))
        h.update(str(inf.get("terminated_reason")).encode())
        if te or tr:
            env.reset()
    return h.hexdigest()


# Frozen against 53ccc06 (the commit this branch started from), before any D32 code change.
_FROZEN_TRACES = {
    "t1_dconfig_random120": "735b1d64f3fb6bb931d797d0a2892aac924f6cba9c26f82ecf7d3cbd1aecb67a",
    "t1_dconfig_zero120": "b90e5e96195986db65988e4040b901eba7149df1cc460d8fe14ca902079d6d63",
    "t2_old_config_random120": "3c75b80aec20e553b759591a6d62b8f152ff1ebb2d1420ece62708358e4c83a9",
    "t2_fixed_d4f4_cartesian_legacy_random120": "e93280af93c5f7bb0e3ceef81a85101d733b0820b2676f9f3215baf341955444",
    "t2_fixed_d4f4_cartesian_legacy_zero120": "5cba171cd500baf0bf7798498f43c6ea693689bd3ec72af1e1fa8b292e721383",
}


def _from_config_file(name):
    path = _ROOT / "configs" / name
    return train.build_env(resolve_config(load_config(path), config_path=str(path)))


def test_frozen_t1_d_config_traces_are_unchanged():
    for kind, key in (("random", "t1_dconfig_random120"), ("zero", "t1_dconfig_zero120")):
        env = _from_config_file("d32_t1_e4e5_seed0.yaml")
        try:
            assert _trace(env, 120, kind) == _FROZEN_TRACES[key], key
        finally:
            env.close()


def test_frozen_flag_off_t2_traces_are_unchanged():
    env = _from_config_file("t2_ppo_shaped_state_seed0.yaml")
    try:
        assert _trace(env, 120, "random") == _FROZEN_TRACES["t2_old_config_random120"]
    finally:
        env.close()
    for kind in ("random", "zero"):
        base = make_t2_env(max_episode_steps=300, action_mode="cartesian")
        env = ShapedPickAndPlaceReward(T2ObstacleTask(
            base, n_obstacles=2, start_square="e4", target_square="e5", obstacle_squares=("d4", "f4")))
        try:
            assert _trace(env, 120, kind) == _FROZEN_TRACES[f"t2_fixed_d4f4_cartesian_legacy_{kind}120"], kind
        finally:
            env.close()
