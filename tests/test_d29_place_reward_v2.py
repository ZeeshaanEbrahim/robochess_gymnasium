"""Tests for D29: posture gate, carry-potential shaping, PPO mean squash,
eval seeding, and the D28-flags-off identity guarantee.

Reuses test_place_reward.py's rig (Stub/make_rig/go/cfg/train) rather than
re-deriving it -- see that file's own docstring for why Stub exists.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from robochess.experiment import ConfigError, resolve_config
from robochess.policies import SquashedMultiInputActorCriticPolicy
from robochess.tasks import PlaceReward
from robochess.tasks.t1_pick_and_place import PLACE_GAMMA, place_carry_potential, place_posture_gate_value

from test_place_reward import PLACE_CFG, S1, _ROOT, cfg, go, make_rig, train  # noqa: E402

import d27_scenarios as sc  # noqa: E402
from scripted_expert import ScriptedExpert  # noqa: E402
import gymnasium as gym  # noqa: E402

GATE_LO, GATE_HI = 10.0, 45.0


# ===================================================================== gate
@pytest.mark.parametrize("tilt, expected", [
    (0.0, 1.0),
    (10.0, 1.0),
    (20.0, 0.5 * (1.0 + math.cos(math.pi * 10.0 / 35.0))),
    (27.5, 0.5),
    (45.0, 0.0),
    (90.0, 0.0),
])
def test_place_posture_gate_value_at_key_tilts(tilt, expected):
    assert place_posture_gate_value(tilt, GATE_LO, GATE_HI) == pytest.approx(expected, abs=1e-12)


def test_s1_equals_d28_when_gate_off_regardless_of_tilt():
    env, rw, stub = make_rig()  # posture_gate defaults False
    try:
        for tilt in (0.0, 10.0, 45.0, 90.0):
            _, info, t = go(rw, stub, xy=0.078, lift=0.0, tilt=tilt, **S1)
            assert t["posture_gate"] == pytest.approx(1.0)
            assert t["state_reward"] == pytest.approx(2.0 + 2 * 0.439637, abs=1e-4)
    finally:
        env.close()


@pytest.mark.parametrize("tilt", [0.0, 5.0, 10.0])
def test_gate_on_upright_s1_equals_d28_formula(tilt):
    env, rw, stub = make_rig(posture_gate=True, gate_tilt_lo_deg=GATE_LO, gate_tilt_hi_deg=GATE_HI)
    try:
        _, info, t = go(rw, stub, xy=0.078, lift=0.0, tilt=tilt, **S1)
        assert t["posture_gate"] == pytest.approx(1.0, abs=1e-12)
        assert t["state_reward"] == pytest.approx(2.0 + 2 * 0.439637, abs=1e-4)
    finally:
        env.close()


@pytest.mark.parametrize("tilt", [45.0, 60.0, 90.0])
def test_gate_on_toppled_s1_equals_position_plus_closing_exactly(tilt):
    env, rw, stub = make_rig(posture_gate=True, gate_tilt_lo_deg=GATE_LO, gate_tilt_hi_deg=GATE_HI)
    try:
        _, info, t = go(rw, stub, xy=0.078, lift=0.03, tilt=tilt, **S1)
        assert info["place_state"] == "S1"
        assert t["posture_gate"] == pytest.approx(0.0, abs=1e-12)
        assert t["state_reward"] == pytest.approx(t["position_term"] + t["closing_term"], abs=1e-12)
    finally:
        env.close()


def test_gate_reported_as_1_outside_s1():
    env, rw, stub = make_rig(posture_gate=True, gate_tilt_lo_deg=GATE_LO, gate_tilt_hi_deg=GATE_HI)
    try:
        _, info, t = go(rw, stub)  # S0
        assert info["place_state"] == "S0" and t["posture_gate"] == pytest.approx(1.0)
    finally:
        env.close()


# =================================================================== carry potential
@pytest.mark.parametrize("d_xy", [0.0, 0.06])
def test_shaping_stationary_piece_matches_closed_form(d_xy):
    """A piece sitting still at a constant d_xy has Phi(s')==Phi(s) each step, so
    F = gamma*Phi - Phi = Phi*(gamma-1) = (1-gamma)*k*d_xy: 0 at the target,
    positive (a small per-step leak) anywhere else."""
    k = 50.0
    env, rw, stub = make_rig(carry_potential=True, carry_potential_k=k)
    try:
        go(rw, stub, xy=d_xy, **S1)                  # settle Phi at d_xy (transient step)
        expected = (1.0 - PLACE_GAMMA) * k * d_xy
        for _ in range(5):
            _, info, t = go(rw, stub, xy=d_xy)
            assert t["shaping"] == pytest.approx(expected, abs=1e-9)
            if d_xy > 0:
                assert t["shaping"] > 0.0
            else:
                assert t["shaping"] == 0.0
    finally:
        env.close()


def test_shaping_positive_toward_target_negative_away():
    env, rw, stub = make_rig(carry_potential=True, carry_potential_k=50.0)
    try:
        go(rw, stub, xy=0.06, **S1)
        _, _, t_close = go(rw, stub, xy=0.03)
        assert t_close["shaping"] > 0.0
        _, _, t_far = go(rw, stub, xy=0.09)
        assert t_far["shaping"] < 0.0
    finally:
        env.close()


def test_invalid_step_no_phi_update_and_shaping_zero():
    env, rw, stub = make_rig(carry_potential=True, carry_potential_k=50.0)
    try:
        go(rw, stub, xy=0.06, **S1)
        phi_before = rw._phi
        _, info, t = go(rw, stub, invalid=True, xy=0.01)
        assert t["shaping"] == 0.0
        assert rw._phi == phi_before
        _, info2, t2 = go(rw, stub, invalid=False, xy=0.03)
        expected = PLACE_GAMMA * place_carry_potential(0.03, 50.0) - phi_before
        assert t2["shaping"] == pytest.approx(expected, abs=1e-9)
    finally:
        env.close()


def test_shaping_telescoping_identity_zero_invalid_steps():
    k = 50.0
    env, rw, stub = make_rig(carry_potential=True, carry_potential_k=k)
    try:
        phi0 = rw._phi
        xys = [0.07, 0.05, 0.02, 0.04, 0.0, 0.01, 0.06]
        shapings, invalid_count = [], 0
        for xy in xys:
            _, info, t = go(rw, stub, xy=xy, **S1)
            shapings.append(t["shaping"])
            invalid_count += bool(info.get("invalid_transition"))
        assert invalid_count == 0
        n = len(shapings)
        disc_sum = sum(PLACE_GAMMA ** i * f for i, f in enumerate(shapings))
        expected = PLACE_GAMMA ** n * place_carry_potential(xys[-1], k) - phi0
        assert abs(disc_sum - expected) < 1e-9
    finally:
        env.close()


def test_shaping_telescoping_on_real_cartesian_replay_reports_invalid_steps():
    """Real physics (not the Stub): report the invalid-step count and only apply
    the 1e-9 telescoping check when it is zero, per the D29 clarification."""
    k = 50.0
    rows = []
    dr = sc.Driver("e4", "e5", budget=300,
                   env_factory=lambda task: PlaceReward(task, carry_potential=True, carry_potential_k=k))
    phi0 = dr.env._phi  # captured right after reset, before any step
    dr.on_step = lambda out: rows.append(out[4])
    try:
        dr.grasp_and_lift()
        dr.place_and_release(height_m=0.0015)
    finally:
        dr.base.close()

    invalid_count = sum(bool(i.get("invalid_transition")) for i in rows)
    shapings = [i["place_terms"]["shaping"] for i in rows]
    xys = [i["place_terms"]["d_xy"] for i in rows]
    n = len(shapings)
    disc_sum = sum(PLACE_GAMMA ** t * f for t, f in enumerate(shapings))
    expected = PLACE_GAMMA ** n * place_carry_potential(xys[-1], k) - phi0
    diff = abs(disc_sum - expected)
    print(f"cartesian expert replay: n={n} steps, invalid={invalid_count}, "
          f"telescoping diff={diff:.3e}")
    if invalid_count == 0:
        assert diff < 1e-9
    # else: reported above, not asserted -- an invalid step (F=0, Phi held)
    # breaks the pure identity by construction; that is the documented behaviour.


# ============================================================= config validation (2e)
@pytest.mark.parametrize("key", ["place_posture_gate", "place_carry_potential", "ppo_mean_squash"])
def test_d29_bool_keys_reject_non_bool(key):
    with pytest.raises(ConfigError, match=f"{key} must be a bool"):
        cfg(**{key: "yes"})


@pytest.mark.parametrize("key", ["place_gate_tilt_lo_deg", "place_gate_tilt_hi_deg", "place_carry_potential_k"])
def test_d29_numeric_keys_reject_non_number(key):
    with pytest.raises(ConfigError, match=f"{key} must be a number"):
        cfg(**{key: "ten"})


def test_gate_lo_must_be_below_hi():
    with pytest.raises(ConfigError, match="place_gate_tilt_lo_deg < place_gate_tilt_hi_deg"):
        cfg(place_gate_tilt_lo_deg=45.0, place_gate_tilt_hi_deg=10.0)
    with pytest.raises(ConfigError, match="place_gate_tilt_lo_deg < place_gate_tilt_hi_deg"):
        cfg(place_gate_tilt_lo_deg=20.0, place_gate_tilt_hi_deg=20.0)


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_carry_potential_k_must_be_positive(bad):
    with pytest.raises(ConfigError, match="place_carry_potential_k must be > 0"):
        cfg(place_carry_potential_k=bad)


@pytest.mark.parametrize("key", ["place_posture_gate", "place_carry_potential"])
def test_gate_and_potential_need_place_mode(key):
    with pytest.raises(ConfigError, match="needs place_mode=True"):
        cfg(**{key: True, "place_mode": False})


def test_ppo_mean_squash_needs_ppo():
    with pytest.raises(ConfigError, match="PPO only"):
        cfg(algorithm="sac", ppo_mean_squash=True)


# =================================================================== gamma guard
def test_check_place_gamma_direct_calls_for_a_carry_potential_config():
    c = resolve_config(dict(PLACE_CFG, place_carry_potential=True), config_path="t.yaml")
    train.check_place_gamma(c, "ppo", 0.99)                       # passes
    with pytest.raises(ValueError, match="gamma"):
        train.check_place_gamma(c, "ppo", 0.95)                   # rejected


def test_build_model_calls_check_place_gamma_for_a_real_carry_potential_config():
    """Building a real SB3 model inside the pytest process segfaults torch
    intermittently in the full suite (DECISIONS D28) -- the monkeypatch and the
    build must both happen in a throwaway subprocess, mirroring
    test_place_eval.py's _make_run."""
    code = f'''
import sys
sys.path.insert(0, {str(_ROOT)!r})
import importlib.util
spec = importlib.util.spec_from_file_location("train", {str(_ROOT / "scripts" / "train.py")!r})
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
from robochess.experiment import resolve_config

calls = []
train.check_place_gamma = lambda c, a, g: calls.append((c.get("place_carry_potential"), a, g))
cfg = resolve_config(dict(task="t1", algorithm="ppo", reward_type="shaped", observation="state", seed=0,
    total_timesteps=1, action_mode="cartesian", max_episode_steps=15, start_square="e4", target_square="e5",
    device="cpu", pickup_obs=True, place_obs=True, place_mode=True, place_carry_potential=True),
    config_path="t.yaml")
venv = train.build_vec_env(cfg)
train.build_model(cfg, venv)
assert len(calls) == 1, calls
assert calls[0] == (True, "ppo", 0.99), calls
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr


# ==================================================================== mean squash
class _ConstantHead(th.nn.Module):
    """Stands in for `action_net`: ignores the latent, returns a constant --
    forces an extreme raw mean regardless of what the feature extractor did."""

    def __init__(self, value: float, dim: int):
        super().__init__()
        self.value = value
        self.dim = dim

    def forward(self, latent_pi: th.Tensor) -> th.Tensor:
        return th.full((latent_pi.shape[0], self.dim), self.value, dtype=th.float32)


def test_mean_squash_bounds_get_distribution_and_evaluate_actions_for_extreme_latents():
    obs_space = spaces.Dict({
        "observation": spaces.Box(-10, 10, (4,), dtype=np.float32),
        "achieved_goal": spaces.Box(-10, 10, (3,), dtype=np.float32),
        "desired_goal": spaces.Box(-10, 10, (3,), dtype=np.float32),
    })
    act_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    policy = SquashedMultiInputActorCriticPolicy(obs_space, act_space, lambda _: 3e-4)
    policy.action_net = _ConstantHead(1e6, 2)

    means = []
    orig = policy._get_action_dist_from_latent

    def spy(latent_pi):
        d = orig(latent_pi)
        means.append(d.distribution.mean.detach().clone())
        return d

    policy._get_action_dist_from_latent = spy

    obs = {k: th.as_tensor(np.stack([s.sample() for _ in range(3)]), dtype=th.float32)
           for k, s in obs_space.spaces.items()}
    policy.get_distribution(obs)
    policy.evaluate_actions(obs, th.zeros((3, 2)))
    assert len(means) == 2                             # both call sites hit the override
    for m in means:
        assert bool(th.all(m <= 1.0)) and bool(th.all(m >= -1.0))
        assert th.allclose(m, th.tanh(th.full_like(m, 1e6)), atol=1e-6)


def test_ppo_mean_squash_build_saves_loads_and_eval_probe_uses_it(tmp_path):
    """Same subprocess constraint as the gamma-guard test above: building (and
    reloading -- PPO.load() reconstructs a model the same way) must not happen
    in this process."""
    run_dir = tmp_path / "runs" / "d29_squash_probe"
    code = f'''
import sys, json
from pathlib import Path
sys.path.insert(0, {str(_ROOT)!r})
import importlib.util
spec = importlib.util.spec_from_file_location("train", {str(_ROOT / "scripts" / "train.py")!r})
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
from robochess.experiment import resolve_config
from robochess.policies import SquashedMultiInputActorCriticPolicy
from stable_baselines3 import PPO

cfg = resolve_config(dict(task="t1", algorithm="ppo", reward_type="shaped", observation="state", seed=0,
    total_timesteps=10, action_mode="cartesian", max_episode_steps=20, start_square="e4", target_square="e5",
    device="cpu", pickup_obs=True, place_obs=True, place_mode=True, ppo_mean_squash=True),
    config_path="t.yaml")
venv = train.build_vec_env(cfg)
model = train.build_model(cfg, venv)
assert isinstance(model.policy, SquashedMultiInputActorCriticPolicy)
run = Path({str(run_dir)!r}); run.mkdir(parents=True)
(run / "manifest.json").write_text(json.dumps(dict(config_resolved=cfg)))
model.save(str(run / "final_model.zip")); venv.save(str(run / "final_vecnormalize.pkl"))

loaded = PPO.load(str(run / "final_model.zip"), device="cpu")
assert isinstance(loaded.policy, SquashedMultiInputActorCriticPolicy)
'''
    build = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stderr

    out = tmp_path / "out.json"
    result = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "eval_probe.py"), "d29_squash_probe",
         "--episodes", "1", "--seed", "1", "--out", str(out),
         "--runs-dir", str(tmp_path / "runs")],
        capture_output=True, text=True, cwd=str(_ROOT))
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert data["n"] == 1


# ==================================================================== eval seeding
# runs/ is gitignored -- a fresh checkout (CI) has no trained models under it.
# These two tests need a real one, so they skip cleanly instead of failing
# when it is absent, rather than depending on it being present.
_D28_PLACE_PPO_SEED0 = _ROOT / "runs" / "_probe_cart_place_ppo_seed0"
requires_d28_place_ppo_seed0 = pytest.mark.skipif(
    not (_D28_PLACE_PPO_SEED0 / "final_model.zip").exists(),
    reason="needs a real trained run under runs/ (gitignored, not present in a fresh checkout)")


def _first_reset_obs(run_dir, seed: int, seed_first: bool, out_path):
    """Loads and resets the real pipeline in a throwaway subprocess (PPO.load()
    reconstructs a model, same segfault-risk pattern as building one)."""
    code = f'''
import json
import numpy as np
import sys
sys.path.insert(0, {str(_ROOT)!r})
import importlib.util
spec = importlib.util.spec_from_file_location("train", {str(_ROOT / "scripts" / "train.py")!r})
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

run_dir = {str(run_dir)!r}
manifest = json.loads(open(run_dir + "/manifest.json").read())["config_resolved"]
venv = DummyVecEnv([lambda: Monitor(train.build_env(manifest))])
venv = VecNormalize.load(run_dir + "/final_vecnormalize.pkl", venv)
venv.training = False; venv.norm_reward = False
model = PPO.load(run_dir + "/final_model.zip", env=venv, device="cpu")
if {seed_first!r}:
    set_random_seed({seed!r})
    model.set_random_seed({seed!r})
venv.seed({seed!r})
obs = venv.reset()
np.savez({str(out_path)!r}, **{{k: np.asarray(v) for k, v in obs.items()}})
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr


@requires_d28_place_ppo_seed0
def test_first_reset_obs_identical_with_or_without_the_new_seed_calls(tmp_path):
    out_without = tmp_path / "without.npz"
    out_with = tmp_path / "with.npz"
    _first_reset_obs(_D28_PLACE_PPO_SEED0, 20000, False, out_without)
    _first_reset_obs(_D28_PLACE_PPO_SEED0, 20000, True, out_with)
    a, b = np.load(out_without), np.load(out_with)
    assert set(a.files) == set(b.files)
    for k in a.files:
        assert np.array_equal(a[k], b[k]), k


@requires_d28_place_ppo_seed0
def test_two_stochastic_evals_same_seed_separate_processes_are_identical(tmp_path):
    run = "_probe_cart_place_ppo_seed0"
    outs = []
    for i in range(2):
        out = tmp_path / f"eval{i}.json"
        result = subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "eval_probe.py"), run,
             "--episodes", "2", "--seed", "20000", "--steps", "--out", str(out)],
            capture_output=True, text=True, cwd=str(_ROOT))
        assert result.returncode == 0, result.stderr
        outs.append(json.loads(out.read_text()))
    assert outs[0]["episodes"] == outs[1]["episodes"]
    assert outs[0]["traj"] == outs[1]["traj"]


# ============================================ D28-flags-off identity (frozen hashes)
# Verified by a real comparison against a git worktree of d28-final (f59027b)
# with every D29 flag left at its default (off): all 4 hashes matched exactly.
# Frozen here per tests/test_cartesian_mode.py's idiom, so the check is CI-safe
# and needs no second worktree at test time. Re-derive and re-freeze if a
# change is meant to alter these paths.
_D29_FLAG_OFF_HASHES = {
    "joint_pickup": "0b873322980bf003cba6551fd3762d07a0537b1e22b6d51cb8b142a3d0e4475b",
    "joint_place": "d27a131a013a2b82a6ebcceecf5d857f6d1a95a2e1c2154b85c5ffe54397c7bd",
    "cart_pickup": "e4128600ee822df039c89da45a523291861e403e92b48b4267d8c8510ef75341",
    "cart_place": "d756c98dec8a55728779d9d3bcba958087fe850d2111fa777bb5e92bea9de043",
}


def _feed_hash(h, obs, reward, term, trunc):
    for k in sorted(obs):
        a = np.asarray(obs[k], dtype=np.float64)
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(np.ascontiguousarray(
        np.array([reward, float(term), float(trunc)], dtype=np.float64)).tobytes())


class _HashRec(gym.Wrapper):
    def __init__(self, env, h):
        super().__init__(env)
        self.h = h

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        _feed_hash(self.h, obs, 0.0, False, False)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        _feed_hash(self.h, obs, r, term, trunc)
        return obs, r, term, trunc, info


def _joint_hash(reward_cls) -> str:
    import hashlib

    from robochess.envs import RoboChessEnv
    from robochess.tasks import T1PickAndPlaceTask

    h = hashlib.sha256()
    base = RoboChessEnv(action_mode="absolute", max_episode_steps=900)
    try:
        task = T1PickAndPlaceTask(base, start_square="e4", target_square="e5")
        rec = _HashRec(reward_cls(task), h)
        ScriptedExpert(rec).run_episode(seed=0)
    finally:
        base.close()
    return h.hexdigest()


def _cart_hash(reward_cls) -> str:
    import hashlib

    h = hashlib.sha256()
    dr = sc.Driver("e4", "e5", budget=300,
                   env_factory=lambda task: _HashRec(reward_cls(task), h))
    try:
        dr.grasp_and_lift()
        dr.place_and_release(height_m=0.0015)
    finally:
        dr.base.close()
    return h.hexdigest()


def test_d29_flags_off_joint_pickup_matches_d28_final():
    from robochess.tasks import PickupReward
    assert _joint_hash(PickupReward) == _D29_FLAG_OFF_HASHES["joint_pickup"]


def test_d29_flags_off_joint_place_matches_d28_final():
    assert _joint_hash(PlaceReward) == _D29_FLAG_OFF_HASHES["joint_place"]


def test_d29_flags_off_cart_pickup_matches_d28_final():
    from robochess.tasks import PickupReward
    assert _cart_hash(PickupReward) == _D29_FLAG_OFF_HASHES["cart_pickup"]


def test_d29_flags_off_cart_place_matches_d28_final():
    assert _cart_hash(PlaceReward) == _D29_FLAG_OFF_HASHES["cart_place"]
