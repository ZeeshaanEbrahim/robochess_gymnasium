"""Tests for D28: the pick-and-place reward (place_mode) and place_obs.

Structure follows test_pickup_reward.py. The per-term tests drive PlaceReward
through a `Stub` that forces exactly the state each case is about (contacts,
the base env's ever_held / release_ok / on-square info, piece height and tilt,
success and termination), because real physics cannot produce "S1 at lift 0.3
and 4 mm off" on demand. Two tests use real physics instead: the S0 terms
against PickupReward on an expert replay, and a clean Cartesian place.

Hand-computed values (design spec): closeness(0.078) = 0.4396, so
  first held at e4, lift 0     2 + 0 + 2(0.4396) + 0           = 2.879
  lifted 30 mm at e4                2 + 2 + 0.879 + 0          = 4.879
  over e5 at 30 mm                  2 + 2 + 2 + 0              = 6.0
  set down pinched on e5            2 + 0 + 2 + 4              = 8.0
  placed (S2), gripper fully open                                =  9.0
  placed (S2), gripper closed on air                             =  8.0  (8 + 1 open_frac)
Success step additionally pays the constant lump 10 / (1 - 0.99) + 20 = 1020.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from robochess.envs import RoboChessEnv
from robochess.experiment import ConfigError, resolve_config
from robochess.tasks import (
    PickupObservation,
    PickupReward,
    PlaceObservation,
    PlaceReward,
    T1PickAndPlaceTask,
)
from robochess.tasks.t1_pick_and_place import (
    N_PLACE_OBS,
    PLACE_GAMMA,
    PLACE_Q_CLOSED,
    PLACE_Q_OPEN,
    find_contacts_owner,
    pickup_closure,
    pickup_geometry,
    pickup_pose_gate,
    pickup_posture,
    pickup_reach,
    place_closeness,
    place_lift_frac,
    place_near,
    place_open_frac,
)

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tests"))
import d27_scenarios as sc  # noqa: E402

_spec = importlib.util.spec_from_file_location("train", _ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

BASE_CFG = dict(task="t1", algorithm="ppo", reward_type="shaped", observation="state",
                seed=0, total_timesteps=1000, max_episode_steps=300,
                action_mode="cartesian", start_square="e4", target_square="e5")
PLACE_CFG = dict(BASE_CFG, pickup_obs=True, place_obs=True, place_mode=True)


def cfg(**over):
    return resolve_config(dict(BASE_CFG, **over), config_path="t.yaml")


def lump(r=10.0, bonus=20.0):
    """D28 (revised): the constant value of staying placed forever, R / (1 - gamma) + bonus."""
    return r / (1.0 - PLACE_GAMMA) + bonus


# --------------------------------------------------------------------- rig
class Stub(gym.Wrapper):
    """Forces the state each case is about, below PlaceReward."""

    def __init__(self, env):
        super().__init__(env)
        self.pinched = False
        self.board = False
        self.grasp = False
        self.ever_held = False
        self.release_ok = False
        self.on_square = False
        self.xy = 0.078
        self.lift = 0.0
        self.tilt = 0.0
        self.success = False
        self.reason = None
        self.terminated = False
        self.invalid = False
        self.aperture = None        # gripper qpos to force (None: leave the physics alone)
        base = self.unwrapped
        self.rest_z = float(base.model.key_qpos[base._home_key][base._piece_qadr + 2])
        find_contacts_owner(env, "no task")._piece_contacts = (
            lambda: (self.pinched, self.pinched, self.board))
        base._piece_tilt_deg = lambda: self.tilt

    def step(self, action):
        obs, reward, _term, trunc, info = self.env.step(action)
        base = self.unwrapped
        base.data.qpos[base._piece_qadr + 2] = self.rest_z + self.lift
        base.data.qvel[base._piece_vadr : base._piece_vadr + 6] = 0.0
        if self.aperture is not None:
            base.data.qpos[base._grip_qadr] = self.aperture
        mujoco.mj_forward(base.model, base.data)
        info = dict(
            info, grasp_success=self.grasp, terminated_reason=self.reason,
            ever_held=self.ever_held, release_ok=self.release_ok,
            piece_on_square=self.on_square, xy_distance_to_goal=self.xy,
            is_success=self.success, invalid_transition=self.invalid,
        )
        return obs, reward, self.terminated, trunc, info


def make_rig(T=300, **weights):
    env = RoboChessEnv(max_episode_steps=T, action_mode="delta")
    task = T1PickAndPlaceTask(env, start_square="e4", target_square="e5")
    stub = Stub(task)
    rw = PlaceReward(stub, **weights)
    rw.reset(seed=0)
    return env, rw, stub


@pytest.fixture
def rig():
    env, rw, stub = make_rig()
    yield rw, stub
    env.close()


def go(rw, stub, **state):
    """One step with `state` applied to the stub; (reward, info, terms)."""
    for k, v in state.items():
        setattr(stub, k, v)
    _o, r, _t, _tr, info = rw.step(np.zeros(rw.action_space.shape, dtype=np.float32))
    return float(r), info, info["place_terms"]


S1 = dict(pinched=True, board=True, ever_held=True)          # pinched, not held: no bonus
S2 = dict(pinched=False, ever_held=True, release_ok=True, on_square=True, xy=0.005, tilt=2.0)


# ------------------------------------------------------- pure term functions
def test_closeness_near_lift_functions():
    assert place_closeness(0.0) == pytest.approx(1.0)
    assert place_closeness(0.078) == pytest.approx(0.439637, abs=1e-5)
    assert place_closeness(0.4) == pytest.approx(0.5 * np.exp(-0.4 / 0.03))
    assert place_closeness(5.0) == pytest.approx(0.5 * np.exp(-5.0 / 0.03))   # ramp clamped
    assert place_near(0.0) == 1.0 and place_near(0.078) < 1e-6
    assert place_near(0.02) == pytest.approx(np.exp(-1.0))
    assert place_lift_frac(-0.01) == 0.0 and place_lift_frac(0.015) == 0.5
    assert place_lift_frac(0.05) == 1.0


# ------------------------------------------------ S1: one +/- control per term
@pytest.mark.parametrize("xy, lift, expected", [
    (0.078, 0.000, 2.0 + 2 * 0.439637),      # first held at e4
    (0.078, 0.030, 4.0 + 2 * 0.439637),      # lifted 30 mm at e4
    (0.000, 0.030, 6.0),                     # over e5 at 30 mm
    (0.000, 0.000, 8.0),                     # set down pinched on e5
])
def test_s1_hand_computed_values(rig, xy, lift, expected):
    rw, stub = rig
    r, info, t = go(rw, stub, xy=xy, lift=lift, **S1)
    assert info["place_state"] == "S1"
    assert t["state_reward"] == pytest.approx(expected, abs=1e-4)
    assert r == pytest.approx(expected, abs=1e-4)      # no bonus in this configuration


def test_s1_base_term_positive_and_negative():
    """2.0 base: present in S1 (positive), absent when not pinched (negative)."""
    env, rw, stub = make_rig(carry_weight=0.0, lower_weight=0.0)
    try:
        _, info, t = go(rw, stub, xy=0.5, lift=0.0, **S1)
        assert t["state_reward"] == pytest.approx(2.0)          # only the base
        _, info, t = go(rw, stub, pinched=False)
        assert info["place_state"] == "S3" and t["state_reward"] != pytest.approx(2.0)
    finally:
        env.close()


def test_s1_lift_term_positive_and_negative(rig):
    rw, stub = rig
    _, _, hi = go(rw, stub, xy=0.5, lift=0.03, **S1)
    _, _, lo = go(rw, stub, xy=0.5, lift=0.0)
    assert hi["state_reward"] - lo["state_reward"] == pytest.approx(2.0)     # 2.0 lift_frac


def test_s1_carry_term_positive_and_negative():
    """C closeness: with C=2 it pays 2 closeness; with C=0 that part is gone."""
    on, off = make_rig(), make_rig(carry_weight=0.0)
    try:
        for (env, rw, stub), C in ((on, 2.0), (off, 0.0)):
            _, _, t = go(rw, stub, xy=0.078, lift=0.03, **S1)
            assert t["state_reward"] == pytest.approx(4.0 + C * 0.439637, abs=1e-4)
    finally:
        on[0].close(); off[0].close()


def test_s1_lower_term_positive_and_negative(rig):
    """L near (1 - lift_frac): pays over the target on the board, zero at full lift
    and zero away from the target."""
    rw, stub = rig
    _, _, t = go(rw, stub, xy=0.0, lift=0.0, **S1)
    base = t["state_reward"]
    assert base == pytest.approx(8.0)
    _, _, t = go(rw, stub, xy=0.0, lift=0.03)
    assert t["state_reward"] == pytest.approx(6.0)             # (1 - lift_frac) = 0
    _, _, t = go(rw, stub, xy=0.078, lift=0.0)
    assert t["state_reward"] == pytest.approx(2.0 + 2 * 0.439637, abs=1e-4)   # near ~ 0


# ------------------------------------------------------------ S2 and its gates
def test_s2_pays_r_placed_and_each_gate_is_needed(rig):
    rw, stub = rig
    r, info, t = go(rw, stub, **S2)
    assert info["place_state"] == "S2" and t["state_reward"] == 9.0
    assert r == 19.0 and t["placed_bonus"] == 10.0             # + first-entry bonus
    for name, off in (("release_ok", dict(release_ok=False)),
                      ("pinched", dict(pinched=True, board=True)),
                      ("on_square", dict(on_square=False)),
                      ("tilt", dict(tilt=15.0)),
                      ("ever_held", dict(ever_held=False))):
        env, rw2, stub2 = make_rig()
        try:
            _, info, t = go(rw2, stub2, **dict(S2, **off))
            assert info["place_state"] != "S2", name
            assert t["state_reward"] != 9.0, name
            assert t["placed_bonus"] == 0.0, name
        finally:
            env.close()


def test_s2_reads_the_env_on_square_flag_not_its_own_distance(rig):
    """One horizontal test shared with success: xy is 1 mm but the flag is False."""
    rw, stub = rig
    _, info, _ = go(rw, stub, **dict(S2, xy=0.001, on_square=False))
    assert info["place_state"] == "S3"
    _, info, _ = go(rw, stub, **dict(S2, xy=0.09, on_square=True))
    assert info["place_state"] == "S2"


# ---------------------------------------------------- S0 / S3 use the D21 terms
def test_s0_and_s3_are_position_plus_closing(rig):
    rw, stub = rig
    r, info, t = go(rw, stub)                                   # S0
    assert info["place_state"] == "S0"
    assert r == pytest.approx(t["position_term"] + t["closing_term"]) and t["position_term"] > 0
    geom = pickup_geometry(rw._base, rw._pinch_sid, rw.dz)
    off = geom["g_to_pinch"]
    assert t["position_term"] == pytest.approx(
        pickup_reach(np.linalg.norm(off)) * pickup_posture(geom["tilt"]), abs=1e-12)
    # closing: the pose gate is ~0 from the home pose, so no closing pay (negative control)
    assert t["closing_term"] == pytest.approx(
        pickup_pose_gate(np.linalg.norm(off[:2]), abs(off[2]), geom["tilt"])
        * pickup_closure(geom["aperture"]), abs=1e-12)
    assert t["closing_term"] < 1e-6
    r, info, t = go(rw, stub, ever_held=True)                   # S3: lost
    assert info["place_state"] == "S3"
    assert r == pytest.approx(t["position_term"] + t["closing_term"])
    r, info, t = go(rw, stub, **S1)                             # pinched: the terms are not paid
    assert info["place_state"] == "S1" and r != pytest.approx(t["position_term"] + t["closing_term"])


# ------------------------------------------------------------------- bonuses
def test_first_held_bonus_fires_exactly_once(rig):
    rw, stub = rig
    fired = []
    for state in (dict(pinched=True, board=False, ever_held=True),) * 3 + (
            dict(pinched=False), dict(pinched=True, board=False)):
        _, _, t = go(rw, stub, **state)
        fired.append(t["first_grasp_bonus"])
    assert fired == [5.0, 0.0, 0.0, 0.0, 0.0]


def test_first_held_bonus_negative_control_pinched_on_board(rig):
    """Both pads on a piece still touching the board is not held: no bonus."""
    rw, stub = rig
    for _ in range(3):
        _, _, t = go(rw, stub, **S1)
        assert t["first_grasp_bonus"] == 0.0
    _, _, t = go(rw, stub, board=False)
    assert t["first_grasp_bonus"] == 5.0


def test_lift_bonus_needs_five_consecutive_steps_and_fires_once(rig):
    rw, stub = rig
    got = [go(rw, stub, grasp=True, lift=0.035, pinched=True, board=False,
              ever_held=True)[2]["lift_bonus"] for _ in range(4)]
    assert got == [0.0] * 4
    assert go(rw, stub)[2]["lift_bonus"] == 5.0                # the fifth
    assert [go(rw, stub)[2]["lift_bonus"] for _ in range(5)] == [0.0] * 5   # never again
    # negative control: a broken streak restarts the count
    env, rw2, stub2 = make_rig()
    try:
        seq = [True, True, True, False, True, True, True, True, True]
        out = [go(rw2, stub2, grasp=g, lift=0.035, pinched=True, board=False,
                  ever_held=True)[2]["lift_bonus"] for g in seq]
        assert out == [0, 0, 0, 0, 0, 0, 0, 0, 5.0]
    finally:
        env.close()
    # below 30 mm never counts
    env, rw3, stub3 = make_rig()
    try:
        assert all(go(rw3, stub3, grasp=True, lift=0.029, pinched=True, board=False,
                      ever_held=True)[2]["lift_bonus"] == 0.0 for _ in range(10))
    finally:
        env.close()


def test_placed_bonus_fires_on_first_entry_only(rig):
    rw, stub = rig
    b = [go(rw, stub, **S2)[2]["placed_bonus"],
         go(rw, stub)[2]["placed_bonus"],
         go(rw, stub, release_ok=False)[2]["placed_bonus"],       # out of S2 (S3)
         go(rw, stub, release_ok=True)[2]["placed_bonus"]]        # re-enter
    assert b == [10.0, 0.0, 0.0, 0.0]


def test_no_bonus_of_any_kind_from_the_approach(rig):
    rw, stub = rig
    for _ in range(5):
        _, _, t = go(rw, stub)
        assert (t["first_grasp_bonus"], t["lift_bonus"], t["placed_bonus"],
                t["success_lump"], t["failure_penalty"]) == (0.0,) * 5


# ------------------------------------------------------------ success and lump
@pytest.mark.parametrize("T, idle", [(300, 0), (300, 149), (300, 299), (12, 0), (12, 5)])
def test_success_lump_is_the_constant_value_of_staying_placed(T, idle):
    """1020 = 10 / (1 - 0.99) + 20 whatever the step and the episode length."""
    env, rw, stub = make_rig(T=T)
    try:
        for _ in range(idle):
            go(rw, stub)
        t_now = idle + 1
        stub.aperture = 0.1          # fully open: S2 pays its ceiling (zero delta actions drift the gripper)
        r, info, t = go(rw, stub, success=True, terminated=True, reason="success", **S2)
        assert t["t"] == t_now                                  # the counter still runs, for info
        assert t["success_lump"] == pytest.approx(1020.0) == pytest.approx(lump())
        # what a success step pays: S2 state reward + first-S2 bonus + lump, no penalty
        assert t["state_reward"] == 9.0 and t["placed_bonus"] == 10.0
        assert t["failure_penalty"] == 0.0
        assert r == pytest.approx(9.0 + 10.0 + 1020.0)
    finally:
        env.close()


def test_lump_follows_r_placed_and_the_bonus_not_the_s2_ceiling():
    env, rw, stub = make_rig(placed_reward=20.0, placed_ceiling=9.0, success_bonus=5.0)
    try:
        stub.aperture = 0.1
        _, _, t = go(rw, stub, success=True, terminated=True, reason="success", **S2)
        assert t["success_lump"] == pytest.approx(20.0 / 0.01 + 5.0)
        assert t["state_reward"] == 9.0
    finally:
        env.close()


def test_no_lump_without_is_success(rig):
    rw, stub = rig
    _, _, t = go(rw, stub, **S2)
    assert t["success_lump"] == 0.0


# ------------------------------------------------------------ failure penalty
@pytest.mark.parametrize("reason", ["bad_contact", "piece_off_board"])
def test_failure_penalty_is_subtracted_from_the_state_reward(rig, reason):
    rw, stub = rig
    r, _, t = go(rw, stub, terminated=True, reason=reason)
    assert t["failure_penalty"] == 5.0
    assert r == pytest.approx(t["position_term"] + t["closing_term"] - 5.0)
    env, rw2, stub2 = make_rig()
    try:
        r, _, t = go(rw2, stub2, terminated=True, reason=reason, **S1)     # in S1: adds to it
        assert r == pytest.approx(t["state_reward"] - 5.0)
    finally:
        env.close()


@pytest.mark.parametrize("terminated, reason", [
    (True, "success"), (True, "solver_instability"), (False, "bad_contact"), (False, None)])
def test_no_penalty_for_other_endings_or_a_mid_episode_flag(rig, terminated, reason):
    rw, stub = rig
    _, _, t = go(rw, stub, terminated=terminated, reason=reason)
    assert t["failure_penalty"] == 0.0


# ----------------------------------------------------- F7: invalid transitions
def test_invalid_step_pays_zero_latches_nothing_and_only_moves_the_counter():
    env, rw, stub = make_rig(T=300)
    try:
        go(rw, stub)                                                    # t = 1
        # an invalid step in which EVERYTHING would fire
        r, info, t = go(rw, stub, invalid=True, success=True, terminated=True, reason="success",
                        pinched=True, board=False, ever_held=True, grasp=True, lift=0.05,
                        release_ok=True, on_square=True, xy=0.0)
        assert r == 0.0
        assert info["place_state"] == "S0"                              # no state transition
        assert (t["first_grasp_bonus"], t["lift_bonus"], t["placed_bonus"],
                t["success_lump"], t["failure_penalty"], t["state_reward"]) == (0.0,) * 6
        assert t["t"] == 2                                              # counter moved
        # nothing latched: the first-held bonus is still available on the next valid step
        r, _, t = go(rw, stub, invalid=False, success=False, terminated=False, reason=None)
        assert t["first_grasp_bonus"] == 5.0 and t["t"] == 3
        # the counter still counts the invalid step (t = 4 here), and the lump does not depend on it
        _, _, t = go(rw, stub, success=True, terminated=True, reason="success", **S2)
        assert t["t"] == 4 and t["success_lump"] == pytest.approx(lump())
    finally:
        env.close()


def test_invalid_step_does_not_advance_the_lift_streak_or_s2_latch():
    env, rw, stub = make_rig()
    try:
        held = dict(grasp=True, lift=0.035, pinched=True, board=False, ever_held=True)
        for _ in range(3):
            go(rw, stub, **held)
        go(rw, stub, invalid=True)                     # would have been the 4th
        assert go(rw, stub, invalid=False)[2]["lift_bonus"] == 0.0     # 4th VALID step
        assert go(rw, stub)[2]["lift_bonus"] == 5.0                    # the 5th valid step
        go(rw, stub, invalid=True, **S2)
        assert go(rw, stub, invalid=False, grasp=False, **S2)[2]["placed_bonus"] == 10.0
    finally:
        env.close()


# ------------------------------------------- step counter and the episode limit
def test_counter_equals_the_env_step_counter_through_truncation():
    T = 12
    env = RoboChessEnv(max_episode_steps=T, action_mode="delta")
    try:
        task = T1PickAndPlaceTask(env, start_square="e4", target_square="e5")
        rw = PlaceReward(task)
        rw.reset(seed=0)
        node, has_timelimit = rw, False
        while node is not None:
            has_timelimit |= isinstance(node, gym.wrappers.TimeLimit)
            node = getattr(node, "env", None)
        assert not has_timelimit                # the limit is the base env's own counter
        base = rw.unwrapped
        assert base.max_episode_steps == T
        n, trunc = 0, False
        zero = np.zeros(rw.action_space.shape, dtype=np.float32)
        while not trunc:
            _o, _r, term, trunc, info = rw.step(zero)
            n += 1
            assert rw._t == base._elapsed_steps == n == info["place_terms"]["t"]
            assert not term, "zero-action episode ended early; the test needs to reach truncation"
        assert n == T
        rw.reset(seed=0)
        assert rw._t == 0
    finally:
        env.close()


# ------------------------------------------------------ real physics: S0 terms
def test_s0_terms_equal_pickup_rewards_on_an_expert_replay():
    """Before the first held step PlaceReward's position/closing terms ARE D21's."""
    rows = []

    def factory(task):
        return PickupReward(PlaceReward(task))

    dr = sc.Driver("e4", "e5", budget=300, env_factory=factory)
    dr.on_step = lambda out: rows.append(out[4])
    dr.grasp_and_lift()
    s0 = [i for i in rows if i["place_state"] == "S0"]
    assert len(s0) >= 50
    for i in s0:
        assert i["place_terms"]["position_term"] == pytest.approx(
            i["pickup_terms"]["position_term"], abs=1e-12)
        assert i["place_terms"]["closing_term"] == pytest.approx(
            i["pickup_terms"]["closing_term"], abs=1e-12)
    assert max(i["place_terms"]["closing_term"] for i in s0) > 1e-3     # not vacuous
    assert any(i["place_state"] == "S1" for i in rows)                  # the grasp happened


# --------------------------------------------- real physics: a clean Cartesian place
def test_clean_cartesian_place_pays_each_bonus_once_and_succeeds():
    rows = []
    dr = sc.Driver("e4", "e5", budget=300,
                   env_factory=lambda task: PlaceReward(task))
    dr.on_step = lambda out: rows.append((out[1], out[4]))
    dr.grasp_and_lift()
    dr.place_and_release(height_m=0.0015)
    last_r, last = rows[-1]
    assert last["is_success"] and last["place_state"] == "S2"
    terms = [i["place_terms"] for _, i in rows]
    assert sum(t["first_grasp_bonus"] > 0 for t in terms) == 1
    assert sum(t["lift_bonus"] > 0 for t in terms) == 1
    assert sum(t["placed_bonus"] > 0 for t in terms) == 1
    assert sum(t["success_lump"] > 0 for t in terms) == 1
    assert sum(t["failure_penalty"] > 0 for t in terms) == 0
    assert terms[-1]["success_lump"] == pytest.approx(1020.0)
    # every step's reward is exactly the reported terms
    for r, i in rows:
        t = i["place_terms"]
        assert r == pytest.approx(t["state_reward"] + t["first_grasp_bonus"] + t["lift_bonus"]
                                  + t["placed_bonus"] + t["success_lump"] - t["failure_penalty"])
    T = dr.base.max_episode_steps
    print(f"clean place: success at step {terms[-1]['t']} of {T}; "
          f"last step reward {last_r:.4f}, terms {terms[-1]}")


# ------------------------------------------------------------ place_obs
def test_place_obs_is_target_minus_piece_and_follows_pickup_obs():
    env = train.build_env(resolve_config(PLACE_CFG, config_path="t.yaml"))
    try:
        base = env.unwrapped
        # T1's achieved_goal IS the piece body's world position, not the pinch point
        piece = base.data.xpos[base._piece_bid]
        obs, _ = env.reset(seed=0)
        assert np.array_equal(obs["achieved_goal"], piece.astype(np.float32))
        assert not np.allclose(obs["achieved_goal"], base.data.site_xpos[base._pinch_sid])
        assert env.observation_space["observation"].shape == (21 + 15 + N_PLACE_OBS,)
        for _ in range(3):
            assert np.allclose(obs["observation"][-3:], obs["desired_goal"] - obs["achieved_goal"],
                               atol=1e-6)
            obs, *_ = env.step(env.action_space.sample())
        # the first 36 entries are exactly what pickup_obs alone would give
        plain = train.build_env(resolve_config(dict(BASE_CFG, pickup_obs=True), config_path="t.yaml"))
        try:
            o2, _ = plain.reset(seed=0)
            o1, _ = env.reset(seed=0)
            assert np.array_equal(o1["observation"][:36], o2["observation"])
        finally:
            plain.close()
    finally:
        env.close()


def test_place_obs_needs_pickup_obs_beneath_it():
    base = RoboChessEnv(action_mode="cartesian")
    try:
        with pytest.raises(TypeError, match="pickup_obs"):
            PlaceObservation(T1PickAndPlaceTask(base))
    finally:
        base.close()


# ------------------------------------------------------------- config / wiring
def test_defaults_leave_everything_off():
    c = cfg()
    assert c["place_mode"] is False and c["place_obs"] is False
    assert (c["place_carry_weight"], c["place_lower_weight"], c["place_placed_reward"],
            c["place_placed_floor"], c["place_placed_ceiling"], c["place_placed_bonus"], c["place_lift_bonus"],
            c["place_success_bonus"], c["place_failure_penalty"]) == (
        2.0, 4.0, 10.0, 8.0, 9.0, 10.0, 5.0, 20.0, 5.0)
    env = train.build_env(c)
    try:
        assert not isinstance(env, PlaceReward)
        assert env.observation_space["observation"].shape == (21,)
        assert "place_state" not in train.monitor_info_keywords(c)
    finally:
        env.close()


def test_place_mode_builds_the_wrapper_and_adds_place_state_and_squares_to_monitor():
    c = resolve_config(PLACE_CFG, config_path="t.yaml")
    env = train.build_env(c)
    try:
        assert isinstance(env, PlaceReward)
    finally:
        env.close()
    assert train.monitor_info_keywords(c) == (
        "terminated_reason", "gripper_self_contact", "ik_rejects",
        "ever_held", "release_height_mm", "place_state", "start_square", "target_square")  # RS29
    assert "place_state" not in train.monitor_info_keywords(
        resolve_config(dict(BASE_CFG, pickup_obs=True, pickup_mode=True), config_path="t.yaml"))


@pytest.mark.parametrize("over", [
    dict(place_mode=True, pickup_obs=True, pickup_mode=True),
    dict(place_mode=True, pickup_obs=True, shaped_staged=True),
    dict(place_mode=True, pickup_obs=True, reward_type="sparse"),
    dict(place_mode=True, pickup_obs=False),
    dict(place_obs=True, pickup_obs=False),
    dict(place_mode="yes", pickup_obs=True),
    dict(place_mode=True, pickup_obs=True, place_carry_weight=-1.0),
])
def test_invalid_place_configs_are_rejected(over):
    with pytest.raises(ConfigError):
        cfg(**over)


def test_gamma_guard_and_the_gammas_the_learners_and_vecnormalize_use():
    """Building a real SB3 model in the full suite segfaults (torch after many mujoco
    tests), so the guard is a function of the gamma and the defaults are read from the
    SB3 constructors' signatures."""
    import inspect

    from stable_baselines3 import PPO, SAC
    for algo in (PPO, SAC):
        assert inspect.signature(algo.__init__).parameters["gamma"].default == PLACE_GAMMA
    c = resolve_config(PLACE_CFG, config_path="t.yaml")
    train.check_place_gamma(c, "ppo", 0.99)                           # passes
    with pytest.raises(ValueError, match="gamma"):
        train.check_place_gamma(c, "sac", 0.95)                       # rejected
    train.check_place_gamma(cfg(), "sac", 0.95)                       # only place_mode cares
    src = (_ROOT / "scripts" / "train.py").read_text()
    assert "gamma=0.99" in src                                        # VecNormalize's
    assert "check_place_gamma(cfg, algorithm, model.gamma)" in src    # and it is wired


# ------------------------------------------------- vecnormalize_clip_reward
def _vecnorm(**over):
    from stable_baselines3.common.vec_env import VecNormalize
    env = train.build_vec_env(resolve_config(dict(BASE_CFG, **over), config_path="t.yaml"))
    assert isinstance(env, VecNormalize)
    return env


def test_vecnormalize_clip_reward_default_is_ten_on_an_existing_config():
    import yaml
    raw = yaml.safe_load((_ROOT / "configs" / "_probe_cart_pickup_ppo_seed0.yaml").read_text())
    c = resolve_config(raw, config_path="x.yaml")
    assert c["vecnormalize_clip_reward"] == 10.0
    env = train.build_vec_env(c)
    try:
        assert env.clip_reward == 10.0 and env.clip_obs == 10.0 and env.gamma == 0.99
    finally:
        env.close()


def test_vecnormalize_clip_reward_1000_reaches_vecnormalize():
    env = _vecnorm(vecnormalize_clip_reward=1000)
    try:
        assert env.clip_reward == 1000.0
    finally:
        env.close()


@pytest.mark.parametrize("bad", [0, -1.0, "10", True, None])
def test_vecnormalize_clip_reward_must_be_positive_number(bad):
    with pytest.raises(ConfigError):
        cfg(vecnormalize_clip_reward=bad)


# ------------------------------------------------------------- S2 open ramp
def test_open_frac_controls():
    assert place_open_frac(PLACE_Q_CLOSED) == 0.0                     # positive: closed -> 0
    assert place_open_frac(PLACE_Q_OPEN) == 1.0                       # released threshold -> 1
    assert place_open_frac(0.0) == 1.0 and place_open_frac(0.05) == 1.0     # clipped above
    assert place_open_frac(0.9) == 0.0                                # negative: past closed, clipped
    mid = 0.5 * (PLACE_Q_CLOSED + PLACE_Q_OPEN)
    assert place_open_frac(mid) == pytest.approx(0.5)
    assert place_open_frac(0.648) == pytest.approx((0.782 - 0.648) / 0.582)   # at a grasp: 0.23
    qs = np.linspace(0.0, 0.8, 50)
    f = [place_open_frac(q) for q in qs]
    assert all(b <= a for a, b in zip(f, f[1:]))                      # monotone in q


def test_q_closed_matches_the_gripper_closed_on_air():
    env = RoboChessEnv(action_mode="cartesian", max_episode_steps=300)
    try:
        env.reset(seed=0)
        for _ in range(80):
            env.step(np.array([0, 0, 0, 1.0], dtype=np.float32))
        q = float(env.unwrapped.data.qpos[env.unwrapped._grip_qadr])
        assert q == pytest.approx(PLACE_Q_CLOSED, abs=0.005)
        assert q > 0.66          # above every grasp qpos measured on the expert pairs (0.647-0.651)
    finally:
        env.close()


@pytest.mark.parametrize("q, expected", [
    (PLACE_Q_CLOSED, 8.0), (0.648, 8.0 + (0.782 - 0.648) / 0.582),
    (0.491, 8.5), (PLACE_Q_OPEN, 9.0), (0.0, 9.0), (0.9, 8.0)])
def test_s2_reward_is_floor_plus_open_frac(rig, q, expected):
    rw, stub = rig
    stub.aperture = q
    _, info, t = go(rw, stub, **S2)
    assert info["place_state"] == "S2"
    assert t["state_reward"] == pytest.approx(expected, abs=1e-9)


def test_s2_never_below_the_floor_and_floor_is_configurable():
    env, rw, stub = make_rig()
    try:
        stub.aperture = 0.9
        for q in np.linspace(0.0, 0.8, 17):
            stub.aperture = float(q)
            assert go(rw, stub, **S2)[2]["state_reward"] >= 8.0
    finally:
        env.close()
    env, rw, stub = make_rig(placed_floor=5.0)
    try:
        stub.aperture = PLACE_Q_CLOSED
        assert go(rw, stub, **S2)[2]["state_reward"] == pytest.approx(5.0)     # the key is live
    finally:
        env.close()
    for bad in (dict(place_placed_floor=9.5),                       # floor > ceiling
                dict(place_placed_floor=-1.0),
                dict(place_placed_ceiling=10.0),                     # ceiling must stay BELOW R_placed
                dict(place_placed_ceiling=11.0),
                dict(place_placed_ceiling=7.0)):                     # ceiling < floor
        with pytest.raises(ConfigError):
            cfg(**bad)
    cfg(place_placed_ceiling=9.99, place_placed_floor=9.0)          # a valid variant
    with pytest.raises(ValueError):
        make_rig(placed_ceiling=10.0)


def test_lump_is_paid_on_top_of_a_floor_level_s2_step():
    env, rw, stub = make_rig()
    try:
        stub.aperture = PLACE_Q_CLOSED                       # S2 pays only 8 this step
        r, _, t = go(rw, stub, success=True, terminated=True, reason="success", **S2)
        assert t["state_reward"] == pytest.approx(8.0)
        assert t["success_lump"] == pytest.approx(1020.0)
    finally:
        env.close()


@pytest.mark.parametrize("start, target", [("e4", "e5"), ("a1", "h8"), ("c1", "f6"), ("b7", "g2")])
def test_s2_reward_is_non_decreasing_through_every_expert_release(start, target):
    rows = []
    dr = sc.Driver(start, target, budget=300, env_factory=lambda task: PlaceReward(task))
    dr.on_step = lambda out: rows.append(out[4]["place_terms"] | {"S": out[4]["place_state"]})
    dr.grasp_and_lift()
    dr.place_and_release(height_m=0.0015)
    s2 = [t["state_reward"] for t in rows if t["S"] == "S2"]
    assert len(s2) >= 3
    assert all(v >= 8.0 for v in s2)
    assert all(b >= a - 1e-12 for a, b in zip(s2, s2[1:])), s2
    assert s2[0] < s2[-1] <= 9.0                            # it does ramp
    pre = [t["state_reward"] for t in rows if t["S"] == "S1"][-1]
    assert s2[0] >= pre - 1.0                                # no cliff at the release step


def test_s2_ceiling_is_configurable_and_finishing_beats_parking():
    env, rw, stub = make_rig(placed_ceiling=9.5)
    try:
        stub.aperture = 0.1
        assert go(rw, stub, **S2)[2]["state_reward"] == pytest.approx(9.5)
    finally:
        env.close()
    # parking in S2 forever is worth at most ceiling / (1 - gamma); success pays R / (1 - gamma) + 20
    assert 9.0 / (1.0 - PLACE_GAMMA) < lump()
