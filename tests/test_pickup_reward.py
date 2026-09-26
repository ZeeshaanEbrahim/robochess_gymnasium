"""Tests for the pick-up training stage: reward, success flag and observation.

Everything here is behind `pickup_mode` / `pickup_obs`, both default off, so
the first test in the file is the one that proves the defaults change nothing.

The reward tests drive the wrapper through a `Stub` that sits between it and
the task and forces exactly the state each case is about -- pad contact, board
contact, piece height, `grasp_success`, termination reason. Real physics cannot
produce "held and lifted exactly 15 mm" on demand, and a test that waits for the
simulator to wander into the state it wants is a slow test that checks the
simulator rather than the reward. The one case built from real physics is the
D18 toppled piece, because the whole point of that case is what the simulator's
contact list actually contains.

`hold()` sends a TRUE hold, which is action-mode dependent: zeros in delta mode,
but `home_action()` in absolute mode, where a zero action does NOT mean "stay
put" -- `normalize_action` maps it to the MID-POINT of every actuator's range,
commanding all six arm joints to 0 rad and the gripper to 127.5. An earlier
version of this file sent zeros in absolute mode, which flew the arm 1.08 rad
off home over six steps and made the position term collapse from 0.18 to 0.00.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from robochess.envs import RoboChessEnv
from robochess.envs.robochess_env import N_ARM_ACTUATORS
from robochess.experiment import ConfigError, resolve_config
from robochess.tasks import PickupObservation, PickupReward, T1PickAndPlaceTask
from robochess.tasks.t1_pick_and_place import (
    CLOSURE_CLOSED,
    CLOSURE_OPEN,
    EXPECTED_PICKUP_DZ,
    N_PICKUP_OBS,
    PINCH_GRASP_Z,
    POSTURE_FLOOR,
    find_contacts_owner,
    pickup_closure,
    pickup_dz,
    pickup_features,
    pickup_geometry,
    pickup_posture,
    pickup_reach,
    pinch_tilt,
)

BASE_CFG = dict(task="t1", algorithm="ppo", reward_type="shaped", observation="state",
                seed=0, total_timesteps=1000, max_episode_steps=100,
                action_mode="delta", start_square="e4", target_square="e5")


def cfg(**over):
    return resolve_config(dict(BASE_CFG, **over), config_path="t.yaml")


# --------------------------------------------------------------------- rig
class Stub(gym.Wrapper):
    """Forces the state each reward case is about, below PickupReward."""

    def __init__(self, env):
        super().__init__(env)
        self.pinched = False
        self.board = False          # piece also touching the board (D18's term)
        self.grasp = False
        self.lift = 0.0
        self.reason = None
        self.terminated = False
        base = self.unwrapped
        self.rest_z = float(base.model.key_qpos[base._home_key][base._piece_qadr + 2])
        task = find_contacts_owner(env, "no task")
        task._piece_contacts = lambda: (self.pinched, self.pinched, self.board)

    def step(self, action):
        obs, reward, _term, trunc, info = self.env.step(action)
        base = self.unwrapped
        base.data.qpos[base._piece_qadr + 2] = self.rest_z + self.lift
        base.data.qvel[base._piece_vadr : base._piece_vadr + 6] = 0.0
        mujoco.mj_forward(base.model, base.data)
        info = dict(info, grasp_success=self.grasp, terminated_reason=self.reason)
        return obs, reward, self.terminated, trunc, info


@pytest.fixture
def rig():
    env = RoboChessEnv(max_episode_steps=200)
    task = T1PickAndPlaceTask(env, start_square="e4", target_square="e5")
    stub = Stub(task)
    rw = PickupReward(stub)
    rw.reset(seed=0)
    yield rw, stub
    env.close()


def hold(rw, stub, n=1):
    """`n` steps of a TRUE hold for this env's action mode; last (reward, info)."""
    base = rw.unwrapped
    action = (np.zeros(rw.action_space.shape, dtype=np.float32)
              if base.action_mode == "delta" else base.home_action())
    out = None
    for _ in range(n):
        _obs, reward, _t, _tr, info = rw.step(action)
        out = (float(reward), info)
    return out


def total_from_terms(info):
    """Every component the wrapper reported, summed.

    Asserted against the returned reward rather than comparing rewards across
    steps: a zero action does not freeze the arm, so the position and closing
    terms drift between consecutive steps and a cross-step difference measures
    that drift as much as the term under test.
    """
    t = info["pickup_terms"]
    return (t["position_term"] + t["closing_term"] + t["lift_term"]
            + t["first_grasp_bonus"] + t["success_term"])


# ------------------------------------------------------------ flag-off identity
def test_defaults_leave_the_reward_stack_untouched():
    """pickup_mode and pickup_obs both default off, and off means absent."""
    c = cfg()
    assert c["pickup_mode"] is False and c["pickup_obs"] is False
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "train", Path(__file__).resolve().parents[1] / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    env = train.build_env(c)
    try:
        assert not isinstance(env, PickupReward)
        assert env.observation_space["observation"].shape == (21,)
        assert train.monitor_info_keywords(c) == (
            "terminated_reason", "gripper_self_contact", "ik_rejects",
            "ever_held", "release_height_mm")
        assert train.monitor_info_keywords(cfg(pickup_mode=True)) == (
            "terminated_reason", "gripper_self_contact", "ik_rejects",
            "ever_held", "release_height_mm", "pickup_success")
    finally:
        env.close()


# ----------------------------------------------------------------- pure terms
def test_dz_is_computed_from_the_model_not_hardcoded():
    env = RoboChessEnv()
    try:
        base = env.unwrapped
        rest_z = float(base.model.key_qpos[base._home_key][base._piece_qadr + 2])
        assert pickup_dz(rest_z) == pytest.approx(PINCH_GRASP_Z - rest_z)
        assert pickup_dz(rest_z) == pytest.approx(EXPECTED_PICKUP_DZ, abs=5e-4)
    finally:
        env.close()
    with pytest.raises(ValueError, match="pickup dz"):
        pickup_dz(rest_z + 0.01)      # a moved piece must be caught, not absorbed


def position(d, tilt_deg):
    return pickup_reach(d) * pickup_posture(np.radians(tilt_deg))


@pytest.mark.parametrize("d, tilt_deg, expected", [
    (0.42036, 0.00, 0.23728),      # (a) home pose
    (0.04762, 44.85, 0.31682),     # (b) 1M policy at step 26
    (0.11319, 21.15, 0.29713),     # (c) expert hover before descend
])
def test_position_term_as_a_pure_function_of_distance_and_tilt(d, tilt_deg, expected):
    assert position(d, tilt_deg) == pytest.approx(expected, abs=1e-4)


def test_posture_is_one_upright_and_decays_to_the_floor():
    assert pickup_posture(0.0) == pytest.approx(1.0)
    assert pickup_posture(np.pi) == pytest.approx(POSTURE_FLOOR, abs=1e-3)
    assert pickup_posture(50.0) == pytest.approx(POSTURE_FLOOR, abs=1e-9)
    assert pickup_posture(0.35) < pickup_posture(0.1) < pickup_posture(0.0)


def test_tilt_is_zero_upright_and_half_pi_horizontal():
    def xmat(approach):
        z = np.asarray(approach, dtype=float)
        z /= np.linalg.norm(z)
        helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(helper, z); x /= np.linalg.norm(x)
        return np.column_stack([x, np.cross(z, x), z]).ravel()

    assert pinch_tilt(xmat([0, 0, -1])) == pytest.approx(0.0, abs=1e-9)
    assert pinch_tilt(xmat([1, 0, 0])) == pytest.approx(np.pi / 2, abs=1e-9)
    assert pinch_tilt(xmat([0, 1, 0])) == pytest.approx(np.pi / 2, abs=1e-9)
    assert pinch_tilt(xmat([0, 0, 1])) == pytest.approx(np.pi, abs=1e-9)


def test_closure_maps_aperture_between_open_and_closed():
    assert pickup_closure(CLOSURE_OPEN) == 0.0
    assert pickup_closure(CLOSURE_CLOSED) == 1.0
    assert pickup_closure(0.0) == 0.0                      # clipped below
    assert pickup_closure(5.0) == 1.0                      # clipped above
    mid = 0.5 * (CLOSURE_OPEN + CLOSURE_CLOSED)
    assert pickup_closure(mid) == pytest.approx(0.5)


# ------------------------------------------------------------- lift and bonus
def test_lift_term_is_dense_from_zero_and_clipped_at_the_target(rig):
    rw, stub = rig
    stub.pinched, stub.board = True, False       # held
    seen = {}
    for lift_mm in (-5.0, 0.0, 7.5, 15.0, 30.0, 60.0):
        stub.lift = lift_mm / 1000.0
        _r, info = hold(rw, stub)
        seen[lift_mm] = info["pickup_terms"]["lift_term"]
    assert seen[-5.0] == 0.0
    assert seen[0.0] == pytest.approx(0.0, abs=1e-5)
    assert seen[7.5] == pytest.approx(0.5, abs=1e-4)
    assert seen[15.0] == pytest.approx(1.0, abs=1e-4)
    assert seen[30.0] == pytest.approx(2.0, abs=1e-4)
    assert seen[60.0] == pytest.approx(2.0, abs=1e-4)       # clipped


def test_lift_term_is_paid_only_while_held(rig):
    """Pads on the piece is not enough: the piece must be clear of the board."""
    rw, stub = rig
    stub.lift = 0.02
    stub.pinched = False
    _r, info = hold(rw, stub)
    assert info["pickup_terms"]["lift_term"] == 0.0

    stub.pinched, stub.board = True, True        # pinched but resting on the board
    _r, info = hold(rw, stub)
    assert info["pickup_terms"]["pinched"] is True
    assert info["pickup_terms"]["held"] is False
    assert info["pickup_terms"]["lift_term"] == 0.0

    stub.board = False                           # now genuinely held
    _r, info = hold(rw, stub)
    assert info["pickup_terms"]["held"] is True
    assert info["pickup_terms"]["lift_term"] > 0.0


def test_first_grasp_bonus_fires_once_on_the_first_held_step(rig):
    rw, stub = rig
    # grasp_success true but NOT pinched: no bonus, because the bonus is
    # keyed on held, not on grasp_success.
    stub.grasp, stub.pinched, stub.lift = True, False, 0.05
    _r, info = hold(rw, stub, 3)
    assert info["pickup_terms"]["first_grasp_bonus"] == 0.0

    # pinched but still on the board: still no bonus.
    stub.pinched, stub.board = True, True
    _r, info = hold(rw, stub, 3)
    assert info["pickup_terms"]["first_grasp_bonus"] == 0.0

    stub.board = False
    r1, i1 = hold(rw, stub)
    assert i1["pickup_terms"]["first_grasp_bonus"] == pytest.approx(5.0)
    assert r1 == pytest.approx(total_from_terms(i1), abs=1e-9)
    r2, i2 = hold(rw, stub)
    assert i2["pickup_terms"]["first_grasp_bonus"] == 0.0
    assert r2 == pytest.approx(total_from_terms(i2), abs=1e-9)
    assert r1 - total_from_terms(i1) + 5.0 == pytest.approx(5.0, abs=1e-9)


def test_first_grasp_bonus_does_not_fire_again_after_release_and_re_pinch(rig):
    rw, stub = rig
    stub.pinched, stub.board = True, False
    _r, info = hold(rw, stub)
    assert info["pickup_terms"]["first_grasp_bonus"] == pytest.approx(5.0)
    stub.pinched = False
    hold(rw, stub, 2)
    stub.pinched = True
    _r, info = hold(rw, stub, 2)
    assert info["pickup_terms"]["first_grasp_bonus"] == 0.0


def test_episode_state_resets(rig):
    rw, stub = rig
    stub.pinched, stub.grasp, stub.lift = True, True, 0.05
    hold(rw, stub, 6)
    assert rw._held_once and rw._hold_streak >= 5 and rw._succeeded
    rw.reset(seed=1)
    assert rw._held_once is False
    assert rw._hold_streak == 0
    assert rw._succeeded is False
    stub.pinched = True
    _r, info = hold(rw, stub)
    assert info["pickup_terms"]["first_grasp_bonus"] == pytest.approx(5.0)
    assert info["pickup_success"] is False


# ------------------------------------------------------------- success state
def test_success_needs_five_consecutive_steps_of_grasp_and_lift(rig):
    rw, stub = rig
    stub.pinched, stub.grasp, stub.lift = True, True, 0.031
    for step in range(1, 5):
        _r, info = hold(rw, stub)
        assert info["pickup_terms"]["success_term"] == 0.0, f"fired early at {step}"
        assert info["pickup_success"] is False
    _r, info = hold(rw, stub)
    assert info["pickup_terms"]["hold_streak"] == 5
    assert info["pickup_terms"]["success_term"] == pytest.approx(1.0)
    assert info["pickup_success"] is True


def test_success_requires_both_grasp_success_and_the_lift_height(rig):
    rw, stub = rig
    stub.pinched, stub.grasp, stub.lift = True, True, 0.029      # just under
    _r, info = hold(rw, stub, 8)
    assert info["pickup_terms"]["success_term"] == 0.0
    stub.lift, stub.grasp = 0.05, False                          # lifted, not grasped
    _r, info = hold(rw, stub, 8)
    assert info["pickup_terms"]["success_term"] == 0.0


def test_a_broken_streak_restarts_the_count(rig):
    rw, stub = rig
    stub.pinched, stub.grasp, stub.lift = True, True, 0.05
    hold(rw, stub, 4)
    stub.lift = 0.01                     # one step below the height
    _r, info = hold(rw, stub)
    assert info["pickup_terms"]["hold_streak"] == 0
    stub.lift = 0.05
    _r, info = hold(rw, stub, 4)
    assert info["pickup_terms"]["success_term"] == 0.0, "streak did not restart"
    _r, info = hold(rw, stub)
    assert info["pickup_terms"]["success_term"] == pytest.approx(1.0)


def test_success_pays_every_step_and_does_not_terminate(rig):
    rw, stub = rig
    stub.pinched, stub.grasp, stub.lift = True, True, 0.05
    action = np.zeros(rw.action_space.shape, dtype=np.float32)
    terminated = []
    rewards = []
    for _ in range(9):
        _obs, reward, term, _trunc, info = rw.step(action)
        terminated.append(term)
        rewards.append(info["pickup_terms"]["success_term"])
    assert not any(terminated), "the success state must not end the episode"
    assert rewards[:4] == [0.0] * 4
    assert all(v == pytest.approx(1.0) for v in rewards[4:])
    assert info["pickup_success"] is True


# ------------------------------------------------------------------- failures
@pytest.mark.parametrize("reason, penalised", [
    ("bad_contact", True),
    ("piece_off_board", True),
    ("solver_instability", False),
    (None, False),
])
def test_failure_penalty_applies_to_the_two_chosen_reasons(rig, reason, penalised):
    rw, stub = rig
    stub.reason, stub.terminated = reason, reason is not None
    reward, info = hold(rw, stub)
    charged = total_from_terms(info) - reward
    if penalised:
        assert charged == pytest.approx(rw.failure_penalty, abs=1e-9)
    else:
        assert charged == pytest.approx(0.0, abs=1e-9)


def test_failure_penalty_is_read_from_config(rig):
    env = RoboChessEnv(max_episode_steps=50)
    task = T1PickAndPlaceTask(env, start_square="e4", target_square="e5")
    stub = Stub(task)
    rw = PickupReward(stub, failure_penalty=12.5)
    rw.reset(seed=0)
    try:
        assert rw.failure_penalty == 12.5
        stub.reason, stub.terminated = "bad_contact", True
        reward, info = hold(rw, stub)
        assert total_from_terms(info) - reward == pytest.approx(12.5, abs=1e-9)
    finally:
        env.close()


# ------------------------------------------------------------------- config
def test_pickup_mode_is_rejected_without_shaped_reward():
    with pytest.raises(ConfigError, match="replaces reward_type"):
        cfg(pickup_mode=True, reward_type="sparse")


def test_pickup_mode_is_rejected_for_non_t1():
    with pytest.raises(ConfigError, match="T1 only"):
        cfg(pickup_mode=True, task="t2", n_obstacles=1)


def test_pickup_mode_is_rejected_alongside_shaped_staged():
    with pytest.raises(ConfigError, match="both replace the reward"):
        cfg(pickup_mode=True, shaped_staged=True)


def test_pickup_obs_is_rejected_for_non_t1():
    with pytest.raises(ConfigError, match="T1 only"):
        cfg(pickup_obs=True, task="t2", n_obstacles=1)


def test_unknown_keys_are_still_rejected():
    with pytest.raises(ConfigError, match="Unknown config key"):
        cfg(pickup_nonsense=True)


# -------------------------------------------------------------- observation
def test_pickup_obs_appends_fifteen_entries_from_the_shared_helper():
    env = RoboChessEnv(max_episode_steps=50)
    task = T1PickAndPlaceTask(env, start_square="e4", target_square="e5")
    obs_wrapper = PickupObservation(task)
    rw = PickupReward(obs_wrapper)
    try:
        assert task.observation_space["observation"].shape == (21,)
        assert obs_wrapper.observation_space["observation"].shape == (21 + N_PICKUP_OBS,)
        obs, _ = rw.reset(seed=0)
        assert obs["observation"].shape == (36,)

        # The appended entries must be the reward's own geometry, not a copy.
        geom = pickup_geometry(rw.unwrapped, rw._pinch_sid, rw.dz)
        np.testing.assert_allclose(obs["observation"][21:], pickup_features(geom),
                                   rtol=0, atol=1e-5)
        assert obs_wrapper.dz == rw.dz

        # ... and the appended block is what it claims to be.
        base = rw.unwrapped
        sid = rw._pinch_sid
        np.testing.assert_allclose(obs["observation"][21:24],
                                   base.data.site_xpos[sid], atol=1e-5)
        np.testing.assert_allclose(
            obs["observation"][27:30],
            base.data.site_xmat[sid].reshape(3, 3)[:, 2], atol=1e-5)
        np.testing.assert_allclose(
            obs["observation"][30:36],
            base.data.ctrl[:N_ARM_ACTUATORS] - base.data.qpos[base._arm_qadr], atol=1e-5)
    finally:
        env.close()


def test_pickup_obs_rejects_a_t2_stack():
    from robochess.tasks import T2ObstacleTask, make_t2_env

    env = make_t2_env(max_episode_steps=50)
    try:
        with pytest.raises(TypeError, match="T1 only"):
            PickupObservation(T2ObstacleTask(env, n_obstacles=1))
    finally:
        env.close()


def test_find_contacts_owner_walks_to_the_task_and_raises_without_one():
    env = RoboChessEnv(max_episode_steps=50)
    task = T1PickAndPlaceTask(env, start_square="e4", target_square="e5")
    try:
        assert find_contacts_owner(PickupObservation(task), "boom") is task
        assert find_contacts_owner(task, "boom") is task
        with pytest.raises(TypeError, match="boom"):
            find_contacts_owner(env, "boom")
    finally:
        env.close()


def test_reward_always_equals_the_sum_of_its_reported_terms(rig):
    """The components in `info` are the reward, not a parallel description."""
    rw, stub = rig
    rng = np.random.default_rng(0)
    for _ in range(12):
        stub.pinched = bool(rng.integers(2))
        stub.board = bool(rng.integers(2))
        stub.grasp = bool(rng.integers(2))
        stub.lift = float(rng.uniform(-0.01, 0.06))
        _r, info = hold(rw, stub)
        assert _r == pytest.approx(total_from_terms(info), abs=1e-9)


# ------------------------------------------------------- held vs pinched (D18)
def _topple_piece_under_closed_pads(task, env):
    """The D18 setup: piece on its side on the board, pads closed across it.

    Copied from test_t1_pick_and_place's toppled-piece case, because the point
    of this test is what MuJoCo's contact list actually holds in that pose.
    """
    import sys
    from pathlib import Path

    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import scripted_expert as se

    ex = se.ScriptedExpert(task)
    m, d = env.model, env.data
    closing = d.xpos[task._left_pad_bid] - d.xpos[task._right_pad_bid]
    closing[2] = 0.0
    closing /= np.linalg.norm(closing)
    axis = np.array([-closing[1], closing[0], 0.0])
    v = np.cross([0.0, 0.0, 1.0], axis)
    qa = env._piece_qadr
    xy = d.qpos[qa:qa + 2].copy()
    d.qpos[qa:qa + 3] = np.array([xy[0], xy[1], env._board_surface_z + 0.015]) - axis * 0.009
    d.qpos[qa + 3:qa + 7] = np.concatenate(
        [[np.cos(np.pi / 4)], np.sin(np.pi / 4) * v / np.linalg.norm(v)])
    d.qvel[env._piece_vadr:env._piece_vadr + 6] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(10):
        task.step(env.home_action())
    centre = d.xpos[env._piece_bid] + d.xmat[env._piece_bid].reshape(3, 3)[:, 2] * 0.009
    for z, grip, budget in ((0.882, 0.0, 25), (centre[2] + 0.014, 0.0, 40),
                            (centre[2] + 0.014, 255.0, 45), (centre[2] + 0.014, 255.0, 30)):
        q = ex.solve_ik(np.array([centre[0], centre[1], z]))
        for _ in ex._drive_to(q, grip, budget):
            pass


def test_toppled_piece_on_the_board_is_pinched_but_not_held():
    """Negative control: D18's false positive must pay nothing.

    Before the fix this pose collected the whole 5.0 first-grasp bonus and about
    1.6 per step, which is the exact case D18's board term exists to exclude.
    """
    env = RoboChessEnv(max_episode_steps=400)
    task = T1PickAndPlaceTask(env, start_square="e4", target_square="e6")
    rw = PickupReward(task)
    rw.reset(seed=0)
    try:
        _topple_piece_under_closed_pads(task, env)
        pinched, held = rw.contact_state()
        left, right, board = task._piece_contacts()
        assert (left and right), "the setup must actually close both pads on the piece"
        assert board, "the toppled piece must still be touching the board"
        assert pinched is True
        assert held is False, "board contact must veto held"
        terms = rw.terms(held)
        assert terms["lift_term"] == 0.0
        assert terms["lift"] > env.rest_tolerance, "D18's 'lifted' term would be satisfied"
        assert rw._held_once is False, "no bonus may have latched"
    finally:
        env.close()


def test_a_genuine_grasp_clear_of_the_board_is_held_and_pays():
    """Positive control: the same board term must not veto a real grasp."""
    import sys
    from pathlib import Path

    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import scripted_expert as se

    env = RoboChessEnv(max_episode_steps=400)
    task = T1PickAndPlaceTask(env, start_square="e4", target_square="e5")
    rw = PickupReward(task)
    seen = []

    class Watch(gym.Wrapper):
        def step(self, action):
            out = self.env.step(action)
            t = out[4]["pickup_terms"]
            seen.append((bool(t["pinched"]), bool(t["held"]), float(t["lift_term"]),
                         float(t["first_grasp_bonus"]), float(t["lift"])))
            return out

    watched = Watch(rw)
    watched.reset(seed=0)
    try:
        se.ScriptedExpert(watched).run_episode()
        held_steps = [r for r in seen if r[1]]
        pinched_on_board = [r for r in seen if r[0] and not r[1]]
        assert held_steps, "the expert must reach a genuinely held state"
        assert all(r[2] > 0.0 or r[4] <= 0.0 for r in held_steps)
        assert sum(r[3] for r in seen) == pytest.approx(5.0), "bonus paid exactly once"
        first_held = next(i for i, r in enumerate(seen) if r[1])
        assert seen[first_held][3] == pytest.approx(5.0), "bonus lands on the first HELD step"
        # every pinched-but-not-held step pays no lift term
        assert all(r[2] == 0.0 for r in pinched_on_board)
    finally:
        env.close()


# --------------------------------------------------- observation step path
def test_pickup_observation_extends_the_observation_on_step_too():
    env = RoboChessEnv(max_episode_steps=50)
    task = T1PickAndPlaceTask(env, start_square="e4", target_square="e5")
    rw = PickupReward(PickupObservation(task))
    try:
        rw.reset(seed=0)
        obs, _r, _t, _tr, _i = rw.step(np.zeros(rw.action_space.shape, dtype=np.float32))
        assert obs["observation"].shape == (36,)
        geom = pickup_geometry(rw.unwrapped, rw._pinch_sid, rw.dz)
        np.testing.assert_allclose(obs["observation"][21:], pickup_features(geom),
                                   rtol=0, atol=1e-5)
    finally:
        env.close()


# ------------------------------------------------------------ defensive raises
class _NotARoboChessEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, (1,))
    action_space = gym.spaces.Box(-1.0, 1.0, (1,))


@pytest.mark.parametrize("wrapper", [PickupObservation, PickupReward])
def test_wrappers_reject_a_non_robochess_base(wrapper):
    with pytest.raises(TypeError, match="Expects a RoboChessEnv underneath"):
        wrapper(_NotARoboChessEnv())


def test_pickup_observation_rejects_a_stack_with_no_t1_task():
    env = RoboChessEnv(max_episode_steps=50)
    try:
        with pytest.raises(TypeError, match="needs a T1PickAndPlaceTask"):
            PickupObservation(env)
    finally:
        env.close()
