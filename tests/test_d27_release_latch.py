"""D27: placement success requires the piece to have been carried (held) and
released on target from at most PLACEMENT_RELEASE_MAX_HEIGHT (15 mm).

Scenarios are real Cartesian-mode runs (tests/d27_scenarios.py); nothing here
patches the task wrapper's `_piece_contacts`, which would not reach the
base-env latch. The two fake-contact tests patch at the base-env level.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import d27_scenarios as sc
from robochess.envs import RoboChessEnv
from robochess.envs.robochess_env import PLACEMENT_RELEASE_MAX_HEIGHT
from robochess.tasks import T1PickAndPlaceTask, T2ObstacleTask, make_t2_env

# Every success term except release_ok (used to prove release_ok is the ONLY
# failing term in the negative scenarios).
OTHER_TERMS = ("piece_on_square", "piece_resting", "gripper_released",
               "piece_settled", "piece_still", "piece_upright")
HEIGHT_TOL_MM = 3.0


def _others_hold(info):
    return {k: info[k] for k in OTHER_TERMS}


def _carried(start, target, height_m, **kw):
    dr = sc.Driver(start, target)
    dr.grasp_and_lift()
    dr.place_and_release(height_m=height_m, **kw)
    return dr


def test_constant_is_locked_at_15mm():
    assert PLACEMENT_RELEASE_MAX_HEIGHT == 0.015


# ------------------------------------------------------------ push, no grasp
def test_push_then_open_is_not_success_e4_e5():
    dr = sc.Driver("e4", "e5")
    sc.push_then_open(dr)
    info = dr.trace.last
    assert all(_others_hold(info).values()), _others_hold(info)  # F2's other terms are met
    assert info["ever_held"] is False
    assert info["release_ok"] is False
    assert info["is_success"] is False


# ------------------------------------------------------------------- drops
@pytest.mark.parametrize("height_mm,expected", [(10.0, True), (30.0, False)])
def test_drop_success_follows_recorded_release_height(height_mm, expected):
    dr = _carried("c1", "f6", height_mm / 1000.0)
    info = dr.trace.last
    assert info["ever_held"] is True
    assert info["release_height_mm"] == pytest.approx(height_mm, abs=HEIGHT_TOL_MM)
    assert all(_others_hold(info).values()), _others_hold(info)
    assert info["release_ok"] is expected
    assert info["is_success"] is expected


def test_clean_place_is_success():
    dr = _carried("c1", "f6", 0.0015)
    info = dr.trace.last
    assert info["release_height_mm"] == pytest.approx(1.5, abs=HEIGHT_TOL_MM)
    assert info["ever_held"] and info["release_ok"] and info["is_success"] is True
    assert info["terminated_reason"] == "success"


# --------------------------------------------------------------- fast set-down
def _fast(hover_m):
    dr = sc.Driver("c1", "f6")
    dr.grasp_and_lift()
    sc.fast_set_down(dr, hover_above_m=hover_m)
    return dr


def test_fast_set_down_at_max_cartesian_step_is_success():
    dr = _fast(0.03)
    info = dr.trace.last
    assert info["release_height_mm"] <= 5.0
    assert info["release_height_mm"] >= -HEIGHT_TOL_MM  # pressed into the board, not through it
    assert info["release_ok"] is True
    assert info["is_success"] is True


@pytest.mark.parametrize("hover_m", [0.015, 0.03, 0.06])
def test_fast_set_down_release_is_low_at_every_hover_height(hover_m):
    """release_ok is about the recorded height, whatever the piece does next: the
    board stops the descent, so the record is ~0 regardless of where it began."""
    info = _fast(hover_m).trace.last
    assert -HEIGHT_TOL_MM <= info["release_height_mm"] <= 5.0
    assert info["release_ok"] is True


# ------------------------------------------------------------------ re-grasp
def test_regrasp_overwrites_an_off_target_release():
    dr = sc.Driver("c1", "f6")
    dr.grasp_and_lift()
    sc.regrasp(dr)
    mid = dr.mid
    # after the first release the piece was far from its target: the record said so
    assert mid["release_xy_error_mm"] > 100.0 and mid["release_ok"] is False
    assert mid["is_success"] is False and mid["ever_held"] is True
    final = dr.trace.last
    assert final["release_xy_error_mm"] < 30.0  # overwritten by the on-target release
    assert final["release_ok"] is True and final["is_success"] is True


# --------------------------------------------------------------------- reset
def test_reset_clears_ever_held_and_the_release_record():
    dr = _carried("c1", "f6", 0.0015)
    base = dr.base
    assert base._ever_held is True and base._release_record is not None
    _, info = dr.env.reset(seed=0)
    assert base._ever_held is False and base._release_record is None
    assert info["ever_held"] is False
    assert info["release_height_mm"] is None and info["release_xy_error_mm"] is None
    assert info["release_ok"] is False


# ---------------------------------------- control: the teleport setup, unprimed
def test_teleport_and_step_without_priming_is_not_success():
    """The setup the existing success tests use (teleport the piece onto the goal,
    gripper open), WITHOUT prime_carried_release: every other term holds and
    release_ok is the term that fails."""
    env = RoboChessEnv()
    env.reset(seed=0)
    q = env._piece_qadr
    env.data.qpos[q : q + 3] = [*env._desired_goal[:2], env._board_surface_z]
    env.data.qvel[env._piece_vadr : env._piece_vadr + 6] = 0.0
    env.data.qpos[env._grip_qadr] = 0.0
    for _ in range(3):
        _, _, _, _, info = env.step(env.home_action())
    assert all(_others_hold(info).values()), _others_hold(info)
    assert info["release_ok"] is False and info["is_success"] is False
    env.close()


# ----------------------------------------------------- pinned contact flags
@pytest.mark.parametrize("mode", ["absolute", "delta", "cartesian"])
def test_contact_flags_at_reset_and_after_the_first_hold_step(mode):
    """Verified equal to the pre-D27 `_piece_contacts()` on ff7ea92. At home the
    piece sits at exactly 0 mm and MuJoCo reports no board contact until the
    first step."""
    base = RoboChessEnv(action_mode=mode)
    task = T1PickAndPlaceTask(base)
    task.reset(seed=0)
    assert task._piece_contacts() == (False, False, False)
    hold = base.home_action() if mode == "absolute" else np.zeros(base.action_space.shape, np.float32)
    task.step(hold)
    assert task._piece_contacts() == (False, False, True)
    base.close()


def test_unstable_step_leaves_flags_latch_and_record_at_their_last_valid_values(monkeypatch):
    """Behaviour change on invalid transitions only: nothing is read from a state the
    solver guard declared corrupt (the returned observation is the last valid one,
    so last-valid flags are consistent with it)."""
    env = RoboChessEnv()
    env.reset(seed=0)
    env.step(env.home_action())
    assert env._contact_flags == (False, False, True)
    before = (env._contact_flags, env._ever_held, env._release_record)

    # A fresh scan would now say "pinched and held". Fake it at the BASE level.
    monkeypatch.setattr(env, "_piece_contact_flags", lambda: (True, True, False))
    monkeypatch.setattr(env, "_solver_instability", lambda: True)
    _, _, _, truncated, info = env.step(env.home_action())
    assert info["invalid_transition"] is True and truncated is True
    assert (env._contact_flags, env._ever_held, env._release_record) == before
    assert info["ever_held"] is False and info["release_height_mm"] is None

    # control: the same fake on a valid step DOES latch, so the fake is live
    monkeypatch.setattr(env, "_solver_instability", lambda: False)
    env.step(env.home_action())
    assert env._contact_flags == (True, True, False)
    assert env._ever_held is True and env._release_record is not None
    env.close()


def _t1():
    return T1PickAndPlaceTask(RoboChessEnv())


def _t2():
    return T2ObstacleTask(make_t2_env(), n_obstacles=2)


@pytest.mark.parametrize("make", [_t1, _t2])
@pytest.mark.parametrize("seed,opts", [(0, {}), (1, {"start_square": "b7", "target_square": "g2"})])
def test_wrapper_flags_after_reset_match_a_fresh_scan(make, seed, opts):
    """The T1/T2 reset() teleports the piece (and places obstacles) after the base
    env's own reset scan, so they refresh the flags; the wrapper must not read a
    scan of the pre-teleport state."""
    task = make()
    task.reset(seed=seed, options=opts)
    assert task._piece_contacts() == task.unwrapped._piece_contact_flags()
    task.unwrapped.close()


# -------------------------------------- info["is_success"] == _is_success()
def _count_mismatches(env_like, base):
    mism, n = [], [0]
    orig = env_like.step

    def checked(a):
        out = orig(a)
        n[0] += 1
        mism.append(bool(out[4]["is_success"]) != base._is_success())
        return out

    env_like.step = checked
    return mism, n


def test_info_is_success_equals__is_success_on_every_step_joint_replay():
    import scripted_expert as se
    base = RoboChessEnv(max_episode_steps=350)
    task = T1PickAndPlaceTask(base, start_square="e4", target_square="e5")
    mism, n = _count_mismatches(task, base)
    rec = se.ScriptedExpert(task).run_episode(seed=0)
    assert n[0] > 100 and sum(mism) == 0, (n[0], sum(mism))
    assert rec["success"] is True  # the expert's own carry-and-place still succeeds
    base.close()


def test_info_is_success_equals__is_success_on_every_step_cartesian_replay():
    dr = sc.Driver("c1", "f6")
    mism, n = _count_mismatches(dr.env, dr.base)
    dr.grasp_and_lift()
    dr.place_and_release(height_m=0.0015)
    assert n[0] > 100 and sum(mism) == 0, (n[0], sum(mism))
    assert dr.trace.last["is_success"] is True
