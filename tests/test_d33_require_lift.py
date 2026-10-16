"""D33: success requires a real lift (place_require_lift, default off).

The lift latch lives in the env (RoboChessEnv: `_lift_latched`, gating `_release_ok`), so S2 entry, the placed
bonus, the success lump and the success flag (is_success and terminated_reason "success") all change together.
PlaceReward additionally pays its one-time lift bonus at the latch instead of at 30 mm held for 5 steps.

The scenarios are real-env ones: a real grasp and lift (Cartesian actions, closed gripper) sets the latch for
real; the placement itself is staged the way the D27/D28/D32 tests stage it (prime a carry-and-release, teleport
the piece onto the target), because a trained policy is not available to a test.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robochess.experiment import ConfigError, load_config, resolve_config
from robochess.tasks.t1_pick_and_place import PLACE_GAMMA, PlaceReward

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tests"))
from _d27_helpers import prime_carried_release  # noqa: E402

_spec = importlib.util.spec_from_file_location("train", _ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

from robochess.tasks import t1_pick_and_place as t1  # noqa: E402

CONTROL = load_config(_ROOT / "configs" / "d32_t1_e4e5_seed0.yaml")
ZG = t1.PINCH_GRASP_Z


def cfg(**over):
    return resolve_config(dict(CONTROL, **over), config_path="t.yaml")


def on(**over):
    return cfg(place_require_lift=True, **over)


# ------------------------------------------------------------------ config
def test_flag_defaults_off_and_the_control_config_is_unchanged():
    c = cfg()
    assert c["place_require_lift"] is False and c["place_lift_latch_m"] == 0.020


def test_flag_on_resolves_with_the_default_latch():
    c = on()
    assert c["place_require_lift"] is True and c["place_lift_latch_m"] == 0.020
    assert on(place_lift_latch_m=0.03)["place_lift_latch_m"] == 0.03


@pytest.mark.parametrize("over, match", [
    (dict(place_require_lift="yes"), "must be a bool"),
    (dict(place_require_lift=True, place_lift_latch_m=0.0), "number > 0"),
    (dict(place_require_lift=True, place_lift_latch_m=-0.01), "number > 0"),
    (dict(place_require_lift=True, place_lift_latch_m=True), "number > 0"),
    (dict(place_lift_latch_m=0.02), "would do nothing"),
    (dict(place_require_lift=True, place_mode=False, place_obs=False, pickup_obs=False,
          place_posture_gate=False, place_carry_potential=False), "needs place_mode"),
])
def test_invalid_flag_configs_are_rejected(over, match):
    with pytest.raises(ConfigError, match=match):
        cfg(**over)


# ------------------------------------------------------------------ rig
class Rig:
    """A full-stack env (train.build_env) with step logging."""

    def __init__(self, **over):
        self.cfg = cfg(**over)
        self.env = train.build_env(self.cfg)
        n = self.env
        while not hasattr(n, "_piece_contacts") and n is not None:
            n = getattr(n, "env", None)
        self.task = n
        self.base = self.env.unwrapped
        self.rewards, self.infos = [], []
        self.stage_idx = None

    def close(self):
        self.env.close()

    def reset(self):
        self.rewards, self.infos, self.stage_idx = [], [], None
        return self.env.reset(seed=0)

    def step(self, a):
        obs, r, te, tr, info = self.env.step(np.asarray(a, dtype=np.float32))
        self.rewards.append(float(r))
        self.infos.append(info)
        return r, te, tr, info

    def pinch(self):
        return self.base.data.site_xpos[self.base._pinch_sid].copy()

    def drive(self, target, grip, n=200, tol=5e-4):
        for _ in range(n):
            err = np.asarray(target) - self.pinch()
            if np.linalg.norm(err) < tol:
                return
            out = self.step(np.concatenate([np.clip(err / self.base.cartesian_step, -1, 1), [grip]]))
            if out[1] or out[2]:
                raise AssertionError(f"episode ended while driving: {out[3].get('terminated_reason')}")

    def grasp(self):
        """Reset, go over e4 at grasp height with the gripper open, close on the piece."""
        self.reset()
        xy = self.task._board_world_xy("e4")
        self.drive(np.array([*xy, ZG + 0.08]), -1.0)
        self.drive(np.array([*xy, ZG]), -1.0)
        for _ in range(7):
            self.step([0, 0, 0, +1.0])
        return xy

    def lift(self, height_m, n=60):
        """Raise the pinch slowly (action 0.25, about 5 mm a step) with the gripper closed until the piece
        is `height_m` up (or n steps), so the height is sampled finely around the latch."""
        for _ in range(n):
            if self.infos[-1]["piece_height_above_board"] >= height_m:
                return
            self.step([0, 0, 0.25, 1.0])

    def stage_placement(self):
        """Prime a carry-and-release at 0 mm height / 0 mm error and put the piece on the target."""
        self.stage_idx = len(self.infos)
        prime_carried_release(self.env, xy_error=0.0, height=0.0)
        self.task._teleport_piece("e5")
        mujoco.mj_forward(self.base.model, self.base.data)
        self.base._refresh_contact_flags()

    def run_out(self, n=400):
        for _ in range(n):
            r, te, tr, info = self.step(self.base.home_action())
            if te or tr:
                return info
        raise AssertionError("the episode did not end")


def discounted(rewards):
    return float(sum(PLACE_GAMMA ** i * r for i, r in enumerate(rewards)))


# ------------------------------------------------------------------ flag off is unchanged
def test_flag_off_builds_the_old_env_and_adds_no_keys():
    rig = Rig()
    try:
        assert rig.base.require_lift is False
        obs, info = rig.reset()
        assert obs["observation"].shape == (39,) and "lift_latched" not in info
        _, _, _, info = rig.step([0, 0, 0, 0])
        assert "lift_latched" not in info and "lift_latched" not in info["place_terms"]
    finally:
        rig.close()


def test_place_reward_and_env_flags_must_agree():
    rig = Rig()
    try:
        inner = rig.env.env   # PlaceObservation
        with pytest.raises(ValueError, match="build both from the same place_require_lift"):
            PlaceReward(inner, require_lift=True)
    finally:
        rig.close()
    rig = Rig(place_require_lift=True)
    try:
        assert rig.base.require_lift is True and rig.base.lift_latch_m == 0.020
        with pytest.raises(ValueError, match="build both from the same place_require_lift"):
            PlaceReward(rig.env.env, require_lift=False)
        assert rig.reset()[0]["observation"].shape == (39,)
    finally:
        rig.close()


def test_latch_threshold_comes_from_the_config():
    rig = Rig(place_require_lift=True, place_lift_latch_m=0.03)
    try:
        assert rig.base.lift_latch_m == 0.03
    finally:
        rig.close()


# ------------------------------------------------------------------ the latch
def test_latch_needs_both_pads_on_the_piece_not_just_height():
    rig = Rig(place_require_lift=True)
    try:
        rig.reset()
        qa = rig.base._piece_qadr
        rig.base.data.qpos[qa + 2] += 0.060            # 60 mm up in free air, no pads on it
        mujoco.mj_forward(rig.base.model, rig.base.data)
        _, _, _, info = rig.step([0, 0, 0, -1.0])
        assert info["piece_height_above_board"] >= 0.020      # it is high enough...
        assert info["lift_latched"] is False                  # ...but nothing is pinching it
    finally:
        rig.close()


def test_latch_sets_once_at_20mm_while_pinched_persists_and_clears_at_reset():
    rig = Rig(place_require_lift=True)
    try:
        rig.grasp()
        assert rig.infos[-1]["lift_latched"] is False            # pinched at rest height: not lifted
        rig.lift(0.045)
        seq = [(i["piece_height_above_board"], i["lift_latched"]) for i in rig.infos]
        first = next(k for k, (h, l) in enumerate(seq) if l)
        assert seq[first][0] >= 0.020, seq[first]                # set only once the piece is 20 mm up
        assert all(not l for h, l in seq[:first])
        assert all(l for h, l in seq[first:])                    # then stays set, even if the piece comes down
        assert max(h for h, l in seq[:first]) < 0.020
        # lower the piece again: the latch does not clear
        rig.drive(np.array([*rig.task._board_world_xy("e4"), ZG]), 1.0)
        assert rig.infos[-1]["lift_latched"] is True
        rig.reset()
        assert rig.base._lift_latched is False                   # cleared only by reset
    finally:
        rig.close()


# ------------------------------------------------------------------ (b) a dragged placement
def _dragged(rig):
    rig.grasp()                                                  # pinched at rest height: never lifted
    assert not rig.base._lift_latched
    rig.stage_placement()
    return rig.run_out()


def test_dragged_placement_earns_nothing_with_the_flag_on():
    rig = Rig(place_require_lift=True, max_episode_steps=120)
    try:
        final = _dragged(rig)
        staged = rig.infos[rig.stage_idx:]                       # the steps after staging
        assert len(staged) > 5
        assert not any(i["release_ok"] for i in staged)
        assert not any(i["place_state"] == "S2" for i in staged)           # no S2 reward
        assert not any(i["place_terms"]["placed_bonus"] > 0 for i in staged)
        assert not any(i["place_terms"]["success_lump"] > 0 for i in staged)
        assert not any(i["is_success"] for i in staged)
        assert final["terminated_reason"] != "success"
        assert {i["place_state"] for i in staged} <= {"S0", "S1", "S3"}
    finally:
        rig.close()


def test_the_same_dragged_placement_is_a_success_with_the_flag_off():
    """The control for the test above: without the flag this staged placement terminates in success."""
    rig = Rig(max_episode_steps=120)
    try:
        final = _dragged(rig)
        assert final["terminated_reason"] == "success" and final["is_success"]
        staged = rig.infos[rig.stage_idx:]
        assert any(i["place_state"] == "S2" for i in staged)
        assert any(i["place_terms"]["placed_bonus"] > 0 for i in staged)
        assert any(i["place_terms"]["success_lump"] > 0 for i in staged)
    finally:
        rig.close()


# ------------------------------------------------------------------ (c) a lifted placement
def test_lifted_placement_earns_the_bonus_s2_the_bonus_the_lump_and_the_success():
    rig = Rig(place_require_lift=True)
    try:
        rig.grasp()
        rig.lift(0.030)
        assert rig.infos[-1]["lift_latched"] is True
        rig.stage_placement()
        final = rig.run_out()
        terms = [i["place_terms"] for i in rig.infos]
        assert final["terminated_reason"] == "success" and final["is_success"] is True
        assert any(i["place_state"] == "S2" for i in rig.infos)                  # S2 entered
        assert sum(t["placed_bonus"] > 0 for t in terms) == 1                    # the placed bonus, once
        assert sum(t["success_lump"] > 0 for t in terms) == 1 and terms[-1]["success_lump"] == pytest.approx(1020.0)
        # the one-time lift bonus (+5) paid exactly once, on the step the latch was set
        paid = [k for k, t in enumerate(terms) if t["lift_bonus"] > 0]
        assert len(paid) == 1 and terms[paid[0]]["lift_bonus"] == 5.0
        latch_step = next(k for k, i in enumerate(rig.infos) if i["lift_latched"])
        assert paid[0] == latch_step
        assert rig.infos[latch_step]["piece_height_above_board"] >= 0.020
    finally:
        rig.close()


def test_the_lift_bonus_is_not_paid_at_30mm_held_for_5_steps_when_the_flag_is_on():
    """With the flag on the bonus follows the latch (20 mm), so it is paid before the piece reaches 30 mm."""
    rig = Rig(place_require_lift=True)
    try:
        rig.grasp()
        rig.lift(0.030)
        terms = [i["place_terms"] for i in rig.infos]
        k = next(k for k, t in enumerate(terms) if t["lift_bonus"] > 0)
        h = rig.infos[k]["piece_height_above_board"]
        assert 0.020 <= h < 0.030, h        # paid at the latch; the D21 rule (30 mm held 5 steps) cannot have fired yet
    finally:
        rig.close()


# ------------------------------------------------------------------ (d) drag-and-wait vs lift-and-finish
def test_drag_and_wait_scores_below_lift_and_finish_over_300_steps_discounted():
    drag = Rig(place_require_lift=True, max_episode_steps=300)
    lift = Rig(place_require_lift=True, max_episode_steps=300)
    drag_off = Rig(max_episode_steps=300)
    try:
        _dragged(drag)
        lift.grasp()
        lift.lift(0.030)
        lift.stage_placement()
        lift.run_out()
        _dragged(drag_off)
        g_drag, g_lift, g_off = discounted(drag.rewards), discounted(lift.rewards), discounted(drag_off.rewards)
        print(f"\ndiscounted return (gamma {PLACE_GAMMA}): drag-and-wait, flag on = {g_drag:.2f} over {len(drag.rewards)} steps; "
              f"lift-and-finish, flag on = {g_lift:.2f} over {len(lift.rewards)} steps; "
              f"drag-and-wait, flag OFF = {g_off:.2f} over {len(drag_off.rewards)} steps")
        assert len(drag.rewards) == 300 and drag.infos[-1]["terminated_reason"] is None
        assert g_drag < g_lift
        assert g_off > g_drag            # without the flag the drag is paid
    finally:
        for r in (drag, lift, drag_off):
            r.close()


# ------------------------------------------------------------------ (e) what a dragged placement pays
def test_print_what_the_env_pays_and_how_it_terminates_at_a_dragged_placement():
    rows, stage_idx = {}, {}
    for label, over in (("flag ON ", dict(place_require_lift=True)), ("flag OFF", dict())):
        rig = Rig(max_episode_steps=200, **over)
        try:
            final = _dragged(rig)
            rows[label] = (final, list(rig.rewards), list(rig.infos))
            stage_idx[label] = rig.stage_idx
        finally:
            rig.close()
    for label, (final, rewards, infos) in rows.items():
        print(f"\n--- dragged placement, {label}: {len(rewards)} steps, ends with terminated_reason={final['terminated_reason']!r}, "
              f"is_success={final['is_success']}, release_ok={final['release_ok']}, place_state={final['place_state']}")
        print("  step: reward  state release_ok is_success placed_bonus success_lump reason  (steps after staging)")
        for k in range(stage_idx[label], min(len(rewards), stage_idx[label] + 6)):
            i = infos[k]
            print(f"  {k+1:4d}: {rewards[k]:8.2f}  {i['place_state']}   {str(i['release_ok']):5s}  {str(i['is_success']):5s}   "
                  f"{i['place_terms']['placed_bonus']:5.1f}  {i['place_terms']['success_lump']:7.1f}  {i['terminated_reason']}")
        print(f"  total reward over the episode: {sum(rewards):.2f}")
    on_final = rows["flag ON "][0]
    assert on_final["terminated_reason"] != "success" and not on_final["is_success"]
    assert rows["flag OFF"][0]["terminated_reason"] == "success"
