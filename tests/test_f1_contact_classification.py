"""F1 (referee review 2026-09-21): the collision detector misses most robot
collision shapes, excludes the board entirely, and has no self-contact
category at all.

`_bad_contact()` classifies robot geoms by a GEOM-NAME `ur_` prefix, which
only 4 of the 23 collidable robot geoms match (the finger pads -- every other
collision geom on the model is unnamed). `_forbidden_geoms` is table/pedestal/
floor only; the board slab is never checked, at any force. There is no
self-contact category.

Round 1 measured all four gaps directly (RoboChess_Code_Review_Report,
section-4/section-1 probes this session): upper-arm/forearm penetrating the
table by 25-45mm with `_bad_contact()` reading False; zero robot self-contact
in 10 regression episodes (nothing to guard, but also nothing built to guard
it); a1->h8's real 166N pad-board contact invisible to the detector because
the board isn't a forbidden geom at all, regardless of force.

Fixtures here reproduce round 1's own scenarios exactly (same seeds/qpos) so
the numbers in these tests are the same ones already reported, not new ones
invented for the test file.
"""

from __future__ import annotations

import numpy as np
import pytest
import mujoco

from robochess.envs import RoboChessEnv
from robochess.tasks import T1PickAndPlaceTask

BOARD_FORCE_CEILING_N = 250.0  # approved in round 2; see round-1 section-4/5 measurements


def _robot_and_board_geoms(model):
    """Ancestry classification (section 1): every geom owned by a body
    descending from the robot's base body, and every geom owned by the board
    body. Not restricted to the named pad geoms -- this is the SAME
    classification the fix uses, so the test measures contact the same way
    `_bad_contact()` will."""
    root = int(model.jnt_bodyid[0])
    while int(model.body_parentid[root]) != 0:
        root = int(model.body_parentid[root])

    def under(bid: int) -> bool:
        while bid != 0:
            if bid == root:
                return True
            bid = int(model.body_parentid[bid])
        return False

    robot_bodies = {i for i in range(model.nbody) if under(i)}
    board_bid = next(
        i for i in range(model.nbody)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) == "board"
    )
    robot_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in robot_bodies}
    board_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == board_bid}
    return robot_geoms, board_geoms


def _combined_robot_board_force(env) -> float:
    """Combined total: the sum of contact-normal force over EVERY simultaneous
    robot-geom-to-board contact at this instant, across all robot geoms (not
    just the pads) and all contact points -- the convention decided for the
    fix (see the divergence-from-`_pad_piece_contacts` comment in
    `robochess_env.py`): the ceiling is about how hard the robot presses on
    the board as a whole, not any single pad or contact point. A per-pad split
    would let two pads each stay under the ceiling while jointly exceeding it.
    """
    m, d = env.model, env.data
    robot_geoms, board_geoms = _robot_and_board_geoms(m)
    total = 0.0
    w = np.zeros(6)
    for k in range(d.ncon):
        c = d.contact[k]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in robot_geoms and g2 in board_geoms) or (g2 in robot_geoms and g1 in board_geoms):
            mujoco.mj_contactForce(m, d, k, w)
            total += abs(float(w[0]))
    return total


def _pad_pad_selftouch(env) -> bool:
    m, d = env.model, env.data
    for k in range(d.ncon):
        c = d.contact[k]
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)) or ""
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)) or ""
        if ("left_pad" in n1 and "right_pad" in n2) or ("right_pad" in n1 and "left_pad" in n2):
            return True
    return False


# --------------------------------------------------------- 1. unclassified geom
def test_unclassified_robot_geom_penetrating_table_is_currently_invisible():
    """Round 1's exact reproduction: seed 0, then the arm driven to
    `rng(22).uniform(_arm_target_low, _arm_target_high)` -- the same
    construction the review's diagnostic harness used. Upper-arm and forearm
    (unnamed collision geoms, no `ur_` prefix on the geom itself) penetrate the
    table by 25-45mm. `_bad_contact()` must see this; it currently does not,
    because those geoms are not in `_arm_geoms`.
    """
    env = RoboChessEnv()
    try:
        env.reset(seed=0)
        env.data.qpos[:6] = np.random.default_rng(22).uniform(
            env._arm_target_low, env._arm_target_high
        )
        mujoco.mj_forward(env.model, env.data)

        # The real, physical contact is there, unmistakably -- this isn't a
        # borderline case.
        penetrations = []
        for k in range(env.data.ncon):
            c = env.data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in env._forbidden_geoms or g2 in env._forbidden_geoms:
                penetrations.append(float(c.dist) * 1000)
        assert penetrations, "fixture must actually produce table/pedestal contact"
        assert min(penetrations) < -20.0, f"expected >20mm penetration, got {penetrations}"

        assert env._bad_contact() is True, (
            "a geom penetrating the table by "
            f"{abs(min(penetrations)):.1f}mm must count as bad contact, "
            "regardless of whether its geom NAME happens to start with 'ur_'"
        )
    finally:
        env.close()


# ------------------------------------------------------------- 2. self-contact
def test_non_adjacent_self_contact_is_currently_undetected():
    """A random arm pose (seed 0 on the ctrl-range uniform draw) folds the
    shoulder link into the forearm link -- ~27mm of overlap between two links
    three joints apart in the kinematic chain, nowhere near the gripper's
    parallel-linkage `<exclude>` pairs and not parent-child (so MuJoCo's
    default filtering does not and should not suppress it; section 1.3
    confirmed empirically that no robot self-contact of any kind, adjacent or
    not, occurs in 10 real regression episodes). There is currently no
    self-contact category in `_bad_contact()` at all.
    """
    env = RoboChessEnv()
    try:
        env.reset(seed=0)
        env.data.qpos[:6] = np.random.default_rng(0).uniform(
            env._ctrl_low[:6], env._ctrl_high[:6]
        )
        mujoco.mj_forward(env.model, env.data)

        m = env.model
        root = int(m.jnt_bodyid[0])
        while int(m.body_parentid[root]) != 0:
            root = int(m.body_parentid[root])

        def under(bid):
            while bid != 0:
                if bid == root:
                    return True
                bid = int(m.body_parentid[bid])
            return False

        self_hits = []
        for k in range(env.data.ncon):
            c = env.data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if under(b1) and under(b2) and b1 != b2:
                self_hits.append((b1, b2, float(c.dist) * 1000))
        assert self_hits, "fixture must actually produce robot self-contact"
        assert min(h[2] for h in self_hits) < -20.0, f"expected >20mm overlap, got {self_hits}"

        assert env._bad_contact() is True, (
            f"non-adjacent robot self-contact ({self_hits}) must count as bad "
            "contact; there is currently no code path that can ever detect it"
        )
    finally:
        env.close()


# --------------------------------------------------- 3 & 4. pad-board ceiling
def _drive_pad_into_board(depth_mm: float, grip_ctrl: float, steps: int):
    env = RoboChessEnv(max_episode_steps=400)
    task = T1PickAndPlaceTask(env, start_square="e4", target_square="e5")
    import sys
    scripts_dir = __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import scripted_expert as se

    expert = se.ScriptedExpert(task)
    task.reset(seed=0)
    goal = env._desired_goal.astype(float)
    target = np.array([goal[0], goal[1], env._board_surface_z - depth_mm / 1000.0])
    q_arm = expert.solve_ik(target)
    for _ in range(steps):
        task.step(expert._to_action(q_arm, grip_ctrl))
    return env, task


def test_pad_board_contact_under_ceiling_is_not_flagged():
    """5mm of pinch-target penetration, a light (30/255) closing command: the
    pad presses the board slab at up to 81.8N over the whole run (measured),
    22.4N COMBINED across both simultaneous contact points at the step checked
    here -- comfortably under the 250N ceiling, and no pad-pad self-touch
    (this is a genuine tolerable placement-adjacent contact, not the
    self-contact case above).

    Combined total, not per-pad (decided convention, see
    `_combined_robot_board_force`): even split two ways this scenario is
    nowhere near the ceiling, so the convention choice doesn't matter for this
    particular fixture, but the measurement uses the decided convention
    throughout for consistency with the over-ceiling test and the fix.

    This assertion already holds against the CURRENT code, but for the wrong
    reason: today `_bad_contact()` never looks at the board at all, at any
    force, so "not flagged" is vacuous here rather than a real tolerance
    decision. It exists as a regression guard for the fix (section 3), not as
    evidence of today's gap -- section 2's report says so explicitly.
    """
    env, task = _drive_pad_into_board(depth_mm=5.0, grip_ctrl=30, steps=40)
    try:
        force = _combined_robot_board_force(env)
        assert 0.0 < force < BOARD_FORCE_CEILING_N, f"fixture force {force:.1f}N not in the intended under-ceiling range"
        assert not _pad_pad_selftouch(env), "fixture must not also trip pad-pad self-contact"

        assert env._bad_contact() is False, (
            f"a {force:.1f}N combined robot-board contact (< {BOARD_FORCE_CEILING_N:.0f}N ceiling) "
            "must be tolerated, not flagged"
        )
    finally:
        task.close()


def test_pad_board_contact_over_ceiling_is_flagged():
    """40mm of pinch-target penetration, a moderate (150/255) closing command:
    settles at two SIMULTANEOUS pad-board contacts -- right pad 200.62N, left
    pad 147.06N, measured separately -- neither of which alone exceeds 250N.
    Combined total: 347.68N, comfortably over the ceiling. No pad-pad
    self-touch, so this is unambiguously a board-force case, not the
    self-contact case above.

    This fixture is exactly why combined-total (not per-pad) is the decided
    convention: a per-pad check would NOT flag this scenario at all (200.62N
    and 147.06N both individually pass a 250N per-pad gate), even though the
    robot is jointly pressing on the board with 347.68N -- which is precisely
    the "two pads each stay under the ceiling while jointly exceeding it"
    failure mode the combined-total decision exists to close.
    """
    env, task = _drive_pad_into_board(depth_mm=40.0, grip_ctrl=150, steps=60)
    try:
        force = _combined_robot_board_force(env)
        assert force > BOARD_FORCE_CEILING_N, f"fixture force {force:.1f}N not comfortably over the ceiling"
        assert not _pad_pad_selftouch(env), "fixture must not also trip pad-pad self-contact (would confound this test)"

        assert env._bad_contact() is True, (
            f"a {force:.1f}N combined robot-board contact (> {BOARD_FORCE_CEILING_N:.0f}N ceiling) must be flagged bad; "
            "the current code never checks the board at all, at any force"
        )
    finally:
        task.close()
