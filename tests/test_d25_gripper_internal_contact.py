"""D25: a contact between the gripper's left and right finger subtrees (closing
on nothing puts pad1 on pad1) is not a referee violation; it is reported in
info["gripper_self_contact"]. Every other robot-to-robot contact still
terminates."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robochess.envs import RoboChessEnv

REPO = Path(__file__).resolve().parents[1]

# Found by sampling arm poses (seed 0, first sample): upper arm through
# ur_gripper_base_mount, a 32 mm penetration -- an arm-to-gripper contact.
ARM_TO_GRIPPER_QPOS = [0.8606, -1.4465, -2.8841, -3.0377, 1.9683, 2.5934]


def _close_gripper(env: RoboChessEnv, steps: int):
    action = env.home_action()
    action[-1] = 1.0
    last = None
    for _ in range(steps):
        last = env.step(action)
    return last


@pytest.mark.parametrize("mode", ["absolute", "delta", "cartesian"])
def test_closing_on_air_does_not_terminate(mode):
    env = RoboChessEnv(action_mode=mode, max_episode_steps=100)
    env.reset(seed=0)
    seen_flag = False
    action = env.home_action()
    action[-1] = 1.0
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated, info["terminated_reason"]
        assert info["bad_contact"] is False
        seen_flag |= info["gripper_self_contact"]
    assert seen_flag and info["gripper_self_contact"]
    aperture = float(env.data.qpos[env._grip_qadr])
    assert aperture == pytest.approx(0.78, abs=0.01)


def test_flag_present_and_false_when_open():
    env = RoboChessEnv(max_episode_steps=10)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.home_action())
    assert info["gripper_self_contact"] is False


def test_arm_to_gripper_contact_still_terminates():
    env = RoboChessEnv(action_mode="delta", max_episode_steps=10)
    env.reset(seed=0)
    env.data.qpos[:6] = ARM_TO_GRIPPER_QPOS
    mujoco.mj_forward(env.model, env.data)
    m = env.model
    pairs = {
        (int(c.geom1), int(c.geom2)) for c in env.data.contact[: env.data.ncon]
        if int(m.geom_bodyid[c.geom1]) != int(m.geom_bodyid[c.geom2])
    }
    assert any(not env._is_gripper_internal(a, b) for a, b in pairs), "fixture no longer touches"
    assert env._bad_contact() is True
    assert env._gripper_self_contact() is False


def test_exemption_is_only_left_against_right():
    env = RoboChessEnv(max_episode_steps=10)
    left, right = env._gripper_left_geoms, env._gripper_right_geoms
    assert left and right and not (left & right)
    # The sets hold every geom on the subtree, visual ones included (contype 0,
    # they never appear in data.contact); 6 per side actually collide.
    m = env.model
    collidable = lambda gs: {g for g in gs if m.geom_contype[g] or m.geom_conaffinity[g]}
    assert len(collidable(left)) == len(collidable(right)) == 6
    for a in left:
        for b in right:
            assert env._is_gripper_internal(a, b) and env._is_gripper_internal(b, a)
    # same side, and gripper base / arm geoms, are not exempt
    a1, a2 = sorted(left)[:2]
    assert not env._is_gripper_internal(a1, a2)
    others = env._robot_geoms - left - right
    assert others
    assert not any(env._is_gripper_internal(o, r) for o in others for r in right)
    assert not any(env._is_gripper_internal(o, l) for o in others for l in left)


def test_monitor_logs_the_flag():
    spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    assert "gripper_self_contact" in train.MONITOR_INFO_KEYWORDS
    assert "ik_rejects" in train.MONITOR_INFO_KEYWORDS
    assert "gripper_self_contact" in train.monitor_info_keywords({})


# Found by sampling arm poses (seed 1): a finger pad against another part of
# the robot, and a finger pad through the table.
PAD_TO_ARM_QPOS = [-2.9286, -1.9679, 1.0976, 0.4434, -2.1454, 2.8402]  # right_pad vs forearm, 10.6 mm
PAD_TO_TABLE_QPOS = [0.0743, 2.8303, -2.2358, 2.8189, -1.1823, -0.4818]  # right_pad vs table, 49.8 mm


def _pose_contacts(qpos):
    env = RoboChessEnv(action_mode="delta", max_episode_steps=10)
    env.reset(seed=0)
    env.data.qpos[:6] = qpos
    mujoco.mj_forward(env.model, env.data)
    m = env.model
    names = set()
    for c in env.data.contact[: env.data.ncon]:
        for g in (int(c.geom1), int(c.geom2)):
            names.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                      or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])))
    return env, names


def test_pad_against_arm_link_still_terminates_and_is_not_gripper_self_contact():
    env, names = _pose_contacts(PAD_TO_ARM_QPOS)
    assert "ur_gripper_right_pad1" in names or "ur_gripper_right_pad" in names
    assert "ur_forearm_link" in names, "fixture no longer touches"
    assert env._bad_contact() is True
    assert env._gripper_self_contact() is False


def test_pad_against_table_still_terminates_and_is_not_gripper_self_contact():
    env, names = _pose_contacts(PAD_TO_TABLE_QPOS)
    assert "table" in names, "fixture no longer touches"
    assert env._bad_contact() is True
    assert env._gripper_self_contact() is False
