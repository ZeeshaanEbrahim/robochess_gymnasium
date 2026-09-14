"""P2 tests for the T1 pick-and-place task wrapper.

Mirrors the style of tests/test_env_basic.py: these validate the wrapper's
own contract (square selection, info augmentation, reward formulations), not
RoboChessEnv's physics, which P1 already covers. Success/termination cases
reuse the direct-teleport technique test_env_basic.py's `_place_piece` uses,
now going through the wrapped env instead of the bare base env.
"""

from __future__ import annotations

import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from robochess.envs import RoboChessEnv
from robochess.envs.robochess_env import PIECE_SETTLE_CONSECUTIVE_STEPS
from robochess.tasks import (
    REACHABLE_SQUARES,
    ShapedPickAndPlaceReward,
    SparsePickAndPlaceReward,
    T1PickAndPlaceTask,
)
from robochess.tasks.squares import ALL_SQUARES, square_to_local_xy
from robochess.tasks.t1_pick_and_place import _reward_from_distances


# --------------------------------------------------------------------- fixtures
@pytest.fixture
def base_env():
    e = RoboChessEnv()
    yield e
    e.close()


@pytest.fixture
def task_env(base_env):
    return T1PickAndPlaceTask(base_env, start_square="a1", target_square="h8")


@pytest.fixture
def sparse_env(base_env):
    return SparsePickAndPlaceReward(
        T1PickAndPlaceTask(base_env, start_square="a1", target_square="h8")
    )


@pytest.fixture
def shaped_env(base_env):
    return ShapedPickAndPlaceReward(
        T1PickAndPlaceTask(base_env, start_square="a1", target_square="h8")
    )


def _place_piece(env, pos, gripper_qpos=0.0):
    """Teleport the tracked piece and set the gripper aperture -- same
    technique as test_env_basic.py's helper, through the wrapped env."""
    base = env.unwrapped
    qadr = base._piece_qadr
    base.data.qpos[qadr : qadr + 3] = pos
    base.data.qvel[base._piece_vadr : base._piece_vadr + 6] = 0.0
    base.data.qpos[base._grip_qadr] = gripper_qpos


# ---------------------------------------------------------------- squares.py
def test_square_to_local_xy_matches_known_board_geoms(base_env):
    """a1 and h8 must match the board's own sq_a1 / (derived) h8 geometry."""
    a1 = square_to_local_xy("a1")
    h8 = square_to_local_xy("h8")
    np.testing.assert_allclose(a1, [-0.273, -0.273], atol=1e-9)
    np.testing.assert_allclose(h8, [0.273, 0.273], atol=1e-9)
    # sq_a1 (dark square geom) is a ground-truth cross-check against scene.xml.
    board_bid = base_env._body_id("board")
    sq_a1_world = base_env.model.geom_pos[base_env._geom_id("sq_a1")][:2]
    np.testing.assert_allclose(a1, sq_a1_world, atol=1e-9)


def test_square_to_local_xy_rejects_invalid_squares():
    for bad in ("z9", "a", "a11", "", "i1"):
        with pytest.raises(ValueError):
            square_to_local_xy(bad)


def test_all_squares_is_64_unique_entries():
    assert len(ALL_SQUARES) == 64
    assert len(set(ALL_SQUARES)) == 64
    assert REACHABLE_SQUARES == ALL_SQUARES


# ----------------------------------------------------------------------- reset
def test_reset_places_piece_on_start_square_and_goal_on_target(task_env):
    obs, info = task_env.reset(seed=0)
    assert info["start_square"] == "a1"
    assert info["target_square"] == "h8"

    expected_start_xy = square_to_local_xy("a1") + task_env.unwrapped.model.body_pos[
        task_env.unwrapped._body_id("board")
    ][:2]
    expected_target_xy = square_to_local_xy("h8") + task_env.unwrapped.model.body_pos[
        task_env.unwrapped._body_id("board")
    ][:2]
    np.testing.assert_allclose(obs["achieved_goal"][:2], expected_start_xy, atol=1e-6)
    np.testing.assert_allclose(obs["desired_goal"][:2], expected_target_xy, atol=1e-6)
    # Piece must be spawned at rest, not mid-air.
    assert obs["achieved_goal"][2] == pytest.approx(task_env.unwrapped._board_surface_z, abs=1e-6)


def test_reset_respects_the_reachable_square_set(base_env):
    """Every start/target square sampled over many resets must come from
    REACHABLE_SQUARES (D2's 64-square set), and start must never equal target."""
    env = T1PickAndPlaceTask(base_env)
    seen_start, seen_target = set(), set()
    for seed in range(60):
        _, info = env.reset(seed=seed)
        assert info["start_square"] in REACHABLE_SQUARES
        assert info["target_square"] in REACHABLE_SQUARES
        assert info["start_square"] != info["target_square"]
        seen_start.add(info["start_square"])
        seen_target.add(info["target_square"])
    # Random sampling over 60 draws should not collapse to a single square.
    assert len(seen_start) > 1
    assert len(seen_target) > 1


def test_reset_honours_a_restricted_reachable_set(base_env):
    restricted = ("a1", "a2", "a3")
    env = T1PickAndPlaceTask(base_env, reachable_squares=restricted)
    for seed in range(20):
        _, info = env.reset(seed=seed)
        assert info["start_square"] in restricted
        assert info["target_square"] in restricted


def test_reset_options_override_constructor_squares(task_env):
    """Per-episode options must win over the constructor's fixed squares."""
    _, info = task_env.reset(seed=0, options={"start_square": "d4", "target_square": "e5"})
    assert info["start_square"] == "d4"
    assert info["target_square"] == "e5"


def test_reset_samples_target_when_only_start_is_fixed(base_env):
    env = T1PickAndPlaceTask(base_env, start_square="a1")
    for seed in range(10):
        _, info = env.reset(seed=seed)
        assert info["start_square"] == "a1"
        assert info["target_square"] in REACHABLE_SQUARES
        assert info["target_square"] != "a1"


def test_reset_samples_start_when_only_target_is_fixed(base_env):
    env = T1PickAndPlaceTask(base_env, target_square="h8")
    for seed in range(10):
        _, info = env.reset(seed=seed)
        assert info["target_square"] == "h8"
        assert info["start_square"] in REACHABLE_SQUARES
        assert info["start_square"] != "h8"


def test_start_and_target_square_must_differ(base_env):
    env = T1PickAndPlaceTask(base_env, start_square="c3", target_square="c3")
    with pytest.raises(ValueError):
        env.reset(seed=0)


def test_unknown_square_rejected_at_construction(base_env):
    with pytest.raises(ValueError):
        T1PickAndPlaceTask(base_env, start_square="z9")


def test_too_few_reachable_squares_rejected(base_env):
    with pytest.raises(ValueError):
        T1PickAndPlaceTask(base_env, reachable_squares=("a1",))


def test_task_wrapper_rejects_a_non_robochess_env():
    with pytest.raises(TypeError):
        T1PickAndPlaceTask(gym.make("CartPole-v1"))


def test_reward_wrapper_rejects_a_non_robochess_env():
    with pytest.raises(TypeError):
        ShapedPickAndPlaceReward(gym.make("CartPole-v1"))


# -------------------------------------------------------------------- info dict
def test_info_dict_always_has_the_new_t1_fields(task_env):
    obs, info = task_env.reset(seed=0)
    for key in ("start_square", "target_square", "grasp_success", "unintended_displacement_count"):
        assert key in info

    obs, reward, terminated, truncated, info = task_env.step(task_env.unwrapped.home_action())
    for key in ("start_square", "target_square", "grasp_success", "unintended_displacement_count"):
        assert key in info
    assert info["unintended_displacement_count"] == 0
    assert isinstance(info["grasp_success"], bool)

    # Original base-env fields must survive augmentation untouched.
    for key in ("is_success", "distance_to_goal", "piece_on_square", "piece_resting", "gripper_released"):
        assert key in info


def test_grasp_success_false_when_piece_lifted_and_gripper_closed_but_not_gripped(task_env):
    """The old definition's false positive: lifted AND closed, with no pad
    touching the piece. This used to assert True -- it is exactly the case
    that flagged 50 non-grasps in the pilot-matrix diagnosis."""
    task_env.reset(seed=0)
    base = task_env.unwrapped
    lifted_pos = [base._desired_goal[0], base._desired_goal[1], base._board_surface_z + 0.03]
    _place_piece(task_env, lifted_pos, gripper_qpos=0.54)  # closed enough to hold, per D3's measured value
    action = base.home_action().copy()
    action[6] = 1.0  # command gripper closed so it does not spring open this step
    _, _, _, _, info = task_env.step(action)
    assert task_env._pad_piece_contact() == (False, False)
    assert info["grasp_success"] is False


def test_grasp_success_false_for_toppled_piece_beside_closed_gripper(task_env):
    """A piece knocked onto its side raises its origin ~18 mm -- past
    rest_tolerance -- which is what fooled the old definition in practice."""
    task_env.reset(seed=0)
    base = task_env.unwrapped
    qadr = base._piece_qadr
    x, y = base._achieved_goal()[:2]
    base.data.qpos[qadr:qadr + 3] = [x, y, base._board_surface_z + 0.018]
    base.data.qpos[qadr + 3:qadr + 7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]  # 90 deg about x
    base.data.qvel[base._piece_vadr:base._piece_vadr + 6] = 0.0
    base.data.qpos[base._grip_qadr] = 0.63   # closed on nothing
    action = base.home_action().copy()
    action[6] = 1.0
    _, _, _, _, info = task_env.step(action)
    assert info["piece_height_above_board"] > base.rest_tolerance   # would have satisfied "lifted"
    assert info["grasp_success"] is False


def test_pad_piece_contact_is_false_at_reset(task_env):
    task_env.reset(seed=0)
    assert task_env._pad_piece_contact() == (False, False)


def test_grasp_success_true_during_a_real_scripted_grasp():
    """End to end on the actual controller: the flag must fire while the
    expert carries the piece, with both pads on it, and never before the
    pads reach the piece."""
    import sys
    from pathlib import Path

    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import scripted_expert as se

    class Record(gym.Wrapper):
        def reset(self, **kw):
            self.rows = []
            return self.env.reset(**kw)

        def step(self, a):
            out = self.env.step(a)
            self.rows.append((out[4]["grasp_success"], *self.env._pad_piece_contact()))
            return out

    env = RoboChessEnv(max_episode_steps=350)
    task = T1PickAndPlaceTask(env, start_square="d1", target_square="a8")
    rec = Record(task)
    result = se.ScriptedExpert(rec).run_episode(seed=0)
    env.close()

    assert result["success"], result
    flags = [r[0] for r in rec.rows]
    assert sum(flags) >= 10, "grasp_success should hold for the whole carry"
    first = flags.index(True)
    assert not any(flags[:first])
    assert all(left and right for g, left, right in rec.rows if g), "every grasp step has both pads on the piece"


def test_grasp_success_false_when_piece_resting(task_env):
    task_env.reset(seed=0)
    base = task_env.unwrapped
    _place_piece(task_env, base._desired_goal, gripper_qpos=0.0)
    _, _, _, _, info = task_env.step(base.home_action())
    assert info["grasp_success"] is False


# -------------------------------------------------------------- termination
def test_not_success_while_piece_still_elevated(sparse_env):
    sparse_env.reset(seed=0)
    base = sparse_env.unwrapped
    goal = base._desired_goal
    _place_piece(sparse_env, [goal[0], goal[1], base._board_surface_z + 0.025], gripper_qpos=0.0)

    _, reward, terminated, _, info = sparse_env.step(base.home_action())
    assert info["piece_on_square"] is True
    assert info["piece_resting"] is False
    assert info["is_success"] is False
    assert terminated is False
    assert reward == 0.0


def test_not_success_while_gripper_still_closed(sparse_env):
    sparse_env.reset(seed=0)
    base = sparse_env.unwrapped
    goal = base._desired_goal
    _place_piece(sparse_env, [goal[0], goal[1], base._board_surface_z], gripper_qpos=0.54)

    action = base.home_action().copy()
    action[6] = 1.0
    _, reward, terminated, _, info = sparse_env.step(action)
    assert info["piece_resting"] is True
    assert info["gripper_released"] is False
    assert info["is_success"] is False
    assert terminated is False
    assert reward == 0.0


def test_success_when_all_three_conditions_hold(sparse_env):
    """D9: is_success also requires PIECE_SETTLE_CONSECUTIVE_STEPS
    consecutive steps of low piece angular velocity, counted from step()
    calls, so drive up to the last one before asserting the reward/success."""
    sparse_env.reset(seed=0)
    base = sparse_env.unwrapped
    _place_piece(sparse_env, base._desired_goal, gripper_qpos=0.0)

    for _ in range(PIECE_SETTLE_CONSECUTIVE_STEPS - 1):
        _, reward, terminated, _, info = sparse_env.step(base.home_action())
        assert terminated is False
        assert reward == 0.0

    _, reward, terminated, truncated, info = sparse_env.step(base.home_action())
    assert info["is_success"] is True
    assert terminated is True
    assert truncated is False
    assert reward == 1.0


def test_episode_truncates_at_exactly_max_episode_steps(base_env):
    """Holding the start pose (start != target by construction) never
    terminates, so the episode must run to truncation and not one step
    either side of it."""
    base_env.max_episode_steps = 25  # RoboChessEnv reads this each step; safe to set post-construction
    env = T1PickAndPlaceTask(base_env, start_square="a1", target_square="h8")
    env.reset(seed=0)
    hold = base_env.home_action()
    for i in range(1, 25):
        _, _, terminated, truncated, _ = env.step(hold)
        assert not terminated, f"unexpected termination at step {i}"
        assert not truncated, f"truncated early at step {i}"

    _, _, terminated, truncated, info = env.step(hold)
    assert truncated is True
    assert terminated is False
    assert info["elapsed_steps"] == 25


# --------------------------------------------------------------- sparse reward
def test_sparse_reward_is_exactly_zero_or_one(sparse_env):
    """D9: is_success also requires PIECE_SETTLE_CONSECUTIVE_STEPS
    consecutive steps of low piece angular velocity, so the 1.0 case needs
    that many steps before it registers."""
    sparse_env.reset(seed=0)
    base = sparse_env.unwrapped

    _place_piece(sparse_env, base._desired_goal, gripper_qpos=0.0)
    for _ in range(PIECE_SETTLE_CONSECUTIVE_STEPS - 1):
        _, reward, _, _, _ = sparse_env.step(base.home_action())
        assert reward == 0.0
    _, reward, _, _, _ = sparse_env.step(base.home_action())
    assert reward in (0.0, 1.0)
    assert reward == 1.0

    sparse_env.reset(seed=0)
    _, reward, _, _, _ = sparse_env.step(base.home_action())
    assert reward in (0.0, 1.0)
    assert reward == 0.0


def test_sparse_compute_reward_is_binary_and_batches(sparse_env):
    sparse_env.reset(seed=0)
    achieved = np.zeros((5, 3))
    desired = np.tile(np.array([0.0, 0.0, 0.0]), (5, 1))
    desired[0] = [10.0, 10.0, 10.0]  # far -> 0
    rewards = sparse_env.compute_reward(achieved, desired, [{}] * 5)
    assert set(np.unique(rewards)).issubset({0.0, 1.0})
    assert rewards[0] == 0.0
    assert rewards[1] == 1.0


# --------------------------------------------------------------- shaped reward
def test_reward_from_distances_is_continuous_and_monotonic():
    """Pure-math check: reward strictly decreases as either distance grows,
    with no discontinuity -- exercised directly, no sim involved."""
    weight = 0.3
    piece_dists = np.linspace(0.0, 1.0, 50)
    rewards = [_reward_from_distances(d, 0.0, weight) for d in piece_dists]
    assert all(np.diff(rewards) < 0), "reward must strictly decrease as piece_to_goal grows"
    assert np.all(np.isfinite(rewards))

    grip_dists = np.linspace(0.0, 1.0, 50)
    rewards2 = [_reward_from_distances(0.0, d, weight) for d in grip_dists]
    assert all(np.diff(rewards2) < 0), "reward must strictly decrease as gripper_to_piece grows"

    # Continuity: no jump larger than the step size warrants (Lipschitz check).
    for d in np.linspace(0.0, 1.0, 200):
        r_a = _reward_from_distances(d, 0.0, weight)
        r_b = _reward_from_distances(d + 1e-6, 0.0, weight)
        assert abs(r_a - r_b) < 1e-3


def test_shaped_reward_decreases_as_piece_approaches_goal(shaped_env):
    """End-to-end version of the monotonicity check, through the real sim:
    moving the piece closer to the goal (gripper held fixed away from both)
    must not decrease the reward."""
    shaped_env.reset(seed=0)
    base = shaped_env.unwrapped
    goal = base._desired_goal.copy()

    far_pos = [goal[0] + 0.3, goal[1] + 0.3, base._board_surface_z]
    near_pos = [goal[0] + 0.05, goal[1] + 0.05, base._board_surface_z]

    _place_piece(shaped_env, far_pos, gripper_qpos=0.0)
    _, reward_far, _, _, _ = shaped_env.step(base.home_action())

    shaped_env.reset(seed=0)
    _place_piece(shaped_env, near_pos, gripper_qpos=0.0)
    _, reward_near, _, _, _ = shaped_env.step(base.home_action())

    assert reward_near > reward_far


def test_shaped_reward_is_a_finite_float(shaped_env):
    shaped_env.reset(seed=0)
    _, reward, _, _, _ = shaped_env.step(shaped_env.unwrapped.home_action())
    assert isinstance(reward, float)
    assert np.isfinite(reward)


# --------------------------------------------------------------- API contract
def test_check_env_passes_on_the_wrapped_t1_env_modulo_known_wrapper_warnings():
    """Wrapping can silently break the Dict observation space contract even
    when the base env passes -- must be checked on the wrapped env, not just
    inferred from the base env's own passing check_env test.

    check_env on a *wrapped* env unconditionally emits two meta-warnings that
    have nothing to do with space correctness: it notices the env differs
    from `.unwrapped` (true by construction -- that's what a wrapper is), and
    it cannot exercise alternate render modes without a `.spec`, which only
    `gymnasium.make()` attaches. tests/test_env_basic.py sidesteps both by
    building through `gym.make(...).unwrapped`; T1 cannot use `.unwrapped`
    here since that would strip the wrapper being tested. So: assert
    check_env raises nothing (the actual Dict-space contract check), and
    assert the *only* warnings are these two known, content-verified ones --
    anything else would mean wrapping broke something real.
    """
    base = RoboChessEnv(render_mode="rgb_array")
    env = SparsePickAndPlaceReward(
        T1PickAndPlaceTask(base, start_square="a1", target_square="h8")
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_env(env)  # raises on an actual space/contract violation

        messages = [str(w.message) for w in caught]
        assert len(caught) == 2, f"expected exactly the 2 known wrapper warnings, got:\n{messages}"
        assert any("is different from the unwrapped version" in m for m in messages)
        assert any("Not able to test alternative render modes" in m for m in messages)
    finally:
        env.close()


def test_wrapped_env_observation_space_matches_base(task_env, base_env):
    assert task_env.observation_space == base_env.observation_space
    assert task_env.action_space == base_env.action_space


def test_wrapped_env_reset_obs_is_contained_in_observation_space(task_env):
    obs, _ = task_env.reset(seed=0)
    assert task_env.observation_space.contains(obs)


def test_wrapped_env_step_obs_is_contained_in_observation_space(task_env):
    task_env.reset(seed=0)
    obs, _, _, _, _ = task_env.step(task_env.unwrapped.home_action())
    assert task_env.observation_space.contains(obs)
