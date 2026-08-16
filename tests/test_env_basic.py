"""P1 contract tests for RoboChessEnv.

These validate the Gymnasium API against the real MuJoCo scene. They are not
task tests -- the reward here is an acknowledged placeholder (see the env
docstring), so nothing asserts anything about reward *quality*, only shape,
finiteness and the terminated/truncated split.
"""

import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import robochess  # noqa: F401  -- registers RoboChess-v0
from robochess.envs import RoboChessEnv


@pytest.fixture
def env():
    e = RoboChessEnv()
    yield e
    e.close()


# --------------------------------------------------------------- API contract
def test_check_env_passes_with_zero_warnings():
    """gymnasium's own checker must pass, and must emit nothing at all.

    Built through gym.make so the env has a `spec`; without one the checker
    warns that it cannot exercise the alternative render modes. `.unwrapped` is
    passed because the checker warns when handed a wrapped env.
    """
    made = gym.make("RoboChess-v0", render_mode="rgb_array")
    env = made.unwrapped
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_env(env)
        assert caught == [], "check_env emitted warnings:\n" + "\n".join(
            f"  [{w.category.__name__}] {w.message}" for w in caught
        )
    finally:
        made.close()


# ---------------------------------------------------------------------- reset
def test_reset_returns_correctly_shaped_obs(env):
    obs, info = env.reset(seed=0)

    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    assert obs["observation"].shape == (21,)
    assert obs["achieved_goal"].shape == (3,)
    assert obs["desired_goal"].shape == (3,)
    for v in obs.values():
        assert v.dtype == np.float32
        assert np.all(np.isfinite(v))

    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    assert "is_success" in info and "distance_to_goal" in info


def test_reset_is_deterministic_and_restores_the_piece(env):
    obs_a, _ = env.reset(seed=0)
    for _ in range(5):
        env.step(env.action_space.sample())
    obs_b, _ = env.reset(seed=0)

    for key in obs_a:
        np.testing.assert_allclose(obs_a[key], obs_b[key], atol=1e-6)


def test_achieved_goal_starts_one_square_from_desired_goal(env):
    """The P1 placeholder goal is the adjacent square, so the opening distance
    should be exactly one 78 mm pitch."""
    obs, info = env.reset(seed=0)
    dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
    assert dist == pytest.approx(0.078, abs=1e-6)
    assert info["distance_to_goal"] == pytest.approx(0.078, abs=1e-6)
    assert info["is_success"] is False


# ----------------------------------------------------------------------- step
def test_zero_action_step_does_not_crash(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))

    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) and np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert reward == pytest.approx(-info["distance_to_goal"], abs=1e-6)


def test_random_actions_stay_finite_and_stable(env):
    """Guards against the sim blowing up under out-of-distribution actions,
    which would surface as NaN observations or MuJoCo warning counters."""
    env.reset(seed=0)
    for _ in range(50):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs["observation"]))
        assert np.isfinite(reward)
        if terminated or truncated:
            env.reset(seed=0)

    counters = [int(w.number) for w in env.data.warning]
    assert not any(counters), f"MuJoCo raised warnings: {counters}"


def test_episode_truncates_at_exactly_max_episode_steps():
    """Holding the home pose never terminates, so the episode must run to the
    truncation limit and not one step either side of it."""
    env = RoboChessEnv(max_episode_steps=25)
    try:
        env.reset(seed=0)
        hold = env.home_action()
        for i in range(1, 25):
            _, _, terminated, truncated, _ = env.step(hold)
            assert not terminated, f"unexpected termination at step {i}"
            assert not truncated, f"truncated early at step {i}"

        _, _, terminated, truncated, info = env.step(hold)
        assert truncated is True
        assert terminated is False
        assert info["elapsed_steps"] == 25
    finally:
        env.close()


def _place_piece(env, pos, gripper_qpos=0.0):
    """Teleport the tracked piece and set the gripper aperture directly."""
    qadr = env._piece_qadr
    env.data.qpos[qadr : qadr + 3] = pos
    env.data.qvel[env._piece_vadr : env._piece_vadr + 6] = 0.0
    env.data.qpos[env._grip_qadr] = gripper_qpos


def test_success_requires_the_piece_to_be_resting(env):
    """A piece held at the right XY but still in the air is NOT a success.

    This is the regression that motivated the criterion: with a plain 3D
    distance test, the scripted expert 'succeeded' in all 50 episodes with the
    piece still clamped 25 mm above the target square.
    """
    env.reset(seed=0)
    goal = env._desired_goal
    _place_piece(env, [goal[0], goal[1], env._board_surface_z + 0.025], gripper_qpos=0.0)

    _, _, terminated, _, info = env.step(env.home_action())
    assert info["piece_on_square"] is True
    assert info["gripper_released"] is True
    assert info["piece_resting"] is False
    assert info["is_success"] is False
    assert terminated is False


def test_success_requires_the_gripper_to_be_open(env):
    """A piece on the square and at board height but still gripped is NOT a success."""
    env.reset(seed=0)
    goal = env._desired_goal
    _place_piece(env, [goal[0], goal[1], env._board_surface_z], gripper_qpos=0.54)

    # Command the gripper closed so it stays shut for this step.
    action = env.home_action().copy()
    action[6] = 1.0
    _, _, terminated, _, info = env.step(action)

    assert info["piece_on_square"] is True
    assert info["piece_resting"] is True
    assert info["gripper_released"] is False
    assert info["is_success"] is False
    assert terminated is False


def test_success_when_all_three_conditions_hold(env):
    env.reset(seed=0)
    goal = env._desired_goal
    _place_piece(env, [goal[0], goal[1], env._board_surface_z], gripper_qpos=0.0)

    _, _, terminated, truncated, info = env.step(env.home_action())
    assert (info["piece_on_square"], info["piece_resting"], info["gripper_released"]) == (
        True,
        True,
        True,
    )
    assert info["is_success"] is True
    assert terminated is True
    assert truncated is False


def test_board_surface_height_is_read_from_the_model(env):
    """The resting height must track the scene, not a hardcoded constant."""
    board_z = env.model.body_pos[env._body_id("board")][2]
    slab_half = env.model.geom_size[env._geom_id("board_slab")][2]
    assert env._board_surface_z == pytest.approx(board_z + slab_half)

    obs, _ = env.reset(seed=0)
    # The piece spawns at rest on the board, so its origin is exactly there.
    assert obs["achieved_goal"][2] == pytest.approx(env._board_surface_z, abs=1e-6)


def test_terminated_and_truncated_are_mutually_exclusive(env):
    """A success on the final step must report terminated, not truncated."""
    env.reset(seed=0)
    # Teleport the piece onto the goal (which is at board height) with the
    # gripper open, so all three success conditions hold, then take one step.
    _place_piece(env, env._desired_goal, gripper_qpos=0.0)

    _, _, terminated, truncated, info = env.step(env.home_action())
    assert terminated is True
    assert truncated is False
    assert info["is_success"] is True
    assert info["terminated_reason"] == "success"


# ---------------------------------------------------------------- action space
def test_action_space_maps_onto_model_ctrlrange(env):
    """The exposed action space is normalised, but its extremes must map exactly
    onto the actuator ctrlrange read from scene.xml -- not onto guessed bounds."""
    low = env.model.actuator_ctrlrange[:, 0]
    high = env.model.actuator_ctrlrange[:, 1]

    np.testing.assert_allclose(env._scale_action(-np.ones(env.model.nu)), low, atol=1e-6)
    np.testing.assert_allclose(env._scale_action(np.ones(env.model.nu)), high, atol=1e-6)

    # Gripper actuator really is the 0..255 range from the 2F-85 model.
    assert low[6] == pytest.approx(0.0)
    assert high[6] == pytest.approx(255.0)


def test_home_action_holds_the_arm_still(env):
    env.reset(seed=0)
    start = env.data.qpos[:6].copy()
    for _ in range(10):
        env.step(env.home_action())
    assert np.max(np.abs(env.data.qpos[:6] - start)) < 0.05


# --------------------------------------------------------------- goal / reward
def test_compute_reward_supports_batched_input(env):
    """HER relabels goals in batches, so compute_reward must broadcast."""
    achieved = np.zeros((4, 3))
    desired = np.tile(np.array([0.0, 0.0, 0.5]), (4, 1))
    rewards = env.compute_reward(achieved, desired, [{}] * 4)

    assert np.shape(rewards) == (4,)
    np.testing.assert_allclose(rewards, -0.5)


def test_goal_space_contains_achievable_positions(env):
    """HER writes achieved_goal into desired_goal, so the two spaces must agree."""
    assert env.observation_space["achieved_goal"] == env.observation_space["desired_goal"]

    obs, _ = env.reset(seed=0)
    assert env.observation_space["desired_goal"].contains(obs["achieved_goal"])


# --------------------------------------------------------------------- render
def test_render_returns_an_rgb_array():
    env = RoboChessEnv(render_mode="rgb_array", render_width=160, render_height=120)
    try:
        env.reset(seed=0)
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8
        assert frame.std() > 1, "frame looks blank"
    finally:
        env.close()


def test_render_returns_none_without_a_render_mode(env):
    env.reset(seed=0)
    assert env.render() is None
