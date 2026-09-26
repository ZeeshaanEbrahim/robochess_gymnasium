"""P2 tests for the T2 obstacle pick-and-place task wrapper.

Same style as tests/test_t1_pick_and_place.py: these validate T2's own
contract -- obstacle placement, the fifth success term, the observation
extension (D12) and the separate scene (D13) -- not RoboChessEnv's physics,
which P1 covers, nor T1's wrapper contract, which its own file covers.

Obstacles are disturbed by teleporting them directly, the same technique
test_env_basic.py's `_place_piece` uses on the tracked piece: it reaches past
the public surface deliberately, because there is no API for "put this body
somewhere" and the task modules are developed alongside the env rather than
across a stable boundary.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import mujoco
import numpy as np
from _d27_helpers import prime_carried_release
import pytest
from gymnasium.utils.env_checker import check_env

from robochess.envs import RoboChessEnv
from robochess.envs.robochess_env import PIECE_SETTLE_CONSECUTIVE_STEPS
from robochess.tasks import (
    N_OBSTACLE_SLOTS,
    OBSTACLE_BODIES,
    OBSTACLE_DISPLACEMENT_TOL,
    PARKED_XY,
    ShapedPickAndPlaceReward,
    SparsePickAndPlaceReward,
    T2_SCENE_PATH,
    T2ObstacleTask,
    file_adjacent_squares,
    make_t2_env,
)
from robochess.tasks.squares import ALL_SQUARES
from robochess.tasks.t1_pick_and_place import _reward_from_distances, _shaped_reward
from robochess.tasks.t2_obstacle_pick_and_place import OBSTACLE_TILT_TOL

BASE_OBS_DIM = 21
T2_OBS_DIM = BASE_OBS_DIM + 2 * N_OBSTACLE_SLOTS  # 29


# --------------------------------------------------------------------- fixtures
@pytest.fixture
def t2_base_env():
    e = make_t2_env()
    yield e
    e.close()


@pytest.fixture
def t2_env(t2_base_env):
    return T2ObstacleTask(
        t2_base_env, start_square="a1", target_square="h8", n_obstacles=2
    )


@pytest.fixture
def t2_sparse(t2_base_env):
    return SparsePickAndPlaceReward(
        T2ObstacleTask(t2_base_env, start_square="a1", target_square="h8", n_obstacles=2)
    )


@pytest.fixture
def t2_shaped(t2_base_env):
    return ShapedPickAndPlaceReward(
        T2ObstacleTask(t2_base_env, start_square="a1", target_square="h8", n_obstacles=2)
    )


def _task_of(env):
    """The T2ObstacleTask inside a reward wrapper (or the task itself)."""
    return env.env if isinstance(env, (ShapedPickAndPlaceReward, SparsePickAndPlaceReward)) else env


def _place_piece(env, pos, gripper_qpos=0.0):
    base = env.unwrapped
    qadr = base._piece_qadr
    base.data.qpos[qadr : qadr + 3] = pos
    base.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    base.data.qpos[base._grip_qadr] = gripper_qpos
    base.data.qvel[:] = 0.0
    mujoco.mj_forward(base.model, base.data)


def _move_obstacle(env, slot, dxy=(0.0, 0.0), quat=None, xy=None):
    """Displace / rotate / relocate one obstacle, then refresh derived state."""
    task = _task_of(env)
    base = env.unwrapped
    qadr = task._obstacle_qadrs[slot]
    if xy is not None:
        base.data.qpos[qadr : qadr + 2] = xy
    else:
        base.data.qpos[qadr : qadr + 2] += np.asarray(dxy, dtype=float)
    if quat is not None:
        base.data.qpos[qadr + 3 : qadr + 7] = quat
    base.data.qvel[task._obstacle_vadrs[slot] : task._obstacle_vadrs[slot] + 6] = 0.0
    mujoco.mj_forward(base.model, base.data)


# ------------------------------------------------------- (a) obstacle placement
def test_obstacles_avoid_start_target_and_file_adjacent_across_many_resets(t2_base_env):
    """Placement exclusion has to hold over the randomized square sampling,
    not for one lucky seed: file-adjacent squares are the ones the open
    gripper measurably overhangs during `descend`."""
    task = T2ObstacleTask(t2_base_env, n_obstacles=N_OBSTACLE_SLOTS)
    seen_obstacles, seen_starts = set(), set()

    for seed in range(40):
        _, info = task.reset(seed=seed)
        start, target = info["start_square"], info["target_square"]
        excluded = {start, target}
        excluded.update(file_adjacent_squares(start))
        excluded.update(file_adjacent_squares(target))

        placed = info["obstacle_squares"]
        assert len(placed) == N_OBSTACLE_SLOTS
        assert len(set(placed)) == len(placed), f"duplicate obstacle squares: {placed}"
        assert not (set(placed) & excluded), (
            f"seed {seed}: obstacles {placed} intersect excluded {sorted(excluded)}"
        )
        for sq in placed:
            assert sq in ALL_SQUARES
        seen_obstacles.update(placed)
        seen_starts.add(start)

    # The sample must actually vary, or the assertion above proves nothing.
    assert len(seen_starts) > 1
    assert len(seen_obstacles) > N_OBSTACLE_SLOTS


def test_file_adjacent_squares_are_same_rank_neighbouring_files():
    assert file_adjacent_squares("e5") == ("d5", "f5")
    assert file_adjacent_squares("a1") == ("b1",)  # no file to the left of a
    assert file_adjacent_squares("h4") == ("g4",)  # none to the right of h


def test_explicit_obstacle_squares_are_honoured_by_constructor_and_reset(t2_base_env):
    task = T2ObstacleTask(
        t2_base_env, start_square="a1", target_square="h8",
        n_obstacles=2, obstacle_squares=("d4", "e5"),
    )
    _, info = task.reset(seed=0)
    assert info["obstacle_squares"] == ("d4", "e5")

    _, info = task.reset(seed=0, options={"obstacle_squares": ("c3", "f6")})
    assert info["obstacle_squares"] == ("c3", "f6")


def test_zero_obstacles_is_a_valid_configuration(t2_base_env):
    """The n_obstacles=0 control: T2's own scene, no obstacles placed. This is
    the honest baseline for T2 (D13), so it has to be constructible."""
    task = T2ObstacleTask(t2_base_env, n_obstacles=0)
    _, info = task.reset(seed=0)
    assert info["obstacle_squares"] == ()
    assert info["obstacles_undisturbed"] is True
    assert info["unintended_displacement_count"] == 0
    assert info["max_obstacle_displacement_mm"] == 0.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_obstacles": 5}, "n_obstacles must be in [0, 4]"),
        ({"n_obstacles": -1}, "n_obstacles must be in [0, 4]"),
        ({"n_obstacles": 2, "obstacle_squares": ("d4",)}, "has 1 entries"),
        ({"n_obstacles": 1, "obstacle_squares": ("z9",)}, "Not a valid square"),
        ({"n_obstacles": 2, "obstacle_squares": ("d4", "d4")}, "must be distinct"),
    ],
)
def test_constructor_rejects_bad_obstacle_configuration(t2_base_env, kwargs, message):
    with pytest.raises(ValueError) as excinfo:
        T2ObstacleTask(t2_base_env, **kwargs)
    assert message in str(excinfo.value)


def test_placement_raises_when_too_few_squares_remain(t2_base_env):
    """Defensive branch: with the exclusions applied there are always far more
    than 4 squares left on a 64-square board, so this can only fire if
    reachable_squares is cut down. Exercised so the message is known to work."""
    task = T2ObstacleTask(
        t2_base_env, start_square="a1", target_square="a2",
        n_obstacles=2, reachable_squares=("a1", "a2", "b1", "b2"),
    )
    with pytest.raises(ValueError, match="Only 0 squares are available"):
        task.reset(seed=0)


# ------------------------------------------------- (b) observation extension
@pytest.mark.parametrize("n_obstacles", [0, 1, 2, 3, 4])
def test_observation_is_always_29_dim_with_parked_sentinels(t2_base_env, n_obstacles):
    """D12: constant width whatever n_obstacles is, and unused slots report
    the documented parked XY -- not zeros, not stale values."""
    task = T2ObstacleTask(
        t2_base_env, start_square="a1", target_square="h8", n_obstacles=n_obstacles
    )
    obs, _ = task.reset(seed=0)

    assert obs["observation"].shape == (T2_OBS_DIM,)
    assert task.observation_space["observation"].shape == (T2_OBS_DIM,)
    assert task.observation_space.contains(obs)

    tail = obs["observation"][BASE_OBS_DIM:].reshape(N_OBSTACLE_SLOTS, 2)
    for slot in range(n_obstacles, N_OBSTACLE_SLOTS):
        assert np.allclose(tail[slot], PARKED_XY[slot], atol=1e-3), (
            f"slot {slot} should report the parked sentinel {PARKED_XY[slot]}, got {tail[slot]}"
        )
    for slot in range(n_obstacles):
        assert not np.allclose(tail[slot], PARKED_XY[slot], atol=1e-3)
        assert np.any(tail[slot] != 0.0)


def test_observation_tail_tracks_the_obstacles_and_survives_a_step(t2_env):
    obs, info = t2_env.reset(seed=0)
    tail = obs["observation"][BASE_OBS_DIM:].reshape(N_OBSTACLE_SLOTS, 2)
    placed = np.array([t2_env._board_world_xy(s) for s in info["obstacle_squares"]])
    assert np.allclose(tail[: len(placed)], placed, atol=1e-3)

    _move_obstacle(t2_env, 0, dxy=(0.05, 0.0))
    obs, _, _, _, _ = t2_env.step(t2_env.unwrapped.home_action())
    assert obs["observation"].shape == (T2_OBS_DIM,)
    moved_tail = obs["observation"][BASE_OBS_DIM:].reshape(N_OBSTACLE_SLOTS, 2)
    assert moved_tail[0][0] == pytest.approx(placed[0][0] + 0.05, abs=2e-3)


def test_base_env_observation_is_left_at_21_dim(t2_base_env):
    """D12's constraint: the extension is the wrapper's, not the env's."""
    obs, _ = t2_base_env.reset(seed=0)
    assert obs["observation"].shape == (BASE_OBS_DIM,)
    assert t2_base_env.observation_space["observation"].shape == (BASE_OBS_DIM,)


def test_check_env_accepts_the_wrapped_t2_env(t2_base_env):
    """Same pattern as T1's
    test_check_env_passes_on_the_wrapped_t1_env_modulo_known_wrapper_warnings:
    check_env must raise nothing, and the ONLY warning allowed is the known
    wrapper meta-warning, content-verified.

    check_env always warns that a wrapped env "is different from the
    unwrapped version" -- true by construction, since the wrapper is what is
    under test. Left uncaught, CI's `pytest -W error::UserWarning` turns that
    into a failure. T1 expects two warnings; this expects one, because
    skip_render_check=True suppresses the second ("Not able to test
    alternative render modes").
    """
    task = T2ObstacleTask(t2_base_env, start_square="a1", target_square="h8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(task, skip_render_check=True)  # raises on an actual space/contract violation

    messages = [str(w.message) for w in caught]
    assert len(caught) == 1, f"expected exactly the 1 known wrapper warning, got:\n{messages}"
    assert "is different from the unwrapped version" in messages[0]


# ------------------------------------------- (c) the fifth success term
def test_undisturbed_obstacles_report_clean(t2_env):
    _, info = t2_env.reset(seed=0)
    assert info["obstacles_undisturbed"] is True
    assert info["unintended_displacement_count"] == 0
    assert info["max_obstacle_displacement_mm"] == pytest.approx(0.0, abs=1e-6)
    assert len(info["obstacle_displacements_mm"]) == 2


def test_displacement_below_tolerance_does_not_count_as_disturbed(t2_env):
    t2_env.reset(seed=0)
    _move_obstacle(t2_env, 0, dxy=(0.002, 0.0))  # 2 mm, under the 5 mm bar
    _, _, terminated, _, info = t2_env.step(t2_env.unwrapped.home_action())
    assert info["max_obstacle_displacement_mm"] < OBSTACLE_DISPLACEMENT_TOL * 1000.0
    assert info["obstacles_undisturbed"] is True
    assert info["unintended_displacement_count"] == 0
    assert terminated is False


def test_displacement_past_tolerance_flips_the_flag_without_terminating(t2_env):
    t2_env.reset(seed=0)
    _move_obstacle(t2_env, 0, dxy=(0.02, 0.0))  # 20 mm, well past the bar
    _, _, terminated, truncated, info = t2_env.step(t2_env.unwrapped.home_action())

    assert info["max_obstacle_displacement_mm"] > OBSTACLE_DISPLACEMENT_TOL * 1000.0
    assert info["obstacles_undisturbed"] is False
    assert info["unintended_displacement_count"] == 1
    assert info["is_success"] is False
    # Non-terminating by design: the trajectory runs to its natural end so RL
    # sees a complete episode.
    assert terminated is False
    assert truncated is False


def test_topple_counts_as_disturbed_even_without_displacement(t2_env):
    """A knocked-over piece whose base barely slides still has to register."""
    t2_env.reset(seed=0)
    # 90 deg about x: the piece is on its side, centre almost unmoved.
    _move_obstacle(t2_env, 0, quat=[np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    _, _, terminated, _, info = t2_env.step(t2_env.unwrapped.home_action())

    assert info["obstacles_undisturbed"] is False
    assert info["unintended_displacement_count"] >= 1
    assert terminated is False


def test_tilt_below_tolerance_does_not_count_as_disturbed(t2_env):
    t2_env.reset(seed=0)
    half = OBSTACLE_TILT_TOL / 4.0  # a small lean, well inside the bar
    _move_obstacle(t2_env, 0, quat=[np.cos(half), np.sin(half), 0.0, 0.0])
    _, _, _, _, info = t2_env.step(t2_env.unwrapped.home_action())
    assert info["obstacles_undisturbed"] is True


def test_obstacle_off_board_counts_as_disturbed(t2_env):
    t2_env.reset(seed=0)
    _move_obstacle(t2_env, 0, xy=(1.5, 1.5))
    _, _, _, _, info = t2_env.step(t2_env.unwrapped.home_action())
    assert info["obstacles_undisturbed"] is False
    assert info["unintended_displacement_count"] == 1


def test_only_the_disturbed_slot_is_counted(t2_base_env):
    task = T2ObstacleTask(
        t2_base_env, start_square="a1", target_square="h8",
        n_obstacles=3, obstacle_squares=("d4", "e5", "f6"),
    )
    task.reset(seed=0)
    _move_obstacle(task, 1, dxy=(0.03, 0.0))
    _, _, _, _, info = task.step(task.unwrapped.home_action())
    assert info["unintended_displacement_count"] == 1
    disp = info["obstacle_displacements_mm"]
    assert len(disp) == 3
    assert disp[0] < 5.0 and disp[2] < 5.0
    assert disp[1] > 5.0


def test_parked_slots_never_count_as_disturbed(t2_base_env):
    """Parked obstacles are off-board by construction; only placed ones are
    consulted, or every episode with n_obstacles < 4 would fail."""
    task = T2ObstacleTask(t2_base_env, start_square="a1", target_square="h8", n_obstacles=1)
    task.reset(seed=0)
    _, _, _, _, info = task.step(task.unwrapped.home_action())
    assert info["obstacles_undisturbed"] is True
    assert len(info["obstacle_displacements_mm"]) == 1


def test_obstacle_term_blocks_an_otherwise_successful_episode(t2_sparse):
    """The whole point of the fifth term: everything RoboChessEnv asks for is
    satisfied, and the episode still is not a success because an obstacle was
    knocked. Compare with test_success_when_all_conditions_hold below."""
    t2_sparse.reset(seed=0)
    base = t2_sparse.unwrapped
    _place_piece(t2_sparse, base._desired_goal, gripper_qpos=0.0)
    prime_carried_release(t2_sparse)  # D27
    _move_obstacle(t2_sparse, 0, dxy=(0.02, 0.0))

    for _ in range(PIECE_SETTLE_CONSECUTIVE_STEPS - 1):
        t2_sparse.step(base.home_action())
    _, reward, terminated, _, info = t2_sparse.step(base.home_action())

    assert info["piece_on_square"] is True
    assert info["piece_resting"] is True
    assert info["gripper_released"] is True
    assert info["piece_settled"] is True
    assert info["obstacles_undisturbed"] is False
    assert info["is_success"] is False
    assert reward == 0.0

    # The base env terminates on its own criterion regardless, so the episode
    # does end here -- but it must not claim "success" while is_success is
    # False. That one combination is relabelled; nothing else is.
    assert terminated is True
    assert info["terminated_reason"] == "success_obstacles_disturbed"


def test_success_when_all_conditions_hold_including_obstacles(t2_sparse):
    t2_sparse.reset(seed=0)
    base = t2_sparse.unwrapped
    _place_piece(t2_sparse, base._desired_goal, gripper_qpos=0.0)
    prime_carried_release(t2_sparse)  # D27

    for _ in range(PIECE_SETTLE_CONSECUTIVE_STEPS - 1):
        t2_sparse.step(base.home_action())
    _, reward, terminated, _, info = t2_sparse.step(base.home_action())

    assert info["obstacles_undisturbed"] is True
    assert info["is_success"] is True
    assert terminated is True
    assert info["terminated_reason"] == "success"  # unrelabelled: obstacles clean
    assert reward == 1.0


# ------------------------------------------------------- (d) reward wrappers
def test_shaped_reward_is_bit_identical_before_and_after_disturbing_an_obstacle(t2_shaped):
    """D12, locked in deliberately: the shaped reward is a pure function of
    piece-to-goal and gripper-to-piece distance and never reads is_success, so
    it gets NO per-step signal about obstacles.

    Scored twice off the SAME unstepped state, with only the obstacle moved in
    between, so nothing but the obstacle can account for a difference. If this
    starts failing, the shaped arm's definition changed and D12's writeup
    caveat needs revisiting.
    """
    t2_shaped.reset(seed=0)
    base = t2_shaped.unwrapped
    _place_piece(t2_shaped, base._desired_goal + np.array([0.05, 0.0, 0.0]))
    weight = t2_shaped.gripper_approach_weight
    pinch_sid = t2_shaped._pinch_sid

    clean = _shaped_reward(base, pinch_sid, base._build_info(None), weight)
    assert _task_of(t2_shaped)._augment_info(base._build_info(None))["obstacles_undisturbed"] is True

    _move_obstacle(t2_shaped, 0, dxy=(0.05, 0.0))

    dirty = _shaped_reward(base, pinch_sid, base._build_info(None), weight)
    assert _task_of(t2_shaped)._augment_info(base._build_info(None))["obstacles_undisturbed"] is False
    assert dirty == clean


def test_shaped_reward_is_exactly_the_two_distance_function_with_obstacles_present(t2_shaped):
    """The functional-form lock behind the test above: whatever the sim state,
    the shaped reward is -(piece_to_goal) - weight * (gripper_to_piece) and
    nothing else. Obstacles cannot enter it because they are not in it."""
    t2_shaped.reset(seed=0)
    base = t2_shaped.unwrapped
    _place_piece(t2_shaped, base._desired_goal + np.array([0.05, 0.0, 0.0]))
    _move_obstacle(t2_shaped, 0, dxy=(0.05, 0.0))

    _, reward, _, _, info = t2_shaped.step(base.home_action())
    gripper_to_piece = float(
        np.linalg.norm(base.data.site_xpos[t2_shaped._pinch_sid] - base._achieved_goal())
    )
    expected = _reward_from_distances(
        info["distance_to_goal"], gripper_to_piece, t2_shaped.gripper_approach_weight
    )
    assert info["obstacles_undisturbed"] is False
    assert reward == pytest.approx(expected, abs=1e-9)


def test_sparse_reward_does_respond_to_obstacle_disturbance(t2_sparse):
    """The counterpart to the shaped case, and the reason no obstacle-specific
    reward wrapper was needed: the sparse arm reads info["is_success"], which
    T2 amends with the fifth term, so obstacles reach it with no code change.
    Shaped and sparse are therefore NOT symmetric here -- that asymmetry is
    the point of D12's writeup caveat."""
    base = t2_sparse.unwrapped

    def run(disturb):
        t2_sparse.reset(seed=0)
        _place_piece(t2_sparse, base._desired_goal, gripper_qpos=0.0)
        prime_carried_release(t2_sparse)  # D27
        if disturb:
            _move_obstacle(t2_sparse, 0, dxy=(0.02, 0.0))
        for _ in range(PIECE_SETTLE_CONSECUTIVE_STEPS - 1):
            t2_sparse.step(base.home_action())
        _, reward, _, _, _ = t2_sparse.step(base.home_action())
        return reward

    assert run(disturb=False) == 1.0
    assert run(disturb=True) == 0.0


def test_shaped_reward_still_tracks_distance_with_obstacles_present(t2_shaped):
    t2_shaped.reset(seed=0)
    base = t2_shaped.unwrapped

    _place_piece(t2_shaped, base._desired_goal + np.array([0.2, 0.0, 0.0]))
    _, far_reward, _, _, _ = t2_shaped.step(base.home_action())
    _place_piece(t2_shaped, base._desired_goal + np.array([0.02, 0.0, 0.0]))
    _, near_reward, _, _, _ = t2_shaped.step(base.home_action())

    assert near_reward > far_reward


# ------------------------------------------------------------ (e) wrong scene
def test_t2_against_t1_scene_fails_loudly_with_the_exact_message():
    """T1's scene deliberately has no obstacle pool (D13). The failure has to
    name the fix, not just raise."""
    env = RoboChessEnv()
    expected = (
        "Scene has no body 'obstacle_1'. T2 needs the obstacle pool, which lives in "
        "assets/scene_t2.xml -- build the env with "
        "RoboChessEnv(scene_path=T2_SCENE_PATH), or use make_t2_env(). "
        "T1's scene.xml deliberately does not have it."
    )
    try:
        with pytest.raises(ValueError, match=re.escape(expected)) as excinfo:
            T2ObstacleTask(env)
        assert type(excinfo.value) is ValueError
        assert str(excinfo.value) == expected
    finally:
        env.close()


def test_make_t2_env_uses_the_t2_scene_and_passes_kwargs_through():
    env = make_t2_env(max_episode_steps=42)
    try:
        assert env.scene_path == T2_SCENE_PATH
        assert env.max_episode_steps == 42
        for name in OBSTACLE_BODIES:
            assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    finally:
        env.close()


def test_t1_scene_has_no_obstacle_pool():
    """The other half of D13: adding the pool to scene.xml is what changed
    T1's sweep, so scene.xml staying clean is the property under test."""
    env = RoboChessEnv()
    try:
        for name in OBSTACLE_BODIES:
            assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name) == -1
    finally:
        env.close()


# --------------------------------------------------- (f) generated-scene sync
def test_scene_t2_matches_what_the_generator_produces():
    """D13: scene_t2.xml is generated from scene.xml and committed. If the two
    drift -- someone edits the board or piece geometry in scene.xml, or edits
    the generated file by hand -- T2 is silently running on a different model
    from T1. Run as part of the suite, not as a separate --check invocation,
    so drift fails CI rather than waiting for someone to remember."""
    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import build_t2_scene

    regenerated = build_t2_scene.build(build_t2_scene.SOURCE.read_text())
    committed = build_t2_scene.GENERATED.read_text()
    assert regenerated == committed, (
        "assets/scene_t2.xml is out of date with assets/scene.xml; "
        "run python scripts/build_t2_scene.py"
    )


def test_generator_parked_positions_match_the_task_module():
    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import build_t2_scene

    assert build_t2_scene.N_OBSTACLES == N_OBSTACLE_SLOTS
    assert build_t2_scene.PARKED == PARKED_XY


def test_generator_refuses_a_scene_it_does_not_recognise():
    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import build_t2_scene

    with pytest.raises(SystemExit, match="no longer ends its worldbody"):
        build_t2_scene.build("<mujoco></mujoco>")
    with pytest.raises(SystemExit, match="home keyframe is not in the expected form"):
        build_t2_scene.build("    </body>\n  </worldbody>")
