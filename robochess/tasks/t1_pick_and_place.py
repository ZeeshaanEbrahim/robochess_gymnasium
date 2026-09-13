"""T1: single-piece pick-and-place, wrapped around RoboChessEnv.

Composition, not subclassing. RoboChessEnv (P1, validated: scripted expert
50/50, `check_env` clean) already owns the arm, gripper, board physics and
the on_square/resting/released success criterion (D3's amendment in
DECISIONS.md) -- this module never touches any of that. T2-T5 will each need
their own start-state logic (obstacles, capture, multi-piece); wrapping
keeps RoboChessEnv the single source of truth for physics and success
instead of re-deriving reset/step in five subclasses.

Two layers, composed rather than merged, so T2-T5 can reuse the reward
layer unchanged:

  T1PickAndPlaceTask        start/target square selection, piece teleport,
                             goal placement, info-dict augmentation.
  ShapedPickAndPlaceReward  dense reward on top of a task wrapper.
  SparsePickAndPlaceReward  {0, 1} reward on top of a task wrapper.

HER decision (locked, not open): T1's HER-compatible configuration is
state-vector observations only. RoboChessEnv currently has no image
observation mode to begin with, so there is nothing to build or omit here --
this note exists so the constraint is written down before P2 adds one. See
DECISIONS.md D5.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from robochess.envs.robochess_env import RoboChessEnv
from robochess.tasks.squares import ALL_SQUARES, square_to_local_xy

# D2 (DECISIONS.md) swept top-down IK over all 64 squares and found every one
# reachable. T1 only ever does top-down grasps, the same condition D2 tested,
# so the full 64-square set is used as-is rather than re-swept here. If a
# square in this set ever turns out not to be reachable in practice, that is
# a D2 regression and belongs back in DECISIONS.md, not a silent shrink of
# this tuple.
REACHABLE_SQUARES: tuple[str, ...] = ALL_SQUARES


def _pinch_site_id(model: mujoco.MjModel) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ur_gripper_pinch")
    if sid < 0:
        raise ValueError("Scene has no 'ur_gripper_pinch' site.")
    return sid


class T1PickAndPlaceTask(gym.Wrapper):
    """Move the tracked piece from a start square to a target square.

    Reset places the piece on ``start_square`` and points ``desired_goal`` at
    ``target_square``, teleporting the piece and writing the goal directly
    through the base env's MuJoCo state -- the same technique
    ``tests/test_env_basic.py``'s ``_place_piece`` helper already uses to
    reach past RoboChessEnv's public surface, since it exposes no API yet for
    "put the piece somewhere other than its keyframe spawn." That coupling is
    deliberate: this task module and RoboChessEnv are developed together, not
    across a stable public boundary.

    Both squares default to random distinct choices from
    ``reachable_squares`` (default: :data:`REACHABLE_SQUARES`, all 64 board
    squares per D2). Pass fixed squares to the constructor, or per-episode via
    ``reset(options={"start_square": ..., "target_square": ...})``.

    Does not change the reward -- ``step`` passes through whatever
    ``RoboChessEnv`` returns (its dense distance placeholder). Stack
    :class:`ShapedPickAndPlaceReward` or :class:`SparsePickAndPlaceReward` on
    top for the study's actual reward formulations.
    """

    metadata = RoboChessEnv.metadata

    def __init__(
        self,
        env: RoboChessEnv,
        start_square: str | None = None,
        target_square: str | None = None,
        reachable_squares: tuple[str, ...] = REACHABLE_SQUARES,
    ):
        super().__init__(env)
        base = self.unwrapped
        if not isinstance(base, RoboChessEnv):
            raise TypeError(
                f"T1PickAndPlaceTask wraps a RoboChessEnv; got {type(base).__name__}."
            )
        self._base: RoboChessEnv = base

        self.reachable_squares = tuple(reachable_squares)
        if len(self.reachable_squares) < 2:
            raise ValueError("Need >= 2 reachable_squares to pick a distinct start/target pair.")
        for name, sq in (("start_square", start_square), ("target_square", target_square)):
            if sq is not None and sq not in self.reachable_squares:
                raise ValueError(f"{name}={sq!r} is not in reachable_squares.")
        self.start_square = start_square
        self.target_square = target_square

        # Piece height at rest, read from the model's own spawn keyframe
        # rather than hardcoded, so a change to piece geometry propagates here.
        self._piece_rest_z = float(base.model.key_qpos[base._home_key][base._piece_qadr + 2])
        self._board_bid = base._body_id("board")

        self.current_start_square: str | None = None
        self.current_target_square: str | None = None

    # ------------------------------------------------------------ placement
    def _board_world_xy(self, square: str) -> np.ndarray:
        return square_to_local_xy(square) + self._base.model.body_pos[self._board_bid][:2]

    def _teleport_piece(self, square: str) -> None:
        base = self._base
        xy = self._board_world_xy(square)
        qadr, vadr = base._piece_qadr, base._piece_vadr
        base.data.qpos[qadr : qadr + 2] = xy
        base.data.qpos[qadr + 2] = self._piece_rest_z
        base.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]  # upright, matches home keyframe
        base.data.qvel[vadr : vadr + 6] = 0.0

    def _set_goal(self, square: str) -> None:
        xy = self._board_world_xy(square)
        self._base._desired_goal = np.array(
            [xy[0], xy[1], self._base._board_surface_z], dtype=np.float32
        )

    def _resolve_squares(self, start: str | None, target: str | None) -> tuple[str, str]:
        rng = self._base.np_random
        if start is not None and target is not None:
            if start == target:
                raise ValueError("start_square and target_square must differ.")
            return start, target
        if start is None and target is None:
            idx = rng.choice(len(self.reachable_squares), size=2, replace=False)
            return self.reachable_squares[idx[0]], self.reachable_squares[idx[1]]
        choices = [s for s in self.reachable_squares if s != (start or target)]
        picked = choices[int(rng.integers(len(choices)))]
        return (start, picked) if start is not None else (picked, target)

    # ---------------------------------------------------------------- gym
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        options = options or {}
        # env.reset() seeds self._base.np_random (via gym.Env.reset(seed=)),
        # so square sampling below is deterministic under a given seed.
        self.env.reset(seed=seed, options=options)

        start, target = self._resolve_squares(
            options.get("start_square", self.start_square),
            options.get("target_square", self.target_square),
        )
        self._teleport_piece(start)
        self._set_goal(target)
        mujoco.mj_forward(self._base.model, self._base.data)

        self.current_start_square, self.current_target_square = start, target

        obs = self._base._get_obs()
        info = self._augment_info(self._base._build_info(terminated_reason=None))
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._augment_info(info)

    # --------------------------------------------------------------- info
    def _grasp_success(self) -> bool:
        """True while the gripper appears to be holding the piece off the board.

        Not part of RoboChessEnv's success criterion (D3) -- this is a
        diagnostic for partial-progress reporting (roadmap P2 notes), built
        from the same measured thresholds D3 already validated: lifted past
        ``rest_tolerance`` above the board, gripper aperture at/above
        ``gripper_open_threshold`` (i.e. not reported as released).
        """
        base = self._base
        piece_z = float(base._achieved_goal()[2])
        lifted = piece_z > base._board_surface_z + base.rest_tolerance
        holding = float(base.data.qpos[base._grip_qadr]) >= base.gripper_open_threshold
        return bool(lifted and holding)

    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        info = dict(info)
        info["start_square"] = self.current_start_square
        info["target_square"] = self.current_target_square
        info["grasp_success"] = self._grasp_success()
        # T1 is single-piece, so this is always 0. The field exists now, not
        # added in P2, so T2+ (multi-piece) extend the same info-dict schema
        # instead of introducing a new key partway through the study.
        info["unintended_displacement_count"] = 0
        return info


def _reward_from_distances(piece_to_goal: float, gripper_to_piece: float, weight: float) -> float:
    """``-(piece_to_goal) - weight * (gripper_to_piece)``.

    Pure function of two scalar distances, kept separate from any MuJoCo
    state so its continuity and monotonicity are unit-testable directly.
    """
    return -float(piece_to_goal) - weight * float(gripper_to_piece)


# --- terminal failure penalty (early-termination exploit fix) --------------
# The per-step shaped reward is strictly negative, and before this penalty a
# failure termination cost nothing extra. So the return-maximising policy was
# to END the episode as fast as possible: a 500k-step T1/PPO dry run converged
# to 0% success with mean episode length 8.9, 85% bad_contact and 13%
# piece_off_board. Ending early simply stopped the negative reward accruing.
#
# Only these two reasons are penalised -- they are the exploit. Time-limit
# truncation is an honest "ran out of time" and is left alone, as is D14's
# solver_instability, which is a physics artefact truncation, not a choice.
PENALISED_TERMINATIONS = frozenset({"bad_contact", "piece_off_board"})

# Upper bounds on the two distance terms over every NON-terminal state, so the
# penalty can be sized against the worst case rather than a typical episode.
# Derived from scene geometry (board centre (0, 0.52), half-extent 0.312 m,
# SQUARE_PITCH 0.078 m; arm base at (0, 0, 0.85); pinch-site reach 1.2616 m
# and max height 2.0959 m from 20k random FK samples), then rounded UP:
#   piece_to_goal:    a non-terminal piece is inside the off-board XY box, so
#                     at most that box's diagonal combined with the highest
#                     height it can be carried to              -> 1.6628 m
#   gripper_to_piece: triangle inequality, reach + base-to-piece -> 2.8217 m
# Measured against it: worst single non-terminal step seen was -1.1378 (random
# actions, 11,411 steps) against this bound's -2.57 -- the bound is loose, as
# a worst-case bound should be. T2 shares the arm and board, so it holds there.
PIECE_TO_GOAL_BOUND = 1.70
GRIPPER_TO_PIECE_BOUND = 2.90


def failure_penalty_magnitude(max_episode_steps: int, weight: float) -> float:
    """Smallest whole-number penalty that makes failing strictly worse than
    enduring a full episode of worst-case per-step reward.

    ``max_episode_steps * (worst-case |per-step reward|)``, rounded up. Tied to
    the episode length rather than hardcoded: a fixed number sized for 100-step
    episodes would silently stop being sufficient at 200.

    Undiscounted, deliberately. Under discounting the remaining per-step cost
    only shrinks while the penalty is paid immediately, so a penalty that beats
    the undiscounted sum beats every discounted one too.
    """
    per_step_bound = PIECE_TO_GOAL_BOUND + weight * GRIPPER_TO_PIECE_BOUND
    return float(np.ceil(max_episode_steps * per_step_bound))


def _shaped_reward(base: RoboChessEnv, pinch_sid: int, info: dict[str, Any], weight: float) -> float:
    """Reads the two live distances out of the sim and scores them."""
    gripper_to_piece = float(
        np.linalg.norm(base.data.site_xpos[pinch_sid] - base._achieved_goal())
    )
    return _reward_from_distances(info["distance_to_goal"], gripper_to_piece, weight)


class ShapedPickAndPlaceReward(gym.Wrapper):
    """Dense reward with a terminal failure penalty::

        r = -(piece_to_goal_dist) - weight * (gripper_to_piece_dist)
            - failure_penalty * [terminated_reason in PENALISED_TERMINATIONS]

    The distance term is continuous and monotonically decreasing in both
    distances, so descent gives a gradient the whole episode rather than only
    at the goal. The penalty is paid once, on the terminating step only; see
    PENALISED_TERMINATIONS for why it exists and failure_penalty_magnitude for
    how it is sized.

    Not HER-safe: the gripper-to-piece term needs the pinch site's live
    position, which is not recoverable from ``(achieved_goal, desired_goal)``
    alone, so ``compute_reward`` is deliberately left unoverridden here and
    falls back to the base env's goal-only placeholder. This mirrors the
    proposal's factorial design, where HER is paired with the sparse arm, not
    the shaped one -- shaped reward is not intended to run under HER.
    """

    def __init__(
        self,
        env: gym.Env,
        gripper_approach_weight: float = 0.3,
        failure_penalty: float | None = None,
    ):
        super().__init__(env)
        base = self.unwrapped
        if not isinstance(base, RoboChessEnv):
            raise TypeError(f"Expects a RoboChessEnv underneath; got {type(base).__name__}.")
        self._base: RoboChessEnv = base
        self._pinch_sid = _pinch_site_id(base.model)
        self.gripper_approach_weight = float(gripper_approach_weight)
        # None -> derived from THIS env's episode length and weight, so the
        # "failing is strictly worse" guarantee holds for whatever episode
        # length the env was actually built with.
        self.failure_penalty = (
            failure_penalty_magnitude(base.max_episode_steps, self.gripper_approach_weight)
            if failure_penalty is None
            else float(failure_penalty)
        )

    def step(self, action: np.ndarray):
        obs, _base_reward, terminated, truncated, info = self.env.step(action)
        reward = _shaped_reward(self._base, self._pinch_sid, info, self.gripper_approach_weight)
        # `terminated` is checked as well as the reason: the penalty is for a
        # failure that ENDED the episode, never for a flag seen mid-episode.
        if terminated and info.get("terminated_reason") in PENALISED_TERMINATIONS:
            reward -= self.failure_penalty
        return obs, reward, terminated, truncated, info


class SparsePickAndPlaceReward(gym.Wrapper):
    """Binary reward: 1.0 on RoboChessEnv's full success criterion, else 0.0.

    HER-safe by construction, with one documented asymmetry (a consequence of
    D3's amendment, not a bug here): the *real* transitions stored by
    ``step`` use the full three-condition criterion (on-square, resting,
    released) from ``info["is_success"]``, but ``compute_reward`` -- used by
    SB3's HerReplayBuffer only to score *relabelled* (synthetic) goals -- can
    only see ``(achieved_goal, desired_goal)`` and therefore approximates
    success as XY distance under ``goal_tolerance``. Gripper-release state
    cannot enter a relabelled reward because it is not part of the goal
    vector (D3). Real transitions are unaffected: SB3 stores the reward
    ``step`` actually returned and only calls ``compute_reward`` for goals it
    invents.
    """

    def step(self, action: np.ndarray):
        obs, _base_reward, terminated, truncated, info = self.env.step(action)
        reward = 1.0 if info["is_success"] else 0.0
        return obs, reward, terminated, truncated, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any] | list | None = None,
    ) -> np.ndarray | float:
        achieved = np.asarray(achieved_goal, dtype=np.float64)
        desired = np.asarray(desired_goal, dtype=np.float64)
        xy_dist = np.linalg.norm(achieved[..., :2] - desired[..., :2], axis=-1)
        tol = self.unwrapped.goal_tolerance
        return (xy_dist < tol).astype(np.float64)
