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

from robochess.envs.robochess_env import N_ARM_ACTUATORS, RoboChessEnv
from robochess.tasks.squares import (
    ALL_SQUARES,
    reject_excluded,
    square_to_local_xy,
    validate_excluded_squares,
)

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
        excluded_squares: tuple[str, ...] = (),
    ):
        super().__init__(env)
        base = self.unwrapped
        if not isinstance(base, RoboChessEnv):
            raise TypeError(
                f"T1PickAndPlaceTask wraps a RoboChessEnv; got {type(base).__name__}."
            )
        self._base: RoboChessEnv = base

        # D26: excluded squares are removed from the sampling pool, order
        # preserved, so with an empty list the pool -- and therefore every RNG
        # draw -- is exactly what it was before the key existed.
        self.excluded_squares = validate_excluded_squares(excluded_squares)
        self.reachable_squares = tuple(
            s for s in reachable_squares if s not in self.excluded_squares
        )
        if len(self.reachable_squares) < 2:
            raise ValueError("Need >= 2 reachable_squares to pick a distinct start/target pair.")
        self.start_square = self._check_fixed("start_square", start_square)
        self.target_square = self._check_fixed("target_square", target_square)

        # Piece height at rest, read from the model's own spawn keyframe
        # rather than hardcoded, so a change to piece geometry propagates here.
        self._piece_rest_z = float(base.model.key_qpos[base._home_key][base._piece_qadr + 2])
        self._board_bid = base._body_id("board")
        # Finger pad bodies for grasp_success's contact terms (the board body,
        # above, is the third).
        self._left_pad_bid = base._left_pad_bid
        self._right_pad_bid = base._right_pad_bid

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

    def _check_fixed(self, name: str, sq: str | None) -> str | None:
        """Shared check for a fixed square, from the constructor or an ``options`` override."""
        if sq is None:
            return None
        reject_excluded(name, sq, self.excluded_squares)
        if sq not in self.reachable_squares:
            raise ValueError(f"{name}={sq!r} is not in reachable_squares.")
        return sq

    def _resolve_squares(self, start: str | None, target: str | None) -> tuple[str, str]:
        rng = self._base.np_random
        start = self._check_fixed("start_square", start)
        target = self._check_fixed("target_square", target)
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
        # The teleport moved the piece after the base env's own reset scan.
        self._base._refresh_contact_flags()
        # F7: the base env snapshotted its own default e4->e5 episode inside
        # env.reset(); the D14 fallback must hand back THIS episode's state.
        self._base._snapshot_last_good_obs()

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
    def _piece_contacts(self) -> tuple[bool, bool, bool]:
        """(left pad, right pad, board): what the tracked piece is touching.

        D27: a thin read of the base env's cached flags, which its step() and
        reset() refresh once per step (the single scan lives in
        ``RoboChessEnv._piece_contact_flags``). Tests that patch THIS method
        change what the wrapper reads but do not reach the base env's
        ever_held / release-record latch; D27 tests must use real replays or
        fake contacts at the base-env level.
        """
        return self._base._contact_flags

    def _pad_piece_contact(self) -> tuple[bool, bool]:
        """(left, right): whether each gripper pad body is touching the piece."""
        left, right, _ = self._piece_contacts()
        return left, right

    def _grasp_success(self) -> bool:
        """True while the gripper is actually holding the piece up.

        Not part of RoboChessEnv's success criterion (D3) -- a diagnostic for
        partial-progress reporting (roadmap P2 notes). All three must hold:

          * lifted:  piece origin more than ``rest_tolerance`` above the board
          * closed:  gripper aperture at/above ``gripper_open_threshold``
          * gripped: BOTH finger pads in contact with the piece
          * clear:   the piece NOT in contact with the board

        The contact term is the fix. Lifted-and-closed alone was satisfied by
        a piece knocked onto its side (its origin rises ~18 mm) next to a
        closed but empty gripper: in the pilot-matrix diagnosis all 50 flagged
        episodes were false positives of exactly that kind, with the pinch
        site 18-68 cm from the piece (results/pilot_diagnosis/s2).

        The board term closes the case the pad term leaves open: a toppled
        piece lying on the board with both pads closed on it still has its
        origin ~18 mm up (so "lifted" holds) and both pads touching, but it
        is resting on the board, not held up by the gripper.
        """
        base = self._base
        piece_z = float(base._achieved_goal()[2])
        lifted = piece_z > base._board_surface_z + base.rest_tolerance
        closed = float(base.data.qpos[base._grip_qadr]) >= base.gripper_open_threshold
        if not (lifted and closed):
            return False
        left, right, board = self._piece_contacts()
        return bool(left and right and not board)

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

# --- staged shaped reward (default OFF) -----------------------------------
# The default shaped reward pays only for two distances, so it has no term for
# closing the gripper, and its piece-to-goal term is 3D -- which means LIFTING
# the piece increases that distance and costs reward. Hovering near the piece
# is therefore a local optimum, which is what the clamped delta 100k run did:
# approach improved (median closest pinch-to-piece 0.427 -> 0.266 m) while real
# grasps stayed at 0 and episode return stayed flat (-53.3 +- 41.5).
#
# The staged variant swaps the 3D piece-to-goal term for its XY component (so
# lifting is free) and adds two bonuses for reaching the intermediate states
# the task requires. Both bonuses are PER-STEP while the condition holds, and
# both are OFF by default; see DECISIONS.md D20 for the sizing argument and for
# the hold-at-goal risk the per-step form carries.
DEFAULT_GRASP_BONUS = 1.0
DEFAULT_LIFT_BONUS = 0.5


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


def find_contacts_owner(env: gym.Env, message: str):
    """The wrapper at or below `env` that owns `_piece_contacts` (T1 or T2).

    One lookup shared by the staged shaped reward and the pick-up reward, so
    both read the same both-pads-on-the-piece test rather than each growing
    its own contact scan.
    """
    node = env
    while node is not None:
        if hasattr(node, "_piece_contacts"):
            return node
        node = getattr(node, "env", None)
    raise TypeError(message)


def _shaped_reward(base: RoboChessEnv, pinch_sid: int, info: dict[str, Any], weight: float) -> float:
    """Reads the two live distances out of the sim and scores them."""
    gripper_to_piece = float(
        np.linalg.norm(base.data.site_xpos[pinch_sid] - base._achieved_goal())
    )
    return _reward_from_distances(info["distance_to_goal"], gripper_to_piece, weight)


def _staged_reward(base: RoboChessEnv, pinch_sid: int, contacts_owner, info: dict[str, Any],
                   weight: float, grasp_bonus: float, lift_bonus: float) -> float:
    """XY distance to goal, gripper approach, plus the grasp and lift bonuses.

    `lift_bonus` is paid only on top of a grasp: "clear of the board" alone is
    also true of a piece knocked into the air, which is not progress.
    """
    gripper_to_piece = float(
        np.linalg.norm(base.data.site_xpos[pinch_sid] - base._achieved_goal())
    )
    reward = _reward_from_distances(info["xy_distance_to_goal"], gripper_to_piece, weight)
    left, right, board = contacts_owner._piece_contacts()
    if left and right:
        reward += grasp_bonus
        if not board:
            reward += lift_bonus
    return reward


class ShapedPickAndPlaceReward(gym.Wrapper):
    """Dense reward with a terminal failure penalty::

        r = -(piece_to_goal_dist) - weight * (gripper_to_piece_dist)
            - failure_penalty * [terminated_reason in PENALISED_TERMINATIONS]

    The distance term is continuous and monotonically decreasing in both
    distances, so descent gives a gradient the whole episode rather than only
    at the goal. The penalty is paid once, on the terminating step only; see
    PENALISED_TERMINATIONS for why it exists and failure_penalty_magnitude for
    how it is sized.

    With ``staged=True`` (default False) the reward becomes::

        r = -d_xy(piece, goal) - weight * (gripper_to_piece_dist)
            + grasp_bonus  if both finger pads contact the piece
            + lift_bonus   if, additionally, the piece is clear of the board
            - failure_penalty * [terminated_reason in PENALISED_TERMINATIONS]

    i.e. the piece-to-goal term drops to its XY component so that lifting no
    longer costs reward, and the two intermediate states the task requires are
    paid for directly. The contact tests are the ones `_grasp_success` already
    uses (`_piece_contacts` on the task wrapper below this one) -- no new
    contact logic. The failure penalty is unchanged in both modes.

    With ``staged=False`` this class is exactly what it was before the staged
    terms existed, arithmetic included.

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
        staged: bool = False,
        grasp_bonus: float = DEFAULT_GRASP_BONUS,
        lift_bonus: float = DEFAULT_LIFT_BONUS,
    ):
        super().__init__(env)
        base = self.unwrapped
        if not isinstance(base, RoboChessEnv):
            raise TypeError(f"Expects a RoboChessEnv underneath; got {type(base).__name__}.")
        self._base: RoboChessEnv = base
        self._pinch_sid = _pinch_site_id(base.model)
        self.gripper_approach_weight = float(gripper_approach_weight)
        self.staged = bool(staged)
        self.grasp_bonus = float(grasp_bonus)
        self.lift_bonus = float(lift_bonus)
        if self.staged:
            # Resolved once, and loudly: the staged terms are defined by the
            # task wrapper's contact test, so a stack without one would
            # silently score every step as "not grasped" rather than fail.
            self._contacts_owner = self._find_contacts_owner()
        else:
            self._contacts_owner = None
        # None -> derived from THIS env's episode length and weight, so the
        # "failing is strictly worse" guarantee holds for whatever episode
        # length the env was actually built with.
        self.failure_penalty = (
            failure_penalty_magnitude(base.max_episode_steps, self.gripper_approach_weight)
            if failure_penalty is None
            else float(failure_penalty)
        )

    def _find_contacts_owner(self):
        """The wrapper below this one that owns `_piece_contacts` (T1 or T2)."""
        return find_contacts_owner(
            self.env,
            "staged=True needs a task wrapper providing _piece_contacts() "
            "(T1PickAndPlaceTask or a subclass) between this wrapper and the env.",
        )

    def step(self, action: np.ndarray):
        obs, _base_reward, terminated, truncated, info = self.env.step(action)
        if self.staged:
            reward = _staged_reward(
                self._base, self._pinch_sid, self._contacts_owner, info,
                self.gripper_approach_weight, self.grasp_bonus, self.lift_bonus,
            )
        else:
            reward = _shaped_reward(self._base, self._pinch_sid, info, self.gripper_approach_weight)
        # `terminated` is checked as well as the reason: the penalty is for a
        # failure that ENDED the episode, never for a flag seen mid-episode.
        if terminated and info.get("terminated_reason") in PENALISED_TERMINATIONS:
            reward -= self.failure_penalty
        if info.get("invalid_transition") and not np.isfinite(reward):
            # F7: computed from a corrupted live state (see _build_info). The
            # flag already marks the transition; this only keeps the value
            # finite so it cannot poison VecNormalize.
            reward = 0.0
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


# --- pick-up stage reward (default OFF) -----------------------------------
# A separate training stage: one piece, grasp it and lift it LIFT_HEIGHT. It
# replaces the reward entirely rather than adding to it -- there is no
# goal-distance term and no `failure_penalty_magnitude` penalty here, because
# the pick-up stage has no target square to travel to and the 771-magnitude
# penalty sized for a 300-step travelling episode is not what this stage wants.
#
# Every term is POSITIVE and bounded, so the per-step reward is in [0, ~6] and
# a failure termination costs a small fixed PICKUP_FAILURE_PENALTY instead of
# dwarfing the episode. Shaping widths live here as module constants, not as
# config keys: they define the shape of the reward, and a run that changes them
# is a different reward, not a different hyper-parameter setting.

# Height of the pinch site above the piece's origin at the pose the scripted
# expert actually grasps from. This is the expert's calibrated GRASP_Z
# (scripts/scripted_expert.GRASP_Z = 0.79925) expressed as an offset; the
# piece's resting height is read from the model's home keyframe, so the offset
# adapts if the board or piece geometry changes.
PINCH_GRASP_Z = 0.79925

# What `pickup_dz` must come out at, with the tolerance the check allows.
# NOTE: 0.03725, not the 0.0353 the stage specification anticipated -- see
# `pickup_dz` for the measurement that settles which is right.
EXPECTED_PICKUP_DZ = 0.03725
PICKUP_DZ_TOL = 5e-4

# reach(d): a wide linear ramp that pays something from anywhere on the board,
# plus a narrow exponential that only pays near the grasp offset. The linear
# span is the order of the arm's board-scale reach; the exponential width is
# the scale over which "close" becomes "grasping".
REACH_LINEAR_SPAN = 0.8
REACH_EXP_WIDTH = 0.03

# posture(tilt): never zero, so a badly-oriented arm still has a reach
# gradient to follow, but worth 2x as much upright. 0.35 rad ~ 20 deg.
#
# Raised from 0.25/0.75 after the Section 3 table: at 0.25/0.75 the multiplier
# was strong enough that the home pose (420 mm away, perfectly upright) outscored
# the expert's own pre-descend hover (113 mm away, 21 deg tilted), so the reward
# FELL along the expert's path. At 0.5/0.5 getting closer always outweighs the
# tilt incurred on the way.
POSTURE_FLOOR = 0.5
POSTURE_SPAN = 0.5
POSTURE_WIDTH = 0.35

# The pose gate multiplying the closing term: all three must be small before
# closing the gripper pays anything. Tighter than the posture width on purpose
# -- closing is only useful from a pose a grasp could actually succeed from.
GATE_LATERAL_WIDTH = 0.02
GATE_VERTICAL_WIDTH = 0.015
GATE_TILT_WIDTH = 0.2

# closure: aperture mapped to [0, 1]. The low end is the env's own
# `gripper_open_threshold` (below it the gripper counts as released); the high
# end is the aperture the pads reach when closed on the piece's stem.
CLOSURE_OPEN = 0.2
CLOSURE_CLOSED = 0.645

WORLD_DOWN = np.array([0.0, 0.0, -1.0])


def pickup_dz(piece_rest_z: float) -> float:
    """Pinch height above the piece origin at the grasp pose, in metres.

    ``PINCH_GRASP_Z - piece_rest_z``, with the resting height read from the
    model rather than written down here.

    MEASURED DISCREPANCY, recorded rather than smoothed over: this returns
    0.03725, while the stage specification expected 0.0353. Both are defensible
    and they disagree by 1.95 mm:

      * 0.03725 is the offset implied by the height the expert COMMANDS.
      * ~0.0360 is where the pinch actually ENDS UP at the close command,
        measured over a replayed e4->e5 expert episode: the descend waypoint
        exits with a ~1.3 mm residual it reports and gives up on
        ("descend: pinch residual 1.83 mm still above 1.0 mm").

    Neither is 0.0353. The commanded value is used because it is the one
    definition that is reproducible from the model plus one calibration
    constant, rather than from whatever the IK residual happened to be.
    """
    dz = float(PINCH_GRASP_Z) - float(piece_rest_z)
    if abs(dz - EXPECTED_PICKUP_DZ) > PICKUP_DZ_TOL:
        raise ValueError(
            f"pickup dz = {dz:.5f} m, expected {EXPECTED_PICKUP_DZ:.5f} "
            f"+/- {PICKUP_DZ_TOL:.4f}. The piece's resting height or the "
            "grasp calibration has moved; re-derive the pick-up reward."
        )
    return dz


def pickup_reach(d: float) -> float:
    """Wide linear ramp plus a narrow exponential peak at d = 0."""
    d = float(d)
    linear = 0.5 * (1.0 - min(d / REACH_LINEAR_SPAN, 1.0))
    return linear + 0.5 * float(np.exp(-d / REACH_EXP_WIDTH))


def pickup_posture(tilt: float) -> float:
    """Multiplier in [POSTURE_FLOOR, 1] for how top-down the gripper is."""
    return POSTURE_FLOOR + POSTURE_SPAN * float(np.exp(-float(tilt) / POSTURE_WIDTH))


def pickup_pose_gate(lateral: float, vertical_error: float, tilt: float) -> float:
    """Product of three Gaussians: 1 only from a pose a grasp could work from."""
    return float(
        np.exp(-((float(lateral) / GATE_LATERAL_WIDTH) ** 2))
        * np.exp(-((float(vertical_error) / GATE_VERTICAL_WIDTH) ** 2))
        * np.exp(-((float(tilt) / GATE_TILT_WIDTH) ** 2))
    )


def pickup_closure(aperture: float) -> float:
    """Gripper aperture mapped onto [0, 1] between open and closed-on-stem."""
    return float(
        np.clip((float(aperture) - CLOSURE_OPEN) / (CLOSURE_CLOSED - CLOSURE_OPEN), 0.0, 1.0)
    )


def pinch_tilt(model_site_xmat: np.ndarray) -> float:
    """Angle in radians between the pinch site's local +Z and world straight down.

    The site's local +Z is the gripper's approach axis: the site sits 145 mm
    along the gripper base's +Z (assets/robotiq_2f85/2f85.xml), which is the
    direction the fingers project, and at the expert's grasp pose that axis
    reads (-0.009, -0.023, -0.9997) in world with the piece at +36 mm along it.
    0 is perfectly top-down, pi/2 is horizontal.
    """
    approach = np.asarray(model_site_xmat, dtype=np.float64).reshape(3, 3)[:, 2]
    return float(np.arccos(np.clip(float(approach @ WORLD_DOWN), -1.0, 1.0)))


# Number of entries `pickup_features` appends to the observation vector.
N_PICKUP_OBS = 15


def pickup_geometry(base: RoboChessEnv, pinch_sid: int, dz: float) -> dict[str, Any]:
    """THE shared pick-up geometry. Both the reward and the observation use it.

    One helper on purpose: the reward is defined in terms of the pinch site's
    pose relative to the grasp point `g`, and the observation exists so the
    policy can see those same quantities. Computing them twice is how the two
    drift apart.

    Returns world-frame quantities:
      pinch      pinch site position
      piece      piece origin (the env's achieved_goal)
      g          piece + (0, 0, dz): where the pinch must be to grasp
      g_to_pinch g -> pinch offset, i.e. `pinch - g`
      approach   pinch site local +Z as a unit vector (see `pinch_tilt`)
      tilt       angle of `approach` from world straight down, rad
      aperture   gripper driver joint position
      lift       piece height above its resting height
      target_error  commanded arm target minus measured arm joint position

    `target_error` is ``data.ctrl[:6] - data.qpos[arm]``. That is the command
    actually sitting on the position actuators, which in delta mode is the
    stored target AFTER the D19 anti-windup clamp and the joint-range clamp
    (`_scale_action` writes exactly that into ctrl), and in absolute mode is
    the scaled action itself. Same definition, same units, both modes -- which
    is the point: the policy sees how far its command is from being achieved
    without needing to know which action space it is in.
    """
    pinch = np.asarray(base.data.site_xpos[pinch_sid], dtype=np.float64).copy()
    piece = np.asarray(base._achieved_goal(), dtype=np.float64)
    g = piece + np.array([0.0, 0.0, float(dz)])
    approach = np.asarray(
        base.data.site_xmat[pinch_sid], dtype=np.float64
    ).reshape(3, 3)[:, 2].copy()
    n = N_ARM_ACTUATORS
    target_error = (
        np.asarray(base.data.ctrl[:n], dtype=np.float64)
        - np.asarray(base.data.qpos[base._arm_qadr], dtype=np.float64)
    )
    return dict(
        pinch=pinch, piece=piece, g=g, g_to_pinch=pinch - g,
        approach=approach,
        tilt=float(np.arccos(np.clip(float(approach @ WORLD_DOWN), -1.0, 1.0))),
        aperture=float(base.data.qpos[base._grip_qadr]),
        lift=float(piece[2] - (base.model.key_qpos[base._home_key][base._piece_qadr + 2])),
        target_error=target_error,
    )


def pickup_features(geom: dict[str, Any]) -> np.ndarray:
    """The 15 appended observation entries, in their documented order.

    21:24 pinch position | 24:27 g - pinch | 27:30 approach axis
    30:36 commanded arm target - measured arm joint position
    """
    return np.concatenate([
        geom["pinch"],
        -geom["g_to_pinch"],       # g - pinch
        geom["approach"],
        geom["target_error"],
    ]).astype(np.float64)


class PickupObservation(gym.Wrapper):
    """Appends `N_PICKUP_OBS` pick-up features to the observation (default off).

    Follows D12's precedent: the extension is declared HERE, on the wrapper,
    and RoboChessEnv's own `observation_space` and `_get_obs` are untouched.
    The base 21 entries keep their indices and their values exactly.

    T1 only. T2 already extends the observation with its obstacle slots at the
    same tail position, so stacking both would produce two different layouts
    behind one shape; that is rejected rather than ordered.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base = self.unwrapped
        if not isinstance(base, RoboChessEnv):
            raise TypeError(f"Expects a RoboChessEnv underneath; got {type(base).__name__}.")
        task = self._find_task(env)
        if type(task) is not T1PickAndPlaceTask:
            raise TypeError(
                f"pickup_obs is T1 only; found task wrapper {type(task).__name__}. "
                "T2 and later tasks extend the observation vector themselves "
                "(D12), so the two extensions would share one tail."
            )
        self._base: RoboChessEnv = base
        self._pinch_sid = _pinch_site_id(base.model)
        self.dz = pickup_dz(
            float(base.model.key_qpos[base._home_key][base._piece_qadr + 2])
        )

        base_space = self.env.observation_space
        obs_space = base_space["observation"]
        # Bounds: positions share the piece's +/-5 m box; the approach axis is a
        # unit vector; the target error is bounded by the joint ranges it is a
        # difference of, rounded out to the same +/-2pi the joint entries use.
        pad_high = np.concatenate([
            np.full(3, 5.0), np.full(3, 5.0), np.ones(3), np.full(N_ARM_ACTUATORS, 2 * np.pi),
        ]).astype(obs_space.dtype)
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(
                low=np.concatenate([obs_space.low, -pad_high]),
                high=np.concatenate([obs_space.high, pad_high]),
                dtype=obs_space.dtype,
            ),
            "achieved_goal": base_space["achieved_goal"],
            "desired_goal": base_space["desired_goal"],
        })

    @staticmethod
    def _find_task(env: gym.Env):
        node = env
        while node is not None:
            if isinstance(node, T1PickAndPlaceTask):
                return node
            node = getattr(node, "env", None)
        raise TypeError("pickup_obs needs a T1PickAndPlaceTask in the stack.")

    def geometry(self) -> dict[str, Any]:
        """The shared geometry at the current sim state."""
        return pickup_geometry(self._base, self._pinch_sid, self.dz)

    def _extend(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        space = self.observation_space["observation"]
        extended = np.concatenate([obs["observation"], pickup_features(self.geometry())])
        obs = dict(obs)
        obs["observation"] = np.clip(extended, space.low, space.high).astype(space.dtype)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._extend(obs)
        self._last_good_obs = {k: v.copy() for k, v in obs.items()}
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("invalid_transition"):
            # F7: the base env hands back its last good observation on an invalid
            # step, but these features are read from the LIVE state, so appending
            # them would splice a corrupted (possibly non-finite) instant onto an
            # old one. Return this wrapper's own last good observation instead.
            obs = {k: v.copy() for k, v in self._last_good_obs.items()}
        else:
            obs = self._extend(obs)
            self._last_good_obs = {k: v.copy() for k, v in obs.items()}
        return obs, reward, terminated, truncated, info

class PickupReward(gym.Wrapper):
    """Pick-up stage reward: grasp the piece and lift it, nothing else.

    Replaces the reward entirely; there is no goal-distance term and no
    travelling failure penalty. Per step, with all weights at their defaults::

        r = reach_weight  * reach(d) * posture(tilt)          # approach
          + close_weight  * pose_gate * closure(aperture)     # close, in pose
          + lift_weight   * clip(lift / lift_height, 0, 1)    # while held
          + success_weight                                    # while succeeding
          + first_grasp_bonus                                 # once per episode
          - failure_penalty                                   # on a failed end

    The penalised failures are the shared PENALISED_TERMINATIONS -- bad_contact
    and piece_off_board only. D14's solver_instability is deliberately NOT
    penalised here, for the same reason the travelling reward excludes it: that
    truncation is a physics artefact the guard raises, not an outcome the
    policy chose, so charging for it would price in the simulator's behaviour.

    `d` is measured to ``g = piece + (0, 0, dz)``, the pinch position a grasp
    actually happens from, NOT to the piece origin. This is the correction the
    e4->e5 diagnosis pointed at: the old term's minimum was the piece origin,
    which is 37 mm inside the piece and unreachable, so "as close as possible"
    and "in a position to grasp" were different poses.

    The first-grasp bonus and the lift term are keyed on HELD -- both finger
    pads touching the piece AND the piece clear of the board. Not on
    `grasp_success`: D18 requires the piece more than `rest_tolerance` (10 mm)
    above the board before that can be true at all, so a lift term keyed on it
    can never pay for the first 10 mm of a lift. And not on PINCHED alone,
    which two measurements showed pays for non-grasps:

      * an incidental pinch at 87 deg off top-down, with a pose gate of 1e-26,
        collected the whole 5.0 bonus for a 3.7 mm nudge (1M policy, episode 12
        step 121 of the 20-episode stochastic sweep);
      * the D18 toppled piece -- lying on the board, pads closed across it --
        paid 6.61 on its first step and ~1.60 per step for 19 steps, which is
        exactly the false positive D18's board-contact term exists to exclude.

    `held` adds that same board-contact test, read from the one
    `_piece_contacts()` call the pinched test already makes, so there is no
    second contact scan. The lift term stays dense from 0 mm, because the piece
    can be clear of the board well below D18's 10 mm threshold.

    The SUCCESS STATE is unchanged and still uses `grasp_success`: paying for a
    lift is shaping, declaring the stage solved is not.

    Success does NOT terminate the episode -- the stage is about holding a
    lift, and terminating on first success would pay a policy for a lift it
    drops immediately. ``info["pickup_success"]`` latches True from the first
    step the success state fires until the episode ends, so the episode-level
    flag reaches Monitor.
    """

    def __init__(
        self,
        env: gym.Env,
        lift_height: float = 0.03,
        hold_steps: int = 5,
        failure_penalty: float = 5.0,
        reach_weight: float = 1.0,
        close_weight: float = 1.0,
        lift_weight: float = 2.0,
        success_weight: float = 1.0,
        first_grasp_bonus: float = 5.0,
    ):
        super().__init__(env)
        base = self.unwrapped
        if not isinstance(base, RoboChessEnv):
            raise TypeError(f"Expects a RoboChessEnv underneath; got {type(base).__name__}.")
        self._base: RoboChessEnv = base
        self._pinch_sid = _pinch_site_id(base.model)
        self.lift_height = float(lift_height)
        self.hold_steps = int(hold_steps)
        self.failure_penalty = float(failure_penalty)
        self.reach_weight = float(reach_weight)
        self.close_weight = float(close_weight)
        self.lift_weight = float(lift_weight)
        self.success_weight = float(success_weight)
        self.first_grasp_bonus = float(first_grasp_bonus)

        # Resting height from the model's home keyframe, so dz adapts to the
        # scene instead of being written down twice.
        self._piece_rest_z = float(
            base.model.key_qpos[base._home_key][base._piece_qadr + 2]
        )
        self.dz = pickup_dz(self._piece_rest_z)

        # The same both-pads-on-the-piece test the staged shaped reward uses.
        self._contacts_owner = find_contacts_owner(
            self.env,
            "pickup_mode needs a task wrapper providing _piece_contacts() "
            "(T1PickAndPlaceTask) between this wrapper and the env.",
        )

        self._held_once = False
        self._hold_streak = 0
        self._succeeded = False

    # ------------------------------------------------------------ per-episode
    def reset(self, **kwargs):
        self._held_once = False
        self._hold_streak = 0
        self._succeeded = False
        out = self.env.reset(**kwargs)
        # F7: what pickup_terms reports if the very first step is invalid.
        self._last_terms = dict(
            self.terms(held=False), first_grasp_bonus=0.0, success_term=0.0,
            hold_streak=0, pinched=False, held=False, grasp_success=False,
        )
        return out

    def contact_state(self) -> tuple[bool, bool]:
        """``(pinched, held)`` from a single contact scan.

        `pinched` is D20's test: both finger pads touching the piece. `held`
        adds D18's board term -- the piece not also resting on the board -- and
        is what the bonus and the lift term are paid on.
        """
        left, right, board = self._contacts_owner._piece_contacts()
        pinched = bool(left and right)
        return pinched, bool(pinched and not board)

    def terms(self, held: bool) -> dict[str, float]:
        """Every reward term and the geometry behind it, for the current state.

        Split out from `step` so the sanity table and the unit tests can score
        a pose without stepping the simulation.
        """
        geom = pickup_geometry(self._base, self._pinch_sid, self.dz)
        offset = geom["g_to_pinch"]

        d = float(np.linalg.norm(offset))
        lateral = float(np.linalg.norm(offset[:2]))
        vertical_error = float(abs(offset[2]))
        tilt = geom["tilt"]
        aperture = geom["aperture"]
        lift = geom["lift"]

        position = self.reach_weight * pickup_reach(d) * pickup_posture(tilt)
        gate = pickup_pose_gate(lateral, vertical_error, tilt)
        closing = self.close_weight * gate * pickup_closure(aperture)
        lifting = (
            self.lift_weight * float(np.clip(lift / self.lift_height, 0.0, 1.0))
            if held else 0.0
        )
        return dict(d=d, lateral=lateral, vertical_error=vertical_error, tilt=tilt,
                    aperture=aperture, lift=lift, pose_gate=gate,
                    position_term=position, closing_term=closing, lift_term=lifting)

    def step(self, action: np.ndarray):
        obs, _base_reward, terminated, truncated, info = self.env.step(action)
        if info.get("invalid_transition"):
            # F7: the state is one the solver guard declared corrupt, so nothing
            # may be read from it -- no reward term, no first-grasp bonus, and no
            # advance of the episode-latched first-grasp / hold-streak /
            # pickup_success state. (Measured before this guard: a spike step
            # with both pads "on" the piece paid 7.0 and latched _held_once.)
            # pickup_terms keeps its shape: the last valid geometry, which is
            # also the state the returned observation describes, with nothing
            # paid this step.
            info = dict(info)
            info["pickup_success"] = self._succeeded
            info["pickup_terms"] = dict(
                self._last_terms, first_grasp_bonus=0.0, success_term=0.0,
                hold_streak=self._hold_streak,
            )
            return obs, 0.0, terminated, truncated, info
        grasped = bool(info.get("grasp_success", False))
        pinched, held = self.contact_state()
        t = self.terms(held)

        reward = t["position_term"] + t["closing_term"] + t["lift_term"]

        # Keyed on `held`, not `pinched`: pads-on-the-piece alone also describes
        # a piece still sitting on the board (3c) and an 87 deg incidental
        # contact (3b), both of which collected the full bonus.
        first_grasp = held and not self._held_once
        if first_grasp:
            self._held_once = True
            reward += self.first_grasp_bonus

        # The hold streak counts CONSECUTIVE steps of grasped-and-high-enough;
        # a single step that breaks either condition restarts it.
        if grasped and t["lift"] >= self.lift_height:
            self._hold_streak += 1
        else:
            self._hold_streak = 0
        holding = self._hold_streak >= self.hold_steps
        if holding:
            reward += self.success_weight
            self._succeeded = True

        if terminated and info.get("terminated_reason") in PENALISED_TERMINATIONS:
            reward -= self.failure_penalty

        info = dict(info)
        info["pickup_success"] = self._succeeded
        info["pickup_terms"] = dict(t, first_grasp_bonus=self.first_grasp_bonus if first_grasp else 0.0,
                                    success_term=self.success_weight if holding else 0.0,
                                    hold_streak=self._hold_streak,
                                    pinched=pinched, held=held, grasp_success=grasped)
        self._last_terms = dict(info["pickup_terms"])
        return obs, float(reward), terminated, truncated, info


# --------------------------------------------------------------------------
# D28: pick-and-place reward (place_mode)
# --------------------------------------------------------------------------
# Discount the success lump is sized against. Fixed, not read from the learner: it is the gamma
# both PPO and SAC train with (SB3 default, and VecNormalize's gamma in train.py). Success pays the
# constant value of staying in the placed state forever, R_placed / (1 - gamma) + success_bonus:
# both learners bootstrap a 300-step truncation (DummyVecEnv marks TimeLimit.truncated; PPO adds
# gamma V(terminal obs), SAC samples the transition as not-done -- audit_truncation.py) and the
# policy cannot see the step index, so the value of parking is R / (1 - gamma), not a
# finite-horizon tail.
PLACE_GAMMA = 0.99
PLACE_LIFT_SPAN = 0.03          # lift_frac = clip(lift / 0.03, 0, 1)
PLACE_NEAR_WIDTH = 0.02         # near = exp(-(d_xy / 0.02)^2)
PLACE_CLOSE_LINEAR_SPAN = 0.4   # closeness linear ramp span
PLACE_CLOSE_EXP_WIDTH = 0.03
PLACE_S2_TILT_DEG = 10.0        # S2 needs the piece within this of upright
# S2 pays floor + (ceiling - floor) * open_frac, open_frac = clip((q_ref - q)/(q_ref - q_open), 0, 1).
# q_ref: the gripper driver's qpos fully closed on air (measured 0.7822; the joint range tops out at
# 0.8). It sits ABOVE the qpos at a grasp (0.647-0.651, pads stopped by the piece's stem), so S2
# starts above the floor with no dip at release. q_open is the F2 released threshold.
PLACE_Q_CLOSED = 0.782
PLACE_Q_OPEN = 0.2
PLACE_HOLD_LIFT = 0.03          # the +5 lift bonus: 30 mm ...
PLACE_HOLD_STEPS = 5            # ... held for 5 consecutive steps
N_PLACE_OBS = 3

PLACE_STATES = ("S0", "S1", "S2", "S3")


def place_closeness(d_xy: float) -> float:
    """0.5 linear ramp over 0.4 m plus 0.5 narrow exponential; 1 at the target."""
    d_xy = float(d_xy)
    return (0.5 * (1.0 - min(d_xy / PLACE_CLOSE_LINEAR_SPAN, 1.0))
            + 0.5 * float(np.exp(-d_xy / PLACE_CLOSE_EXP_WIDTH)))


def place_near(d_xy: float) -> float:
    """Gaussian bump, 1 directly over the target."""
    return float(np.exp(-((float(d_xy) / PLACE_NEAR_WIDTH) ** 2)))


def place_open_frac(q: float) -> float:
    """How far the gripper has opened: 0 fully closed, 1 at the released threshold or beyond."""
    return float(np.clip((PLACE_Q_CLOSED - float(q)) / (PLACE_Q_CLOSED - PLACE_Q_OPEN), 0.0, 1.0))


def place_lift_frac(lift: float) -> float:
    return float(np.clip(float(lift) / PLACE_LIFT_SPAN, 0.0, 1.0))


def place_posture_gate_value(tilt_deg: float, lo_deg: float, hi_deg: float) -> float:
    """D29 posture gate: 1 at/below `lo_deg` (upright), 0 at/above `hi_deg`
    (toppled), cosine ease-out between."""
    tilt_deg = float(tilt_deg)
    if tilt_deg <= lo_deg:
        return 1.0
    if tilt_deg >= hi_deg:
        return 0.0
    return 0.5 * (1.0 + float(np.cos(np.pi * (tilt_deg - lo_deg) / (hi_deg - lo_deg))))


def place_carry_potential(d_xy: float, k: float) -> float:
    """D29 carry-potential shaping function: Phi(s) = -k * d_xy(piece, target)."""
    return -float(k) * float(d_xy)


class PlaceObservation(gym.Wrapper):
    """Appends ``desired_goal - achieved_goal`` (`N_PLACE_OBS` entries) to the observation.

    Sits on top of `PickupObservation`, so the entries land after its 15. The
    target is already visible to the policy as ``desired_goal``; this gives it
    the relative form inside ``observation`` (which VecNormalize covers).
    T1 only, like the pick-up features. F7: an invalid step returns this
    wrapper's own last good observation.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(self.unwrapped, RoboChessEnv):
            raise TypeError(f"Expects a RoboChessEnv underneath; got {type(self.unwrapped).__name__}.")
        node = env
        while node is not None and not isinstance(node, PickupObservation):
            if isinstance(node, gym.Wrapper) and type(node).__name__ == "T2ObstacleTask":
                break
            node = getattr(node, "env", None)
        if not isinstance(node, PickupObservation):
            raise TypeError("place_obs needs pickup_obs (PickupObservation) beneath it, T1 only.")
        obs_space = self.env.observation_space["observation"]
        pad = np.full(N_PLACE_OBS, 5.0).astype(obs_space.dtype)
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(
                low=np.concatenate([obs_space.low, -pad]),
                high=np.concatenate([obs_space.high, pad]),
                dtype=obs_space.dtype,
            ),
            "achieved_goal": self.env.observation_space["achieved_goal"],
            "desired_goal": self.env.observation_space["desired_goal"],
        })

    def _extend(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        space = self.observation_space["observation"]
        rel = np.asarray(obs["desired_goal"], dtype=np.float64) - np.asarray(
            obs["achieved_goal"], dtype=np.float64)
        obs = dict(obs)
        obs["observation"] = np.clip(
            np.concatenate([obs["observation"], rel]), space.low, space.high
        ).astype(space.dtype)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._extend(obs)
        self._last_good_obs = {k: v.copy() for k, v in obs.items()}
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("invalid_transition"):
            obs = {k: v.copy() for k, v in self._last_good_obs.items()}
        else:
            obs = self._extend(obs)
            self._last_good_obs = {k: v.copy() for k, v in obs.items()}
        return obs, reward, terminated, truncated, info


class PlaceReward(gym.Wrapper):
    """D28 pick-and-place reward (place_mode, default off). T1 only.

    Design rule: finishing the move pays more than any way of stalling or
    cheating, in total and discounted return. Per-step reward by state::

        S0 approach   not ever_held          D21 position + D21 closing
        S1 grasped    pinched and ever_held  2.0 + 2.0 lift_frac + C closeness
                                             + L near (1 - lift_frac)
        S2 placed     release_ok, not pinched, piece_on_square, tilt < 10 deg
                                             floor + (ceiling - floor) open_frac
                                             (8 at a closed gripper, 9 fully open)
        S3 lost       ever_held, not pinched, not S2   D21 position + closing

    Plus one-time bonuses (+5 first held step, +5 first 30 mm held for 5
    steps, +10 first entry to S2); on success (env is_success, which
    terminates) the constant ``R_placed / (1 - gamma) + success_bonus`` (1020 by
    default), the value of staying placed forever; ``-failure_penalty`` on a
    penalised termination. S2's per-step reward is capped below R_placed, so
    finishing always beats parking in S2.

    The D21 term functions are used unchanged (weights 1.0). S2's horizontal
    test is ``info["piece_on_square"]`` -- the same test success uses.

    F7: an invalid-transition step pays 0, fires nothing and latches nothing;
    only the step counter moves. The counter is 1-based, counts every step and
    is reset in reset(); it is reported in ``place_terms["t"]`` and no longer
    enters the reward.

    D29 additions, each its own constructor flag, default off and
    byte-identical to the above when off:

      * ``posture_gate``: in S1 only, blends the S1 recurring term above with
        the S0/S3 D21 position+closing terms by a cosine gate on piece tilt
        (1 upright, 0 toppled -- see ``place_posture_gate_value``), so a
        held-but-toppled piece stops collecting the S1 rate.
      * ``carry_potential``: potential-based shaping, ``Phi(s) = -k * d_xy``,
        added to every valid step's reward as ``gamma * Phi(s') - Phi(s)``
        (all states, including a terminal step, where Phi(s') is the actual
        post-step value -- not zeroed). An invalid-transition step applies no
        update and reports ``shaping=0.0``.

    ``place_terms`` gains ``posture_gate`` (1.0 when the gate is off or the
    state is not S1) and ``shaping`` (0.0 when the potential is off) either way.
    """

    def __init__(
        self,
        env: gym.Env,
        carry_weight: float = 2.0,
        lower_weight: float = 4.0,
        placed_reward: float = 10.0,
        placed_floor: float = 8.0,
        placed_ceiling: float = 9.0,
        placed_bonus: float = 10.0,
        lift_bonus: float = 5.0,
        success_bonus: float = 20.0,
        failure_penalty: float = 5.0,
        first_grasp_bonus: float = 5.0,
        posture_gate: bool = False,
        gate_tilt_lo_deg: float = 10.0,
        gate_tilt_hi_deg: float = 45.0,
        carry_potential: bool = False,
        carry_potential_k: float = 50.0,
    ):
        super().__init__(env)
        base = self.unwrapped
        if not isinstance(base, RoboChessEnv):
            raise TypeError(f"Expects a RoboChessEnv underneath; got {type(base).__name__}.")
        self._base: RoboChessEnv = base
        self._pinch_sid = _pinch_site_id(base.model)
        self.carry_weight = float(carry_weight)
        self.lower_weight = float(lower_weight)
        self.placed_reward = float(placed_reward)
        self.placed_floor = float(placed_floor)
        self.placed_ceiling = float(placed_ceiling)
        if not self.placed_floor <= self.placed_ceiling < self.placed_reward:
            raise ValueError(f"need placed_floor <= placed_ceiling < placed_reward, got "
                             f"{placed_floor}, {placed_ceiling}, {placed_reward}")
        self.placed_bonus = float(placed_bonus)
        self.lift_bonus = float(lift_bonus)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.first_grasp_bonus = float(first_grasp_bonus)
        self.posture_gate = bool(posture_gate)
        self.gate_tilt_lo_deg = float(gate_tilt_lo_deg)
        self.gate_tilt_hi_deg = float(gate_tilt_hi_deg)
        if not self.gate_tilt_lo_deg < self.gate_tilt_hi_deg:
            raise ValueError(f"need gate_tilt_lo_deg < gate_tilt_hi_deg, got "
                             f"{gate_tilt_lo_deg}, {gate_tilt_hi_deg}")
        self.carry_potential = bool(carry_potential)
        self.carry_potential_k = float(carry_potential_k)
        if not self.carry_potential_k > 0:
            raise ValueError(f"need carry_potential_k > 0, got {carry_potential_k}")
        self.dz = pickup_dz(float(base.model.key_qpos[base._home_key][base._piece_qadr + 2]))
        self._contacts_owner = find_contacts_owner(
            self.env,
            "place_mode needs a task wrapper providing _piece_contacts() "
            "(T1PickAndPlaceTask) between this wrapper and the env.",
        )
        self._reset_latches()

    def _reset_latches(self) -> None:
        self._t = 0
        self._held_once = False
        self._lift_streak = 0
        self._lift_bonus_paid = False
        self._s2_entered = False
        self._last_state = "S0"
        self._last_terms: dict[str, Any] = {}
        self._phi = 0.0

    def reset(self, **kwargs):
        self._reset_latches()
        out = self.env.reset(**kwargs)
        # D29: Phi(s) at reset, read from the same piece-position source
        # (`_achieved_goal`) that `info["xy_distance_to_goal"]` is built from
        # after every step's F10 refresh -- the task wrapper's own reset has
        # already teleported the piece and called mj_forward by the time
        # `self.env.reset()` above returns.
        if self.carry_potential:
            d_xy0 = float(np.linalg.norm(
                self._base._achieved_goal()[:2] - self._base._desired_goal[:2]))
            self._phi = place_carry_potential(d_xy0, self.carry_potential_k)
        self._last_terms = dict(
            state="S0", d_xy=0.0, lift=0.0, lift_frac=0.0, closeness=0.0, near=0.0, open_frac=0.0,
            position_term=0.0, closing_term=0.0, state_reward=0.0, first_grasp_bonus=0.0,
            lift_bonus=0.0, placed_bonus=0.0, success_lump=0.0, failure_penalty=0.0,
            posture_gate=1.0, shaping=0.0,
            pinched=False, held=False, t=0,
        )
        return out

    def step(self, action: np.ndarray):
        obs, _base_reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        if info.get("invalid_transition"):
            info = dict(info)
            info["place_state"] = self._last_state
            info["place_terms"] = dict(
                self._last_terms, state_reward=0.0, first_grasp_bonus=0.0, lift_bonus=0.0,
                placed_bonus=0.0, success_lump=0.0, failure_penalty=0.0, shaping=0.0, t=self._t,
            )
            return obs, 0.0, terminated, truncated, info

        left, right, board = self._contacts_owner._piece_contacts()
        pinched = bool(left and right)
        held = bool(pinched and not board)
        ever_held = bool(info["ever_held"])

        geom = pickup_geometry(self._base, self._pinch_sid, self.dz)
        offset = geom["g_to_pinch"]
        tilt = geom["tilt"]
        lift = geom["lift"]
        d_xy = float(info["xy_distance_to_goal"])
        lift_frac = place_lift_frac(lift)
        closeness = place_closeness(d_xy)
        near = place_near(d_xy)

        position = pickup_reach(float(np.linalg.norm(offset))) * pickup_posture(tilt)
        gate = pickup_pose_gate(float(np.linalg.norm(offset[:2])), float(abs(offset[2])), tilt)
        closing = gate * pickup_closure(geom["aperture"])

        open_frac = place_open_frac(geom["aperture"])
        piece_tilt = self._base._piece_tilt_deg()
        in_s2 = bool(
            info["release_ok"] and not pinched and info["piece_on_square"]
            and piece_tilt < PLACE_S2_TILT_DEG
        )
        if not ever_held:
            state = "S0"
        elif pinched:
            state = "S1"
        elif in_s2:
            state = "S2"
        else:
            state = "S3"

        gate = 1.0
        if state == "S1":
            s1_recurring = (2.0 + 2.0 * lift_frac + self.carry_weight * closeness
                            + self.lower_weight * near * (1.0 - lift_frac))
            if self.posture_gate:
                gate = place_posture_gate_value(piece_tilt, self.gate_tilt_lo_deg, self.gate_tilt_hi_deg)
                state_reward = gate * s1_recurring + (1.0 - gate) * (position + closing)
            else:
                state_reward = s1_recurring
        elif state == "S2":
            state_reward = self.placed_floor + (self.placed_ceiling - self.placed_floor) * open_frac
        else:  # S0 and S3: the D21 approach terms
            state_reward = position + closing
        reward = state_reward

        first_grasp = held and not self._held_once
        if first_grasp:
            self._held_once = True
            reward += self.first_grasp_bonus

        # D21's hold test: grasp_success with the piece >= 30 mm up, consecutive.
        if bool(info.get("grasp_success", False)) and lift >= PLACE_HOLD_LIFT:
            self._lift_streak += 1
        else:
            self._lift_streak = 0
        lift_fired = self._lift_streak >= PLACE_HOLD_STEPS and not self._lift_bonus_paid
        if lift_fired:
            self._lift_bonus_paid = True
            reward += self.lift_bonus

        placed_fired = state == "S2" and not self._s2_entered
        if placed_fired:
            self._s2_entered = True
            reward += self.placed_bonus

        lump = 0.0
        if info.get("is_success", False):
            lump = self.placed_reward / (1.0 - PLACE_GAMMA) + self.success_bonus
            reward += lump

        penalty = 0.0
        if terminated and info.get("terminated_reason") in PENALISED_TERMINATIONS:
            penalty = self.failure_penalty
            reward -= penalty

        # D29 carry potential: F = gamma * Phi(s') - Phi(s), every valid step,
        # every state, including a terminal step (Phi(s') is the actual
        # post-step value here, never zeroed for termination).
        shaping = 0.0
        if self.carry_potential:
            phi_next = place_carry_potential(d_xy, self.carry_potential_k)
            shaping = PLACE_GAMMA * phi_next - self._phi
            self._phi = phi_next
            reward += shaping

        self._last_state = state
        terms = dict(
            state=state, d_xy=d_xy, lift=lift, lift_frac=lift_frac, closeness=closeness,
            near=near, open_frac=open_frac, position_term=position, closing_term=closing,
            state_reward=state_reward,
            first_grasp_bonus=self.first_grasp_bonus if first_grasp else 0.0,
            lift_bonus=self.lift_bonus if lift_fired else 0.0,
            placed_bonus=self.placed_bonus if placed_fired else 0.0,
            success_lump=lump, failure_penalty=penalty,
            posture_gate=gate, shaping=shaping,
            pinched=pinched, held=held, piece_tilt_deg=piece_tilt, t=self._t,
        )
        self._last_terms = dict(terms)
        info = dict(info)
        info["place_state"] = state
        info["place_terms"] = terms
        return obs, float(reward), terminated, truncated, info
