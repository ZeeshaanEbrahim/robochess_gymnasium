"""T3 capture/remove skeleton: one controller, fixed roles, no T3 reward.

Role mapping: piece_a is the capturing piece; piece_b is the captured piece.

Only this wrapper owns the two-stage success and post-step termination.
Existing base helpers run on a separate captured-piece context, never by
patching base methods or changing its tracked-piece handles. The shared
base class and T1/T2 are untouched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from types import SimpleNamespace

import gymnasium as gym
import mujoco
import numpy as np

from robochess.envs.robochess_env import (
    PLACEMENT_RELEASE_MAX_HEIGHT,
    PLACEMENT_REST_TOLERANCE,
    PIECE_ANG_SETTLE_TOL,
    PIECE_LIN_SETTLE_TOL,
    PIECE_SETTLE_CONSECUTIVE_STEPS,
    PIECE_TILT_TOL_DEG,
    RoboChessEnv,
)
from robochess.tasks.t1_pick_and_place import T1PickAndPlaceTask
from robochess.tasks.t2_obstacle_pick_and_place import (
    OBSTACLE_DISPLACEMENT_TOL,
    OBSTACLE_TILT_TOL,
)

T3_SCENE_PATH = Path(__file__).resolve().parents[2] / "assets" / "scene_t3.xml"
T3_EXCLUDED_SQUARES = ("d1", "e1")


def reject_t3_training_options(options: dict[str, Any]) -> None:
    """Reject unsupported task/reward options before constructing any model."""
    keys = set(options)
    replay = options.get("replay_buffer_class")
    if (any(key == "her" or key.startswith("her_") or key.endswith("_her") for key in keys)
            or getattr(replay, "__name__", replay) == "HerReplayBuffer"):
        raise ValueError("T3 goal covers one piece; HER not supported")
    if "pickup_obs" in keys or any(key.startswith("place_") for key in keys):
        raise ValueError("T3 does not support pickup_obs or place-reward keys")
    if any("reward" in key or key in ("pickup_mode", "shaped_staged") for key in keys):
        raise ValueError("T3 reward not implemented")


def make_t3_env(
    *, start_square: str | None = None, target_square: str | None = None,
    **env_kwargs: Any,
) -> "T3CaptureTask":
    """Build the task-only environment; reward and HER options are errors.

    step() passes through the base's diagnostic distance placeholder. This
    is not a T3 reward formulation, and training configs are rejected.
    """
    reject_t3_training_options(env_kwargs)
    env_kwargs.setdefault("scene_path", T3_SCENE_PATH)
    base = RoboChessEnv(**env_kwargs)
    try:
        return T3CaptureTask(base, start_square=start_square, target_square=target_square)
    except Exception:
        base.close()
        raise


class T3CaptureTask(T1PickAndPlaceTask):
    """Remove piece_b to the pad, then place piece_a on its vacated square."""

    def __init__(
        self, env: RoboChessEnv, start_square: str | None = None,
        target_square: str | None = None, **unsupported: Any,
    ):
        reject_t3_training_options(unsupported)
        if unsupported:
            raise TypeError(f"Unknown T3 option(s): {sorted(unsupported)}")
        if not isinstance(env, RoboChessEnv):
            raise TypeError("T3 wraps RoboChessEnv directly; no reward/observation wrappers below it.")
        super().__init__(env, start_square=start_square, target_square=target_square,
                         excluded_squares=T3_EXCLUDED_SQUARES)
        b = self._base
        if b._piece_bid != b._body_id("piece_a"):
            raise ValueError("T3's capturing piece must be the base tracked piece_a.")
        self._captured_bid = b._body_id("piece_b")
        joint = int(b.model.body_jntadr[self._captured_bid])
        self._captured_qadr = int(b.model.jnt_qposadr[joint])
        self._captured_vadr = int(b.model.jnt_dofadr[joint])
        # The original helpers can run as plain functions on this context.
        # It shares the live model/data but has piece_b's own addresses.
        self._captured_view = SimpleNamespace(
            model=b.model, data=b.data, _piece_bid=self._captured_bid,
            _piece_qadr=self._captured_qadr, _piece_vadr=self._captured_vadr,
            _left_pad_bid=b._left_pad_bid, _right_pad_bid=b._right_pad_bid,
            _board_bid=b._board_bid, _body_id=b._body_id, _geom_id=b._geom_id,
        )
        self._captured_view._piece_kinetic_energy = (
            lambda: RoboChessEnv._piece_kinetic_energy(self._captured_view)
        )
        pad = b._geom_id("capture_pad")
        if int(b.model.geom_bodyid[pad]) != self._board_bid:
            raise ValueError("T3 capture_pad must belong to the board body.")
        centre = b.model.body_pos[self._board_bid] + b.model.geom_pos[pad]
        half = b.model.geom_size[pad]
        self._zone_low = centre[:2] - half[:2]
        self._zone_high = centre[:2] + half[:2]
        self._zone_top = float(centre[2] + half[2])
        self.zone_position = np.array([*centre[:2], self._zone_top], dtype=np.float64)
        self._captured_radius = float(b.model.geom_size[b._geom_id("piece_b_base")][0])

        space = env.observation_space["observation"]
        # Reuse the base piece-pose and goal bounds (D12), then a binary stage.
        low = np.concatenate([space.low, space.low[-7:],
                              env.observation_space["desired_goal"].low, [0]])
        high = np.concatenate([space.high, space.high[-7:],
                               env.observation_space["desired_goal"].high, [1]])
        self.observation_space = gym.spaces.Dict({
            **env.observation_space.spaces,
            "observation": gym.spaces.Box(low.astype(space.dtype), high.astype(space.dtype),
                                           dtype=space.dtype),
        })
        self._reset_t3_state()

    def _reset_t3_state(self) -> None:
        self.stage = 0
        self.order_violation = False
        self._order_reason = None
        self._captured_ever_held = False
        self._captured_release_record = None
        self._captured_streaks = [0, 0, 0]
        self._captured_terms: dict[str, Any] = {}
        self._captured_removal = False
        self._off_board_piece = None
        self._capturing_start_xyz = None
        self._last_good_t3_obs = None

    def _for_captured(self, method):
        """Call the unchanged helper on the separate piece_b context."""
        return method(self._captured_view)

    def _captured_contacts(self) -> tuple[bool, bool, bool]:
        return RoboChessEnv._piece_contact_flags(self._captured_view)

    def _inside_pad(self, xy: np.ndarray, *, whole_base: bool = False) -> bool:
        margin = self._captured_radius if whole_base else 0.0
        return bool(np.all(xy >= self._zone_low + margin)
                    and np.all(xy <= self._zone_high - margin))

    def _either_piece_off_board(self) -> bool:
        b = self._base
        # The ordinary off-board envelope includes half a square of margin.
        # The explicit pad check prevents that overlap giving piece_a an exemption.
        if (RoboChessEnv._piece_off_board(b)
                or self._inside_pad(b.data.xpos[b._piece_bid, :2])):
            self._off_board_piece = "piece_a"
            return True
        captured_off = self._for_captured(RoboChessEnv._piece_off_board)
        pos = b.data.xpos[self._captured_bid]
        below = bool(pos[2] < b.data.xpos[self._board_bid, 2] - 0.05)
        if below or (captured_off and not self._inside_pad(pos[:2])):
            self._off_board_piece = "piece_b"
            return True
        self._off_board_piece = None
        return False

    def _read_captured_terms(self, *, advance: bool) -> None:
        b = self._base
        pos = b.data.xpos[self._captured_bid].copy()
        velocity = b.data.qvel[self._captured_vadr:self._captured_vadr + 6]
        left, right, board = self._captured_contacts()
        pinched = left and right
        held = pinched and not board
        height = float(pos[2] - self._zone_top)
        lin_speed = float(np.linalg.norm(velocity[:3]))
        ang_speed = float(np.linalg.norm(velocity[3:]))
        tilt = self._for_captured(RoboChessEnv._piece_tilt_deg)
        if advance:
            self._captured_ever_held = self._captured_ever_held or held
            if pinched:
                self._captured_release_record = (*map(float, pos[:2]), height)
            conditions = (lin_speed < PIECE_LIN_SETTLE_TOL,
                          ang_speed < PIECE_ANG_SETTLE_TOL, tilt < PIECE_TILT_TOL_DEG)
            self._captured_streaks = [
                streak + 1 if valid else 0
                for streak, valid in zip(self._captured_streaks, conditions)
            ]
        record = self._captured_release_record
        release_ok = bool(
            self._captured_ever_held and record is not None
            and self._inside_pad(np.array(record[:2]), whole_base=True)
            and record[2] <= PLACEMENT_RELEASE_MAX_HEIGHT
        )
        still, settled, upright = (
            streak >= PIECE_SETTLE_CONSECUTIVE_STEPS for streak in self._captured_streaks
        )
        terms = dict(
            in_zone=self._inside_pad(pos[:2], whole_base=True),
            resting=abs(height) < PLACEMENT_REST_TOLERANCE,
            still=still, settled=settled, upright=upright,
            released=not left and not right, pinched=bool(pinched), held=bool(held),
            board_contact=bool(board), ever_held=bool(self._captured_ever_held),
            release_ok=release_ok, height_mm=height * 1000,
            linear_speed=lin_speed, angular_speed=ang_speed, tilt_deg=tilt,
        )
        self._captured_terms = terms
        self._captured_removal = all(
            terms[key] for key in ("in_zone", "resting", "still", "settled",
                                  "upright", "released", "release_ok")
        )

    def _update_order(self) -> None:
        if self.stage != 0 or self.order_violation:
            return
        b = self._base
        left, right, _ = b._contact_flags
        displacement = np.linalg.norm(
            b.data.xpos[b._piece_bid] - self._capturing_start_xyz)
        tilt = np.radians(b._piece_tilt_deg())
        if left and right:
            reason = "capturing_piece_pinched"
        elif displacement > OBSTACLE_DISPLACEMENT_TOL:
            reason = "capturing_piece_displaced"
        elif tilt > OBSTACLE_TILT_TOL:
            reason = "capturing_piece_tilted"
        else:
            return
        self.order_violation = True
        self._order_reason = reason

    def _extend_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        b = self._base
        captured = b.data.qpos[self._captured_qadr:self._captured_qadr + 7]
        space = self.observation_space["observation"]
        obs = dict(obs)
        obs["observation"] = np.clip(
            np.concatenate([obs["observation"], captured, self.zone_position, [self.stage]]),
            space.low, space.high,
        ).astype(space.dtype)
        return obs

    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        info = super()._augment_info(info)
        record = self._captured_release_record
        info.update(
            stage=self.stage, order_violation=self.order_violation,
            order_violation_reason=self._order_reason,
            captured_removal=bool(self._captured_removal),
            captured_terms=dict(self._captured_terms),
            captured_ever_held=bool(self._captured_ever_held),
            captured_release_height_mm=record[2] * 1000 if record is not None else None,
            captured_release_position=record[:2] if record is not None else None,
            capture_zone_position=tuple(map(float, self.zone_position)),
            off_board_piece=self._off_board_piece,
        )
        return info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        reject_t3_training_options(options or {})
        self._reset_t3_state()
        super().reset(seed=seed, options=options)
        b = self._base
        q, v = self._captured_qadr, self._captured_vadr
        b.data.qpos[q:q + 3] = [*self._board_world_xy(self.current_target_square),
                                self._piece_rest_z]
        b.data.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        b.data.qvel[v:v + 6] = 0
        mujoco.mj_forward(b.model, b.data)
        b._refresh_contact_flags()
        self._capturing_start_xyz = b.data.xpos[b._piece_bid].copy()
        self._read_captured_terms(advance=False)
        b._snapshot_last_good_obs()
        obs = self._extend_obs(b._get_obs())
        self._last_good_t3_obs = {key: value.copy() for key, value in obs.items()}
        info = self._augment_info(b._build_info(terminated_reason=None))
        info["capturing_t1_success"] = bool(info["is_success"])
        info["is_success"] = False
        info["captured_solver_instability"] = False
        return obs, info

    def step(self, action: np.ndarray):
        # Run the base normally. Its success is a candidate T1 result only;
        # the T3 wrapper owns the externally returned termination decision.
        obs, reward, _terminated, _truncated, info = self.env.step(action)
        t1_success = bool(info["is_success"])
        captured_unstable = RoboChessEnv._solver_instability(self._captured_view)
        invalid = bool(info["invalid_transition"] or captured_unstable)
        if invalid:
            # D14/F7: return the complete last valid T3 observation, and do
            # not advance captured history, settle streaks, stage or order.
            obs = {key: value.copy() for key, value in self._last_good_t3_obs.items()}
            terminated, truncated, reason, success = False, True, "solver_instability", False
            info = dict(info, solver_instability=True, invalid_transition=True)
            if not np.isfinite(reward):
                reward = 0.0
        else:
            off_board = self._either_piece_off_board()
            self._update_order()
            self._read_captured_terms(advance=True)
            if self._captured_removal:
                self.stage = 1
            success = bool(
                self.stage == 1 and self._captured_removal
                and not self.order_violation and t1_success
                and not off_board and not info["bad_contact"]
            )
            if off_board:
                reason = "piece_off_board"
            elif info["bad_contact"]:
                reason = "bad_contact"
            elif success:
                reason = "success"
            else:
                reason = None
            terminated = reason is not None
            truncated = (not terminated and self._base._elapsed_steps
                         >= self._base.max_episode_steps)
            info = dict(info, piece_off_board=off_board)
            obs = self._extend_obs(obs)
            self._last_good_t3_obs = {key: value.copy() for key, value in obs.items()}
        info = self._augment_info(info)
        info.update(capturing_t1_success=t1_success, captured_solver_instability=captured_unstable,
                    is_success=success, terminated_reason=reason)
        return obs, reward, terminated, truncated, info

    def compute_reward(self, *args, **kwargs):
        raise ValueError("T3 goal covers one piece; HER not supported")
