"""P1 scaffold environment for RoboChessGymnasium.

This is deliberately *mechanics only*: enough of a Gymnasium environment to
validate the API contract against the real MuJoCo scene. The actual task
family (T1-T5) and the study's shaped reward arrive in P2 -- see the
placeholder notes on `_compute_reward` and `_sample_desired_goal` below.

Scene: ``assets/scene.xml`` (UR5e + Robotiq 2F-85 over a 78 mm board).
Goal representation is the tracked piece's 3D position -- decision D3 in
DECISIONS.md.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# Board square pitch. 78 mm, not the 57 mm tournament standard -- decision D1.
SQUARE_PITCH = 0.078

DEFAULT_SCENE = Path(__file__).resolve().parents[2] / "assets" / "scene.xml"

# Number of arm actuators; the 7th actuator is the gripper. Split out because
# the two halves are scaled completely differently (see __init__).
N_ARM_ACTUATORS = 6

# D9 (DECISIONS.md): is_success previously had no requirement that the piece
# itself be at rest, and was found to register success while the piece still
# had substantial angular velocity (observed up to 17.7 rad/s during D8's
# release-instability investigation). These are the same threshold and
# streak length established as the corrected criterion throughout that
# investigation (and used identically in scripts/scripted_expert.py's own
# D8 release fix) -- not re-derived here, must stay in sync if ever revisited.
PIECE_ANG_SETTLE_TOL = 0.05  # rad/s
PIECE_SETTLE_CONSECUTIVE_STEPS = 3


class RoboChessEnv(gym.Env):
    """UR5e + Robotiq 2F-85 manipulating a chess piece on a 78 mm board.

    Observation is a goal-conditioned ``Dict``:

    ==================  =====  =============================================
    key                 shape  contents
    ==================  =====  =============================================
    ``observation``     (21,)  arm qpos (6), arm qvel (6), gripper driver
                               joint pos+vel (2), piece pose (3 pos + 4 quat)
    ``achieved_goal``   (3,)   tracked piece's current world position
    ``desired_goal``    (3,)   target world position for that piece
    ==================  =====  =============================================

    Actions are 7-dimensional position targets: 6 arm joints + 1 gripper.

    Success requires the piece to be *placed*, not merely carried near the
    target: within ``goal_tolerance`` in XY, within ``rest_tolerance`` of the
    board surface in z, and with the gripper open past
    ``gripper_open_threshold``. See :meth:`_success_terms`.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        scene_path: str | os.PathLike | None = None,
        max_episode_steps: int = 200,
        frame_skip: int = 20,
        goal_tolerance: float = 0.03,
        rest_tolerance: float = 0.01,
        gripper_open_threshold: float = 0.2,
        normalize_action: bool = True,
        render_mode: str | None = None,
        render_width: int = 640,
        render_height: int = 480,
        piece_body: str = "piece_a",
    ):
        super().__init__()

        self.scene_path = Path(scene_path) if scene_path is not None else DEFAULT_SCENE
        if not self.scene_path.exists():
            raise FileNotFoundError(
                f"Scene not found: {self.scene_path}. If the mesh assets are missing, "
                "run scripts/fetch_assets.sh (the meshes are gitignored)."
            )

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)

        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.goal_tolerance = float(goal_tolerance)
        self.rest_tolerance = float(rest_tolerance)
        self.gripper_open_threshold = float(gripper_open_threshold)
        self.normalize_action = bool(normalize_action)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self._renderer: mujoco.Renderer | None = None
        self._camera: mujoco.MjvCamera | None = None

        # --- model handles -------------------------------------------------
        self._piece_bid = self._body_id(piece_body)
        self._home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self._home_key < 0:
            raise ValueError(f"Scene {self.scene_path} has no 'home' keyframe.")

        # Piece free joint: 7 qpos (pos + quat), 6 qvel.
        piece_jid = self.model.body_jntadr[self._piece_bid]
        if piece_jid < 0 or self.model.jnt_type[piece_jid] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Body {piece_body!r} must have a free joint.")
        self._piece_qadr = int(self.model.jnt_qposadr[piece_jid])
        self._piece_vadr = int(self.model.jnt_dofadr[piece_jid])

        # Gripper aperture proxy: one of the two driver joints. Range [0, 0.8]
        # rad, 0 = fully open. Measured reference points: 0.003 fully open,
        # 0.542 gripping a 30 mm piece, 0.630 closed on nothing -- so the
        # default gripper_open_threshold of 0.2 sits well clear of any aperture
        # at which a 30 mm piece could still be held.
        self._grip_qadr, self._grip_vadr = self._joint_adr("ur_gripper_right_driver_joint")

        # Top face of the board slab: the height a piece's origin sits at when
        # resting on the playing surface. Derived from the model, not hardcoded.
        self._board_surface_z = float(
            self.model.body_pos[self._body_id("board")][2]
            + self.model.geom_size[self._geom_id("board_slab")][2]
        )

        # Geoms whose contact with the arm/gripper means the episode has gone
        # wrong (the arm has driven into the furniture).
        self._forbidden_geoms = {
            self._geom_id(n) for n in ("table", "pedestal", "floor")
        }
        self._arm_geoms = {
            g
            for g in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("ur_")
        }

        # --- action space --------------------------------------------------
        # Bounds come from the model's actuator ctrlrange, never hardcoded.
        # Measured from assets/scene.xml:
        #   arm     shoulder_pan/lift  +/- 6.2831 rad
        #           elbow              +/- 3.1415 rad  (size3_limited)
        #           wrist_1/2/3        +/- 6.2831 rad
        #   gripper fingers_actuator   0 .. 255        (open .. closed)
        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float64).copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float64).copy()
        if not np.all(self.model.actuator_ctrllimited):
            raise ValueError("Every actuator must be ctrllimited for the action space to be finite.")

        if self.normalize_action:
            # The raw ctrlrange spans +/-6.28 for the arm but 0..255 for the
            # gripper -- a 40x scale mismatch across one Box. Exposing that
            # directly makes the action space neither symmetric nor normalized,
            # which both gymnasium's check_env and SB3 flag. So the *exposed*
            # space is [-1, 1]^7 and `_scale_action` maps it onto the ctrlrange
            # read above. Set normalize_action=False to expose the raw ranges.
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=self._ctrl_low.astype(np.float32),
                high=self._ctrl_high.astype(np.float32),
                dtype=np.float32,
            )

        # --- observation space ---------------------------------------------
        # All bounds are finite: an unbounded Box trips check_env's
        # "minimum value is -infinity" warning. Observations are clipped into
        # these bounds before being returned, so containment is guaranteed even
        # if the simulation does something extreme.
        arm_pos_lim = np.full(N_ARM_ACTUATORS, 2 * np.pi, dtype=np.float32)
        arm_vel_lim = np.full(N_ARM_ACTUATORS, 50.0, dtype=np.float32)
        grip_lim = np.array([np.pi, 50.0], dtype=np.float32)
        piece_pos_lim = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        piece_quat_lim = np.ones(4, dtype=np.float32)

        obs_high = np.concatenate(
            [arm_pos_lim, arm_vel_lim, grip_lim, piece_pos_lim, piece_quat_lim]
        )
        # Goal bounds must cover every position the piece can actually reach,
        # not just the board: HER relabels desired_goal from achieved_goal, so a
        # tight board-sized goal box would reject relabelled goals for a piece
        # that fell on the floor.
        goal_high = piece_pos_lim

        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-obs_high, obs_high, dtype=np.float32),
                "achieved_goal": spaces.Box(-goal_high, goal_high, dtype=np.float32),
                "desired_goal": spaces.Box(-goal_high, goal_high, dtype=np.float32),
            }
        )

        self._elapsed_steps = 0
        self._desired_goal = np.zeros(3, dtype=np.float32)
        # D9 fix: consecutive-low-angular-velocity streak gating is_success's
        # new piece-settle requirement. Must be reset in reset() (below) --
        # a forgotten reset here would silently let one episode's settled
        # state leak into the next episode's early steps.
        self._piece_ang_settled_streak = 0

    # ------------------------------------------------------------------ ids
    def _body_id(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise ValueError(f"No body named {name!r} in {self.scene_path}")
        return i

    def _geom_id(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if i < 0:
            raise ValueError(f"No geom named {name!r} in {self.scene_path}")
        return i

    def _joint_adr(self, name: str) -> tuple[int, int]:
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise ValueError(f"No joint named {name!r} in {self.scene_path}")
        return int(self.model.jnt_qposadr[j]), int(self.model.jnt_dofadr[j])

    # --------------------------------------------------------------- action
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Map an agent action onto the model's actuator ctrlrange."""
        action = np.clip(np.asarray(action, dtype=np.float64), self.action_space.low, self.action_space.high)
        if not self.normalize_action:
            return action
        # [-1, 1] -> [ctrl_low, ctrl_high]
        return self._ctrl_low + (action + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)

    def home_action(self) -> np.ndarray:
        """The action that reproduces the home keyframe's ctrl (i.e. hold still).

        Useful for tests and for P2 baselines that need a no-op action.
        """
        ctrl = self.model.key_ctrl[self._home_key].astype(np.float64)
        if not self.normalize_action:
            return ctrl.astype(np.float32)
        span = np.where(self._ctrl_high > self._ctrl_low, self._ctrl_high - self._ctrl_low, 1.0)
        return (2.0 * (ctrl - self._ctrl_low) / span - 1.0).astype(np.float32)

    # ---------------------------------------------------------- observation
    def _get_obs(self) -> dict[str, np.ndarray]:
        qpos, qvel = self.data.qpos, self.data.qvel
        pq = self._piece_qadr
        obs = np.concatenate(
            [
                qpos[:N_ARM_ACTUATORS],
                qvel[:N_ARM_ACTUATORS],
                [qpos[self._grip_qadr], qvel[self._grip_vadr]],
                qpos[pq : pq + 7],
            ]
        ).astype(np.float32)

        achieved = self._achieved_goal()
        space = self.observation_space
        return {
            "observation": np.clip(obs, space["observation"].low, space["observation"].high),
            "achieved_goal": np.clip(
                achieved, space["achieved_goal"].low, space["achieved_goal"].high
            ),
            "desired_goal": np.clip(
                self._desired_goal, space["desired_goal"].low, space["desired_goal"].high
            ),
        }

    def _achieved_goal(self) -> np.ndarray:
        """Tracked piece's world position -- the goal representation (D3)."""
        return self.data.xpos[self._piece_bid].astype(np.float32).copy()

    # --------------------------------------------------------------- reward
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any] | list | None = None,
    ) -> np.ndarray | float:
        """Dense placeholder reward: negative Euclidean distance to the goal.

        NOT the study's shaped-reward formulation -- this exists only so P1 can
        validate the Gymnasium contract end to end. P2 replaces it per task class.

        Batched (HER relabelling) and single-transition inputs both work.
        """
        achieved = np.asarray(achieved_goal, dtype=np.float64)
        desired = np.asarray(desired_goal, dtype=np.float64)
        dist = np.linalg.norm(achieved - desired, axis=-1)
        return -dist

    # ---------------------------------------------------------------- reset
    def _sample_desired_goal(self) -> np.ndarray:
        """Fixed target: one square along +y from the spawn square.

        P1 placeholder. Goal randomisation over legal destination squares comes
        with the T1-T5 task classes in P2.
        """
        spawn = self.model.key_qpos[self._home_key][self._piece_qadr : self._piece_qadr + 3]
        goal = np.array(spawn, dtype=np.float32).copy()
        goal[1] += SQUARE_PITCH
        return goal

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        # Home keyframe restores the arm pose, the gripper (open) and the
        # piece's spawn pose in one shot, since the keyframe carries full qpos.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        mujoco.mj_forward(self.model, self.data)

        self._desired_goal = self._sample_desired_goal()
        self._elapsed_steps = 0
        self._piece_ang_settled_streak = 0  # D9: must not leak across episodes

        obs = self._get_obs()
        info = self._build_info(terminated_reason=None)
        return obs, info

    # ----------------------------------------------------------------- step
    def _piece_off_board(self) -> bool:
        """True once the piece has left the playing surface."""
        board_bid = self._body_id("board")
        slab = self._geom_id("board_slab")
        centre = self.data.xpos[board_bid]
        half = self.model.geom_size[slab][:2]
        pos = self.data.xpos[self._piece_bid]
        outside_xy = bool(np.any(np.abs(pos[:2] - centre[:2]) > half + 0.5 * SQUARE_PITCH))
        below = bool(pos[2] < centre[2] - 0.05)
        return outside_xy or below

    def _bad_contact(self) -> bool:
        """True if any arm/gripper geom is touching the table, pedestal or floor."""
        for k in range(self.data.ncon):
            c = self.data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self._arm_geoms and g2 in self._forbidden_geoms) or (
                g2 in self._arm_geoms and g1 in self._forbidden_geoms
            ):
                return True
        return False

    # -------------------------------------------------------------- success
    def _success_terms(self) -> tuple[bool, bool, bool, bool, float]:
        """The four independent conditions for a successful placement.

        A 3D distance-to-goal test alone is not enough: with a 30 mm tolerance a
        piece still clamped in the gripper 25 mm above its target square scores
        as a success, which is exactly what the scripted expert did before this
        was fixed -- every episode "succeeded" mid-air and never released. So
        success requires all four of:

          on_square  piece within goal_tolerance of the goal in XY
          resting    piece origin at board height (not still elevated)
          released   gripper open wide enough that it cannot be holding the piece
          settled    piece's own angular speed has stayed below
                     PIECE_ANG_SETTLE_TOL for PIECE_SETTLE_CONSECUTIVE_STEPS
                     consecutive step() calls (D9, DECISIONS.md)

        `settled` exists because on_square+resting+released alone can be
        satisfied while the piece is still spinning (D8/D9): a chess piece
        gripped by a rotationally-underconstrained head-only grasp can pass
        through a valid position/height/release state mid-spin, not only once
        genuinely at rest. `_piece_ang_settled_streak` (updated in `step()`)
        tracks this across steps; it is reset to 0 in `reset()` and must never
        be allowed to leak between episodes.

        Note for HER: `is_success` is therefore NOT a pure function of
        (achieved_goal, desired_goal) -- `released` and `settled` depend on
        gripper and velocity state respectively. `compute_reward` remains a
        pure function of the goals, so goal relabelling is unaffected, but a
        relabelled transition's success flag cannot be recomputed from goals
        alone.
        """
        achieved = self._achieved_goal()
        xy_dist = float(np.linalg.norm(achieved[:2] - self._desired_goal[:2]))
        on_square = xy_dist < self.goal_tolerance
        resting = abs(float(achieved[2]) - self._board_surface_z) < self.rest_tolerance
        released = float(self.data.qpos[self._grip_qadr]) < self.gripper_open_threshold
        settled = self._piece_ang_settled_streak >= PIECE_SETTLE_CONSECUTIVE_STEPS
        return on_square, resting, released, settled, xy_dist

    def _is_success(self) -> bool:
        on_square, resting, released, settled, _ = self._success_terms()
        return bool(on_square and resting and released and settled)

    def _build_info(self, terminated_reason: str | None) -> dict[str, Any]:
        dist = float(np.linalg.norm(self._achieved_goal() - self._desired_goal))
        on_square, resting, released, settled, xy_dist = self._success_terms()
        return {
            "is_success": bool(on_square and resting and released and settled),
            "distance_to_goal": dist,
            # The success terms are reported separately so a failure says
            # which condition was missing rather than just "not close enough".
            "xy_distance_to_goal": xy_dist,
            "piece_on_square": bool(on_square),
            "piece_resting": bool(resting),
            "gripper_released": bool(released),
            "piece_settled": bool(settled),
            "piece_height_above_board": float(self._achieved_goal()[2] - self._board_surface_z),
            "piece_off_board": bool(self._piece_off_board()),
            "bad_contact": bool(self._bad_contact()),
            "terminated_reason": terminated_reason,
            "elapsed_steps": int(self._elapsed_steps),
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.data.ctrl[:] = self._scale_action(action)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        # mj_step leaves derived quantities (xpos, contacts) current for the
        # final substep, so no extra mj_forward is needed here.

        self._elapsed_steps += 1

        # D9: consecutive-low-angular-velocity streak for is_success's piece-
        # settle requirement. Updated every step regardless of the other
        # success terms, so it reflects the piece's own physical state, not
        # whether a placement attempt happens to be in progress.
        piece_ang_speed = float(
            np.linalg.norm(self.data.qvel[self._piece_vadr + 3: self._piece_vadr + 6])
        )
        self._piece_ang_settled_streak = (
            self._piece_ang_settled_streak + 1 if piece_ang_speed < PIECE_ANG_SETTLE_TOL else 0
        )

        achieved = self._achieved_goal()
        reward = float(self.compute_reward(achieved, self._desired_goal))
        dist = -reward

        success = self._is_success()
        off_board = self._piece_off_board()
        bad_contact = self._bad_contact()

        reason = None
        if success:
            reason = "success"
        elif off_board:
            reason = "piece_off_board"
        elif bad_contact:
            reason = "bad_contact"

        terminated = reason is not None
        truncated = (not terminated) and self._elapsed_steps >= self.max_episode_steps

        info = self._build_info(reason)
        return self._get_obs(), reward, terminated, truncated, info

    # --------------------------------------------------------------- render
    def _make_camera(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.azimuth, cam.elevation, cam.distance = 140.0, -28.0, 2.1
        cam.lookat[:] = [0.0, 0.45, 0.85]
        return cam

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
            self._camera = self._make_camera()
        self._renderer.update_scene(self.data, self._camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._camera = None
