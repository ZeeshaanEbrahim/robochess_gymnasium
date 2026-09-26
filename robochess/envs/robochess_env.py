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

# --- D14: solver-instability guard ---------------------------------------
# D14 (DECISIONS.md) measured the contact solver injecting ~563 J into the
# tracked piece inside a single macro-step, taking it to 4224 rad/s and
# 257 m/s, while the arm was stationary at a converged joint target with every
# commanded value inside its actuator range and only ~1.75 W of actuator power
# available. The controller cannot produce that, so no controller-side change
# prevents it, and a learning policy driving the same actuators is exposed to
# the same event -- with the corrupted transition landing in its replay buffer.
#
# These are sanity bounds, not task tolerances: anything past them is not a
# hard episode, it is a physics artefact, and the episode is ended and
# labelled rather than allowed to continue into the next observation.
#
# Sizing, against what this project has actually measured:
#   linear   the tool itself peaks at ~2.6 m/s during `traverse`, and a piece
#            dropped from transit height lands at ~1.5 m/s. 10 m/s is ~4x any
#            legitimate figure and ~26x below D14's 257 m/s.
#   angular  D8 measured release spin up to ~50 rad/s at per-substep
#            resolution, the highest legitimate reading in the project.
#            200 rad/s is 4x that and ~21x below D14's 4224 rad/s.
#   energy   a 50 rad/s spin on this piece is ~0.005 J; 1 J is ~200x that
#            and ~680x below D14's 682 J.
PIECE_LIN_SPEED_SANITY = 10.0    # m/s
PIECE_ANG_SPEED_SANITY = 200.0   # rad/s
PIECE_KINETIC_ENERGY_SANITY = 1.0  # joules

# F1 (referee round 2): the ceiling on ROBOT-TO-BOARD contact force -- tolerated
# below this, flagged as bad contact at or above it. Calibrated against what
# real operation has actually produced: the review's own accepted a1->h8
# release case peaked at 166.4N on a single pad (round-1 section-4
# measurement); the scripted expert's clean placements never exceeded ~93N of
# piece-board force. 250N gives headroom above the single largest real-world
# reading this project has measured, without being so high it stops meaning
# anything.
#
# COMBINED TOTAL across every simultaneous robot-geom-to-board contact at the
# read, not per-pad, and not restricted to the two named pad geoms -- this is
# a DELIBERATE divergence from scripts/scripted_expert.py's
# _pad_piece_contacts(), which sums per SIDE (left total, right total) so a
# left/right grasp imbalance is visible to a human reading a diagnostic dump.
# That is a reporting convention for a different purpose. This ceiling is a
# SAFETY gate: the question is how hard the robot presses on the board as a
# whole, not which geom is doing it. A per-pad split would let two pads each
# sit just under 250N while jointly applying far more -- exactly the failure
# mode a safety ceiling exists to catch. Measured directly (referee round 2):
# a real fixture (40mm pinch-target penetration, moderate gripper close)
# settles at 200.62N on the right pad and 147.06N on the left, NEITHER of
# which alone crosses 250N, but which combine to 347.68N -- a per-pad gate
# would never fire on this case; a combined-total gate correctly does.
BOARD_CONTACT_FORCE_CEILING_N = 250.0  # newtons

# F7. The bounds above cannot see a non-finite value: every comparison against
# NaN is False. And a NaN that reaches mj_step is not left in the state for a
# post-step check to find -- MuJoCo does not raise, it resets its own MjData
# and only increments one of these warning counters. Measured: after one
# env.step() with a NaN piece velocity, sim time went 0.12 -> 0.04, the state
# was back at the home pose, every value was finite, and the step looked
# ordinary. So the guard checks both the state (a non-finite value present now)
# and these counters (one that was already consumed). reset() clears them.
_NUMERIC_FAILURE_WARNINGS = tuple(
    int(getattr(mujoco.mjtWarning, name))
    for name in ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC", "mjWARN_BADCTRL")
)

DEFAULT_SCENE = Path(__file__).resolve().parents[2] / "assets" / "scene.xml"

# Number of arm actuators; the 7th actuator is the gripper. Split out because
# the two halves are scaled completely differently (see __init__).
N_ARM_ACTUATORS = 6

# --- action modes ------------------------------------------------------------
# "absolute" (default): each arm action dimension maps linearly across the
#   actuator's full ctrlrange (+-2 pi on most joints), so every step commands an
#   absolute joint target. This is what the 16-cell pilot matrix trained on.
#   Measured there: trained policies sat at +-1, commanding targets hundreds of
#   degrees from the actual joints, with the big-joint actuators force-saturated
#   on 88-99% of steps (results/pilot_diagnosis/s4).
# "delta": each arm action dimension scales to a per-step CHANGE in the joint
#   target, capped at max_joint_delta and applied to the previous target, then
#   clamped to the joint's valid range. The gripper (7th dimension) stays
#   absolute in both modes, so a policy can open or close in one step.
# "cartesian": a 4-dimensional action -- (dx, dy, dz, gripper). The first three
#   scale to a per-step displacement of the PINCH POINT in world axes, added to
#   the pinch's MEASURED position, clamped to a workspace box, and turned into
#   arm joint targets by a top-down-locked mink IK solve. The tool orientation
#   is never commanded: it is pinned to the home keyframe's straight-down pose
#   (D2), which is the only orientation the reach envelope and the scripted
#   expert were ever validated under. The gripper (4th dimension) stays
#   absolute, exactly as in the other two modes.
ACTION_MODES = ("absolute", "delta", "cartesian")

# Delta cap, rad per env step (40 ms). Calibrated from the scripted expert's
# ACTUAL joint motion over 23 successful episodes spanning near and far squares
# (results/delta_calibration/expert_deltas.json): p95 0.04-0.08 on every joint,
# p99 0.11-0.14 on pan/lift/wrist_1/wrist_3 (elbow 0.19, max 0.31). Its COMMANDED
# targets could not be used: they are waypoint jumps (zero on 95.6% of steps,
# up to 1.85 rad otherwise). 0.1 also keeps a maximal step close to the point
# where the PD servos saturate (forcerange/kp = 0.075 rad big joints, 0.056 wrists).
DEFAULT_MAX_JOINT_DELTA = 0.1

# D19: delta-mode target may lead the actual joint by at most this multiple of
# the actuator's saturation error (forcerange/kp). See RoboChessEnv.__init__.
WINDUP_MARGIN_FORCE_MULTIPLE = 2.0

# --- cartesian mode ----------------------------------------------------------
# (dx, dy, dz, gripper).
CARTESIAN_ACTION_DIM = 4

# Metres of pinch displacement per unit action per axis, per env step (40 ms).
# 0.020 is the step size the Cartesian feasibility sweep actually measured
# (results/cartesian_feasibility/): warm-started 20 mm solves moved no joint
# more than 0.0429 rad and produced zero configuration flips over a 30-point
# grid x 6 directions. At the 40 ms macro step it commands at most 0.5 m/s.
DEFAULT_CARTESIAN_STEP = 0.020

# How a solve's joint motion is limited, rad. A solve asking for more than
# CARTESIAN_JOINT_SCALE_TO on some joint is SCALED -- the whole change vector
# multiplied by one factor so its largest component is exactly that -- and only
# a solve above CARTESIAN_JOINT_REJECT_ABOVE is thrown away.
#
# Scaling rather than clipping, and scaling the whole vector rather than each
# joint, is the entire point. Clipping joints independently executes a
# DIFFERENT DIRECTION from the one IK solved for, which rotates the tool out of
# top-down; the next solve then spends its whole budget undoing that rotation,
# a per-joint reject threshold throws that correction away, and the condition
# never clears. Measured on the first implementation, which clipped per joint
# and rejected above 0.15 rad: driving to e4 latched at step 13 with 10.68 deg
# of tilt and 48 of 60 steps rejected, and never recovered. With uniform
# scaling on both, the same drive arrives in 56 steps to 1.34 mm with zero
# rejects and a peak tilt of 0.57 deg.
#
# 0.5 still sits an order of magnitude below the two documented IK hazards it
# exists for: after an IKFailure mink's Configuration is left 1.571 rad from
# the robot, and a target driven down through the table returns a pose
# 2.016 rad away WITHOUT raising anything at all.
CARTESIAN_JOINT_SCALE_TO = 0.1
CARTESIAN_JOINT_REJECT_ABOVE = 0.5

# Workspace box. x/y: the outermost square CENTRE plus one square of margin.
# That margin is not arbitrary -- 3.5*pitch + pitch is identically
# (slab half-extent + half a pitch), i.e. exactly the bound _piece_off_board
# uses, so the box the arm may command coincides with the region where a piece
# still counts as being on the board.
CARTESIAN_XY_MARGIN = SQUARE_PITCH

# The box is axis-aligned but the arm reaches in a disc, so its far corners sit
# outside the workspace: measured from the home configuration, the corners at
# radial 0.9391 m do not converge in 400 iterations (residual 13-34 mm, and a
# retry seeded from a converged neighbour returns a bit-identical residual, so
# it is geometry and not a seeding accident), while radial 0.8710 m converges
# in 6. The target's xy is therefore also clipped radially about the arm base
# axis, whose xy is read from the model in _setup_cartesian. 0.86 is kept as an
# outer backstop, but see CARTESIAN_REACH_R below: reach also shrinks with
# height, and 0.86 alone is not tight enough near the top of the box (measured:
# a sustained push toward high z/large radius hit the 20-iteration cap on
# 102/150 steps, 0 of which converged even at 400 iterations -- a real
# workspace boundary, not a slow-convergence artifact).
CARTESIAN_MAX_RADIUS = 0.86

# Reach as a function of pinch height z, about _base_xy, tool locked top-down.
# From a static-solve bisection (5 mm tolerance): seeded from the home
# configuration, cap 400, along +y and along the direction to a stress-test
# push target -- both directions gave the same radius at every height tested,
# so one table serves both. CARTESIAN_REACH_MARGIN (20 mm) is the smallest of
# {20, 30, 40, 50} mm for which every one of the 5 heights, driven from a seed
# 20 mm further out, converges the SHIPPED cap-20 solve in at most 10
# iterations (measured: 2 iterations at every height, at every margin tried --
# 20 mm was not a close call). Interpolated linearly between the table's
# heights in _scale_action.
CARTESIAN_REACH_Z = (0.7935, 0.90, 1.00, 1.10, 1.1822)
CARTESIAN_REACH_R = (0.9250, 0.9125, 0.8906, 0.8531, 0.8094)
CARTESIAN_REACH_MARGIN = 0.020

# INNER radial limit, about _base_xy: the arm folds on itself near the base, and
# ur_upper_arm_link presses into ur_wrist_2_link. Measured by a static map (IK
# cap 400, tool top-down, seeded from home or the previous radius, then
# mj_forward and the deepest robot-to-robot penetration) over five rays at
# azimuths -45..+45 deg from +y and radii 0.15-0.45 m in 10 mm steps. Each
# CARTESIAN_INNER_R entry is the largest r_min across the rays at that height
# (r_min = smallest radius converging with no self-penetration) minus
# CARTESIAN_INNER_MARGIN, or 0.0 where no penetration was found on any ray
# (z = 1.10 and 1.1822: the sweep's first radius, 0.15 m, was already clean, so
# those heights' 0.15 is a sweep bound, not a zone edge). A target closer to the
# axis than the interpolated limit is scaled outward along its radius. Driven
# stall this prevents: a (0,-1,-1) push wedged the two links together at
# 395-441 N from step 60 on. The nearest squares, d1/e1 at 0.2501 m, only graze
# the zone (65-68 N at GRASP_Z, static clean radius 0.255 m); the limit at
# GRASP_Z, 0.2409 m, sits inside them so they stay reachable.
CARTESIAN_INNER_Z = (0.7935, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.1822)
CARTESIAN_INNER_MARGIN = 0.015
_INNER_MAX_RMIN = (0.26, 0.22, 0.19, 0.19, 0.19, 0.17, None, None)  # None: no zone
CARTESIAN_INNER_R = tuple(
    0.0 if r is None else round(r - CARTESIAN_INNER_MARGIN, 4) for r in _INNER_MAX_RMIN
)

# z floor, as a height above the board's top face (which is itself read from
# the model): the scripted expert's LOWEST commanded pinch height, PLACE_Z =
# 0.7935 with the surface at 0.762. The expert never goes below it, and below
# it is where a target starts passing through the table -- the hazard that
# returns a 2.016 rad joint jump without raising anything.
CARTESIAN_Z_FLOOR_ABOVE_BOARD = 0.0315

# z ceiling: the home keyframe's own pinch height, read from the model in
# _setup_cartesian. It has to be at least that high or the arm would start
# OUTSIDE its own workspace box, with every step clamped to a quarter-metre
# jump down and rejected (measured: 100/100 steps rejected at a 0.932 ceiling).
# Nothing in the task wants to go higher: the expert's transit height CLEAR_Z
# is 0.882, well below the home pose's 1.182.

# IK settings. The frame, solver, timestep, damping and both tolerances are
# the scripted expert's (scripts/scripted_expert.py) deliberately: the expert
# is the only IK configuration this project has validated, and a policy solving
# against different tolerances would not be driving the same interface the
# expert replays through. Only the iteration cap differs; see below.
CARTESIAN_IK_SITE = "ur_attachment_site"
CARTESIAN_PINCH_SITE = "ur_gripper_pinch"
CARTESIAN_IK_SOLVER = "daqp"
CARTESIAN_IK_DT = 0.02
# 20, NOT the expert's 400. The expert solves a handful of waypoints per
# episode and can afford to grind; this solver runs once per env step, and an
# unreachable target costs the full cap every time. Measured: every reachable
# target in the box converges in 4-17 iterations from home, while unreachable
# ones burn all 400 at 180-218 ms each -- 70x the 2.6 ms median step. Capping
# at 20 turns that into a cheap rejection. The expert's own IK is untouched.
CARTESIAN_IK_MAX_ITERS = 20
CARTESIAN_IK_POS_TOL = 1e-3   # metres
CARTESIAN_IK_ORI_TOL = 1e-2   # radians
CARTESIAN_IK_DAMPING = 1e-3

# D9 (DECISIONS.md): is_success previously had no requirement that the piece
# itself be at rest, and was found to register success while the piece still
# had substantial angular velocity (observed up to 17.7 rad/s during D8's
# release-instability investigation). These are the same threshold and
# streak length established as the corrected criterion throughout that
# investigation (and used identically in scripts/scripted_expert.py's own
# D8 release fix) -- not re-derived here, must stay in sync if ever revisited.
PIECE_ANG_SETTLE_TOL = 0.05  # rad/s
PIECE_SETTLE_CONSECUTIVE_STEPS = 3

# F2 (referee review 2026-09-21): the placement criterion accepted a piece 10 mm
# off the board, still translating, or tilted well off vertical. Thresholds
# approved by Zee. `rest_tolerance` deliberately stays 10 mm: it is also T1's
# `lifted` term and D18's grasp check, so the tighter placement height is its own
# constant. The linear-speed and tilt terms reuse PIECE_SETTLE_CONSECUTIVE_STEPS
# and the same streak-in-step() mechanism as the angular-speed term.
PLACEMENT_REST_TOLERANCE = 0.005  # m
PIECE_LIN_SETTLE_TOL = 0.01  # m/s
PIECE_TILT_TOL_DEG = 10.0  # degrees between the piece's z axis and world z

# D27 (referee sweep on experiments/adv-sweep, a8517ba/eefd05e): F2 passed a
# piece pushed onto its square and then "released" with no grasp, and a piece
# dropped from up to 81 mm. Success now also requires the piece to have been
# HELD (both pads on it, off the board) at some point, and to have been let go
# on target from no more than this height: the piece base's height above the
# board at the last step both pads were still on it. Locked at 15 mm before
# any result was seen; do not tune it against outcomes.
PLACEMENT_RELEASE_MAX_HEIGHT = 0.015  # m


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
    target: within ``goal_tolerance`` in XY, within ``PLACEMENT_REST_TOLERANCE``
    of the board surface in z, upright and still, and with the gripper open past
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
        action_mode: str = "absolute",
        max_joint_delta: float = DEFAULT_MAX_JOINT_DELTA,
        cartesian_step: float = DEFAULT_CARTESIAN_STEP,
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
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}")
        if action_mode == "delta" and not self.normalize_action:
            raise ValueError("action_mode='delta' requires normalize_action=True: the delta "
                             "is defined as a fraction of max_joint_delta per unit action.")
        if action_mode == "cartesian" and not self.normalize_action:
            raise ValueError("action_mode='cartesian' requires normalize_action=True: the "
                             "displacement is defined as a fraction of cartesian_step per "
                             "unit action, and the raw ctrlrange has no Cartesian meaning.")
        if not max_joint_delta > 0:
            raise ValueError(f"max_joint_delta must be > 0, got {max_joint_delta!r}")
        if not cartesian_step > 0:
            raise ValueError(f"cartesian_step must be > 0, got {cartesian_step!r}")
        self.action_mode = action_mode
        self.max_joint_delta = float(max_joint_delta)
        self.cartesian_step = float(cartesian_step)

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

        # D27: bodies for the piece-contact scan, so `held` / `pinched` exist in
        # every mode and not only under PickupReward. The task wrappers read the
        # cached result; nothing else scans.
        self._board_bid = self._body_id("board")
        self._left_pad_bid = self._body_id("ur_gripper_left_pad")
        self._right_pad_bid = self._body_id("ur_gripper_right_pad")

        # Geoms whose contact with the robot means the episode has gone wrong
        # (the arm has driven into the furniture). Zero tolerance, unchanged
        # by F1 -- presence alone is bad, no force is read.
        self._forbidden_geoms = {
            self._geom_id(n) for n in ("table", "pedestal", "floor")
        }

        # F1: every collidable geom belonging to the robot, by body ANCESTRY
        # from the robot's base body -- not by a `ur_` geom-name prefix. Only
        # 4 of the 23 collidable robot geoms are named at all (the finger
        # pads); every other collision geom on every link is unnamed, so the
        # prefix filter missed 19 of them. Measured directly: upper-arm and
        # forearm penetrating the table by 25-45mm with the old filter reading
        # no contact at all (round-1 finding, reproduced in round 2's
        # test_f1_contact_classification.py).
        root_body = int(self.model.jnt_bodyid[0])
        while int(self.model.body_parentid[root_body]) != 0:
            root_body = int(self.model.body_parentid[root_body])

        def _under_root(body_id: int) -> bool:
            while body_id != 0:
                if body_id == root_body:
                    return True
                body_id = int(self.model.body_parentid[body_id])
            return False

        self._robot_geoms = {
            g for g in range(self.model.ngeom) if _under_root(int(self.model.geom_bodyid[g]))
        }
        # Every geom on the board body (the slab; its square markings are
        # visual-only, non-colliding), for the tolerant ceiling check below.
        self._board_geoms = {
            g for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) == self._body_id("board")
        }

        # D25: the gripper's two finger subtrees, for the gripper-internal
        # exemption in _bad_contact(). LEFT is everything under the left driver
        # and the left spring link (which carries follower and pad); RIGHT is
        # the mirror. A robot-to-robot contact with one geom in each is
        # gripper-internal: it is what closing on nothing produces (pad1 on
        # pad1). Same-side pairs, the gripper base/mount and every arm link are
        # in neither set, so those contacts still terminate.
        def _subtree_geoms(*root_names: str) -> frozenset[int]:
            roots = {self._body_id(n) for n in root_names}

            def under(body_id: int) -> bool:
                while body_id != 0:
                    if body_id in roots:
                        return True
                    body_id = int(self.model.body_parentid[body_id])
                return False

            return frozenset(
                g for g in self._robot_geoms if under(int(self.model.geom_bodyid[g]))
            )

        self._gripper_left_geoms = _subtree_geoms("ur_gripper_left_driver", "ur_gripper_left_spring_link")
        self._gripper_right_geoms = _subtree_geoms("ur_gripper_right_driver", "ur_gripper_right_spring_link")

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

        # Delta mode clamps each arm target to the valid range of the joint its
        # actuator drives, intersected with the actuator's own ctrlrange (on this
        # model the two coincide; the intersection keeps that from being assumed).
        arm_joints = self.model.actuator_trnid[:N_ARM_ACTUATORS, 0]
        self._arm_target_low = np.maximum(self.model.jnt_range[arm_joints, 0], self._ctrl_low[:N_ARM_ACTUATORS])
        self._arm_target_high = np.minimum(self.model.jnt_range[arm_joints, 1], self._ctrl_high[:N_ARM_ACTUATORS])
        self._arm_target = self.model.key_ctrl[self._home_key][:N_ARM_ACTUATORS].astype(np.float64).copy()
        self._arm_qadr = self.model.jnt_qposadr[arm_joints].copy()

        # D19 anti-windup: in delta mode the target is also held within
        # WINDUP_MARGIN_FORCE_MULTIPLE * forcerange/kp of the ACTUAL joint
        # position, so it cannot run away from a joint that cannot follow
        # (measured before this clamp: shoulder_lift target a median 100 deg
        # ahead at 100k steps). forcerange/kp is the position error at which
        # the PD servo saturates; derived per actuator from the live model.
        # Measured on assets/scene.xml: 0.15 rad pan/lift/elbow, 0.112 wrists.
        kp = self.model.actuator_gainprm[:N_ARM_ACTUATORS, 0].astype(np.float64)
        forcerange = self.model.actuator_forcerange[:N_ARM_ACTUATORS].astype(np.float64)
        if not (np.all(self.model.actuator_forcelimited[:N_ARM_ACTUATORS]) and np.all(kp > 0)):
            raise ValueError("Arm actuators must be force-limited position servos (kp > 0) "
                             "to derive the delta-mode windup margin.")
        self._windup_margin = (
            WINDUP_MARGIN_FORCE_MULTIPLE * np.minimum(-forcerange[:, 0], forcerange[:, 1]) / kp
        )

        # Counts solves rejected this episode (see _solve_cartesian_ik). Set
        # unconditionally so info's schema does not depend on the action mode.
        self._ik_rejects = 0
        if self.action_mode == "cartesian":
            self._setup_cartesian()

        if self.action_mode == "cartesian":
            # 4 dimensions, not the model's 7 actuators: the arm's six joints
            # are produced by IK from three Cartesian numbers.
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(CARTESIAN_ACTION_DIM,), dtype=np.float32
            )
        elif self.normalize_action:
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
        # F2: same pattern and same reset requirement, for linear speed / tilt.
        self._piece_lin_settled_streak = 0
        self._piece_upright_streak = 0
        # D27 per-episode state; set here so the first _build_info() (from
        # reset()) can read it, and cleared again in reset().
        self._contact_flags: tuple[bool, bool, bool] = (False, False, False)
        self._ever_held = False
        self._release_record: tuple[float, float] | None = None

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

    # ------------------------------------------------------------ cartesian
    def _setup_cartesian(self) -> None:
        """Build the IK solver and the workspace box for "cartesian" mode.

        Called from __init__ only when the mode is selected, so absolute and
        delta pay neither the import nor the extra MjData.
        """
        import mink  # local: only this mode needs it

        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, CARTESIAN_IK_SITE)
        pid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, CARTESIAN_PINCH_SITE)
        if sid < 0 or pid < 0:
            raise ValueError(
                f"Scene {self.scene_path} needs both the {CARTESIAN_IK_SITE!r} and "
                f"{CARTESIAN_PINCH_SITE!r} sites for action_mode='cartesian'."
            )
        self._ik_sid, self._pinch_sid = int(sid), int(pid)

        self._mink = mink
        self._ik_configuration = mink.Configuration(self.model)
        self._ik_task = mink.FrameTask(
            frame_name=CARTESIAN_IK_SITE,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )

        # Tool offset and the locked top-down rotation, both read from the home
        # keyframe on a scratch MjData so construction does not disturb
        # self.data. Same derivation as the scripted expert's, and measured
        # from the model rather than hardcoded so it tracks the gripper mount.
        scratch = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, scratch, self._home_key)
        mujoco.mj_forward(self.model, scratch)
        self._tool_length = float(scratch.site_xpos[sid][2] - scratch.site_xpos[pid][2])
        self._tool_rotation = mink.SO3.from_matrix(scratch.site_xmat[sid].reshape(3, 3).copy())

        # Arm base axis, read from the model (not hardcoded): the centre the
        # radial reach limit is measured about.
        self._base_xy = scratch.xpos[self._body_id("arm_base")][:2].copy()

        # Workspace box, derived from the board's own geometry.
        board = self.model.body_pos[self._body_id("board")]
        half = 3.5 * SQUARE_PITCH + CARTESIAN_XY_MARGIN
        self._cartesian_low = np.array(
            [board[0] - half, board[1] - half,
             self._board_surface_z + CARTESIAN_Z_FLOOR_ABOVE_BOARD],
            dtype=np.float64,
        )
        self._cartesian_high = np.array(
            [board[0] + half, board[1] + half,
             float(scratch.site_xpos[pid][2])],
            dtype=np.float64,
        )

    def _solve_cartesian_ik(self, pinch_target: np.ndarray) -> np.ndarray | None:
        """Joint motion putting the pinch point at `pinch_target`, top-down.

        Returns the per-joint CHANGE from the arm's current qpos, or None if
        the solve is rejected -- either it did not converge (the scripted
        expert's IKFailure condition, plus anything mink itself raises) or it
        asks for more than CARTESIAN_JOINT_REJECT_ABOVE on some joint. A change
        between CARTESIAN_JOINT_SCALE_TO and that bound is not rejected: the
        whole vector is scaled down by one factor, which slows the motion
        without altering the direction IK chose.

        A change rather than an absolute pose: the solve is seeded from the
        live qpos, so its answer carries the arm's standing servo error (the
        D7 gravity droop) baked in. Committing that pose as the new target
        would hand the servo a zero-error command and let the arm free-fall a
        droop's worth every step. The DISPLACEMENT is what the solve actually
        determined; it is applied to the previous joint target, which keeps
        whatever offset is holding the arm up.
        """
        site_target = np.asarray(pinch_target, dtype=np.float64).copy()
        site_target[2] += self._tool_length  # site sits tool_length above the pinch
        self._ik_task.set_target(
            self._mink.SE3.from_rotation_and_translation(self._tool_rotation, site_target)
        )

        # Re-sync from the live robot BEFORE the solve, every time, never from
        # whatever mink retained. After an IKFailure its Configuration is left
        # 1.571 rad away from the robot, and warm-starting from that is how a
        # single unreachable target poisons every later step.
        seed = self.data.qpos.copy()
        n = N_ARM_ACTUATORS
        try:
            self._ik_configuration.update(seed)
            solved = None
            for _ in range(CARTESIAN_IK_MAX_ITERS):
                err = self._ik_task.compute_error(self._ik_configuration)
                if (np.linalg.norm(err[:3]) < CARTESIAN_IK_POS_TOL
                        and np.linalg.norm(err[3:]) < CARTESIAN_IK_ORI_TOL):
                    solved = self._ik_configuration.q[:n].copy()
                    break
                vel = self._mink.solve_ik(
                    self._ik_configuration, [self._ik_task], CARTESIAN_IK_DT,
                    CARTESIAN_IK_SOLVER, damping=CARTESIAN_IK_DAMPING,
                )
                # Only the arm may move: the gripper's closed-loop linkage and
                # the piece's free joint are masked out of the integration.
                masked = np.zeros_like(vel)
                masked[:n] = vel[:n]
                self._ik_configuration.integrate_inplace(masked, CARTESIAN_IK_DT)
        except self._mink.exceptions.MinkError:
            return None
        if solved is None:
            return None  # ran out of iterations: the expert's IKFailure

        change = solved - seed[self._arm_qadr]
        largest = float(np.max(np.abs(change)))
        if largest > CARTESIAN_JOINT_REJECT_ABOVE:
            return None
        if largest > CARTESIAN_JOINT_SCALE_TO:
            change = change * (CARTESIAN_JOINT_SCALE_TO / largest)
        return change

    @staticmethod
    def _windup_scale(gap: np.ndarray, change: np.ndarray, margin: np.ndarray) -> float:
        """Largest f in [0, 1] with |gap + f*change| <= max(margin, |gap|) on every joint.

        The uniform replacement for delta mode's per-joint np.clip. Same D19
        margins, same guarantee -- the target still cannot lead a joint that
        cannot follow -- but enforced by slowing the whole motion rather than
        by truncating some of its components and not others, which is what bent
        the tool out of top-down.

        A joint already outside its own margin gets its OWN |gap| as the bound
        instead of margin: moving it further inward is unrestricted, and moving
        it further outward forces f = 0 for the whole vector. A joint inside its
        margin is unchanged (max(margin, |gap|) is just margin there). Without
        this, a joint that starts beyond its margin can never be corrected in
        one step short of a full return to margin, and the function fell back
        to f = 0 -- discarding a solve that was moving the joint the right way,
        the same class of defect as the per-joint clamp this replaced.
        """
        lo, hi = 0.0, 1.0
        for g, c, m in zip(gap, change, margin):
            bound = max(m, abs(g))
            if abs(c) < 1e-15:
                if abs(g) > bound:
                    return 0.0
                continue
            a, b = (-bound - g) / c, (bound - g) / c
            if a > b:
                a, b = b, a
            lo, hi = max(lo, a), min(hi, b)
        if hi < lo or hi < 0.0:
            return 0.0
        return max(0.0, hi)

    # --------------------------------------------------------------- action
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Map an agent action onto the model's actuator ctrlrange.

        In delta mode this ADVANCES the stored arm target, so it must be called
        exactly once per env step (RoboChessEnv.step is its only caller).
        """
        action = np.clip(np.asarray(action, dtype=np.float64), self.action_space.low, self.action_space.high)
        if self.action_mode == "cartesian":
            n = N_ARM_ACTUATORS
            # D19's lesson: the Cartesian target is rebuilt from the MEASURED
            # pinch position every step and never accumulated, so a clamped or
            # rejected command cannot leave a standing error for the next one
            # to add to. Where the arm actually is IS the state.
            target = self.data.site_xpos[self._pinch_sid] + action[:3] * self.cartesian_step
            target = np.clip(target, self._cartesian_low, self._cartesian_high)
            # Then radially, about the arm base axis: the box's far corners are
            # outside the arm's reach and cost a full iteration cap to discover.
            # The limit itself shrinks with height (CARTESIAN_REACH_Z/_R):
            # reach at the top of the box is 115.6 mm less than at the floor
            # (CARTESIAN_REACH_R[0] - CARTESIAN_REACH_R[-1]), and a fixed radius
            # alone let a sustained push toward high z/large radius stall on
            # unreachable targets (measured: 102/150 steps capped, 0 converging
            # even at 400 iterations). It recovers on the first reverse step
            # once the command reverses (measured on 4 of 4 pushes).
            r_limit = min(CARTESIAN_MAX_RADIUS,
                          float(np.interp(target[2], CARTESIAN_REACH_Z, CARTESIAN_REACH_R))
                          - CARTESIAN_REACH_MARGIN)
            offset = target[:2] - self._base_xy
            radius = float(np.linalg.norm(offset))
            if radius > r_limit:
                target[:2] = self._base_xy + offset * (r_limit / radius)
            else:
                # Inner limit (CARTESIAN_INNER_Z/_R): near the base the upper
                # arm and wrist_2 link press into each other. Scaled outward
                # along the radius; radius > 0 guards the division (the box
                # keeps y >= 0.169, so the target cannot sit on the axis).
                r_inner = float(np.interp(target[2], CARTESIAN_INNER_Z, CARTESIAN_INNER_R))
                if 0.0 < radius < r_inner:
                    target[:2] = self._base_xy + offset * (r_inner / radius)
            change = self._solve_cartesian_ik(target)
            if change is None:
                self._ik_rejects += 1  # hold the previous joint targets
            else:
                # D19 anti-windup, same margins as delta mode but applied as
                # ONE factor on the whole change vector rather than per joint.
                actual = self.data.qpos[self._arm_qadr]
                f = self._windup_scale(self._arm_target - actual, change, self._windup_margin)
                self._arm_target = np.clip(self._arm_target + f * change,
                                           self._arm_target_low, self._arm_target_high)
            gripper = self._ctrl_low[n:] + (action[3:] + 1.0) * 0.5 * (self._ctrl_high[n:] - self._ctrl_low[n:])
            return np.concatenate([self._arm_target, gripper])
        if self.action_mode == "delta":
            n = N_ARM_ACTUATORS
            delta = action[:n] * self.max_joint_delta
            # D19 anti-windup first, then the joint-range clamp. The range clamp
            # cannot push the target back outside the margin unless the joint
            # itself sits more than a margin beyond its own range.
            actual = self.data.qpos[self._arm_qadr]
            target = np.clip(self._arm_target + delta,
                             actual - self._windup_margin, actual + self._windup_margin)
            self._arm_target = np.clip(target, self._arm_target_low, self._arm_target_high)
            gripper = self._ctrl_low[n:] + (action[n:] + 1.0) * 0.5 * (self._ctrl_high[n:] - self._ctrl_low[n:])
            return np.concatenate([self._arm_target, gripper])
        if not self.normalize_action:
            return action
        # [-1, 1] -> [ctrl_low, ctrl_high]
        return self._ctrl_low + (action + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)

    def home_action(self) -> np.ndarray:
        """The action that reproduces the home keyframe's ctrl (i.e. hold still).

        Useful for tests and for P2 baselines that need a no-op action. In delta
        mode "hold still" is a zero arm delta; the gripper stays absolute.
        """
        ctrl = self.model.key_ctrl[self._home_key].astype(np.float64)
        if self.action_mode == "cartesian":
            n = N_ARM_ACTUATORS
            span = self._ctrl_high[n:] - self._ctrl_low[n:]
            gripper = 2.0 * (ctrl[n:] - self._ctrl_low[n:]) / span - 1.0
            return np.concatenate([np.zeros(3), gripper]).astype(np.float32)
        if self.action_mode == "delta":
            n = N_ARM_ACTUATORS
            span = self._ctrl_high[n:] - self._ctrl_low[n:]
            gripper = 2.0 * (ctrl[n:] - self._ctrl_low[n:]) / span - 1.0
            return np.concatenate([np.zeros(n), gripper]).astype(np.float32)
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

        D27: this cannot recompute the ever_held / release-height history term
        of the success criterion (it is a function of the goals only), so a
        relabelled transition's success flag is not reproducible from here. HER
        stays off (known issue F3).
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
        # Delta mode's running arm target starts from the keyframe's ctrl. Reset
        # unconditionally (absolute mode never reads it) so no episode inherits
        # the previous one's target.
        self._arm_target = self.data.ctrl[:N_ARM_ACTUATORS].astype(np.float64).copy()
        self._ik_rejects = 0  # cartesian mode; must not leak across episodes

        self._desired_goal = self._sample_desired_goal()
        self._elapsed_steps = 0
        self._piece_ang_settled_streak = 0  # D9: must not leak across episodes
        self._piece_lin_settled_streak = 0  # F2: likewise
        self._piece_upright_streak = 0
        # D27: no latch or release record may leak across episodes. The contact
        # flags are NOT zeroed but scanned on the reset state, so a wrapper's
        # _augment_info reads real flags (home: no pad, board contact).
        self._ever_held = False
        self._release_record = None
        self._refresh_contact_flags()

        obs = self._get_obs()
        self._snapshot_last_good_obs(obs)
        info = self._build_info(terminated_reason=None)
        return obs, info

    def _snapshot_last_good_obs(self, obs: dict[str, np.ndarray] | None = None) -> None:
        """Record the observation the D14 guard hands back if a step is invalid.

        reset() seeds it so an instability on the very first step still has
        something uncorrupted to return, and step() refreshes it after every
        valid step. F7: anything that changes the episode's start state AFTER
        reset() returns -- the T1 task teleports the piece and rewrites the goal
        -- must call this again, or the fallback is the base env's default
        e4->e5 episode instead of the one actually being played.
        """
        if obs is None:
            obs = self._get_obs()
        self._last_good_obs = {k: v.copy() for k, v in obs.items()}

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

    def _is_gripper_internal(self, g1: int, g2: int) -> bool:
        """D25: True for a contact between the gripper's left and right finger
        subtrees (either order). Not a referee violation."""
        return (g1 in self._gripper_left_geoms and g2 in self._gripper_right_geoms) or (
            g2 in self._gripper_left_geoms and g1 in self._gripper_right_geoms
        )

    def _gripper_self_contact(self) -> bool:
        """D25: whether any left-finger/right-finger contact is present now.
        Reported in info so closing on nothing stays visible even though it no
        longer terminates. Its own pass over data.contact, so it is not cut
        short by _bad_contact()'s early return."""
        for k in range(self.data.ncon):
            c = self.data.contact[k]
            if self._is_gripper_internal(int(c.geom1), int(c.geom2)):
                return True
        return False

    def _bad_contact(self) -> bool:
        """True if the robot is in a physically unsafe contact state.

        Three independent categories, checked over the SAME single pass of
        `data.contact` (F1):

          * furniture -- a robot geom touching the table, pedestal or floor.
            Zero tolerance, unchanged.
          * self-contact -- a robot geom touching a DIFFERENT robot geom,
            EXCEPT gripper-internal contact (D25: left finger subtree against
            right finger subtree), which is reported as
            info["gripper_self_contact"] and never terminates. Everything
            else, arm-to-arm and arm-to-gripper included, is unchanged.
            Zero tolerance: there is no legitimate reason for the robot to
            touch itself during T1. No adjacency exclusion is added here on
            top of MuJoCo's own: `filterparent` (enabled, the default) already
            drops every parent-child pair, and the gripper's 6 explicit
            `<exclude>` pairs cover its parallel-linkage siblings that aren't
            parent-child but overlap by design. Confirmed empirically (section
            1.3, round 2): zero robot self-contact of any kind, adjacent or
            not, across 10 regression episodes checked at full sub-step
            fidelity. Any robot-robot contact that reaches `data.contact` at
            all is therefore a genuine candidate, safe to flag outright.
          * board -- robot geoms touching the board slab are tolerated up to
            `BOARD_CONTACT_FORCE_CEILING_N` (see that constant for the
            combined-total-not-per-pad reasoning), logged via the accumulated
            force but not, on their own, below the ceiling, grounds for
            failure.

        Read once per macro-step, at the boundary, after `step()`'s
        `mj_forward()` (F10) has refreshed derived state to match the
        just-integrated qpos. KNOWN, ACCEPTED LIMITATION: a contact that
        begins and fully resolves within the macro-step's interior substeps --
        never present at the final substep -- is invisible to this read.
        Measured directly (round 2 cadence work): of three real flagged-
        contact events in one regression episode, only the one still active at
        the final substep was caught this way; the other two, each 3-4
        substeps long out of 20, were missed. Full sub-step checking would
        catch them but costs roughly 2x per env.step() (measured) and was not
        adopted. This is not a TODO -- it is the accepted cost of checking
        once per macro-step rather than 20 times.
        """
        board_force = 0.0
        board_contact_present = False
        wrench = np.zeros(6)
        for k in range(self.data.ncon):
            c = self.data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)

            if (g1 in self._robot_geoms and g2 in self._forbidden_geoms) or (
                g2 in self._robot_geoms and g1 in self._forbidden_geoms
            ):
                self._last_board_contact_force_n = board_force if board_contact_present else None
                return True

            if (
                g1 in self._robot_geoms
                and g2 in self._robot_geoms
                and int(self.model.geom_bodyid[g1]) != int(self.model.geom_bodyid[g2])
                and not self._is_gripper_internal(g1, g2)
            ):
                self._last_board_contact_force_n = board_force if board_contact_present else None
                return True

            if (g1 in self._robot_geoms and g2 in self._board_geoms) or (
                g2 in self._robot_geoms and g1 in self._board_geoms
            ):
                mujoco.mj_contactForce(self.model, self.data, k, wrench)
                board_force += abs(float(wrench[0]))
                board_contact_present = True

        # Diagnostic aid (info["board_contact_force_n"], via _build_info): the
        # combined total measured here, or None when no robot-board contact
        # was present at all -- not the same as 0.0, which is a real contact
        # that happened to read no force. Set even when this call returns
        # early above (furniture/self-contact took precedence): the value
        # then reflects whatever board contact had already been scanned by
        # that point in the loop, which may undercount. Acceptable for a
        # diagnostic field on an already-bad, episode-ending step.
        self._last_board_contact_force_n = board_force if board_contact_present else None
        return board_force > BOARD_CONTACT_FORCE_CEILING_N

    # -------------------------------------------------------------- success
    def _piece_contact_flags(self) -> tuple[bool, bool, bool]:
        """(left pad, right pad, board): what the tracked piece is touching.

        The one contact scan in the project (D27; it used to live on the T1
        task wrapper). Read from MuJoCo's active contact list (``data.contact``
        geom pairs, mapped to bodies via ``model.geom_bodyid``). Bodies, not
        geom names, so every collision geom on a pad counts; on the board that
        is only ``board_slab`` (its square geoms are visual). Callers that want
        the current step's answer read ``self._contact_flags``, which ``step()``
        and ``reset()`` refresh.
        """
        m, d = self.model, self.data
        piece = self._piece_bid
        left = right = board = False
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if piece == b1:
                other = b2
            elif piece == b2:
                other = b1
            else:
                continue
            left = left or other == self._left_pad_bid
            right = right or other == self._right_pad_bid
            board = board or other == self._board_bid
        return left, right, board

    def _refresh_contact_flags(self) -> None:
        """Scan the current state and cache the result (see _piece_contact_flags)."""
        self._contact_flags = self._piece_contact_flags()

    def _piece_height_above_board(self) -> float:
        """Signed height of the piece's base above the board top, in metres.

        The piece origin is its base (0 mm at the home keyframe). This is the
        height F2's ``resting`` term and ``info["piece_height_above_board"]``
        use, and the one the D27 release record stores.
        """
        return float(self._achieved_goal()[2] - self._board_surface_z)

    def _release_ok(self) -> bool:
        """D27: was the piece held once, and let go on target from low down?

        The record is the (xy error, height) at the LAST step both pads were on
        the piece, so a re-grasp overwrites an earlier release and a push that
        was never a grasp fails ``ever_held``.
        """
        if not self._ever_held or self._release_record is None:
            return False
        xy_error, height = self._release_record
        return bool(xy_error <= self.goal_tolerance and height <= PLACEMENT_RELEASE_MAX_HEIGHT)

    def _success_terms(self) -> tuple[bool, bool, bool, bool, float, bool, bool, bool]:
        """The six independent conditions for a successful placement.

        A 3D distance-to-goal test alone is not enough: with a 30 mm tolerance a
        piece still clamped in the gripper 25 mm above its target square scores
        as a success, which is exactly what the scripted expert did before this
        was fixed -- every episode "succeeded" mid-air and never released. So
        success requires all four of:

          on_square  piece within goal_tolerance of the goal in XY
          resting    piece origin within PLACEMENT_REST_TOLERANCE (5 mm, F2) of
                     board height (not still elevated). Deliberately not
                     `rest_tolerance`, which stays 10 mm for T1's `lifted`.
          released   gripper open wide enough that it cannot be holding the piece
          settled    piece's own angular speed has stayed below
                     PIECE_ANG_SETTLE_TOL for PIECE_SETTLE_CONSECUTIVE_STEPS
                     consecutive step() calls (D9, DECISIONS.md)
          still      (F2) piece's linear speed below PIECE_LIN_SETTLE_TOL for
                     PIECE_SETTLE_CONSECUTIVE_STEPS consecutive step() calls
          upright    (F2) piece's tilt from vertical below PIECE_TILT_TOL_DEG
                     for PIECE_SETTLE_CONSECUTIVE_STEPS consecutive step() calls

        The tuple is (on_square, resting, released, settled, xy_dist, still,
        upright, release_ok): the F2 and D27 terms are appended so existing
        positional readers are unaffected. `still` / `upright` exist because `settled` only watches
        ANGULAR speed, so a piece sliding across the board, or resting tilted on
        its rim, satisfied all four original terms.

        `settled` exists because on_square+resting+released alone can be
        satisfied while the piece is still spinning (D8/D9): a chess piece
        gripped by a rotationally-underconstrained head-only grasp can pass
        through a valid position/height/release state mid-spin, not only once
        genuinely at rest. `_piece_ang_settled_streak` (updated in `step()`)
        tracks this across steps, as do `_piece_lin_settled_streak` and
        `_piece_upright_streak`; all are reset to 0 in `reset()` and must never
        be allowed to leak between episodes.

          release_ok (D27) the piece was HELD (both pads on it, off the board) at
                     some step, and at the last step both pads were on it its
                     XY error was within `goal_tolerance` and its base at most
                     PLACEMENT_RELEASE_MAX_HEIGHT (15 mm) above the board. See
                     `_release_ok`.

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
        resting = abs(float(achieved[2]) - self._board_surface_z) < PLACEMENT_REST_TOLERANCE
        released = float(self.data.qpos[self._grip_qadr]) < self.gripper_open_threshold
        settled = self._piece_ang_settled_streak >= PIECE_SETTLE_CONSECUTIVE_STEPS
        still = self._piece_lin_settled_streak >= PIECE_SETTLE_CONSECUTIVE_STEPS
        upright = self._piece_upright_streak >= PIECE_SETTLE_CONSECUTIVE_STEPS
        return on_square, resting, released, settled, xy_dist, still, upright, self._release_ok()

    def _piece_tilt_deg(self) -> float:
        """Angle between the tracked piece's z axis and world z, in degrees."""
        zz = float(self.data.xmat[self._piece_bid][8])
        return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))

    def _is_success(self) -> bool:
        on_square, resting, released, settled, _, still, upright, release_ok = self._success_terms()
        return bool(
            on_square and resting and released and settled and still and upright and release_ok
        )

    def _piece_kinetic_energy(self) -> float:
        """Translational + rotational KE of the tracked piece, in joules."""
        v = self.data.qvel[self._piece_vadr : self._piece_vadr + 6]
        mass = float(self.model.body_mass[self._piece_bid])
        inertia = self.model.body_inertia[self._piece_bid]
        return float(
            0.5 * mass * float(np.dot(v[:3], v[:3]))
            + 0.5 * float(np.dot(inertia, np.square(v[3:])))
        )

    def _solver_instability(self) -> bool:
        """True if the piece's own state is physically implausible (D14).

        A guard, not a task condition: it fires on states the contact solver
        produced rather than on anything the controller commanded, and it is
        deliberately set far above every legitimate figure this project has
        measured (see the constants' sizing note).

        F7: also fires on any non-finite simulator state, and on a MuJoCo
        numeric-failure warning since the last reset (see
        ``_NUMERIC_FAILURE_WARNINGS`` for why both are needed). The
        finite check comes first because the bounds below are all False for NaN.
        """
        d = self.data
        if not (
            np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(d.qacc).all()
        ):
            return True
        if any(d.warning[i].number for i in _NUMERIC_FAILURE_WARNINGS):
            return True
        v = self.data.qvel[self._piece_vadr : self._piece_vadr + 6]
        return bool(
            float(np.linalg.norm(v[:3])) > PIECE_LIN_SPEED_SANITY
            or float(np.linalg.norm(v[3:])) > PIECE_ANG_SPEED_SANITY
            or self._piece_kinetic_energy() > PIECE_KINETIC_ENERGY_SANITY
        )

    def _build_info(self, terminated_reason: str | None) -> dict[str, Any]:
        """The info dict for the current state.

        ``invalid_transition`` (F7) is the downstream contract: True means the
        transition this info belongs to must not be used as a learning sample.
        It is always present, like ``solver_instability`` (D14), which is the
        specific cause behind it today. When it is True the measurements below
        describe the CORRUPTED live state, not the observation that was returned
        (the last good one), and ``is_success`` is forced False so a corrupted
        step can never read as a success -- matching ``step()``'s own precedence.
        """
        dist = float(np.linalg.norm(self._achieved_goal() - self._desired_goal))
        on_square, resting, released, settled, xy_dist, still, upright, release_ok = (
            self._success_terms()
        )
        unstable = self._solver_instability()
        record = self._release_record
        return {
            "is_success": bool(
                on_square and resting and released and settled and still and upright
                and release_ok and not unstable
            ),
            "distance_to_goal": dist,
            # The success terms are reported separately so a failure says
            # which condition was missing rather than just "not close enough".
            "xy_distance_to_goal": xy_dist,
            "piece_on_square": bool(on_square),
            "piece_resting": bool(resting),
            "gripper_released": bool(released),
            "piece_settled": bool(settled),
            "piece_still": bool(still),
            "piece_upright": bool(upright),
            # D27. None = no release recorded yet; not NaN because check_env's
            # data_equivalence fails on NaN.
            "ever_held": bool(self._ever_held),
            "release_height_mm": float(record[1] * 1000.0) if record else None,
            "release_xy_error_mm": float(record[0] * 1000.0) if record else None,
            "release_ok": bool(release_ok),
            "piece_height_above_board": self._piece_height_above_board(),
            "piece_off_board": bool(self._piece_off_board()),
            "bad_contact": bool(self._bad_contact()),
            # D25: the gripper's fingers touching each other (closing on air).
            # Not a termination; always present so the schema is fixed.
            "gripper_self_contact": bool(self._gripper_self_contact()),
            # F1 diagnostic aid: the combined robot-to-board contact force
            # _bad_contact() just measured (see its docstring/the
            # BOARD_CONTACT_FORCE_CEILING_N comment for the combined-total
            # convention), or None when no such contact was present at all.
            # Always present, like the keys below, so the info schema does
            # not change shape depending on whether contact occurred.
            "board_contact_force_n": self._last_board_contact_force_n,
            # D14. Always present so the info schema does not change shape
            # depending on whether the guard fired.
            "solver_instability": bool(unstable),
            "invalid_transition": bool(unstable),
            "piece_kinetic_energy_j": self._piece_kinetic_energy(),
            # Cartesian mode's rejected-solve count for the episode so far.
            # Always present, and always 0 in the other modes, so the info
            # schema does not change shape with the action mode.
            "ik_rejects": int(self._ik_rejects),
            "terminated_reason": terminated_reason,
            "elapsed_steps": int(self._elapsed_steps),
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.data.ctrl[:] = self._scale_action(action)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        # F10: mj_step does NOT leave derived quantities current. It computes
        # xpos / site_xpos / geom_xpos and the contact list from the state at the
        # START of its last substep and then integrates qpos/qvel, so after it
        # returns they describe the instant one substep before qpos does. Measured
        # by calling mj_forward on the returned state: up to ~1.0 mm after five
        # delta steps, 13 mm in absolute mode, and 0.785 mm on a falling piece's
        # height -- against 5-10 mm referee tolerances. Everything below (success
        # terms, contacts, the observation) must read the state being returned,
        # so refresh once here. mj_forward does not advance time.
        mujoco.mj_forward(self.model, self.data)

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
        # F2: same mechanism for translational speed and tilt (read from the
        # refreshed xmat, per F10).
        piece_lin_speed = float(
            np.linalg.norm(self.data.qvel[self._piece_vadr : self._piece_vadr + 3])
        )
        self._piece_lin_settled_streak = (
            self._piece_lin_settled_streak + 1 if piece_lin_speed < PIECE_LIN_SETTLE_TOL else 0
        )
        self._piece_upright_streak = (
            self._piece_upright_streak + 1 if self._piece_tilt_deg() < PIECE_TILT_TOL_DEG else 0
        )

        achieved = self._achieved_goal()
        reward = float(self.compute_reward(achieved, self._desired_goal))
        dist = -reward

        # D14: checked before anything else reads the state. A solver blow-up
        # can satisfy the ordinary conditions on its way past them (the piece
        # can cross its goal square at 250 m/s), so this takes precedence over
        # success, off-board and bad-contact alike -- otherwise the episode
        # would be recorded under whichever of those it happened to trip.
        unstable = self._solver_instability()
        if unstable and not np.isfinite(reward):
            # A NaN reward would poison VecNormalize's running statistics for the
            # rest of the run. The transition is flagged invalid regardless; this
            # only keeps the value finite. (Finite corrupted rewards are left as
            # they were -- see invalid_transition in _build_info.)
            reward = 0.0

        # D27: on a corrupted step nothing is read from the contact list -- the
        # cached flags, the ever_held latch and the release record all keep their
        # last valid values (the same skip PickupReward makes on
        # invalid_transition). Scan -> cache -> latch, then the success terms
        # below read the updated record.
        if not unstable:
            self._refresh_contact_flags()
            left, right, board = self._contact_flags
            if left and right:  # pinched
                if not board:  # held
                    self._ever_held = True
                # Overwritten on EVERY pinched step, so it is the state at the
                # last step the pads were on the piece, i.e. the release.
                self._release_record = (
                    float(np.linalg.norm(achieved[:2] - self._desired_goal[:2])),
                    self._piece_height_above_board(),
                )

        success = (not unstable) and self._is_success()
        off_board = self._piece_off_board()
        bad_contact = self._bad_contact()

        reason = None
        if unstable:
            reason = "solver_instability"
        elif success:
            reason = "success"
        elif off_board:
            reason = "piece_off_board"
        elif bad_contact:
            reason = "bad_contact"

        # Truncation, not termination: the state is an artefact, not an MDP
        # terminal state, and labelling it terminal would teach a value
        # function that this configuration is a real end-state of the task.
        if unstable:
            terminated = False
            truncated = True
        else:
            terminated = reason is not None
            truncated = (not terminated) and self._elapsed_steps >= self.max_episode_steps

        info = self._build_info(reason)

        if unstable:
            # D14: hand back the last uncorrupted observation, not this step's.
            # SB3 bootstraps the value function off the FINAL observation when
            # an episode is truncated (unlike termination, where it does not),
            # so returning the spike state would feed a 250 m/s piece position
            # straight into the value estimate -- the one place the corrupted
            # state would still reach the learner despite the episode ending.
            # truncated/info are unchanged: the episode still ends here and is
            # still labelled solver_instability. The live state is not read for
            # the observation at all: it may be non-finite.
            obs = {k: v.copy() for k, v in self._last_good_obs.items()}
        else:
            obs = self._get_obs()
            self._snapshot_last_good_obs(obs)
        return obs, reward, terminated, truncated, info

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
