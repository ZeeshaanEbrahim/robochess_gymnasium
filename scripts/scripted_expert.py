#!/usr/bin/env python
"""Scripted expert for the P1 single-piece pick-and-place task.

Roadmap check, not a baseline: if a hand-coded controller cannot reliably solve
single-piece pick-and-place, the environment or its tolerances are wrong, and no
amount of RL will fix that. This exists to find that out before any RL runs.

The controller is an open-loop waypoint sequence. Each waypoint is an IK-solved
joint target for the UR5e's `attachment_site` frame, locked to the same top-down
orientation the P1 reach-envelope sweep validated (decision D2). Targets are fed
through `RoboChessEnv.step()` as ordinary actions -- the env's physics, action
scaling and termination logic are never bypassed.

Nothing here tunes the environment. Success is whatever the env's own
`info["is_success"]` says.

Usage:
    python scripts/scripted_expert.py --episodes 50
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import mink
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robochess.envs import RoboChessEnv  # noqa: E402
from robochess.tasks import T1PickAndPlaceTask  # noqa: E402

# --- controller geometry -------------------------------------------------
# Heights are for the *pinch* point (between the fingertips), in world z.
# The board playing surface is at z = 0.762.
#
# GRASP_Z puts the pads around the piece's WAIST: the narrow stem (r 8 mm,
# 0.780-0.796) sits between them while they clamp the base's top edge and the
# head's underside (0.7885-0.8073 with the 9.4 mm half-height pads). That is a
# form-closure grasp, and the contact patch straddles the centre of mass at
# 0.7795 rather than sitting under it -- a pure base grasp puts the whole patch
# below the COM and the piece pivots out during the lift (measured: 58 deg of
# tilt, ending on its side); a pure head grasp is a single sphere and simply
# rotates in the fingers (measured: piece placed 38 mm off target).
#
# RECALIBRATED from 0.8035 to 0.79925 (D7, attempt 5). The original value was
# tuned by outcome while the arm still had an uncorrected gravity-droop error,
# so what P1 validated as a stem grasp was really the COMMAND 0.8035 minus
# ~5.6 mm of sag at e4 -- i.e. an actual pinch height of 0.7979. The sag was
# doing the calibration. With task-space residual compensation the arm now
# tracks the command, so commanding 0.8035 lands the pads on the head alone.
# 0.7979 is therefore not a new grasp geometry: it is the geometry P1 actually
# validated at 100/100, now commanded explicitly and reach-independently
# instead of arrived at by however far the arm happened to sag on one square.
GRASP_Z = 0.79925
CLEAR_Z = 0.882  # 12 cm above the board: transit height
# Set the piece down before opening. Releasing from higher drops it and topples it.
PLACE_Z = 0.7935

GRIPPER_OPEN_CTRL = 0.0
# Full close command: the 16 mm stem is narrower than the gripper's 17 mm
# free-space gap at ctrl=200, so anything less never actually clamps it.
# Grip force stays bounded by the actuator's own forcerange either way.
GRIPPER_GRIP_CTRL = 255.0


# General settle-check fix (D7 mitigation follow-up): a waypoint is only
# declared "settled" after this many CONSECUTIVE macro-steps of low arm and
# gripper velocity, not a single instantaneous reading. 3 is chosen as the
# smallest value that comfortably exceeds a one-step contact-deceleration
# transient (the specific failure observed) while staying small relative to
# every waypoint's budget (>=25 macro-steps) so it doesn't meaningfully
# lengthen episodes that were already genuinely settling.
SETTLE_CONSECUTIVE_STEPS = 3

# --- task-space residual compensation (D7 fix, attempt 5) ----------------
# The UR5e's actuators are plain affine PD position servos with no gravcomp
# (gainprm 2000/500, confirmed inherited unchanged from mujoco_menagerie), so
# holding a joint target against gravity leaves a STEADY-STATE error of
# roughly qfrc_bias/kp. That is a static equilibrium, not a slow transient:
# measured 1.54 deg at shoulder_lift on the far rank vs 1.615 deg predicted.
# In task space it reaches ~27 mm of vertical sag and ~12 mm of horizontal
# under-reach at rank 8 -- enough to drop the grasp off the piece's stem and
# back onto its base (below the centre of mass, the exact failure P1 already
# diagnosed), and to push the piece past the gripper pad's 11.00 mm
# half-extent, which is what separated catastrophic from surviving squares
# across all 18 diagnosed squares.
#
# Attempts 1-4 all failed because none of them could move a static
# equilibrium: two adjusted the CLOSING phase (the piece is already
# mispositioned before closing starts), one retargeted which site IK aims at
# (the error is downstream of IK, in physical tracking), and one simply
# waited longer for a floor that never converges.
#
# This corrects the only thing that can be corrected: the COMMANDED target.
# Drive to the waypoint, measure where the pinch site actually ended up, add
# the shortfall back into the IK target, re-solve, re-drive. Because the
# error is a stable static offset, the correction converges in one or two
# passes rather than oscillating. Measuring the real pinch site also absorbs
# wrist-tilt error automatically, which is what retargeting IK could not do.
RESIDUAL_TOL = 1e-3          # 1 mm: the "sub-mm" convergence bar
MAX_RESIDUAL_ITERS = 2       # hard cap; warn rather than loop on
RESIDUAL_BUDGET = 10         # macro-steps per correction pass (small move)

# Waypoints where the pinch point's absolute accuracy actually determines the
# outcome, so they are worth spending correction steps on:
#   descend  the pre-close grasp pose -- sets grip height on the piece AND
#            the in-pad alignment that D7 turned on.
# "place" was tried and rejected: by the time it runs, the piece is already
# resting in contact with the board, so the measured "shortfall" partly
# reflects the board pushing back rather than free-space tracking error, and
# the correction loop fights that contact -- at larger RESIDUAL_BUDGET values
# it drove the piece back off the board entirely. The transit waypoints
# (approach, lift, traverse, retreat) only need to clear obstacles, and
# "close"/"release" deliberately REUSE the pose their predecessor converged
# to, rather than re-solving back to the uncompensated target.
COMPENSATED_WAYPOINTS = frozenset({"descend"})

IK_SOLVER = "daqp"
IK_DT = 0.02
IK_MAX_ITERS = 400
IK_POS_TOL = 1e-3  # metres
IK_ORI_TOL = 1e-2  # radians

ARM_DOF = 6

# D7 diagnostics (DECISIONS.md): phases logged when `run_episode(log=...)`
# is given a list. "Through the approach and grasp-close phase" is read as
# the whole pre-lift span -- approach, descend, close -- since descend is
# where the arm actually travels down to grasp height and any tilt drift or
# torque buildup could start there, not only once the gripper starts closing.
LOGGED_PHASES = frozenset({"approach", "descend", "close"})


class IKFailure(RuntimeError):
    """Raised when the IK solver cannot reach a waypoint."""


class ScriptedExpert:
    """Open-loop waypoint controller driving RoboChessEnv through step()."""

    # settle_tol: BASELINE 0.02 rad. D7 fix attempt 4 tightened this to 0.002
    # and it failed its own verification -- every waypoint on every square
    # exhausted its full step budget without ever reading "settled", and the
    # catastrophic squares stayed catastrophic even at full budget. Reverted;
    # 0.02 is the value all pre-attempt-4 measurements were taken under and is
    # the base config for any new D7 data.
    SETTLE_TOL_DEFAULT = 0.02

    def __init__(self, env: RoboChessEnv, settle_tol: float = SETTLE_TOL_DEFAULT, max_steps_per_waypoint: int = 40):
        self.env = env
        # `env` may be RoboChessEnv directly or wrapped (e.g. T1PickAndPlaceTask):
        # step()/reset() go through whatever wrapping is present, via self.env,
        # but MuJoCo internals (model, data, private joint/keyframe handles) are
        # only ever on the base env, and gymnasium's Wrapper no longer forwards
        # arbitrary attribute access -- so those go through `self.base` instead.
        self.base: RoboChessEnv = env.unwrapped
        self.model = self.base.model
        self.settle_tol = settle_tol
        self.max_steps_per_waypoint = max_steps_per_waypoint

        self.configuration = mink.Configuration(self.model)
        self.task = mink.FrameTask(
            frame_name="ur_attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        # Only the arm may move during IK. The gripper's closed-loop linkage and
        # the piece's free joint are masked out of the integration step below.
        self._arm_dofs = np.arange(ARM_DOF)

        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        # Tool offset: distance from the attachment site down to the pinch point.
        # Measured from the model rather than hardcoded, so it tracks any change
        # to how the gripper is mounted.
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, self.base._home_key)
        mujoco.mj_forward(self.model, data)
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ur_attachment_site")
        pid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ur_gripper_pinch")
        self.tool_length = float(data.site_xpos[sid][2] - data.site_xpos[pid][2])
        # Needed by the residual compensation below: the pinch point is what
        # has to land accurately, and it is not the frame IK regulates.
        self._pinch_sid = pid
        # Top-down tool orientation, taken from the home keyframe (site +z points
        # straight down there) and held fixed for every waypoint.
        self.tool_rotation = mink.SO3.from_matrix(data.site_xmat[sid].reshape(3, 3).copy())
        self._ee_sid = sid

        # --- D7 diagnostics setup ------------------------------------
        # Pad geoms (for contact-force logging) and piece geoms (any geom on
        # the tracked piece's body -- robust to which geoms make up the piece,
        # not hardcoded to today's base/stem/head split).
        self._left_pad_geom_ids = self._geoms_named_like("ur_gripper_left_pad")
        self._right_pad_geom_ids = self._geoms_named_like("ur_gripper_right_pad")
        self._pad_geom_ids = self._left_pad_geom_ids | self._right_pad_geom_ids
        self._piece_geom_ids = {
            g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self.base._piece_bid
        }
        # Joint limits, for the joint-limit-proximity check in step 3.
        self._jnt_limited = self.model.jnt_limited[:ARM_DOF].copy()
        self._jnt_range = self.model.jnt_range[:ARM_DOF].copy()

    def _geoms_named_like(self, prefix: str) -> set[int]:
        return {
            g
            for g in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(prefix)
        }

    # -------------------------------------------------------------- actions
    def _to_action(self, q_arm: np.ndarray, gripper_ctrl: float) -> np.ndarray:
        """Normalised action reproducing these joint targets (env uses [-1, 1])."""
        ctrl = np.concatenate([q_arm, [gripper_ctrl]])
        ctrl = np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        span = self._ctrl_high - self._ctrl_low
        return (2.0 * (ctrl - self._ctrl_low) / span - 1.0).astype(np.float32)

    # ------------------------------------------------------------------ ik
    def solve_ik(self, pinch_target: np.ndarray) -> np.ndarray:
        """Arm joint angles putting the pinch point at `pinch_target`, top-down.

        Raises IKFailure if the solver does not converge.
        """
        site_target = np.asarray(pinch_target, dtype=float).copy()
        site_target[2] += self.tool_length  # site sits tool_length above the pinch

        self.task.set_target(
            mink.SE3.from_rotation_and_translation(self.tool_rotation, site_target)
        )
        # Seed from the arm's current pose for continuity between waypoints.
        self.configuration.update(self.base.data.qpos.copy())

        for _ in range(IK_MAX_ITERS):
            err = self.task.compute_error(self.configuration)
            if np.linalg.norm(err[:3]) < IK_POS_TOL and np.linalg.norm(err[3:]) < IK_ORI_TOL:
                return self.configuration.q[:ARM_DOF].copy()

            vel = mink.solve_ik(
                self.configuration, [self.task], IK_DT, IK_SOLVER, damping=1e-3
            )
            masked = np.zeros_like(vel)
            masked[self._arm_dofs] = vel[self._arm_dofs]
            self.configuration.integrate_inplace(masked, IK_DT)

        err = self.task.compute_error(self.configuration)
        raise IKFailure(
            f"IK did not converge for pinch target {np.round(pinch_target, 4)}: "
            f"pos err {np.linalg.norm(err[:3])*1000:.1f} mm, "
            f"ori err {np.linalg.norm(err[3:]):.3f} rad"
        )

    # --------------------------------------------------------- diagnostics
    def _pad_piece_contacts(self) -> dict:
        """Contact force between each gripper pad and the tracked piece.

        Uses `mj_contactForce`, which returns the 6D wrench (normal,
        2 tangential, torsional, 2 rolling) in the contact frame -- summed
        per pad side so a left/right imbalance is directly visible.
        """
        base = self.base
        data = base.data
        totals = {"left": [0.0, 0.0], "right": [0.0, 0.0]}  # [normal, tangential]
        n_contacts = 0
        wrench = np.zeros(6)
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self._piece_geom_ids and g2 in self._pad_geom_ids:
                pad_geom = g2
            elif g2 in self._piece_geom_ids and g1 in self._pad_geom_ids:
                pad_geom = g1
            else:
                continue
            n_contacts += 1
            mujoco.mj_contactForce(base.model, data, k, wrench)
            side = "left" if pad_geom in self._left_pad_geom_ids else "right"
            totals[side][0] += abs(float(wrench[0]))
            totals[side][1] += float(np.hypot(wrench[1], wrench[2]))
        return {
            "n_pad_piece_contacts": n_contacts,
            "left_pad_normal_force": totals["left"][0],
            "right_pad_normal_force": totals["right"][0],
            "left_pad_tangent_force": totals["left"][1],
            "right_pad_tangent_force": totals["right"][1],
        }

    def _ee_tilt(self) -> dict:
        """End-effector orientation deviation from the fixed top-down target.

        Decomposed into roll/pitch/yaw of the *deviation* rotation (current
        relative to the nominal `self.tool_rotation`, same for every
        waypoint), plus a single combined tilt-from-vertical angle (angle
        between the site's local +z axis and world straight-down) as a
        sanity-check figure against the decomposition.
        """
        ee_xmat = self.base.data.site_xmat[self._ee_sid].reshape(3, 3)
        r_dev = ee_xmat @ self.tool_rotation.as_matrix().T
        roll = np.degrees(np.arctan2(r_dev[2, 1], r_dev[2, 2]))
        pitch = np.degrees(np.arctan2(-r_dev[2, 0], np.hypot(r_dev[2, 1], r_dev[2, 2])))
        yaw = np.degrees(np.arctan2(r_dev[1, 0], r_dev[0, 0]))
        z_axis = ee_xmat[:, 2]
        total_tilt = np.degrees(np.arccos(np.clip(np.dot(z_axis, [0.0, 0.0, -1.0]), -1.0, 1.0)))
        return {
            "ee_roll_deg": float(roll),
            "ee_pitch_deg": float(pitch),
            "ee_yaw_deg": float(yaw),
            "ee_tilt_from_vertical_deg": float(total_tilt),
        }

    def _joint_limit_margins(self) -> list[float | None]:
        """Distance (rad) from each arm joint's current angle to its nearer
        limit; None for unlimited joints."""
        qpos = self.base.data.qpos[:ARM_DOF]
        margins: list[float | None] = []
        for j in range(ARM_DOF):
            if not self._jnt_limited[j]:
                margins.append(None)
                continue
            lo, hi = self._jnt_range[j]
            margins.append(float(min(qpos[j] - lo, hi - qpos[j])))
        return margins

    def _diagnostics_sample(self, phase: str, phase_step: int) -> dict:
        base = self.base
        data = base.data
        piece_vel = data.qvel[base._piece_vadr : base._piece_vadr + 6]
        sample = {
            "global_step": int(base._elapsed_steps),
            "sim_time": float(data.time),
            "phase": phase,
            "phase_step": int(phase_step),
            "qpos": data.qpos[:ARM_DOF].copy().tolist(),
            "qvel": data.qvel[:ARM_DOF].copy().tolist(),
            "qfrc_actuator": data.qfrc_actuator[:ARM_DOF].copy().tolist(),
            "joint_limit_margin_rad": self._joint_limit_margins(),
            "gripper_aperture": float(data.qpos[base._grip_qadr]),
            "gripper_velocity": float(data.qvel[base._grip_vadr]),
            "ee_pos": data.site_xpos[self._ee_sid].copy().tolist(),
            "piece_pos": data.xpos[base._piece_bid].copy().tolist(),
            "piece_lin_speed": float(np.linalg.norm(piece_vel[:3])),
            "piece_ang_speed": float(np.linalg.norm(piece_vel[3:])),
        }
        sample.update(self._ee_tilt())
        sample.update(self._pad_piece_contacts())
        return sample

    # --------------------------------------------------------------- drive
    def _drive_to(
        self,
        q_arm: np.ndarray,
        gripper_ctrl: float,
        budget: int,
        label: str | None = None,
        log: list[dict] | None = None,
    ):
        """Hold a joint target until arm *and* gripper settle, or the budget runs out.

        The gripper condition matters: the 2F-85's closing stroke takes far longer
        than the arm takes to reach a nearby waypoint, so an arm-only settle check
        exits the "close" waypoint after a single step and the fingers never reach
        the piece.

        Settling requires SETTLE_CONSECUTIVE_STEPS in a row of low arm/gripper
        velocity, not a single reading (general fix, D7 mitigation follow-up):
        a single instantaneous low-velocity reading can be a transient -- e.g.
        the brief deceleration right as a pad first contacts the piece -- and
        was previously misread as "the grip has settled," exiting before the
        gripper had actually pressed in to a firm hold. This showed up two
        ways: silently, as the likely cause of the false-negative early exit
        during the D7 frame_skip=1 diagnostic probe (`_drive_to` was checked
        20x more often there, making a one-step transient far more likely to
        be caught); and visibly, as an earlier ramp-based mitigation attempt
        stopping at a shallow aperture instead of a firm one.

        If `log` is a list, one diagnostics dict (D7) is appended per step,
        captured immediately after `env.step()` so it reflects the resulting
        state. Purely additive: reads state, writes nothing back into the sim.

        Yields the env's step tuple so the caller can react to termination.
        """
        action = self._to_action(q_arm, gripper_ctrl)
        settled_streak = 0
        for i in range(budget):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if log is not None:
                log.append(self._diagnostics_sample(label, i))
            yield obs, reward, terminated, truncated, info
            if terminated or truncated:
                return

            arm_settled = np.max(np.abs(self.base.data.qpos[:ARM_DOF] - q_arm)) < self.settle_tol
            grip_settled = abs(self.base.data.qvel[self.base._grip_vadr]) < 0.02
            settled_streak = settled_streak + 1 if (arm_settled and grip_settled) else 0
            if settled_streak >= SETTLE_CONSECUTIVE_STEPS:
                return

    def pinch_pos(self) -> np.ndarray:
        """Where the pinch point ACTUALLY is right now, in world coordinates."""
        return self.base.data.site_xpos[self._pinch_sid].copy()

    def _run_waypoint(
        self,
        pinch_target: np.ndarray,
        grip: float,
        budget: int,
        label: str,
        log: list[dict] | None,
        q_arm: np.ndarray,
        compensate: bool,
    ) -> dict:
        """Drive one waypoint, optionally correcting its task-space residual.

        Drives to `q_arm`, then -- if `compensate` -- measures how far the
        pinch point actually ended up from `pinch_target` and adds that
        shortfall back into the commanded IK target before re-solving and
        re-driving. Capped at MAX_RESIDUAL_ITERS passes; if the residual is
        still above RESIDUAL_TOL after that, it warns and gives up rather
        than looping or eating into later waypoints' budgets.

        Returns a dict with the final joint target (so a following waypoint
        holding the SAME pose can reuse it instead of re-solving back to the
        uncompensated one), plus the env's last step tuple and diagnostics.
        """
        for obs, _, terminated, truncated, info in self._drive_to(
            q_arm, grip, budget, label=label, log=log
        ):
            pass

        result = {
            "q_arm": q_arm, "obs": obs, "info": info,
            "terminated": terminated, "truncated": truncated,
            "iters": 0, "ik_failed": False,
            "residual_mm": float(np.linalg.norm(pinch_target - self.pinch_pos())) * 1000.0,
        }
        if not compensate or terminated or truncated:
            return result

        # The commanded target drifts away from the true target by exactly the
        # tracking shortfall; the true target itself never moves.
        commanded = np.asarray(pinch_target, dtype=float).copy()
        while result["iters"] < MAX_RESIDUAL_ITERS and result["residual_mm"] > RESIDUAL_TOL * 1000.0:
            commanded = commanded + (pinch_target - self.pinch_pos())
            result["iters"] += 1
            try:
                q_arm = self.solve_ik(commanded)
            except IKFailure as exc:
                # The compensated target sits slightly further out than the
                # original, which can push a corner square past the arm's
                # reach. Degrade to the uncompensated pose rather than failing
                # the episode outright -- and say so.
                print(f"  [warn] {label}: IK failed on compensated target "
                      f"(iter {result['iters']}): {exc}")
                result["ik_failed"] = True
                result["iters"] -= 1
                break
            for obs, _, terminated, truncated, info in self._drive_to(
                q_arm, grip, RESIDUAL_BUDGET, label=label, log=log
            ):
                pass
            result.update(q_arm=q_arm, obs=obs, info=info,
                          terminated=terminated, truncated=truncated)
            result["residual_mm"] = float(
                np.linalg.norm(pinch_target - self.pinch_pos())) * 1000.0
            if terminated or truncated:
                return result

        if result["residual_mm"] > RESIDUAL_TOL * 1000.0 and not result["ik_failed"]:
            print(f"  [warn] {label}: pinch residual {result['residual_mm']:.2f} mm still "
                  f"above {RESIDUAL_TOL*1000:.1f} mm after {result['iters']} correction "
                  f"pass(es); continuing uncorrected.")
        return result

    def waypoints(self, piece_xy: np.ndarray, goal_xy: np.ndarray):
        """(label, pinch position, gripper ctrl, step budget) for the pick-and-place.

        Budgets are per-waypoint caps, not fixed holds -- a waypoint exits as soon
        as it settles. The gripper phases get the largest budgets because the
        2F-85's stroke is the slowest thing in the sequence.
        """
        px, py = piece_xy
        gx, gy = goal_xy
        return [
            ("approach", np.array([px, py, CLEAR_Z]), GRIPPER_OPEN_CTRL, 25),
            ("descend", np.array([px, py, GRASP_Z]), GRIPPER_OPEN_CTRL, 25),
            ("close", np.array([px, py, GRASP_Z]), GRIPPER_GRIP_CTRL, 45),
            ("lift", np.array([px, py, CLEAR_Z]), GRIPPER_GRIP_CTRL, 25),
            ("traverse", np.array([gx, gy, CLEAR_Z]), GRIPPER_GRIP_CTRL, 30),
            ("place", np.array([gx, gy, PLACE_Z]), GRIPPER_GRIP_CTRL, 25),
            ("release", np.array([gx, gy, PLACE_Z]), GRIPPER_OPEN_CTRL, 35),
            ("retreat", np.array([gx, gy, CLEAR_Z]), GRIPPER_OPEN_CTRL, 25),
        ]

    # ------------------------------------------------------------- episode
    def run_episode(
        self, seed: int | None = None, jitter: float = 0.0, log: list[dict] | None = None
    ) -> dict:
        """`log`, if a list, is appended in place with one diagnostics dict
        (D7) per timestep of the LOGGED_PHASES ("approach", "descend",
        "close") -- opt-in, no effect on control flow or on the returned
        record when omitted."""
        obs, info = self.env.reset(seed=seed)
        piece_xy = obs["achieved_goal"][:2].astype(float)
        goal_xy = obs["desired_goal"][:2].astype(float)

        # `jitter` perturbs only the CONTROLLER's belief about where the piece and
        # target are -- it stands in for pose-estimation error. The environment is
        # untouched. With jitter=0 the env's reset is fully deterministic, so every
        # episode is bit-identical; jitter is the only way to get a spread.
        if jitter > 0.0:
            rng = np.random.default_rng(seed)
            piece_xy = piece_xy + rng.normal(0.0, jitter, size=2)
            goal_xy = goal_xy + rng.normal(0.0, jitter, size=2)

        record = {
            "success": False,
            "steps": 0,
            "outcome": None,
            "failed_phase": None,
            "final_distance": float(info["distance_to_goal"]),
            "piece_z_after_lift": None,
            "grasped": None,
        }

        terminated = truncated = False
        record["compensation_iters"] = {}
        record["residual_mm"] = {}
        prev_target = None
        prev_q = None
        for label, pinch_target, grip, budget in self.waypoints(piece_xy, goal_xy):
            # A waypoint that holds the SAME pinch target as its predecessor
            # ("close" after "descend", "release" after "place") must reuse the
            # pose that predecessor actually converged to. Re-solving would
            # command the uncompensated target again and throw the correction
            # away at exactly the moment it matters.
            reuse = prev_target is not None and np.allclose(pinch_target, prev_target)
            if reuse:
                q_arm = prev_q
            else:
                try:
                    q_arm = self.solve_ik(pinch_target)
                except IKFailure as exc:
                    record["outcome"] = "ik_failure"
                    record["failed_phase"] = label
                    record["detail"] = str(exc)
                    record["steps"] = self.base._elapsed_steps
                    return record

            phase_log = log if (log is not None and label in LOGGED_PHASES) else None
            wp = self._run_waypoint(
                pinch_target, grip, budget, label, phase_log, q_arm,
                compensate=(label in COMPENSATED_WAYPOINTS and not reuse),
            )
            obs, info = wp["obs"], wp["info"]
            terminated, truncated = wp["terminated"], wp["truncated"]
            prev_target, prev_q = np.asarray(pinch_target, dtype=float), wp["q_arm"]
            if label in COMPENSATED_WAYPOINTS:
                record["compensation_iters"][label] = wp["iters"]
                record["residual_mm"][label] = wp["residual_mm"]
                if wp["ik_failed"]:
                    record.setdefault("compensation_ik_failures", []).append(label)

            if label == "lift":
                # Diagnostic only: did the piece actually come up with the gripper?
                piece_z = float(obs["achieved_goal"][2])
                record["piece_z_after_lift"] = piece_z
                record["grasped"] = bool(piece_z > 0.762 + 0.02)

            if terminated or truncated:
                record["failed_phase"] = label
                break

        record["steps"] = self.base._elapsed_steps
        record["final_distance"] = float(info["distance_to_goal"])
        record["success"] = bool(info["is_success"])

        if record["success"]:
            record["outcome"] = "success"
            record["failed_phase"] = None
        elif truncated:
            record["outcome"] = "truncated_step_limit"
        elif info.get("terminated_reason"):
            record["outcome"] = info["terminated_reason"]
        elif info.get("piece_off_board"):
            record["outcome"] = "piece_off_board"
        else:
            record["outcome"] = "sequence_finished_not_at_goal"
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--verbose", action="store_true", help="print per-episode results")
    parser.add_argument(
        "--jitter", type=float, default=0.0,
        help="std-dev (m) of noise added to the CONTROLLER's estimate of the piece and "
             "target positions. Stands in for pose-estimation error; the env is untouched. "
             "The env reset is deterministic, so jitter=0 makes every episode identical.")
    parser.add_argument(
        "--task", choices=["base", "t1"], default="base",
        help="'base' drives RoboChessEnv directly (P1's fixed adjacent-square goal). "
             "'t1' wraps it in T1PickAndPlaceTask (P2), so start/target squares vary "
             "per episode -- this is the regression check that T1's wrapping did not "
             "change the underlying pick-and-place behaviour RoboChessEnv already validated.")
    args = parser.parse_args()

    env = RoboChessEnv(max_episode_steps=args.max_episode_steps)
    if args.task == "t1":
        env = T1PickAndPlaceTask(env)
    expert = ScriptedExpert(env)

    print(f"tool length (attachment_site -> pinch): {expert.tool_length*1000:.1f} mm")
    print(f"grasp height {GRASP_Z:.3f}, transit height {CLEAR_Z:.3f}, "
          f"goal tolerance {expert.base.goal_tolerance*1000:.0f} mm")
    print(f"controller waypoint jitter: {args.jitter*1000:.1f} mm"
          + ("  (env reset is deterministic -> all episodes identical)" if args.jitter == 0 else "")
          + "\n")

    records = []
    for ep in range(args.episodes):
        rec = expert.run_episode(seed=ep, jitter=args.jitter)
        records.append(rec)
        if args.verbose:
            print(
                f"  ep {ep:3d}  {'OK ' if rec['success'] else 'FAIL'}  "
                f"{rec['outcome']:<28} steps={rec['steps']:3d} "
                f"dist={rec['final_distance']*1000:6.1f} mm "
                f"phase={rec['failed_phase']}"
            )

    env.close()

    n = len(records)
    successes = sum(r["success"] for r in records)
    steps = [r["steps"] for r in records]

    print("=" * 62)
    print(f"episodes            : {n}")
    print(f"success rate        : {successes}/{n} = {100.0*successes/n:.1f}%")
    print(f"avg episode length  : {np.mean(steps):.1f} steps (min {min(steps)}, max {max(steps)})")
    print(f"avg final distance  : {np.mean([r['final_distance'] for r in records])*1000:.1f} mm")

    print("\noutcome breakdown (env's own signals):")
    for outcome, count in collections.Counter(r["outcome"] for r in records).most_common():
        print(f"  {outcome:<32} {count:3d}  ({100.0*count/n:.0f}%)")

    failures = [r for r in records if not r["success"]]
    if failures:
        print("\nfailures by phase reached:")
        for phase, count in collections.Counter(r["failed_phase"] for r in failures).most_common():
            print(f"  {str(phase):<32} {count:3d}")
        grasp_known = [r for r in failures if r["grasped"] is not None]
        if grasp_known:
            lifted = sum(r["grasped"] for r in grasp_known)
            print(
                f"\ngrasp diagnostic: piece left the board on {lifted}/{len(grasp_known)} "
                "failures that reached the lift phase"
            )
        detail = next((r.get("detail") for r in failures if r.get("detail")), None)
        if detail:
            print(f"\nfirst IK failure detail: {detail}")

    return 0 if successes == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
