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
# GRASP_Z targets the piece's STEM (18-34 mm above its base), not the wider
# base cylinder. The piece's centre of mass is 17.5 mm up -- the top edge of
# the base -- so a base grasp puts the whole contact patch *below* the COM and
# the piece pivots out of the fingers during the lift (measured: 58 degrees of
# tilt by the top of the lift, ending on its side). Gripping the stem puts the
# contact above the COM and it hangs stably: 1.1 degrees of tilt instead.
GRASP_Z = 0.8035
CLEAR_Z = 0.882  # 12 cm above the board: transit height
# Set the piece down before opening. Releasing from higher drops it and topples it.
PLACE_Z = 0.7935

GRIPPER_OPEN_CTRL = 0.0
# Full close command: the 16 mm stem is narrower than the gripper's 17 mm
# free-space gap at ctrl=200, so anything less never actually clamps it.
# Grip force stays bounded by the actuator's own forcerange either way.
GRIPPER_GRIP_CTRL = 255.0

IK_SOLVER = "daqp"
IK_DT = 0.02
IK_MAX_ITERS = 400
IK_POS_TOL = 1e-3  # metres
IK_ORI_TOL = 1e-2  # radians

ARM_DOF = 6


class IKFailure(RuntimeError):
    """Raised when the IK solver cannot reach a waypoint."""


class ScriptedExpert:
    """Open-loop waypoint controller driving RoboChessEnv through step()."""

    def __init__(self, env: RoboChessEnv, settle_tol: float = 0.02, max_steps_per_waypoint: int = 40):
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
        # Top-down tool orientation, taken from the home keyframe (site +z points
        # straight down there) and held fixed for every waypoint.
        self.tool_rotation = mink.SO3.from_matrix(data.site_xmat[sid].reshape(3, 3).copy())

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

    # --------------------------------------------------------------- drive
    def _drive_to(self, q_arm: np.ndarray, gripper_ctrl: float, budget: int):
        """Hold a joint target until arm *and* gripper settle, or the budget runs out.

        The gripper condition matters: the 2F-85's closing stroke takes far longer
        than the arm takes to reach a nearby waypoint, so an arm-only settle check
        exits the "close" waypoint after a single step and the fingers never reach
        the piece.

        Yields the env's step tuple so the caller can react to termination.
        """
        action = self._to_action(q_arm, gripper_ctrl)
        for i in range(budget):
            obs, reward, terminated, truncated, info = self.env.step(action)
            yield obs, reward, terminated, truncated, info
            if terminated or truncated:
                return
            arm_settled = np.max(np.abs(self.base.data.qpos[:ARM_DOF] - q_arm)) < self.settle_tol
            grip_settled = abs(self.base.data.qvel[self.base._grip_vadr]) < 0.02
            if i >= 2 and arm_settled and grip_settled:
                return

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
    def run_episode(self, seed: int | None = None, jitter: float = 0.0) -> dict:
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
        for label, pinch_target, grip, budget in self.waypoints(piece_xy, goal_xy):
            try:
                q_arm = self.solve_ik(pinch_target)
            except IKFailure as exc:
                record["outcome"] = "ik_failure"
                record["failed_phase"] = label
                record["detail"] = str(exc)
                record["steps"] = self.base._elapsed_steps
                return record

            for obs, _, terminated, truncated, info in self._drive_to(q_arm, grip, budget):
                pass

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
