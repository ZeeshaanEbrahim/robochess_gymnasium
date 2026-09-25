#!/usr/bin/env python
"""Close-gated Cartesian expert-replay gate, harness v3 (roadmap 5b.3 step 4 / 5b.8 step E).

Drives the scripted expert's LIVE phase list through the 4-dimensional
Cartesian action interface (dx, dy, dz, gripper) under the repaired referee.
The phase sequence and targets are the expert's own (`ScriptedExpert.waypoints`:
approach, descend, close, lift, traverse, place, release, retreat) and the
release ramp is the expert's own `_release_ramp` (D8), unchanged. What is new is
the transitions: every phase is left on a MEASURED CONDITION, and a phase that
does not meet its condition within its timeout is a failure reported with its
phase.

v1 -> v2 (harness-only; no environment, referee or expert change):
  (a) arrival tolerance by phase: 2 mm / 2 mm / 1 deg for descend and place,
      10 mm / 10 mm / 2 deg for approach, lift, traverse, retreat;
  (b) close gate is `pinched` + steady aperture (no `held`); in lift, `held`
      must appear within 10 steps or the episode is `grasp_lost_at_lift`, and
      lift arrival also requires `held`;
  (c) timeouts scale with distance for the four transit phases:
      ceil(start distance / 5 mm) + 40; descend / close / place / release keep 60;
  (d) braking: the commanded 3D step is limited to
      max(1.5 mm, min(20 mm, 0.4 x remaining distance)), direction unchanged;
  (e) budget rule: judged at 300 steps; a pair that ends on `budget:time_limit`
      with no other failure label is rerun once at 400 as F8 information
      (reported separately, not counted toward the pass rule).

v2 -> v3 (harness-only; final revision; no environment, referee or expert change):
  (a) place target from the live in-hand offset: at the start of place,
      target pinch z = current pinch z - (piece z - piece rest z) + 1.5 mm, so the
      piece base stops 1.5 mm above the board (xy unchanged; the expert's fixed
      PLACE_Z is no longer used);
  (b) no pushing once arrived: a phase whose pinch is inside its arrival
      position tolerance (lateral and vertical) commands ZERO displacement (no
      1.5 mm floor); release commands zero displacement throughout. The close
      phase, which holds the descend target, uses the fine tolerance for this;
  (c) descend braking factor 0.2 instead of 0.4 (descend only);
  (d) reporting columns (not gates) at the first release step: piece base height
      above rest, piece tilt, joint-target lead as implied pinch z;
  (e) verdict: `clean_place` = no referee termination in the episode AND piece
      base <= 5 mm above its resting height at the first release step. Only
      clean_place counts; F2 `success` is reported as a separate column.

The Cartesian mode's workspace-box / radial clipping is attributed by capturing
the target the env hands its IK solver (a wrapper on the instance) and
re-deriving which stage of the clip pipeline moved it; a mismatch is logged.

    python scripts/gate_expert_cartesian.py --show-params
    python scripts/gate_expert_cartesian.py --brake-test
    python scripts/gate_expert_cartesian.py --out /path/to/gate.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import scripted_expert as se  # noqa: E402
from replay_expert_cartesian import _build_env  # noqa: E402  (helper reused, not modified)
from robochess.envs import RoboChessEnv  # noqa: E402
from robochess.envs import robochess_env as renv  # noqa: E402

# =========================================================================
# FROZEN v3 PARAMETERS
# =========================================================================
HARNESS_VERSION = "v3"

# (a) Arrival tolerance by phase: (lateral m, vertical m, tilt deg). All phases
# need ARRIVE_CONSECUTIVE consecutive steps inside all three bounds.
FINE_TOL = (2.0e-3, 2.0e-3, 1.0)       # descend, place
COARSE_TOL = (10.0e-3, 10.0e-3, 2.0)   # approach, lift, traverse, retreat
ARRIVE_CONSECUTIVE = 2

# (b) close->lift: pinched AND aperture steady. lift: held within 10 steps.
STEADY_APERTURE_EPS = 0.002   # rad; |aperture[t] - aperture[t-1]| below this is "steady"
STEADY_CONSECUTIVE = 2
LIFT_HELD_WITHIN = 10         # steps from lift start; else `grasp_lost_at_lift`

# release->retreat: the expert's own D8/D9 rule, reused.
RELEASE_ANG_TOL = se.PIECE_ANG_SETTLE_TOL             # 0.05 rad/s
RELEASE_SETTLE_STEPS = se.SETTLE_CONSECUTIVE_STEPS    # 3

# (c) Timeouts.
TRANSIT_TIMEOUT_MM_PER_STEP = 5.0     # ceil(start distance / 5 mm) + 40
TRANSIT_TIMEOUT_EXTRA = 40
FIXED_TIMEOUT = 60                    # descend, close, place, release

# (d) Braking on the commanded 3D step.
BRAKE_MIN_M = 1.5e-3
BRAKE_MAX_M = 20.0e-3
BRAKE_FRACTION = 0.4
BRAKE_FRACTION_DESCEND = 0.2          # v3 (c): descend only

# v3 (a): place target from the live in-hand offset.
PLACE_BASE_CLEARANCE_M = 1.5e-3       # piece base stops this far above its resting height

# v3 (b): a phase inside its position tolerance commands zero displacement.
ZERO_WHEN_ARRIVED = True
ZERO_TOL_PHASES = {"close": "fine"}   # phases with no tolerance class of their own that hold a target

# v3 (a) verdict / (d) reporting.
CLEAN_BASE_MM = 5.0                   # verdict: base above rest at first release step
STRICT_BASE_MM = 3.0                  # reporting only (the earlier "clean" definition)
STRICT_TILT_DEG = 10.0

# (e) Budget.
EPISODE_BUDGET = 300
RERUN_BUDGET = 400
CLIP_TOL_M = 1.0e-6           # a clip that moved the target by less than 1 um is not counted
OVERSHOOT_BAND_M = 10.0e-3    # ring_mm: worst distance after first entering this band

# phase -> (target key, gripper mode, transition gate, tolerance class)
PHASES = {
    "approach": ("hover_piece", "open", "arrive", "coarse"),
    "descend": ("grasp", "open", "arrive", "fine"),
    "close": ("grasp", "closed", "steady_pinched", None),
    "lift": ("hover_piece", "closed", "arrive_held", "coarse"),
    "traverse": ("hover_goal", "closed", "arrive", "coarse"),
    "place": ("place", "closed", "arrive", "fine"),
    "release": ("place", "ramp", "released_settled", None),
    "retreat": ("hover_goal", "open", "arrive", "coarse"),
}
TRANSIT_PHASES = ("approach", "lift", "traverse", "retreat")
TOL = {"fine": FINE_TOL, "coarse": COARSE_TOL}

DEFAULT_PAIRS = "e4:e5,a1:h8,h8:a1,a8:h1,h1:h8,h8:h1,c1:f6,f1:c6,b7:g2,d8:e3"


def braked_step_m(remaining_m: float, fraction: float = BRAKE_FRACTION) -> float:
    """(d) magnitude of the commanded 3D step for a given remaining distance."""
    return max(BRAKE_MIN_M, min(BRAKE_MAX_M, fraction * remaining_m))


def phase_timeout(phase: str, start_distance_m: float) -> int:
    """(c) steps a phase may wait for its gate."""
    if phase in TRANSIT_PHASES:
        return math.ceil(start_distance_m / (TRANSIT_TIMEOUT_MM_PER_STEP * 1e-3)) + TRANSIT_TIMEOUT_EXTRA
    return FIXED_TIMEOUT


def show_params() -> None:
    print(f"FROZEN {HARNESS_VERSION} PARAMETERS")
    print(f"  arrival tolerance (lateral, vertical, tilt), {ARRIVE_CONSECUTIVE} consecutive steps:")
    print(f"    descend, place                      {FINE_TOL[0]*1e3:.0f} mm, {FINE_TOL[1]*1e3:.0f} mm, {FINE_TOL[2]:.0f} deg")
    print(f"    approach, lift, traverse, retreat   {COARSE_TOL[0]*1e3:.0f} mm, {COARSE_TOL[1]*1e3:.0f} mm, {COARSE_TOL[2]:.0f} deg")
    print(f"  close->lift    pinched (both pads on piece) AND |d aperture| < {STEADY_APERTURE_EPS} "
          f"rad/step for {STEADY_CONSECUTIVE} consecutive steps (no `held`)")
    print(f"  lift           `held` must be true within {LIFT_HELD_WITHIN} steps of lift start, else "
          f"`grasp_lost_at_lift`; lift arrival also requires `held`")
    print(f"  release->retreat  ramp complete ({se.RELEASE_RAMP_MACRO_STEPS} steps) AND piece angular speed < "
          f"{RELEASE_ANG_TOL} rad/s for {RELEASE_SETTLE_STEPS} consecutive steps")
    print(f"  timeouts       approach/lift/traverse/retreat: ceil(start distance / "
          f"{TRANSIT_TIMEOUT_MM_PER_STEP:.0f} mm) + {TRANSIT_TIMEOUT_EXTRA} steps; "
          f"descend/close/place/release: {FIXED_TIMEOUT}")
    print(f"  braking        commanded 3D step = max({BRAKE_MIN_M*1e3:.1f} mm, min({BRAKE_MAX_M*1e3:.0f} mm, "
          f"f x remaining distance)), direction unchanged; f = {BRAKE_FRACTION_DESCEND} in descend, "
          f"{BRAKE_FRACTION} elsewhere")
    print(f"  no pushing     inside a phase's position tolerance (lateral AND vertical) the commanded displacement is "
          f"ZERO (no {BRAKE_MIN_M*1e3:.1f} mm floor); close uses the fine tolerance; release commands zero throughout")
    print(f"  place target   pinch z = pinch z at place start - (piece z - piece rest z) + "
          f"{PLACE_BASE_CLEARANCE_M*1e3:.1f} mm; xy = goal square centre (unchanged)")
    print(f"  verdict        clean_place = no referee termination AND piece base <= {CLEAN_BASE_MM:.0f} mm above "
          f"rest at the first release step; F2 success reported separately; only clean_place counts (>= 8/10)")
    print(f"  reporting only at release start: base above rest, piece tilt, implied pinch z of the joint target; "
          f"strict clean (base < {STRICT_BASE_MM:.0f} mm and tilt < {STRICT_TILT_DEG:.0f} deg) shown as info")
    print(f"  budget         judged at {EPISODE_BUDGET} steps; a pair ending on `budget:time_limit` with no other "
          f"failure label is rerun once at {RERUN_BUDGET} (F8 information, not counted)")
    print(f"  overshoot      overshoot_mm = furthest excursion past the target along the direction of travel; "
          f"vertical_overshoot_mm = furthest excursion past the target height in the direction of z "
          f"travel, anywhere in the phase; ring_mm = worst distance to target after first entering "
          f"{OVERSHOOT_BAND_M*1e3:.0f} mm")
    print("  braked step for remaining distance (mm -> commanded mm): "
          + ", ".join(f"{d}->{braked_step_m(d*1e-3)*1e3:.1f}" for d in (500, 100, 50, 25, 10, 5, 2, 1))
          + f"   [descend: " + ", ".join(f"{d}->{braked_step_m(d*1e-3, BRAKE_FRACTION_DESCEND)*1e3:.1f}"
                                          for d in (500, 100, 50, 25, 10, 5, 2, 1)) + "]")
    print("  phases:")
    for label, (tgt, grip, gate, tol) in PHASES.items():
        print(f"    {label:9s} target={tgt:11s} gripper={grip:6s} gate={gate:16s} tol={tol}")


# ---------------------------------------------------------------- helpers
def _task_of(env):
    """The wrapper that owns `_piece_contacts` (the T1 task), found by walking down."""
    e = env
    while e is not None and not hasattr(e, "_piece_contacts"):
        e = getattr(e, "env", None)
    if e is None:
        raise RuntimeError("no wrapper with _piece_contacts in the env chain")
    return e


def _name(model, g: int) -> str:
    return (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            or "<" + (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                        int(model.geom_bodyid[g])) or "?") + ">")


def contact_list(base: RoboChessEnv) -> list[dict]:
    """Every active contact: geom names (body name if the geom is unnamed),
    normal force, signed distance in mm."""
    m, d = base.model, base.data
    f = np.zeros(6)
    out = []
    for k in range(d.ncon):
        c = d.contact[k]
        mujoco.mj_contactForce(m, d, k, f)
        out.append(dict(a=_name(m, int(c.geom1)), b=_name(m, int(c.geom2)),
                        force_n=round(abs(float(f[0])), 3), dist_mm=round(float(c.dist) * 1e3, 4)))
    return out


def robot_robot_contacts(base: RoboChessEnv) -> list[dict]:
    m, d = base.model, base.data
    f = np.zeros(6)
    out = []
    for k in range(d.ncon):
        c = d.contact[k]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in base._robot_geoms and g2 in base._robot_geoms
                and m.geom_bodyid[g1] != m.geom_bodyid[g2]):
            mujoco.mj_contactForce(m, d, k, f)
            out.append(dict(a=_name(m, g1), b=_name(m, g2), force_n=round(abs(float(f[0])), 3),
                            dist_mm=round(float(c.dist) * 1e3, 4),
                            gripper_internal=bool(base._is_gripper_internal(g1, g2))))
    return out


def board_forces(base: RoboChessEnv) -> dict:
    """Summed normal force (N) of robot-board and piece-board contacts."""
    m, d = base.model, base.data
    f = np.zeros(6)
    robot = piece = 0.0
    for k in range(d.ncon):
        c = d.contact[k]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in base._board_geoms or g2 in base._board_geoms:
            other = g2 if g1 in base._board_geoms else g1
            mujoco.mj_contactForce(m, d, k, f)
            if other in base._robot_geoms:
                robot += abs(float(f[0]))
            elif int(m.geom_bodyid[other]) == base._piece_bid:
                piece += abs(float(f[0]))
    return dict(robot_board_n=round(robot, 2), piece_board_n=round(piece, 2), total_n=round(robot + piece, 2))


def _tilt_deg(base: RoboChessEnv, sid: int) -> float:
    z = base.data.site_xmat[sid].reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(-z[2], -1.0, 1.0))))


def _clip_stages(base: RoboChessEnv, pinch: np.ndarray, action3: np.ndarray) -> dict:
    """Re-derive which stage of RoboChessEnv._scale_action's Cartesian clip
    pipeline moved the target. Copy of that pipeline's logic; the caller compares
    the result with the target the env actually used."""
    raw = pinch + action3 * base.cartesian_step
    boxed = np.clip(raw, base._cartesian_low, base._cartesian_high)
    box_axes = [ax for ax in range(3) if abs(boxed[ax] - raw[ax]) > CLIP_TOL_M]
    final = boxed.copy()
    r_limit = min(renv.CARTESIAN_MAX_RADIUS,
                  float(np.interp(boxed[2], renv.CARTESIAN_REACH_Z, renv.CARTESIAN_REACH_R))
                  - renv.CARTESIAN_REACH_MARGIN)
    offset = boxed[:2] - base._base_xy
    radius = float(np.linalg.norm(offset))
    outer = inner = False
    if radius > r_limit:
        final[:2] = base._base_xy + offset * (r_limit / radius)
        outer = float(np.linalg.norm(final[:2] - boxed[:2])) > CLIP_TOL_M
    else:
        r_inner = float(np.interp(boxed[2], renv.CARTESIAN_INNER_Z, renv.CARTESIAN_INNER_R))
        if 0.0 < radius < r_inner:
            final[:2] = base._base_xy + offset * (r_inner / radius)
            inner = float(np.linalg.norm(final[:2] - boxed[:2])) > CLIP_TOL_M
    return dict(final=final, box_axes=[("x", "y", "z")[a] for a in box_axes], outer=outer, inner=inner)


def _phase_motion_stats(rows: list[dict], start: np.ndarray, target: np.ndarray) -> dict:
    """overshoot_mm and ring_mm for one phase (see show_params)."""
    if not rows:
        return dict(overshoot_mm=None, ring_mm=None)
    travel = target - start
    n = float(np.linalg.norm(travel))
    pos = np.array([r["_pinch"] for r in rows])
    if n > 1e-9:
        u = travel / n
        over = float(max(0.0, np.max((pos - target) @ u))) * 1e3
    else:
        over = 0.0
    dist = np.linalg.norm(pos - target, axis=1)
    inside = np.flatnonzero(dist <= OVERSHOOT_BAND_M)
    ring = float(np.max(dist[inside[0]:])) * 1e3 if inside.size else None
    # Vertical excursion past the target height in the direction of z travel,
    # anywhere in the phase (it can happen while still far off laterally).
    if abs(travel[2]) > 1e-3:
        vert_over = float(max(0.0, np.max((pos[:, 2] - target[2]) * np.sign(travel[2])))) * 1e3
    else:
        vert_over = None
    return dict(overshoot_mm=round(over, 2), vertical_overshoot_mm=None if vert_over is None else round(vert_over, 2),
                ring_mm=None if ring is None else round(ring, 2))


# ---------------------------------------------------------------- episode
def run_pair(start: str, target_sq: str, expert: se.ScriptedExpert, budget: int = EPISODE_BUDGET,
             braking: bool = True, stop_after: str | None = None,
             descend_fraction: float = BRAKE_FRACTION_DESCEND) -> dict:
    env = _build_env("cartesian", budget)
    base = env.unwrapped
    task = _task_of(env)
    sid = base._pinch_sid
    env.reset(seed=0, options={"start_square": start, "target_square": target_sq})

    piece_xy = base._achieved_goal()[:2].astype(float).copy()
    goal_xy = base._desired_goal[:2].astype(float).copy()
    wps = {lab: (pos, grip) for lab, pos, grip, _ in expert.waypoints(piece_xy, goal_xy)}
    targets = {
        "hover_piece": wps["approach"][0], "grasp": wps["descend"][0],
        "hover_goal": wps["traverse"][0], "place": wps["place"][0],
    }
    order = list(PHASES)

    captured: dict = {}
    orig_solve = base._solve_cartesian_ik

    def wrapped(target):
        captured["target"] = np.asarray(target, dtype=np.float64).copy()
        return orig_solve(target)

    base._solve_cartesian_ik = wrapped

    scratch = mujoco.MjData(base.model)

    def implied_pinch_z_mm() -> float:
        """Pinch z (mm, relative to the workspace-box floor) if the arm reached its joint target."""
        scratch.qpos[:] = base.data.qpos
        scratch.qpos[base._arm_qadr] = base._arm_target
        mujoco.mj_kinematics(base.model, scratch)
        return float((scratch.site_xpos[sid][2] - base._cartesian_low[2]) * 1e3)

    span = base._ctrl_high[6:] - base._ctrl_low[6:]
    to_action = lambda ctrl: float(2.0 * (ctrl - base._ctrl_low[6]) / span[0] - 1.0)

    rows: list[dict] = []
    phase_steps = {p: 0 for p in order}
    phase_stats: dict = {}
    result = dict(harness=HARNESS_VERSION, start=start, target=target_sq, budget=budget,
                  braking=braking, success=False, failure=None, failure_phase=None,
                  descend_fraction=descend_fraction, clean_place=False, release_start=None,
                  place_target=None, referee_termination=None, grasp_contacts=None, grasp_step=None, first_pinched_step=None,
                  last_held_step=None, lost_grasp=None, phase_lengths=phase_steps,
                  phase_stats=phase_stats)
    prev_aperture = float(base.data.qpos[base._grip_qadr])
    done = False
    info = {}

    for phase in order:
        tkey, gmode, gate, tolclass = PHASES[phase]
        if phase == "place":
            # v3 (a): target from the live in-hand offset.
            piece_z = float(base._achieved_goal()[2])
            above_rest = piece_z - task._piece_rest_z
            pz = float(base.data.site_xpos[sid][2])
            tz = pz - above_rest + PLACE_BASE_CLEARANCE_M
            targets["place"] = np.array([goal_xy[0], goal_xy[1], tz])
            result["place_target"] = dict(
                pinch_z_mm=round(pz * 1e3, 2), piece_above_rest_mm=round(above_rest * 1e3, 2),
                target_z_mm=round(tz * 1e3, 2), workspace_floor_mm=round(float(base._cartesian_low[2]) * 1e3, 2),
                below_floor_mm=round(max(0.0, float(base._cartesian_low[2]) - tz) * 1e3, 2),
                expert_place_z_mm=round(float(wps["place"][0][2]) * 1e3, 2))
        target = targets[tkey]
        phase_start_pinch = base.data.site_xpos[sid].copy()
        start_dist = float(np.linalg.norm(target - phase_start_pinch))
        timeout = phase_timeout(phase, start_dist)
        tol = TOL.get(tolclass)
        zero_tol = TOL[ZERO_TOL_PHASES[phase]] if phase in ZERO_TOL_PHASES else tol
        if phase == "descend":
            descend_start_piece = base._achieved_goal().astype(float).copy()
        if phase == "close":
            # (a) error at the close command step (state before the first close step);
            # (c) piece displacement since the start of descend.
            pc = base.data.site_xpos[sid] - target
            piece_now = base._achieved_goal().astype(float)
            dp = (piece_now - descend_start_piece) * 1e3
            result["close_start"] = dict(
                step=int(base._elapsed_steps), lateral_err_mm=round(float(np.hypot(pc[0], pc[1])) * 1e3, 3),
                vertical_err_mm=round(float(pc[2]) * 1e3, 3),
                descend_gate_consecutive_steps=ARRIVE_CONSECUTIVE,
                piece_disp_from_descend_start_mm=dict(
                    total=round(float(np.linalg.norm(dp)), 3), xy=round(float(np.hypot(dp[0], dp[1])), 3),
                    z=round(float(dp[2]), 3)))
        if phase == "release":
            result["release_start"] = dict(
                step=int(base._elapsed_steps),
                base_above_rest_mm=round((float(base._achieved_goal()[2]) - task._piece_rest_z) * 1e3, 3),
                base_above_board_mm=round(float(base._achieved_goal()[2] - base._board_surface_z) * 1e3, 3),
                piece_tilt_deg=round(float(base._piece_tilt_deg()), 2),
                implied_pinch_z_vs_floor_mm=round(implied_pinch_z_mm(), 2),
                pinch_z_vs_floor_mm=round(float((base.data.site_xpos[sid][2] - base._cartesian_low[2]) * 1e3), 2),
                board_force=board_forces(base),
                arm_target_lead_rad=round(float(np.max(np.abs(base._arm_target - base.data.qpos[base._arm_qadr]))), 4),
                peak_arm_target_lead_in_place_rad=round(max(
                    (r["arm_lead_rad"] for r in rows if r["phase"] == "place"), default=0.0), 4))
        arrive_streak = steady_streak = settle_streak = 0
        ramp_final_reached = False
        held_seen_in_lift = False
        phase_rows: list[dict] = []
        for i in range(timeout):
            if gmode == "open":
                gctrl = se.GRIPPER_OPEN_CTRL
            elif gmode == "closed":
                gctrl = se.GRIPPER_GRIP_CTRL
            else:
                gctrl = se._release_ramp(i)
            pinch = base.data.site_xpos[sid].copy()
            to_target = target - pinch
            dist = float(np.linalg.norm(to_target))
            arrived_pos = (zero_tol is not None
                           and float(np.hypot(to_target[0], to_target[1])) <= zero_tol[0]
                           and abs(float(to_target[2])) <= zero_tol[1])
            if phase == "release" or (ZERO_WHEN_ARRIVED and arrived_pos):
                disp = np.zeros(3)                    # v3 (b): no pushing
            elif braking:
                frac = descend_fraction if phase == "descend" else BRAKE_FRACTION
                mag = braked_step_m(dist, frac)
                vec = to_target / dist * mag if dist > 1e-12 else np.zeros(3)
                disp = np.clip(vec / base.cartesian_step, -1.0, 1.0)
            else:
                disp = np.clip(to_target / base.cartesian_step, -1.0, 1.0)
            action = np.concatenate([disp, [to_action(gctrl)]]).astype(np.float32)
            rejects_before = int(base._ik_rejects)
            captured.clear()
            _, reward, term, trunc, info = env.step(action)

            # ---- per-step measurements (state AFTER the step) -------------
            pinch2 = base.data.site_xpos[sid].copy()
            err = pinch2 - target
            lateral = float(np.hypot(err[0], err[1]))
            vertical = float(err[2])
            tilt = _tilt_deg(base, sid)
            aperture = float(base.data.qpos[base._grip_qadr])
            left, right, board = task._piece_contacts()
            pinched = bool(left and right)
            held = bool(pinched and not board)
            piece = base._achieved_goal().astype(float)
            R = base.data.site_xmat[sid].reshape(3, 3)
            rel = (R.T @ (piece - pinch2)) * 1e3
            ang = float(np.linalg.norm(base.data.qvel[base._piece_vadr + 3: base._piece_vadr + 6]))
            stages = _clip_stages(base, pinch, disp.astype(np.float64))
            used = captured.get("target")
            mismatch = (used is not None
                        and float(np.linalg.norm(used - stages["final"])) > 1e-9)
            rr = robot_robot_contacts(base)
            row = dict(step=int(base._elapsed_steps), phase=phase, i=i,
                       lateral_mm=round(lateral * 1e3, 3), vertical_mm=round(vertical * 1e3, 3),
                       tilt_deg=round(tilt, 3), aperture=round(aperture, 5),
                       grip_cmd=round(float(gctrl), 2), pinched=pinched, held=held,
                       pads=(bool(left), bool(right)), piece_on_board=bool(board),
                       piece_z_mm=round(float(info["piece_height_above_board"]) * 1e3, 3),
                       piece_in_pinch_mm=np.round(rel, 2).tolist(),
                       piece_ang_speed=round(ang, 4),
                       cmd_step_mm=round(float(np.linalg.norm(disp)) * base.cartesian_step * 1e3, 3),
                       gripper_self_contact=bool(info["gripper_self_contact"]),
                       robot_contacts=rr, ik_rejected=int(base._ik_rejects) - rejects_before,
                       clip_box=stages["box_axes"], clip_outer=stages["outer"],
                       clip_inner=stages["inner"],
                       piece_tilt_deg=round(float(base._piece_tilt_deg()), 2),
                       implied_pinch_z_vs_floor_mm=round(implied_pinch_z_mm(), 2),
                       pinch_z_vs_floor_mm=round(float((pinch2[2] - base._cartesian_low[2]) * 1e3), 2),
                       cmd_zero=bool(np.all(disp == 0.0)),
                       arm_lead_rad=round(float(np.max(np.abs(base._arm_target - base.data.qpos[base._arm_qadr]))), 4),
                       board_force_n=round(float(info.get("board_contact_force_n") or 0.0), 2), clip_attribution_mismatch=bool(mismatch),
                       terminated=bool(term), truncated=bool(trunc),
                       reason=info.get("terminated_reason"))
            rows.append(row)
            phase_rows.append(dict(row, _pinch=pinch2.tolist()))
            phase_steps[phase] += 1

            if pinched and result["first_pinched_step"] is None:
                result["first_pinched_step"] = row["step"]
            if held:
                result["last_held_step"] = row["step"]
            if (result["last_held_step"] is not None and phase in ("lift", "traverse", "place")
                    and not pinched and result["lost_grasp"] is None):
                result["lost_grasp"] = dict(after_step=result["last_held_step"],
                                            noticed_step=row["step"], phase=phase)

            if term or trunc:
                done = True
                reason = info.get("terminated_reason") or "time_limit"
                result["success"] = bool(info["is_success"])
                if not result["success"]:
                    result["failure"] = ("budget:time_limit" if reason == "time_limit"
                                         else f"terminated:{reason}")
                    result["failure_phase"] = phase
                break

            # ---- gate ------------------------------------------------------
            passed = False
            if gate in ("arrive", "arrive_held"):
                inside = (lateral <= tol[0] and abs(vertical) <= tol[1] and tilt <= tol[2])
                if gate == "arrive_held":
                    held_seen_in_lift = held_seen_in_lift or held
                    inside = inside and held
                    if i + 1 >= LIFT_HELD_WITHIN and not held_seen_in_lift:
                        result["failure"] = "grasp_lost_at_lift"
                        result["failure_phase"] = phase
                        done = True
                        break
                arrive_streak = arrive_streak + 1 if inside else 0
                passed = arrive_streak >= ARRIVE_CONSECUTIVE
            elif gate == "steady_pinched":
                steady = abs(aperture - prev_aperture) < STEADY_APERTURE_EPS
                steady_streak = steady_streak + 1 if steady else 0
                passed = steady_streak >= STEADY_CONSECUTIVE and pinched
                if passed:
                    result["grasp_step"] = row["step"]
                    result["grasp_contacts"] = contact_list(base)
                    result["grasp_held_at_gate"] = held
            elif gate == "released_settled":
                if gctrl == se.GRIPPER_OPEN_CTRL:
                    ramp_final_reached = True
                if ramp_final_reached:
                    settle_streak = settle_streak + 1 if ang < RELEASE_ANG_TOL else 0
                passed = settle_streak >= RELEASE_SETTLE_STEPS
            prev_aperture = aperture
            if passed:
                break
        else:
            # ran the phase's whole timeout without meeting the gate
            result["failure"] = "timeout"
            result["failure_phase"] = phase
            done = True

        stats = _phase_motion_stats(phase_rows, phase_start_pinch, target)
        stats.update(start_distance_mm=round(start_dist * 1e3, 1), timeout=timeout,
                     steps=len(phase_rows))
        phase_stats[phase] = stats
        if done:
            break
        if stop_after == phase:
            result["stopped_after"] = phase
            done = True
            break

    if not done:                      # every phase gated through; retreat finished
        result["success"] = bool(info.get("is_success"))
        if not result["success"]:
            result["failure"] = "sequence_finished_not_success"
            result["failure_phase"] = "retreat"

    result["steps"] = int(base._elapsed_steps)
    # A referee termination is a failure label; "success" (the task's own terminal) and
    # "time_limit" (budget truncation) are not.
    term_rows = [r for r in rows if r["terminated"] and r["reason"] not in (None, "success", "time_limit")]
    result["descend_peak_overshoot_mm"] = phase_stats.get("descend", {}).get("vertical_overshoot_mm")
    result["place_descent_peak_overshoot_mm"] = phase_stats.get("place", {}).get("vertical_overshoot_mm")
    result["referee_termination"] = term_rows[0]["reason"] if term_rows else None
    rs = result["release_start"]
    if rs is not None:
        rel_rows = [r for r in rows if r["phase"] == "release"]
        rs["board_force_max_in_release_n"] = max((r["board_force_n"] for r in rel_rows), default=None)
        rs["strict_clean"] = bool(rs["base_above_rest_mm"] < STRICT_BASE_MM and rs["piece_tilt_deg"] < STRICT_TILT_DEG)
        result["clean_place"] = bool(result["referee_termination"] is None
                                     and rs["base_above_rest_mm"] <= CLEAN_BASE_MM)
    result["budget_ended_episode"] = result["failure"] == "budget:time_limit"
    result["ik_rejects"] = int(base._ik_rejects)
    result["gripper_self_contact_steps"] = sum(r["gripper_self_contact"] for r in rows)
    result["clip_steps"] = dict(
        box=sum(bool(r["clip_box"]) for r in rows),
        box_by_axis={ax: sum(ax in r["clip_box"] for r in rows) for ax in "xyz"},
        outer=sum(r["clip_outer"] for r in rows), inner=sum(r["clip_inner"] for r in rows),
        attribution_mismatches=sum(r["clip_attribution_mismatch"] for r in rows))
    result["final_info"] = {k: info.get(k) for k in (
        "is_success", "terminated_reason", "distance_to_goal", "piece_on_square", "piece_resting",
        "gripper_released", "piece_settled", "piece_still", "piece_upright",
        "board_contact_force_n", "invalid_transition")}
    result["rows"] = rows
    env.close()
    return result


def brake_test(expert: se.ScriptedExpert) -> dict:
    """v3 (c): approach + descend only, descend braking factor 0.4 vs 0.2, from h8 and a8."""
    out = {}
    for start, target in (("h8", "a1"), ("a8", "h1")):
        for frac in (0.4, 0.2):
            r = run_pair(start, target, expert, stop_after="descend", descend_fraction=frac)
            s_ = r["phase_stats"]["descend"]
            d = [x for x in r["rows"] if x["phase"] == "descend"]
            out[f"{start}_f{frac}"] = dict(
                steps=s_["steps"], overshoot_mm=s_["overshoot_mm"], vertical_overshoot_mm=s_["vertical_overshoot_mm"],
                ring_mm=s_["ring_mm"], worst_vertical_mm=max((x["vertical_mm"] for x in d), key=abs),
                worst_lateral_mm=max(x["lateral_mm"] for x in d), failure=r["failure"])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", default=DEFAULT_PAIRS, help="start:target,... (default: the 10 gate pairs)")
    p.add_argument("--out", help="write full per-step results as JSON")
    p.add_argument("--show-params", action="store_true", help="print the frozen parameters and exit")
    p.add_argument("--brake-test", action="store_true",
                   help="descend braking factor 0.4 vs 0.2 from h8 and a8, then exit")
    args = p.parse_args()
    if args.show_params:
        show_params()
        return 0

    warnings.simplefilter("ignore")
    # ScriptedExpert is used only for its waypoint list (a pure function of the
    # piece and goal xy); it needs an env to construct.
    expert = se.ScriptedExpert(RoboChessEnv())
    if args.brake_test:
        for name, r in brake_test(expert).items():
            print(name, r)
        return 0

    results = []
    for pair in args.pairs.split(","):
        start, target = pair.split(":")
        r = run_pair(start, target, expert, budget=EPISODE_BUDGET)
        if r["failure"] == "budget:time_limit":
            # F8 information only: never counted toward the pass rule.
            rr = run_pair(start, target, expert, budget=RERUN_BUDGET)
            r["rerun_400"] = {k: rr[k] for k in ("success", "failure", "failure_phase", "steps",
                                                 "phase_lengths", "phase_stats")}
        results.append(r)
        rs = r["release_start"] or {}
        print(f"{start}->{target}: CLEAN={r['clean_place']} F2success={r['success']} failure={r['failure']}@{r['failure_phase']} "
              f"rel_base={rs.get('base_above_rest_mm')}mm tilt={rs.get('piece_tilt_deg')} lead={rs.get('implied_pinch_z_vs_floor_mm')} "
              f"steps={r['steps']}/{r['budget']} phases={r['phase_lengths']} rejects={r['ik_rejects']} "
              f"grip_self={r['gripper_self_contact_steps']} clips={r['clip_steps']}"
              + (f" rerun400={r['rerun_400']['success']}/{r['rerun_400']['failure']}" if "rerun_400" in r else ""),
              flush=True)
    print(f"CLEAN PLACE {sum(r['clean_place'] for r in results)}/{len(results)}  "
          f"F2 success {sum(r['success'] for r in results)}/{len(results)} "
          f"(budget ended {sum(r['budget_ended_episode'] for r in results)} episodes)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        meta = dict(
            harness=HARNESS_VERSION,
            preregistered_check_failed=dict(
                what="v3 (c): descend braking factor 0.2 must bring descend overshoot under 5 mm (h8, a8) or stop",
                result="FAILED: h8 0.4 -> 19.8 mm, 0.2 -> 11.2 mm; a8 0.4 -> 19.8 mm, 0.2 -> 11.0 mm "
                       "(vertical overshoot; along-travel 11.2 / 10.9 mm)",
                disposition="Withdrawn as a stop condition by the user before any v3 result (it was a proxy for "
                            "grasp height); replaced by per-episode reporting columns that do not change the verdict."),
            verdict_rule=f">= 8/10 clean_place at {EPISODE_BUDGET} steps",
            clean_place=sum(r["clean_place"] for r in results), f2_success=sum(r["success"] for r in results),
            close_gate_note=f"descend->close needs {ARRIVE_CONSECUTIVE} consecutive in-tolerance steps; "
                            f"close->lift needs pinched + {STEADY_CONSECUTIVE} consecutive steady-aperture steps")
        Path(args.out).write_text(json.dumps(dict(meta=meta, pairs=results), indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
