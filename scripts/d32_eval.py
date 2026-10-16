"""D32 evaluation recorder helpers (pure functions; no rollouts, no environment edits).

Used by scripts/eval_probe.py when it is run with --d32. Records, per T2 episode, the obstacle
displacement and tilt the disturbance latch is built from, and WHICH latch condition fired; and, for
T1 and T2, the lift shares among placed episodes. The lift method is the one the RS32 diagnosis used
(results/rs32_48_pairs/eval/diagnosis/diag_eval_tables.py table E and diag_addendum_tables.py X2):
`max_lift_mm` per episode, thresholds on it, restricted to placed episodes. Those scripts are tied to
the RS32 file layout, so the method is re-implemented here rather than imported.

The latch (T2ObstacleTask._augment_info) is set at the first env step where, for any placed obstacle,
xy displacement from its reset position exceeds `displacement_tol` (5 mm), OR its tilt exceeds
`tilt_tol` (0.26 rad), OR it is off the board. The cause recorded here is read at that first step:
"displacement" if only the displacement exceeds its tolerance, "tilt" if only the tilt does, "both",
"off_board" if neither of the two does but an obstacle is off the board, "unresolved" if none does.

Definitions used by the D32 pre-registration:
  success         terminated_reason == "success" (for T2 this is is_success with no obstacle disturbed)
  placement only  terminated_reason in ("success", "success_obstacles_disturbed")
"""
from __future__ import annotations

import collections
from typing import Any

import numpy as np

LIFT_THRESHOLDS_MM = (10.0, 50.0)
PLACEMENT_ONLY_REASONS = ("success", "success_obstacles_disturbed")


def find_t2(env):
    """The T2ObstacleTask in a wrapper stack, or None (T1)."""
    node = env
    while node is not None:
        if type(node).__name__ == "T2ObstacleTask":
            return node
        node = getattr(node, "env", None)
    return None


def obstacle_row(t2, info: dict[str, Any]) -> dict[str, Any]:
    """The obstacle readings at the current state, from the task's own measurement methods.

    `latch` is the env's own `obstacles_ever_disturbed` for this step. Displacement and tilt are the
    maxima over the placed obstacles (parked slots are never read)."""
    n = len(t2.current_obstacle_squares)
    latch = bool(info.get("obstacles_ever_disturbed", False))
    if n == 0:
        return dict(disp_mm=0.0, tilt=0.0, offb=False, latch=latch)
    moved = np.linalg.norm(t2._obstacle_xy()[:n] - t2._placed_xy[:n], axis=1)
    return dict(disp_mm=float(moved.max() * 1e3), tilt=float(t2._obstacle_tilts()[:n].max()),
                offb=bool(t2._obstacle_off_board()[:n].any()), latch=latch)


def obstacle_episode_summary(ob_rows: list[dict[str, Any]], disp_tol_mm: float, tilt_tol: float) -> dict[str, Any]:
    """Per-episode obstacle summary: peaks over the episode, and the latch cause at its first step."""
    peak_disp = max(r["disp_mm"] for r in ob_rows)
    peak_tilt = max(r["tilt"] for r in ob_rows)
    k = next((i for i, r in enumerate(ob_rows) if r["latch"]), None)
    if k is None:
        return dict(peak_disp_mm=peak_disp, peak_tilt_rad=peak_tilt, latch_fired=False, latch_step=None,
                    disp_at_latch_mm=None, tilt_at_latch_rad=None, offboard_at_latch=None, cause=None)
    r = ob_rows[k]
    d, t = r["disp_mm"] > disp_tol_mm, r["tilt"] > tilt_tol
    cause = ("both" if d and t else "displacement" if d else "tilt" if t
             else "off_board" if r["offb"] else "unresolved")
    return dict(peak_disp_mm=peak_disp, peak_tilt_rad=peak_tilt, latch_fired=True, latch_step=k + 1,
                disp_at_latch_mm=r["disp_mm"], tilt_at_latch_rad=r["tilt"], offboard_at_latch=r["offb"],
                cause=cause)


def episode_extra(summary: dict[str, Any], ob_rows: list[dict[str, Any]] | None,
                  disp_tol_mm: float | None, tilt_tol: float | None) -> dict[str, Any]:
    """What --d32 adds to one episode summary: lift flags (T1 and T2), and the obstacle block (T2)."""
    ml = float(summary["max_lift_mm"])
    out: dict[str, Any] = {f"lift_gt{int(t)}mm": bool(ml > t) for t in LIFT_THRESHOLDS_MM}
    if ob_rows is not None:
        out["obstacle"] = obstacle_episode_summary(ob_rows, disp_tol_mm, tilt_tol)
    return out


def _lift_share(eps: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(eps)}
    for t in LIFT_THRESHOLDS_MM:
        k = sum(1 for e in eps if e["max_lift_mm"] > t)
        out[f"gt{int(t)}mm"] = dict(k=k, n=len(eps), share=(k / len(eps) if eps else None))
    return out


def d32_report(summaries: list[dict[str, Any]], is_t2: bool) -> dict[str, Any]:
    """The D32 numbers, from the per-episode summaries (each carrying a `d32` block)."""
    success = [e for e in summaries if e["reason"] == "success"]
    placement = [e for e in summaries if e["reason"] in PLACEMENT_ONLY_REASONS]
    rep: dict[str, Any] = dict(
        n=len(summaries), success=len(success), placement_only=len(placement),
        reasons=dict(collections.Counter(e["reason"] for e in summaries)),
        lift_among_success=_lift_share(success), lift_among_placement_only=_lift_share(placement))
    if is_t2:
        fired = [e for e in summaries if e["d32"]["obstacle"]["latch_fired"]]
        failures = [e for e in summaries if e["reason"] != "success"]
        cause = lambda es: dict(collections.Counter(e["d32"]["obstacle"]["cause"] for e in es))
        rep["latch"] = dict(
            episodes_fired=len(fired), cause_all_fired=cause(fired),
            failures=len(failures), failures_with_latch=sum(1 for e in failures if e["d32"]["obstacle"]["latch_fired"]),
            cause_among_failures=cause([e for e in failures if e["d32"]["obstacle"]["latch_fired"]]),
            placed_not_success=sum(1 for e in summaries if e["reason"] == "success_obstacles_disturbed"),
            cause_among_placed_not_success=cause([e for e in summaries if e["reason"] == "success_obstacles_disturbed"]))
    return rep
