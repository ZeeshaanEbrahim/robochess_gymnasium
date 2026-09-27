"""Place-mode (D28) episode metrics and their aggregation, shared by eval_probe.py / eval_probe_agg.py.

Pure functions over per-step rows so they can be tested without a model or an env.
A row is a dict with: st (place_state S0..S3), held, pinched (bool), lift (m), dxy (m),
ap (gripper qpos), succ (env is_success); optionally pxy (piece x, y in m) and gxy (target x, y),
which the push-assisted diagnostic needs; optionally ptilt (piece tilt, deg) and pg
(place_terms["posture_gate"], D29), which the gate diagnostic needs.
"""
from __future__ import annotations

import math

LIFT_M = 0.03          # "lifted": piece 30 mm up while pinched
CARRY_TOL_M = 0.03     # "carried within 3 cm": the env's goal_tolerance


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def place_episode_summary(rows: list[dict], reason: str) -> dict:
    """Per-episode booleans, in the order of the task: held -> lifted -> carried -> released -> placed."""
    # "held" is the D27 condition (both pads on the piece, no piece-board contact); it is NOT D18
    # grasp_success, which eval_probe.py reports separately as grasp_ever.
    held = any(r["held"] for r in rows)
    lift_k = next((k for k, r in enumerate(rows) if r["st"] == "S1" and r["pinched"] and r["lift"] >= LIFT_M), None)
    carried = lift_k is not None and any(
        r["st"] == "S1" and r["dxy"] < CARRY_TOL_M for r in rows[lift_k:])
    released = any(r["st"] == "S2" for r in rows)
    placed = bool(rows[-1]["succ"])
    held_idx = [k for k, r in enumerate(rows) if r["held"]]
    carry_mm = start_to_target_mm = None
    if "pxy" in rows[0] and "gxy" in rows[0]:
        start_to_target_mm = 1e3 * math.dist(rows[0]["pxy"], rows[0]["gxy"])
        if held_idx:
            carry_mm = 1e3 * math.dist(rows[held_idx[0]]["pxy"], rows[held_idx[-1]]["pxy"])
    # D29, reported not gated: S1 steps and how many the posture gate discounted
    # (pg < 1), plus the max piece tilt reached in S1 -- 0/None when the gate is
    # off (posture_gate then defaults to 1.0 every step, so gate_lt1 is always 0).
    s1_rows = [r for r in rows if r["st"] == "S1"]
    s1_gate_lt1 = sum(1 for r in s1_rows if r.get("pg", 1.0) < 1.0)
    s1_tilts = [r["ptilt"] for r in s1_rows if "ptilt" in r]
    max_s1_tilt = max(s1_tilts) if s1_tilts else None
    return dict(held=held, held_steps=len(held_idx), carry_mm=carry_mm,
                start_to_target_mm=start_to_target_mm, lifted_30mm=lift_k is not None, carried_within_3cm=carried,
                released=released, placed=placed,
                truncated_in_s2=bool(reason == "time_limit" and rows[-1]["st"] == "S2"),
                final_state=rows[-1]["st"], final_gripper_qpos=float(rows[-1]["ap"]),
                s2_steps=sum(r["st"] == "S2" for r in rows), n=len(rows),
                s1_steps=len(s1_rows), s1_gate_lt1_steps=s1_gate_lt1, max_s1_tilt=max_s1_tilt)


PUSH_ASSISTED_FRACTION = 0.5


def push_assisted(summaries: list[dict]) -> dict:
    """Reported, not a gate: of the SUCCESSFUL episodes, those whose held phase moved the piece
    less than half the start-to-target distance (the piece got most of the way by being pushed,
    with at most a token lift). Successful episodes with no held step count as push-assisted."""
    ok = [e for e in summaries if e["placed"] and e.get("start_to_target_mm") is not None]
    flagged = [e for e in ok if e["carry_mm"] is None
               or e["carry_mm"] < PUSH_ASSISTED_FRACTION * e["start_to_target_mm"]]
    return dict(label="push-assisted", successful=len(ok), push_assisted=len(flagged),
                fraction=(len(flagged) / len(ok) if ok else None),
                rule=f"carry_mm < {PUSH_ASSISTED_FRACTION:.0%} of start-to-target distance",
                per_episode=[dict(held_steps=e["held_steps"], carry_mm=e["carry_mm"],
                                  start_to_target_mm=e["start_to_target_mm"]) for e in ok])


def gate_diagnostics(summaries: list[dict]) -> dict:
    """D29, reported not gated: of the PLACED episodes, the fraction of S1 steps
    the posture gate discounted (posture_gate < 1) and the max S1 tilt reached.
    Both are None/0 with no placed episodes, and fraction_gate_lt1 is always 0.0
    for a flag-off (D28) replay, since posture_gate then defaults to 1.0."""
    ok = [e for e in summaries if e["placed"]]
    total_s1 = sum(e["s1_steps"] for e in ok)
    total_gate_lt1 = sum(e["s1_gate_lt1_steps"] for e in ok)
    tilts = [e["max_s1_tilt"] for e in ok if e["max_s1_tilt"] is not None]
    return dict(label="gate (placed episodes only)", placed=len(ok),
                s1_steps=total_s1, s1_gate_lt1_steps=total_gate_lt1,
                fraction_gate_lt1=(total_gate_lt1 / total_s1 if total_s1 else None),
                max_s1_tilt_deg=(max(tilts) if tilts else None))


STAGES = ("held", "lifted_30mm", "carried_within_3cm", "released", "placed")


def place_report(summaries: list[dict], deterministic: bool = False, identical: bool | None = None) -> dict:
    """Rates with Wilson intervals; the S2-truncation fraction and its final gripper qpos.

    A deterministic policy from a fixed start repeats one episode, so the 20 episodes are
    ONE observation: `effective_n` is 1 when they are all identical, and the intervals are
    computed at that n (not 20).
    """
    n = len(summaries)
    eff = 1 if (deterministic and identical) else n
    scale = eff / n if n else 1
    rates = {}
    for s in STAGES:
        k = sum(e[s] for e in summaries)
        ke = round(k * scale) if eff != n else k
        lo, hi = wilson(ke, eff)
        rates[s] = dict(k=k, n=n, rate=(k / n if n else 0.0), effective_k=ke, effective_n=eff,
                        wilson_lo=lo, wilson_hi=hi)
    t2 = [e for e in summaries if e["truncated_in_s2"]]
    push = push_assisted(summaries)
    gate = gate_diagnostics(summaries)
    lo, hi = wilson(len(t2) if eff == n else round(len(t2) * scale), eff)
    return dict(n=n, effective_n=eff, deterministic=deterministic, identical_episodes=identical,
                rates=rates, push_assisted=push, gate=gate,
                truncated_in_s2=dict(k=len(t2), n=n, wilson_lo=lo, wilson_hi=hi,
                                     final_gripper_qpos=sorted(round(e["final_gripper_qpos"], 4) for e in t2)))
