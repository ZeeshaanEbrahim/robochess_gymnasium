"""Place-mode (D28) episode metrics and their aggregation, shared by eval_probe.py / eval_probe_agg.py.

Pure functions over per-step rows so they can be tested without a model or an env.
A row is a dict with: st (place_state S0..S3), held, pinched (bool), lift (m), dxy (m),
ap (gripper qpos), succ (env is_success).
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
    return c - h, c + h


def place_episode_summary(rows: list[dict], reason: str) -> dict:
    """Per-episode booleans, in the order of the task: grasp -> lifted -> carried -> released -> placed."""
    grasp = any(r["held"] for r in rows)
    lift_k = next((k for k, r in enumerate(rows) if r["st"] == "S1" and r["pinched"] and r["lift"] >= LIFT_M), None)
    carried = lift_k is not None and any(
        r["st"] == "S1" and r["dxy"] < CARRY_TOL_M for r in rows[lift_k:])
    released = any(r["st"] == "S2" for r in rows)
    placed = bool(rows[-1]["succ"])
    return dict(grasp=grasp, lifted_30mm=lift_k is not None, carried_within_3cm=carried,
                released=released, placed=placed,
                truncated_in_s2=bool(reason == "time_limit" and rows[-1]["st"] == "S2"),
                final_state=rows[-1]["st"], final_gripper_qpos=float(rows[-1]["ap"]),
                s2_steps=sum(r["st"] == "S2" for r in rows), n=len(rows))


STAGES = ("grasp", "lifted_30mm", "carried_within_3cm", "released", "placed")


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
    lo, hi = wilson(len(t2) if eff == n else round(len(t2) * scale), eff)
    return dict(n=n, effective_n=eff, deterministic=deterministic, identical_episodes=identical,
                rates=rates,
                truncated_in_s2=dict(k=len(t2), n=n, wilson_lo=lo, wilson_hi=hi,
                                     final_gripper_qpos=sorted(round(e["final_gripper_qpos"], 4) for e in t2)))
