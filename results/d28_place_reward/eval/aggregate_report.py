#!/usr/bin/env python
"""Read the 8 custom_eval.py JSONs and print the Section-2 report in the
requested order. Read-only; does not touch tracked files."""
import json, math, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

JOBS = [
    ("ppo_seed0", "final_model", "ppo_seed0_final.json"),
    ("ppo_seed1", "final_model", "ppo_seed1_final.json"),
    ("ppo_seed2", "final_model", "ppo_seed2_final.json"),
    ("ppo_seed0", "rc_500000", "ppo_seed0_ck500000.json"),
    ("ppo_seed1", "rc_500000", "ppo_seed1_ck500000.json"),
    ("ppo_seed2", "rc_500000", "ppo_seed2_ck500000.json"),
    ("sac_seed0", "final_model", "sac_seed0_final.json"),
    ("sac_seed0", "rc_500000", "sac_seed0_ck500000.json"),
]


def wilson(k, n, z=1.959963985):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fmt_rate(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k/n:.4f}  Wilson95=[{lo:.4f},{hi:.4f}]" if n else "n=0"


def mmm(v):
    if not v:
        return "n=0"
    return f"min={min(v):.2f} median={st.median(v):.2f} max={max(v):.2f} (n={len(v)})"


for model, ck, fname in JOBS:
    path = HERE / fname
    if not path.exists():
        print(f"\n{'='*70}\n{model} {ck}: MISSING ({fname})\n{'='*70}")
        continue
    d = json.loads(path.read_text())
    print(f"\n{'='*78}\n{model}  {ck}\n{'='*78}")

    st_ = d["stochastic"]["summary"]
    rms_same = d["stochastic"]["obs_rms_identical_before_after"]
    print(f"-- stochastic, n={st_['n']}, seed={d['seed']}, obs_rms_identical_before_after={rms_same} --")

    pr = st_["place_report"]["rates"]
    print("D18 grasp_ever:            ", fmt_rate(st_["grasp_ever_k"], st_["grasp_ever_n"]))
    for stage, label in [("held", "held (D27)"), ("lifted_30mm", "lifted 30mm"),
                         ("carried_within_3cm", "carried within 3cm"), ("released", "released"),
                         ("placed", "placed (D27 success)")]:
        r = pr[stage]
        print(f"{label:26s}", fmt_rate(r["k"], r["n"]))

    tr2 = st_["trunc_s2_true_flag"]
    print(f"\nTruncated in S2 (env truncated flag): {fmt_rate(tr2['k'], tr2['n'])}")
    print(f"  final gripper qpos (sorted): {tr2['final_gripper_qpos']}")

    push = st_["place_report"]["push_assisted"]
    print(f"\nPush-assisted (of {push['successful']} successes): {push['push_assisted']}/{push['successful']} "
          f"= {push['fraction']:.4f}" if push["fraction"] is not None else f"\nPush-assisted: n/a (0 successes)")

    print(f"\nRelease height (mm), among successes: {mmm(st_['release_height_mm'])}")
    print(f"Release XY error (mm), among successes: {mmm(st_['release_xy_error_mm'])}")
    print(f"Success step (median): {mmm(st_['success_step'])}")

    print(f"\nTermination reason counts: {st_['reason_counts']}  (n_truncated={st_['n_truncated']}, n_terminated={st_['n_terminated']})")

    print(f"\nEpisode-end place_state counts (ALL episodes):       {st_['end_state_all']}")
    print(f"Episode-end place_state counts (TRUNCATED episodes): {st_['end_state_truncated']}")

    t1 = st_["truncated_in_s1"]
    print(f"\nTruncated-in-S1 episodes: n={t1['n']}")
    if t1["n"]:
        print(f"  final piece horiz. dist. to target (mm): {mmm(t1['final_dxy_mm'])}")
        print(f"  final piece lift above start height (mm): {mmm(t1['final_lift_mm'])}")
        print(f"  final pinched: {sum(t1['final_pinched'])}/{t1['n']}  final held: {sum(t1['final_held'])}/{t1['n']}")
        print(f"  final gripper qpos: {mmm(t1['final_gripper_qpos'])}")
        print(f"  min horiz. dist. to target reached at any step (mm): {mmm(t1['min_dxy_mm'])}")
        print(f"  max lift reached (mm): {mmm(t1['max_lift_mm'])}")
        print(f"  final-step recurring reward (place_terms.state_reward): {mmm(t1['final_state_reward'])}")
        print(f"  fraction within 3cm of target at final step: {t1['frac_within_3cm_final']:.4f}")

    e3 = st_["ever_within_3cm_while_held"]
    print(f"\nEver within 3cm of target while held: {fmt_rate(e3['k'], e3['n'])}")

    det = d["deterministic"]["summary"]
    det_rms_same = d["deterministic"]["obs_rms_identical_before_after"]
    identical = det["place_report"]["identical_episodes"]
    eff_n = det["place_report"]["effective_n"]
    placed_k = det["place_report"]["rates"]["placed"]["k"]
    print(f"\n-- deterministic, n={det['n']}, obs_rms_identical_before_after={det_rms_same} --")
    print(f"placed: {placed_k}/{det['n']}  identical_episodes={identical}  effective_n={eff_n}")
