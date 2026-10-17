#!/usr/bin/env python
"""D32 results: reads results/d32_t1_t2/eval/*.json (scripts/eval_probe.py --d32 output) and applies the
pre-registered rule in results/d32_t1_t2/PREREGISTRATION.md. Read-only; no rollouts.

    python results/d32_t1_t2/analyse.py > results/d32_t1_t2/analysis_output.txt

Rule (fixed before any run): a seed learns if success (terminated_reason "success") is >= 100 of 200 episodes;
the claim holds for an arm if >= 3 of 5 seeds learn. Wilson 95% interval: z 1.96, no continuity correction.
Prediction (not a gate): pooled T2 success within 10 percentage points of the pooled T1 control.
D33 (results/d33_t1/PREREGISTRATION.md) is evaluated only if all five D33 eval files exist.
"""
import collections
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV = HERE / "eval"
SEEDS = (0, 100, 200, 300, 400)
ARMS = {"T1 control": "d32_t1_e4e5_seed{}", "T2 (d4,f4)": "d32_t2_d4f4_seed{}", "D33 (lift latch)": "d33_t1_e4e5_seed{}"}
LEARN_AT, CLAIM_AT, PRED_PTS = 100, 3, 10.0
PLACEMENT_ONLY = ("success", "success_obstacles_disturbed")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*k/n:5.1f}% [{100*lo:5.1f}, {100*hi:5.1f}]"


def load(name):
    p = EV / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def arm_report(label, pattern):
    data = {s: load(pattern.format(s)) for s in SEEDS}
    have = [s for s in SEEDS if data[s] is not None]
    if not have:
        print(f"\n##### {label}: no evaluation files")
        return None
    print(f"\n##### {label}")
    if len(have) < len(SEEDS):
        print(f"  evaluated seeds: {have} (not all five)")
    k_succ_tot = n_tot = k_place_tot = 0
    learned = 0
    allreasons = collections.Counter()
    eps_all = []
    for s in have:
        d = data[s]
        eps = d["episodes"]
        n = len(eps)
        reasons = collections.Counter(e["reason"] for e in eps)
        k = reasons["success"]
        kp = sum(reasons[r] for r in PLACEMENT_ONLY)
        ok = k >= LEARN_AT
        learned += ok
        k_succ_tot += k
        k_place_tot += kp
        n_tot += n
        allreasons.update(reasons)
        eps_all += eps
        print(f"  seed {s:>3}: success {fmt(k, n)}   placement-only {fmt(kp, n)}   learns (>= {LEARN_AT}/200): {'YES' if ok else 'no'}")
        print(f"            terminated_reason counts: {dict(reasons)}   [eval n={d['n']}, seed {d['seed']}, deterministic={d['deterministic']}, "
              f"model={d['model']}, obs_rms_identical_before_after={d['obs_rms_identical_before_after']}]")
    print(f"  seeds learning: {learned} out of {len(have)}   claim (>= {CLAIM_AT} of 5): {'HOLDS' if learned >= CLAIM_AT and len(have) == 5 else 'does not hold' if len(have) == 5 else 'n/a (not all seeds)'}")
    print(f"  pooled success        {fmt(k_succ_tot, n_tot)}")
    print(f"  pooled placement-only {fmt(k_place_tot, n_tot)}")
    print(f"  pooled terminated_reason counts: {dict(allreasons)}")

    # reported, not gated: lift shares among placed episodes (max_lift_mm thresholds)
    for tag, sel in (("success", lambda e: e["reason"] == "success"), ("placement-only", lambda e: e["reason"] in PLACEMENT_ONLY)):
        placed = [e for e in eps_all if sel(e)]
        if not placed:
            print(f"  lift among {tag} episodes: none")
            continue
        g10 = sum(e["max_lift_mm"] > 10 for e in placed)
        g50 = sum(e["max_lift_mm"] > 50 for e in placed)
        ml = sorted(e["max_lift_mm"] for e in placed)
        med = ml[len(ml) // 2]
        print(f"  lift among {tag} episodes (pooled, n={len(placed)}): max_lift > 10 mm {fmt(g10, len(placed))}; > 50 mm {fmt(g50, len(placed))}; median max lift {med:.1f} mm")
        per = []
        for s in have:
            pe = [e for e in data[s]["episodes"] if sel(e)]
            per.append(f"s{s}: {sum(e['max_lift_mm'] > 10 for e in pe)}/{len(pe)} >10, {sum(e['max_lift_mm'] > 50 for e in pe)}/{len(pe)} >50")
        print("      per seed: " + "; ".join(per))
    # T2: latch causes
    if "latch" in (data[have[0]].get("d32_report") or {}):
        tot = collections.Counter(); fail = collections.Counter(); pns = collections.Counter(); nf = nfl = 0
        for s in have:
            r = data[s]["d32_report"]["latch"]
            tot.update(r["cause_all_fired"]); fail.update(r["cause_among_failures"]); pns.update(r["cause_among_placed_not_success"])
            nf += r["failures"]; nfl += r["failures_with_latch"]
        print(f"  T2 latch: episodes where it fired (pooled) by cause: {dict(tot)}")
        print(f"            failures (reason != success): {nf}, of which the latch fired in {nfl}; cause among those: {dict(fail)}")
        print(f"            placed but not success (success_obstacles_disturbed): {sum(pns.values())}; cause: {dict(pns)}")
        for s in have:
            r = data[s]["d32_report"]["latch"]
            print(f"      seed {s}: fired in {r['episodes_fired']} episodes, causes {r['cause_all_fired']}; failures {r['failures']}, with latch {r['failures_with_latch']}")
        peaks = [(e["d32"]["obstacle"]["peak_disp_mm"], e["d32"]["obstacle"]["peak_tilt_rad"]) for e in eps_all]
        print(f"            peak obstacle displacement over all episodes: max {max(p[0] for p in peaks):.2f} mm; peak tilt: max {max(p[1] for p in peaks):.3f} rad")
    return dict(learned=learned, k=k_succ_tot, n=n_tot, kp=k_place_tot, all_seeds=len(have) == 5)


print("D32 evaluation analysis (rule: seed learns at >= 100/200 success; claim at >= 3 of 5 seeds; thresholds fixed in PREREGISTRATION.md)")
res = {label: arm_report(label, pat) for label, pat in ARMS.items()}

t1, t2 = res["T1 control"], res["T2 (d4,f4)"]
if t1 and t2:
    p1, p2 = 100 * t1["k"] / t1["n"], 100 * t2["k"] / t2["n"]
    print("\n##### Prediction check (not a gate): pooled T2 within 10 percentage points of the pooled T1 control")
    print(f"  T1 control pooled success {p1:.1f}%   T2 pooled success {p2:.1f}%   difference (T2 - T1) {p2 - p1:+.1f} points   -> "
          f"{'WITHIN 10 points' if abs(p2 - p1) <= PRED_PTS else 'OUTSIDE 10 points'}")
    q1, q2 = 100 * t1["kp"] / t1["n"], 100 * t2["kp"] / t2["n"]
    print(f"  placement-only (secondary): T1 {q1:.1f}%   T2 {q2:.1f}%   difference {q2 - q1:+.1f} points")

d33 = res["D33 (lift latch)"]
print("\n##### D33")
if d33 and d33["all_seeds"]:
    print(f"  D33 meets the rule on {d33['learned']} of 5 seeds -> "
          f"{'at least 3: T2 switches to the D33 reward in a new pre-registered cell' if d33['learned'] >= CLAIM_AT else 'fewer than 3: dragging is reported as a finding and T2 stays on D32'}")
    if t1:
        print(f"  D32 control pooled {100*t1['k']/t1['n']:.1f}% next to D33 pooled {100*d33['k']/d33['n']:.1f}%")
else:
    print("  not evaluated: the D33 runs are not all finished (none was launched at the time of this analysis)." if not d33 else "  not all five D33 seeds evaluated.")
