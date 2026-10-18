#!/usr/bin/env python
"""D33 results: reads results/d33_t1/eval/*.json (scripts/eval_probe.py --d32) and applies the rule in
results/d33_t1/PREREGISTRATION.md. Read-only; no rollouts. The D32 control evaluation (the "before") is read from git
at commit d4dde28 (results/d32_t1_t2/eval/d32_t1_e4e5_seed*.json), not copied.

    python results/d33_t1/analyse.py > results/d33_t1/analysis_output.txt

Rule (fixed before any D33 run): a seed learns if success (terminated_reason "success") is >= 100 of 200 episodes; the claim
holds if >= 3 of 5 seeds learn, and then T2 switches to the D33 reward in a new pre-registered cell; otherwise dragging is
reported as a finding and T2 stays on the D32 cell. Wilson 95%: z 1.96, no continuity correction.

Failure classes (PREREGISTRATION.md "Reported, not gated"; mutually exclusive, applied in this order to every episode whose
terminated_reason is not "success"):
  toppled        piece_off_board, or the piece tilt in S1 above 45 degrees (place.max_s1_tilt)
  bad contact    bad_contact
  stuck low      time limit, max_lift_mm below 20 mm (the latch was never reached), not toppled
  lifted, unfinished   time limit, max_lift_mm at or above 20 mm, not toppled
  other          anything else
"""
import collections
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EV = HERE / "eval"
SEEDS = (0, 100, 200, 300, 400)
LEARN_AT, CLAIM_AT = 100, 3
CONTROL_COMMIT = "d4dde284827cd248dbb4fe6411449b4de6760e7d"
TOPPLE_DEG, LATCH_MM = 45.0, 20.0


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*k/n:5.1f}% [{100*lo:5.1f}, {100*hi:5.1f}]"


def d33(seed):
    p = EV / f"d33_t1_e4e5_seed{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None


def control(seed):
    out = subprocess.run(["git", "-C", str(REPO), "show", f"{CONTROL_COMMIT}:results/d32_t1_t2/eval/d32_t1_e4e5_seed{seed}.json"],
                         capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 else None


def failure_class(e):
    tilt = (e.get("place") or {}).get("max_s1_tilt")
    if e["reason"] == "piece_off_board" or (tilt is not None and tilt > TOPPLE_DEG):
        return "toppled"
    if e["reason"] == "bad_contact":
        return "bad contact"
    if e["reason"] == "time_limit":
        return "stuck low" if e["max_lift_mm"] < LATCH_MM else "lifted, unfinished"
    return "other"


print("D33 evaluation analysis (rule: seed learns at >= 100/200 success; claim at >= 3 of 5 seeds; thresholds fixed in PREREGISTRATION.md)")
dd = {s: d33(s) for s in SEEDS}
cc = {s: control(s) for s in SEEDS}
have = [s for s in SEEDS if dd[s] is not None]
if len(have) < 5:
    print(f"  evaluated seeds: {have} (not all five)")

learned = 0
K = N = KC = NC = 0
placed_all, eps_all = [], []
fail_all = collections.Counter()
reasons_all = collections.Counter()
print("\n##### D33 (place_require_lift, 20 mm latch) next to the D32 T1 control (same cell without the flag)")
for s in have:
    d = dd[s]
    eps = d["episodes"]
    n = len(eps)
    reasons = collections.Counter(e["reason"] for e in eps)
    k = reasons["success"]
    kp = sum(reasons[r] for r in ("success", "success_obstacles_disturbed"))
    ok = k >= LEARN_AT
    learned += ok
    K += k
    N += n
    reasons_all.update(reasons)
    eps_all += eps
    placed = [e for e in eps if e["reason"] == "success"]
    placed_all += placed
    g10 = sum(e["max_lift_mm"] > 10 for e in placed)
    g50 = sum(e["max_lift_mm"] > 50 for e in placed)
    reached = sum(e["max_lift_mm"] >= LATCH_MM for e in eps)
    fails = [e for e in eps if e["reason"] != "success"]
    fc = collections.Counter(failure_class(e) for e in fails)
    fail_all.update(fc)
    print(f"  seed {s:>3}: success {fmt(k, n)}   placement-only {fmt(kp, n)}   learns (>= {LEARN_AT}/200): {'YES' if ok else 'no'}")
    print(f"            terminated_reason counts: {dict(reasons)}   [eval n={d['n']}, seed {d['seed']}, deterministic={d['deterministic']}, model={d['model']}, obs_rms_identical_before_after={d['obs_rms_identical_before_after']}]")
    print(f"            placed episodes (n={len(placed)}): max_lift > 10 mm {g10}, > 50 mm {g50}; episodes with max_lift >= 20 mm (any time): {reached}/{n}")
    print(f"            failures ({len(fails)}): {dict(fc)}")
    c = cc[s]
    if c:
        ce = c["episodes"]
        kc = sum(e["reason"] == "success" for e in ce)
        KC += kc
        NC += len(ce)
        print(f"            D32 control seed {s}: success {fmt(kc, len(ce))}")
print(f"\n  D33 seeds learning: {learned} out of {len(have)}   claim (>= {CLAIM_AT} of 5): {'HOLDS' if learned >= CLAIM_AT and len(have) == 5 else 'does not hold' if len(have) == 5 else 'n/a (not all seeds)'}")
print(f"  D33 pooled success    {fmt(K, N)}")
if NC:
    print(f"  D32 control pooled    {fmt(KC, NC)}   (control seeds learning: {sum(sum(e['reason']=='success' for e in cc[s]['episodes']) >= LEARN_AT for s in SEEDS if cc[s])} of 5)")
print(f"  D33 pooled terminated_reason counts: {dict(reasons_all)}")
if placed_all:
    g10 = sum(e["max_lift_mm"] > 10 for e in placed_all)
    g50 = sum(e["max_lift_mm"] > 50 for e in placed_all)
    ml = sorted(e["max_lift_mm"] for e in placed_all)
    print(f"  D33 placed episodes pooled (n={len(placed_all)}): max_lift > 10 mm {fmt(g10, len(placed_all))}; > 50 mm {fmt(g50, len(placed_all))}; median max lift {ml[len(ml)//2]:.1f} mm; min {ml[0]:.1f} mm")
else:
    print("  D33 placed episodes pooled: none")
reached_all = sum(e["max_lift_mm"] >= LATCH_MM for e in eps_all)
print(f"  D33 episodes whose piece ever reached 20 mm lift (any time, pinched or not): {fmt(reached_all, N)}")
print(f"  D33 failure classes pooled ({sum(fail_all.values())} failures): {dict(fail_all)}")

print("\n##### D33 rule verdict")
if len(have) == 5:
    if learned >= CLAIM_AT:
        print(f"  D33 meets the rule on {learned} of 5 seeds (>= 3): T2 switches to the D33 reward in a new pre-registered cell.")
    else:
        print(f"  D33 meets the rule on {learned} of 5 seeds (< 3): dragging is reported as a finding and T2 stays on the D32 cell.")
else:
    print("  not all five seeds evaluated; no verdict.")
