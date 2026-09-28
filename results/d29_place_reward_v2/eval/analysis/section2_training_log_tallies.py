#!/usr/bin/env python
"""D29 section 2 (read-only): training-log tallies from Monitor CSV (the
stochastic MOVING policy during training, not the frozen final-model eval).

Success column: `terminated_reason == "success"` -- a standard Monitor keyword
logged for every run regardless of mode. The env sets it exactly when
info["is_success"] is True at termination (robochess_env.py's step()), the
same flag eval_probe.py/place_eval.py read as `succ`/`placed`. Empty
terminated_reason = truncated at max_episode_steps (300).

Reads runs/ only. Writes only --out. No trace directory needed (kept as an
accepted argument for interface consistency).
"""
import argparse, csv, json, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

RUNS = [f"_probe_cart_place29_ppo_seed{s}" for s in range(5)] + \
       [f"_probe_cart_place29_ppotanh_seed{s}" for s in range(5)] + \
       ["_probe_cart_place29_sac_seed0"]

p = argparse.ArgumentParser()
p.add_argument("--trace-dir", type=Path, default=Path.home() / "robochess_archive" / "d29_eval_traces",
                help="unused by this script; accepted for interface consistency")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--runs-dir", type=Path, default=REPO / "runs")
a = p.parse_args()

out = {}
for run in RUNS:
    path = a.runs_dir / run / "logs" / "monitor.monitor.csv"
    with open(path) as f:
        next(f)  # comment/header line with t_start json
        rows = list(csv.DictReader(f))
    n = len(rows)
    reasons = {}
    for row in rows:
        tr = row["terminated_reason"]
        reasons[tr] = reasons.get(tr, 0) + 1
    succ_all = sum(1 for row in rows if row["terminated_reason"] == "success")
    last100 = rows[-100:]
    last500 = rows[-500:]
    succ_100 = sum(1 for row in last100 if row["terminated_reason"] == "success")
    succ_500 = sum(1 for row in last500 if row["terminated_reason"] == "success")
    mean_ret_100 = statistics.mean(float(row["r"]) for row in last100) if last100 else None
    out[run] = dict(
        episodes=n,
        placement_successes_all=succ_all,
        placement_successes_all_rate=succ_all / n if n else None,
        placement_successes_last100=succ_100,
        placement_successes_last100_rate=succ_100 / len(last100) if last100 else None,
        placement_successes_last500=succ_500,
        placement_successes_last500_rate=succ_500 / len(last500) if last500 else None,
        terminated_reason_counts=reasons,
        mean_return_last100=mean_ret_100,
        n_last100=len(last100),
        n_last500=len(last500),
    )

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
