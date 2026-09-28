#!/usr/bin/env python
"""D29 section 4, item 4 (read-only): terminated_reason counts among the
NON-PLACED episodes of each of the 10 PPO eval_probe.py runs (empty/"time_limit"
= ran to 300 steps without terminating).

Reads the small eval_probe.py outputs already in --eval-dir. Writes only --out.
"""
import argparse, collections, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUNS = [f"ppo_seed{s}" for s in range(5)] + [f"ppotanh_seed{s}" for s in range(5)]

p = argparse.ArgumentParser()
p.add_argument("--trace-dir", type=Path, default=Path.home() / "robochess_archive" / "d29_eval_traces",
                help="unused by this script; accepted for interface consistency")
p.add_argument("--eval-dir", type=Path, default=REPO / "results" / "d29_place_reward_v2" / "eval")
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

out = {}
for r in RUNS:
    with open(a.eval_dir / f"{r}_final.json") as f:
        d = json.load(f)
    eps = d["episodes"]
    non_placed = [e for e in eps if not e["place"]["placed"]]
    reasons = collections.Counter(e["reason"] for e in non_placed)
    out[r] = dict(n_non_placed=len(non_placed), reasons=dict(reasons))

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
