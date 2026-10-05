#!/usr/bin/env python
"""D29 section 4 follow-up, item 4 (read-only): squash s1's 87 episodes that
end in state S2. Ever held, piece displacement from the start square, and
which success terms are false at the final step -- separating "push without
grasp" from "fragile grip" with numbers rather than asserting either.

Reads section4_followup_ppotanh_seed1.json from --trace-dir. Writes only --out.
"""
import argparse, json, math
from pathlib import Path

import numpy as np

TERMS = ["piece_on_square", "piece_resting", "gripper_released", "piece_settled",
         "piece_still", "piece_upright", "release_ok"]

p = argparse.ArgumentParser()
p.add_argument("--trace-dir", type=Path, default=Path.home() / "robochess_archive" / "d29_eval_traces")
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

with open(a.trace_dir / "section4_followup_ppotanh_seed1.json") as f:
    d = json.load(f)

eps = d["episodes"]
s2_end = [ep for ep in eps if ep["rows"][-1]["state"] == "S2"]

ever_held = [any(r["held"] for r in ep["rows"]) for ep in s2_end]

disp_mm = []
for ep in s2_end:
    start = ep["start_xy"]
    final = ep["rows"][-1]["piece_xy"]
    disp_mm.append(1000 * math.dist(start, final))

false_counts = {t: 0 for t in TERMS}
for ep in s2_end:
    last = ep["rows"][-1]
    for t in TERMS:
        if not last[t]:
            false_counts[t] += 1

detail = []
for ep, eh, disp in zip(s2_end, ever_held, disp_mm):
    last = ep["rows"][-1]
    false_terms = [t for t in TERMS if not last[t]]
    detail.append(dict(ever_held=eh, disp_mm=round(disp, 1), n=len(ep["rows"]), false_terms=false_terms,
                        terminated_reason=last["terminated_reason"]))

out = dict(n_s2_end=len(s2_end), n_total=len(eps), ever_held_count=sum(ever_held),
           disp_mm_stats=dict(mean=float(np.mean(disp_mm)), median=float(np.median(disp_mm)),
                               min=float(np.min(disp_mm)), max=float(np.max(disp_mm))),
           false_at_final=false_counts, per_episode=detail)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
