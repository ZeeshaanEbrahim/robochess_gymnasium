#!/usr/bin/env python
"""D29 section 4 follow-up, item 3 (read-only): per-dimension action statistics
while held (dx, dy, dz, gripper), for default s0/s1/s3 and squash s0/s1/s3.
Median Gaussian mean, median std, and the analytic probability of sampling the
opposite sign to the mean (Phi(-|mean|/std), symmetric under mean<0 by
construction). Plus the gripper-open probability restricted to held steps
within 30mm of the target (threshold: action[3] < 0, the midpoint of the
exposed [-1,1] action space -- see section4_followup2's gripper-conversion
derivation for why the qpos-based gripper_open_threshold=0.2 can't be used
directly here, different unit).

Reads section4_followup_{run}.json for all 6 runs from --trace-dir.
Writes only --out.
"""
import argparse, json
from pathlib import Path

import numpy as np
from scipy.stats import norm

RUNS = ["ppo_seed0", "ppo_seed1", "ppo_seed3", "ppotanh_seed0", "ppotanh_seed1", "ppotanh_seed3"]
DIMS = ["dx", "dy", "dz", "gripper"]

p = argparse.ArgumentParser()
p.add_argument("--trace-dir", type=Path, default=Path.home() / "robochess_archive" / "d29_eval_traces")
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()


def load(run):
    with open(a.trace_dir / f"section4_followup_{run}.json") as f:
        return json.load(f)


results = {}
for run in RUNS:
    d = load(run)
    held_rows = [r for ep in d["episodes"] for r in ep["rows"] if r["held"]]
    held30_rows = [r for r in held_rows if r["d_xy"] < 0.03]
    means = np.array([r["mean"] for r in held_rows])
    stds = np.array([r["std"] for r in held_rows])
    p_opp = norm.cdf(-np.abs(means) / stds)
    per_dim = {}
    for i, dim in enumerate(DIMS):
        per_dim[dim] = dict(
            median_mean=float(np.median(means[:, i])),
            median_std=float(np.median(stds[:, i])),
            median_p_opposite_sign=float(np.median(p_opp[:, i])),
        )
    if held30_rows:
        m30 = np.array([r["mean"][3] for r in held30_rows])
        s30 = np.array([r["std"][3] for r in held30_rows])
        p_open = norm.cdf(-m30 / s30)
        gripper_open = dict(n_held30=len(held30_rows), median_p_open=float(np.median(p_open)),
                             mean_p_open=float(np.mean(p_open)),
                             frac_steps_p_open_gt_half=float((p_open > 0.5).mean()))
    else:
        gripper_open = dict(n_held30=0, median_p_open=None, mean_p_open=None, frac_steps_p_open_gt_half=None)
    results[run] = dict(n_held=len(held_rows), per_dim=per_dim, gripper_open_within_30mm=gripper_open)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
