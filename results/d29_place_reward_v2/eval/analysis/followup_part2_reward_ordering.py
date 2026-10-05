#!/usr/bin/env python
"""D29 section 4 follow-up, item 2 (read-only): discounted-return reward
ordering. Gamma 0.99, env reward before VecNormalize (the "reward" field in
the traces below, reconstructed from place_terms and cross-checked against
venv.step()'s raw r with norm_reward=False -- see section4_followup_diag.py).

G_from(rewards, k) re-bases discounting at step k (matches D29's own
results/d29_place_reward_v2/analyze.py); boot() adds the D29 acceptance
test's truncation bootstrap for episodes that ran the full 300 steps without
success.

(a) default s0, ALL episodes: G0 and G from the first held-and-within-30mm
    step, plain and bootstrapped.
(b) placed episodes of default s3 and squash s3: same four measures.

Reads section4_followup_ppo_seed0.json, section4_followup_ppo_seed3.json,
section4_followup_ppotanh_seed3.json from --trace-dir. Writes only --out.
"""
import argparse, json
from pathlib import Path

import numpy as np

GAMMA = 0.99
T = 300

p = argparse.ArgumentParser()
p.add_argument("--trace-dir", type=Path, default=Path.home() / "robochess_archive" / "d29_eval_traces")
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()


def load(run):
    with open(a.trace_dir / f"section4_followup_{run}.json") as f:
        return json.load(f)


def G_from(rewards, k):
    x = np.array(rewards[k:])
    return float(np.sum(x * GAMMA ** np.arange(len(x))))


def boot(rewards, k, success, steps):
    g = G_from(rewards, k)
    if not success and steps >= T:
        g += GAMMA ** (steps - k) * rewards[-1] / (1 - GAMMA)
    return g


def first_held30(rows):
    for i, r in enumerate(rows):
        if r["held"] and r["d_xy"] < 0.03:
            return i
    return None


results = {}

# (a) default s0, ALL episodes
d = load("ppo_seed0")
g0s, gh30s, b0s, bh30s = [], [], [], []
n_no_held30 = 0
for ep in d["episodes"]:
    rows = ep["rows"]
    rewards = [r["reward"] for r in rows]
    success = rows[-1]["is_success"]
    steps = len(rows)
    k = first_held30(rows)
    g0 = G_from(rewards, 0)
    b0 = boot(rewards, 0, success, steps)
    g0s.append(g0)
    b0s.append(b0)
    if k is not None:
        gh30s.append(G_from(rewards, k))
        bh30s.append(boot(rewards, k, success, steps))
    else:
        n_no_held30 += 1
results["ppo_seed0_all"] = dict(
    n=len(d["episodes"]), n_no_held30=n_no_held30,
    G0_mean=float(np.mean(g0s)), G0_median=float(np.median(g0s)),
    Gheld30_mean=float(np.mean(gh30s)) if gh30s else None, Gheld30_median=float(np.median(gh30s)) if gh30s else None,
    n_held30=len(gh30s),
    boot0_mean=float(np.mean(b0s)), boot0_median=float(np.median(b0s)),
    bootheld30_mean=float(np.mean(bh30s)) if bh30s else None, bootheld30_median=float(np.median(bh30s)) if bh30s else None,
)

# (b) placed episodes of default s3, squash s3
for run, label in [("ppo_seed3", "ppo_seed3_placed"), ("ppotanh_seed3", "ppotanh_seed3_placed")]:
    d = load(run)
    g0s, gh30s, b0s, bh30s = [], [], [], []
    n_no_held30 = 0
    n_placed = 0
    for ep in d["episodes"]:
        rows = ep["rows"]
        if not rows[-1]["is_success"]:
            continue
        n_placed += 1
        rewards = [r["reward"] for r in rows]
        success = True
        steps = len(rows)
        k = first_held30(rows)
        g0 = G_from(rewards, 0)
        b0 = boot(rewards, 0, success, steps)
        g0s.append(g0)
        b0s.append(b0)
        if k is not None:
            gh30s.append(G_from(rewards, k))
            bh30s.append(boot(rewards, k, success, steps))
        else:
            n_no_held30 += 1
    results[label] = dict(
        n_placed=n_placed, n_no_held30=n_no_held30,
        G0_mean=float(np.mean(g0s)), G0_median=float(np.median(g0s)),
        Gheld30_mean=float(np.mean(gh30s)) if gh30s else None, Gheld30_median=float(np.median(gh30s)) if gh30s else None,
        n_held30=len(gh30s),
        boot0_mean=float(np.mean(b0s)), boot0_median=float(np.median(b0s)),
        bootheld30_mean=float(np.mean(bh30s)) if bh30s else None, bootheld30_median=float(np.median(bh30s)) if bh30s else None,
    )

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
