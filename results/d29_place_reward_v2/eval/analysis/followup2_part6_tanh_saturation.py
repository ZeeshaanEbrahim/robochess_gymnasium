#!/usr/bin/env python
"""D29 section 4 follow-up 2, item 6 (read-only): tanh-saturation check on the
pre-tanh gripper mean around each episode's last pinched step -- squash s1's
87 S2-ending episodes vs squash s3's placed episodes.

Two windows are reported: the original +-10 offsets (as first reported), and
-10..0 only (pre-release, requested afterward to exclude s3's post-release
negative swing from the "saturated fraction" comparison -- keys suffixed
`_pre_release_only`, additional to the original ones, not a replacement).

Reads section4_followup2_ppotanh_seed1.json / section4_followup2_ppotanh_seed3.json
from --trace-dir. Writes only --out.
"""
import argparse, json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--trace-dir", type=Path, default=Path.home() / "robochess_archive" / "d29_eval_traces")
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()


def load(run):
    with open(a.trace_dir / f"section4_followup2_{run}.json") as f:
        return json.load(f)


THRESH = 0.5 * np.log((1 + np.sqrt(0.95)) / (1 - np.sqrt(0.95)))


def last_pinched_idx(rows):
    idxs = [i for i, r in enumerate(rows) if r["pinched"]]
    return idxs[-1] if idxs else None


def collect_window_pretanh(episodes, offsets):
    vals = []
    for ep in episodes:
        rows = ep["rows"]
        c = last_pinched_idx(rows)
        if c is None:
            continue
        for o in offsets:
            idx = c + o
            if 0 <= idx < len(rows):
                vals.append(rows[idx]["mean_pretanh"][3])
    return np.array(vals)


def summarize(v):
    tanhp = 1 - np.tanh(v) ** 2
    return dict(n=len(v), median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                frac_saturated=float(np.mean(np.abs(v) > THRESH)), frac_negative=float(np.mean(v < 0)),
                median_tanh_prime=float(np.median(tanhp)))


d1 = load("ppotanh_seed1")
s2_end_eps = [ep for ep in d1["episodes"] if ep["rows"][-1]["state"] == "S2"]
d3 = load("ppotanh_seed3")
placed_eps = [ep for ep in d3["episodes"] if ep["rows"][-1]["is_success"]]

v1_full = collect_window_pretanh(s2_end_eps, range(-10, 11))
v3_full = collect_window_pretanh(placed_eps, range(-10, 11))
v1_pre = collect_window_pretanh(s2_end_eps, range(-10, 1))
v3_pre = collect_window_pretanh(placed_eps, range(-10, 1))

out = dict(
    threshold_abs_x=float(THRESH),
    squash_s1=summarize(v1_full),
    squash_s3=summarize(v3_full),
    squash_s1_pre_release_only=summarize(v1_pre),
    squash_s3_pre_release_only=summarize(v3_pre),
)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
