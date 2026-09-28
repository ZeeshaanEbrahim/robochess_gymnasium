#!/usr/bin/env python
"""D29 section 4 follow-up 2, item 5 (read-only): release windows, 10 steps
before/after a reference step, median action[3]/mean[3]/std[3]/mean_pretanh[3]/
qpos per offset.

- default s3, squash s3: placed episodes, window around each episode's last
  pinched step.
- squash s1: its 87 S2-ending episodes, window around the last pinched step,
  plus its gripper action/qpos on its own S2 steps.
- default s0: window around every S1->S3 transition event (pooled across all
  300 episodes).

Reads section4_followup2_{run}.json from --trace-dir. Writes only --out
(--out for the main windows, --out-extra for squash s1 / default s0, matching
this repo's two existing files).
"""
import argparse, json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--trace-dir", type=Path, default=Path.home() / "robochess_archive" / "d29_eval_traces")
p.add_argument("--out", type=Path, required=True, help="main window output (default s3 / squash s3)")
p.add_argument("--out-extra", type=Path, required=True, help="squash s1 / default s0 output")
a = p.parse_args()


def load(run):
    with open(a.trace_dir / f"section4_followup2_{run}.json") as f:
        return json.load(f)


def last_pinched_idx(rows):
    idxs = [i for i, r in enumerate(rows) if r["pinched"]]
    return idxs[-1] if idxs else None


def window_stats(episodes, center_fn, offsets=range(-10, 11)):
    by_offset = {o: {"action3": [], "mean3": [], "std3": [], "mean_pretanh3": [], "qpos": []} for o in offsets}
    n_events = 0
    for ep in episodes:
        rows = ep["rows"]
        c = center_fn(rows)
        if c is None:
            continue
        n_events += 1
        for o in offsets:
            idx = c + o
            if 0 <= idx < len(rows):
                r = rows[idx]
                by_offset[o]["action3"].append(r["action"][3])
                by_offset[o]["mean3"].append(r["mean"][3])
                by_offset[o]["std3"].append(r["std"][3])
                by_offset[o]["mean_pretanh3"].append(r["mean_pretanh"][3])
                by_offset[o]["qpos"].append(r["qpos"])
    out = {}
    for o in offsets:
        b = by_offset[o]
        if b["action3"]:
            out[o] = dict(n=len(b["action3"]), median_action3=float(np.median(b["action3"])),
                           median_mean3=float(np.median(b["mean3"])), median_std3=float(np.median(b["std3"])),
                           median_mean_pretanh3=float(np.median(b["mean_pretanh3"])),
                           median_qpos=float(np.median(b["qpos"])))
        else:
            out[o] = None
    return n_events, out


# --- main: default s3, squash s3, placed episodes ---
main_results = {}
for run in ["ppo_seed3", "ppotanh_seed3"]:
    d = load(run)
    placed_eps = [ep for ep in d["episodes"] if ep["rows"][-1]["is_success"]]
    n_ev, out = window_stats(placed_eps, lambda rows: last_pinched_idx(rows))
    main_results[f"{run}_placed_release_window"] = dict(n_events=n_ev, window=out)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(main_results, indent=2))

# --- extra: squash s1 (87 S2-ending episodes), default s0 (S1->S3 transitions) ---
extra = {}

d1 = load("ppotanh_seed1")
s2_end_eps = [ep for ep in d1["episodes"] if ep["rows"][-1]["state"] == "S2"]
n_ev, out = window_stats(s2_end_eps, lambda rows: last_pinched_idx(rows))
extra["ppotanh_seed1_s2end_release_window"] = dict(n_events=n_ev, window=out)

s2_rows = [r for ep in s2_end_eps for r in ep["rows"] if r["state"] == "S2"]
extra["ppotanh_seed1_s2_steps"] = dict(
    n=len(s2_rows),
    action3_median=float(np.median([r["action"][3] for r in s2_rows])),
    action3_min=float(np.min([r["action"][3] for r in s2_rows])), action3_max=float(np.max([r["action"][3] for r in s2_rows])),
    qpos_median=float(np.median([r["qpos"] for r in s2_rows])),
    qpos_min=float(np.min([r["qpos"] for r in s2_rows])), qpos_max=float(np.max([r["qpos"] for r in s2_rows])),
)

d0 = load("ppo_seed0")
def s1_to_s3_transitions(rows):
    return [i for i in range(1, len(rows)) if rows[i - 1]["state"] == "S1" and rows[i]["state"] == "S3"]

by_offset = {o: {"action3": [], "mean3": [], "std3": [], "qpos": []} for o in range(-10, 11)}
n_transitions = 0
for ep in d0["episodes"]:
    rows = ep["rows"]
    for t in s1_to_s3_transitions(rows):
        n_transitions += 1
        for o in range(-10, 11):
            idx = t + o
            if 0 <= idx < len(rows):
                r = rows[idx]
                by_offset[o]["action3"].append(r["action"][3])
                by_offset[o]["mean3"].append(r["mean"][3])
                by_offset[o]["std3"].append(r["std"][3])
                by_offset[o]["qpos"].append(r["qpos"])
out0 = {}
for o in range(-10, 11):
    b = by_offset[o]
    if b["action3"]:
        out0[o] = dict(n=len(b["action3"]), median_action3=float(np.median(b["action3"])),
                        median_mean3=float(np.median(b["mean3"])), median_std3=float(np.median(b["std3"])),
                        median_qpos=float(np.median(b["qpos"])))
extra["ppo_seed0_s1_to_s3_transition_window"] = dict(n_events=n_transitions, window=out0)

a.out_extra.parent.mkdir(parents=True, exist_ok=True)
a.out_extra.write_text(json.dumps(extra, indent=2))
print("main:", json.dumps(main_results)[:200])
print("extra:", json.dumps(extra)[:200])
