#!/usr/bin/env python
"""D29 section 4 (read-only): combine each PPO run's place_report (from
eval_probe.py's *_final.json) with its diagnostics (state fractions,
saturation, gripper command, toppled-hold, shaping total, S1 medians,
first-held step -- from section4_full_diag.py's per-run output) into one
summary dict; SAC gets the diagnostics subset only (no place_report merge
needed beyond what's already reported elsewhere).

Reads the small eval_probe.py and section4_full_diag.py outputs already in
--eval-dir. Writes only --out.
"""
import argparse, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PPO_RUNS = [f"ppo_seed{s}" for s in range(5)]
TANH_RUNS = [f"ppotanh_seed{s}" for s in range(5)]

p = argparse.ArgumentParser()
p.add_argument("--trace-dir", type=Path, default=Path.home() / "robochess_archive" / "d29_eval_traces",
                help="unused by this script; accepted for interface consistency")
p.add_argument("--eval-dir", type=Path, default=REPO / "results" / "d29_place_reward_v2" / "eval")
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

runs = PPO_RUNS + TANH_RUNS
summary = {}
for r in runs:
    with open(a.eval_dir / f"{r}_final.json") as f:
        d3 = json.load(f)
    with open(a.eval_dir / f"section4_{r}.json") as f:
        d4 = json.load(f)
    pr = d3["place_report"]
    summary[r] = dict(
        placed_rate=pr["rates"]["placed"]["rate"],
        truncated_in_s2=pr["truncated_in_s2"],
        push_assisted=pr["push_assisted"]["fraction"],
        gate_fraction_lt1=pr["gate"]["fraction_gate_lt1"],
        gate_max_s1_tilt=pr["gate"]["max_s1_tilt_deg"],
        state_fraction=d4["state_fraction"],
        saturation=d4["saturation"],
        gripper_cmd_lt_0p9_while_held=d4["gripper_cmd_lt_0p9_while_held"],
        toppled_hold_step150=d4["toppled_hold_step150"],
        shaping_total_per_episode_median=d4["shaping_total_per_episode_median"],
        s1=d4["s1"],
        first_held=d4["first_held"],
    )

with open(a.eval_dir / "section4_sac_seed0.json") as f:
    sac = json.load(f)
summary["sac_seed0"] = dict(
    state_fraction=sac["state_fraction"], saturation=sac["saturation"],
    gripper_cmd_lt_0p9_while_held=sac["gripper_cmd_lt_0p9_while_held"],
    toppled_hold_step150=sac["toppled_hold_step150"],
    shaping_total_per_episode_median=sac["shaping_total_per_episode_median"],
    s1=sac["s1"], first_held=sac["first_held"],
)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
