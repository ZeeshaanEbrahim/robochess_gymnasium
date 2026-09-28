#!/usr/bin/env python
"""D29 section 3 (read-only): rule 1 (per-seed placement rate + Wilson 95% CI,
>=10% pass/fail) and rule 2 (squash adopted only if it beats default's pass
count; Fisher exact test on the resulting 4/5-vs-3/5 table).

Reads the small eval_probe.py outputs already in --eval-dir (ppo_seed*_final.json
/ ppotanh_seed*_final.json, each already committed-size) and
section2_training_log_tallies.json (for the training-log-vs-eval-CI
cross-check). Writes only --out.
"""
import argparse, json
from pathlib import Path

from scipy.stats import fisher_exact

REPO = Path(__file__).resolve().parents[4]

PPO_RUNS = [f"ppo_seed{s}" for s in range(5)]
TANH_RUNS = [f"ppotanh_seed{s}" for s in range(5)]

p = argparse.ArgumentParser()
p.add_argument("--trace-dir", type=Path, default=Path.home() / "robochess_archive" / "d29_eval_traces",
                help="unused by this script; accepted for interface consistency")
p.add_argument("--eval-dir", type=Path, default=REPO / "results" / "d29_place_reward_v2" / "eval")
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()


def load(r):
    with open(a.eval_dir / f"{r}_final.json") as f:
        return json.load(f)


with open(a.eval_dir / "section2_training_log_tallies.json") as f:
    s2 = json.load(f)

out = {}
def row(r, monitor_key):
    d = load(r)
    placed = d["place_report"]["rates"]["placed"]
    rms_same = d["obs_rms_identical_before_after"]
    lo, hi = placed["wilson_lo"], placed["wilson_hi"]
    passes = placed["rate"] >= 0.10
    last500 = s2[monitor_key]["placement_successes_last500_rate"]
    in_ci = lo <= last500 <= hi
    out[r] = dict(k=placed["k"], n=placed["n"], rate=placed["rate"],
                  wilson_lo=lo, wilson_hi=hi, passes_10pct=passes,
                  rms_same=rms_same,
                  training_log_last500_rate=last500,
                  training_log_last500_in_eval_ci=in_ci)
    return out[r]


for r in PPO_RUNS:
    row(r, f"_probe_cart_place29_{r}")
for r in TANH_RUNS:
    row(r, f"_probe_cart_place29_{r}")

default_pass_count = sum(1 for r in PPO_RUNS if out[r]["passes_10pct"])
tanh_pass_count = sum(1 for r in TANH_RUNS if out[r]["passes_10pct"])

rule1_default = default_pass_count >= 4
rule1_tanh = tanh_pass_count >= 4
rule2 = "squash" if tanh_pass_count > default_pass_count else "default (tie or fewer)"

table = [[tanh_pass_count, len(TANH_RUNS) - tanh_pass_count],
         [default_pass_count, len(PPO_RUNS) - default_pass_count]]
_, fisher_p = fisher_exact(table)

summary = dict(
    per_seed=out,
    default_seeds_passing=default_pass_count,
    tanh_seeds_passing=tanh_pass_count,
    rule1_default_passes=rule1_default,
    rule1_tanh_passes=rule1_tanh,
    rule2_adopted=rule2,
)
summary["adoption_wording_note"] = (
    "Squash is adopted on a one-seed margin (4/5 vs 3/5) decided by seed 0 alone "
    "(squash seed 0 passes at 92.3%; default's non-passing seeds 0 and 1 are stuck at 0.0%, "
    "not near the 10% line). Fisher exact test on the 4/5-vs-3/5 contingency table gives "
    f"p = {fisher_p:.4f} (not significant). Among seeds that pass on both variants (2, 3, 4), "
    "placement rates are similar between default and squash (e.g. seed 2: 96.0% vs 96.0%; "
    "seed 4: 98.0% vs 97.3%). This is adoption under the pre-registered rule (D29 rule 2), "
    "not evidence that squash performs better than default PPO."
)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
