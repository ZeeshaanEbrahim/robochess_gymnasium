# D29 place-reward-v2 evaluation (results/d29_place_reward_v2/eval/)

Rule 1/rule 2 read-out and diagnostics for D29 (branch `place-reward-v2` @
`41ae8e542c60a17802f92d668a872896bf6b7026`). All commands below run from the repo root with
`source ~/venvs/robochess/bin/activate` first. Every number in `section6_summary.md` traces to one of
the scripts below; none were hand-edited into a JSON file.

## Raw per-run eval outputs (produced by the tracked `scripts/eval_probe.py`)

```bash
python3 scripts/eval_probe.py _probe_cart_place29_<run> --episodes 300 --seed 20000 \
  --out results/d29_place_reward_v2/eval/<run>_final.json
```
for `<run>` in `ppo_seed{0..4}`, `ppotanh_seed{0..4}`, `sac_seed0`. Stochastic (no `--deterministic`),
torch-seeded, frozen VecNormalize for PPO (SAC carries none). Produces `<run>_final.json` -- per-episode
`place_report` (Wilson-CI rates, gate diagnostics, push-assisted, truncated-in-S2).

## Per-run diagnostics (`section4_full_diag.py`, in this directory)

```bash
python3 results/d29_place_reward_v2/eval/section4_full_diag.py _probe_cart_place29_<run> --episodes 300 \
  --out results/d29_place_reward_v2/eval/section4_<run>.json
```
State fractions, S1 medians, saturation (PPO only), gripper-command-<0.9-while-held, toppled-hold@step150,
shaping-total-per-episode-median. Also used, pointed at the D28 run directly, for the SAC toppled-hold
baseline: `python3 results/d29_place_reward_v2/eval/section4_full_diag.py _probe_cart_place_sac_seed0
--episodes 300 --out results/d29_place_reward_v2/eval/section5_d28_sac_seed0_toppled.json --runs-dir runs`.

## Full per-step traces (archived, NOT committed -- see `.gitignore`)

`section4_followup_diag.py` and `section4_followup2_diag.py` (both in this directory) produce full
300-episode per-step traces (13-64 MB each), too large to commit. They live at
`~/robochess_archive/d29_eval_traces/`, with their own `README.md` (exact regeneration commands) and
`SHA256SUMS.txt`. The `analysis/` scripts below read them from there by default (`--trace-dir`).

`section4_followup3_pad_height.py` (in this directory, 150 episodes, small aggregate output only, no
trace file) measures median pad-contact height while held for the tip-grip-lead check.

## `analysis/` -- scripts that turn the above into every number section 6 cites

Each takes `--trace-dir` (default `~/robochess_archive/d29_eval_traces/`) and `--out`; several also take
`--eval-dir` (default this directory) for the small already-committed inputs they read. All 10 were
verified by re-running and diffing against the JSON already in this directory: every pre-existing number
matched exactly (max abs diff 0.0); `followup2_part6_tanh_saturation.py` additionally writes 14 new
`_pre_release_only` keys (the offset -10..0 recompute) that were not in the file before.

| script | reads | writes | section 6 numbers it backs |
|---|---|---|---|
| `section1_state_check.py` | `runs/*/manifest.json`, `logs/progress.csv`, (SAC) the saved model + Monitor CSV | `section1_state_check.json` | -- (state check, not directly cited in section 6) |
| `section2_training_log_tallies.py` | `runs/*/logs/monitor.monitor.csv` | `section2_training_log_tallies.json` | training-log cross-check in section 3's per-seed table |
| `section3_rule1_evaluation.py` | `<run>_final.json`, `section2_training_log_tallies.json` | `section3_rule1_evaluation.json` | (a) rule 1 table, (b) rule 2 / Fisher p=1.0 wording |
| `section4_summary.py` | `<run>_final.json`, `section4_<run>.json`, `section4_sac_seed0.json` | `section4_summary.json` | (c) SAC place/toppled-hold row; (d) failure-taxonomy state fractions |
| `section4_terminated_reason_breakdown.py` | `<run>_final.json` | `section4_terminated_reason_breakdown.json` | (background for the taxonomy's failure-mode descriptions) |
| `followup_part2_reward_ordering.py` | trace: `section4_followup_ppo_seed{0,3}.json`, `section4_followup_ppotanh_seed3.json` | `section4_followup_part2_reward_ordering.json` | (d) default s0 "local optimum" G0/G-held-30mm numbers (234 vs 555 etc.) |
| `followup_part3_action_stats.py` | trace: `section4_followup_{ppo,ppotanh}_seed{0,1,3}.json` | `section4_followup_part3_action_stats.json` | (background for the gripper-saturation discussion, not directly in section 6) |
| `followup_part4_squash_s1_s2end.py` | trace: `section4_followup_ppotanh_seed1.json` | `section4_followup_part4_squash_s1_s2end.json` | (d) squash s1 "places correctly but never opens", 87/87 `gripper_released` False |
| `followup2_part5_release_window.py` | trace: `section4_followup2_{ppo_seed0,ppo_seed3,ppotanh_seed1,ppotanh_seed3}.json` | `section4_followup2_part5_release_window.json`, `..._extra.json` | (background for the release-window analysis; item 4's pad-height context) |
| `followup2_part6_tanh_saturation.py` | trace: `section4_followup2_ppotanh_seed{1,3}.json` | `section4_followup2_part6_tanh_saturation.json` | (open items) tip-grip / tanh-saturation lead, tested and not confirmed |

## `section4_followup3_pad_height.py` (this directory, not in `analysis/` -- no trace file to read)

```bash
python3 results/d29_place_reward_v2/eval/section4_followup3_pad_height.py _probe_cart_place29_<run> \
  --episodes 150 --out results/d29_place_reward_v2/eval/section4_followup3_<run>.json
```
for `<run>` in `ppo_seed3`, `ppotanh_seed0`, `ppotanh_seed1`, `ppotanh_seed3`. Backs open item (i)/(ii)'s
pad-contact-height numbers (squash s1 31.4mm, squash s0 57.9mm, squash s3 30.5mm, default s3 29.4mm).

## `section6_summary.md`

Hand-written, drawing only on the JSON files above (and `DECISIONS.md`'s D29 entry for the pre-registered
rule text). No number in it was computed outside of the scripts listed here.
