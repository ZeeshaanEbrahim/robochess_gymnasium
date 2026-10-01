# RS29: random-squares read-out

Experiment: PPO with mean squash (`ppo_mean_squash`) and the D29 reward, trained on random start and target squares
(`excluded_squares` [d1, e1], `max_episode_steps` 400, 3,000,000 steps, seeds 0 to 5). Rules were fixed before any
training in `PREREGISTRATION.md` (commit `ffd0313`, unchanged since). Runs were launched at `1c215fe` (clean tree, all six
manifests). All evaluation below ran at `1c215fe`. All numbers come from the outputs listed in the last section.

## 1. Pre-registered result: RS29 FAILS (0 of 6 seeds learn)

Protocol: final model, `scripts/eval_probe.py --pairs` with the 20 pairs of PREREGISTRATION.md in listed order,
`--episodes 300` (15 per pair), stochastic, eval seed 20000, torch-seeded, frozen VecNormalize (obs_rms identical before and
after, all six). Placement = env `is_success`.

| Seed | Placed / 300 | Wilson 95% | Pairs with >= 1 placement | Learns (>= 20% and >= 10/20 pairs) |
|---|---|---|---|---|
| 0 | 0 | [0.00, 1.26]% | 0/20 | no |
| 1 | 0 | [0.00, 1.26]% | 0/20 | no |
| 2 | 0 | [0.00, 1.26]% | 0/20 | no |
| 3 | 0 | [0.00, 1.26]% | 0/20 | no |
| 4 | 0 | [0.00, 1.26]% | 0/20 | no |
| 5 | 0 | [0.00, 1.26]% | 0/20 | no |

Seeds that learn: 0 of 6. The pass rule needs >= 4 of 6, so RS29 fails. Per the pre-registration, a fail is followed by a
diagnosis-only prompt before any change. Training was complete and clean for all six runs (3,000,320 steps, exit 0, no
errors, 30 checkpoints each). Whole-run `success` terminations in training: 0, 0, 0, 0, 0 and 3 (seed 5).

## 2. Pre-registered diagnostics (reported, not gated)

- **3a. Placement per pair, per band, and by distance.** Every cell is 0/15 for every seed. Per band (short n=90, mid 60,
  long 90, diagonal 60 per seed) and for pairs with start-to-target distance < 0.4 m (8 pairs, n=120) vs >= 0.4 m
  (12 pairs, n=180): 0 placements everywhere, Wilson upper bounds 4.1%, 6.0%, 4.1%, 6.0%, 3.1% and 2.1%. Held episodes by
  seed (D27 held, out of 300) were 35, 108, 1, 83, 30 and 135 in total. Full per-pair table in `eval/rs29_analysis_output.txt`.
- **3b. Funnel per seed (of 300).** grasp (D18) / held (D27) / lifted 30 mm / carried within 3 cm / released / placed:
  seed 0: 0 / 35 / 0 / 0 / 0 / 0; seed 1: 5 / 108 / 0 / 0 / 2 / 0; seed 2: 1 / 1 / 0 / 0 / 0 / 0; seed 3: 6 / 83 / 0 / 0 / 0 / 0;
  seed 4: 1 / 30 / 0 / 0 / 0 / 0; seed 5: 43 / 135 / 3 / 0 / 3 / 0. Computed with `scripts/place_eval.py` through
  `scripts/eval_probe.py`. **Do not use `lifted_30mm` and `carried_within_3cm` as funnel stages: they read 0/300 for the D29
  control that places 99.3% (section 3), so they cannot separate a working policy from a non-working one.**
- **3c. Failure classification.** The D29 taxonomy has no numeric definitions; the criteria here are this read-out's own:
  "never grasps" = held in fewer than 5 of 300 episodes; "grasps but never carries" = held in at least 5, zero carried
  within 3 cm, and median held-phase displacement under 50% of the start-to-target distance. Seed 2: never grasps (held in
  1/300). Seeds 0, 1, 3, 4, 5: grasps but never carries (held in 35, 108, 83, 30, 135; median held displacement / distance
  0.1%, 7.2%, 6.5%, 2.4%, 22.6%; median max lift while held 0.2, 2.7, 1.9, 3.0, 7.7 mm). No seed fits "carries but never
  releases" or "holds high, never descends".
- **3d. Truncated-in-S2 and shaping.** Truncated-in-S2 is 0/300 for all six seeds (tripwire at 10% not tripped). Median
  shaping total per episode is 81.93, 86.23, 84.01, 55.76, 90.96, 65.87 for seeds 0 to 5, each within about 0.7 of the
  idle baseline 0.5 x d x steps (median difference -0.00, -0.00, -0.00, -0.64, -0.00, -0.30).
- **3e. Training-log placement by band, last 10% of episodes.** 0 successes in every seed. The band buckets were chosen
  by this read-out for training pairs (all 3782 ordered pairs): short <= 110 mm, mid 111 to 499, long 500 to 599,
  diagonal >= 600. Details in `eval/rs29_monitor_analysis_output.txt`.
- **3f. e4->e5 anchor.** 0/15 placed for all six seeds, against D29 squash 92.3 / 0 / 96.0 / 99.3 / 97.3 (seeds 0 to 4).
  Confounded by the 400 vs 300 step budget and by multi-pair training.
- **3g. Checkpoint curve (same protocol).** Placed /300 at 1M, 2M, 3M: seeds 0 to 4: 0, 0, 0 at all three; seed 5: 4
  (1.33%, 1 pair), 0, 0.

## 3. Exploratory items (NOT pre-registered)

- **Positive control (j).** D29 squash seed 3 final model (`_probe_cart_place29_ppotanh_seed3`, 300-step budget from its
  own manifest), `--pairs e4:e5 --episodes 300`, seed 20000, at `1c215fe`: **298/300 = 99.33%** placed (Wilson
  [97.60, 99.82]), matching D29's 99.3% at `8cf881b`. Funnel: held 300, **lifted_30mm 0, carried_within_3cm 0**, released
  298, placed 298. These two stages read 0 for a policy that places 99.3% of episodes, so they are not usable as funnel
  stages (see section 2, 3b).
- **Ten time chunks (h).** Per seed, from the Monitor CSV, ever-held rate, end `place_state`, `piece_off_board`,
  `bad_contact` and success count per tenth of the episodes. For example seed 1's ever-held rises from 11.5% to 89.2%
  (chunk 4) and falls to 27.8% (chunk 10); seed 3 ends at 44.4% with `bad_contact` rising to 10.8%; seed 4 peaks at 65.7%
  (chunk 4) and ends at 11.9%; seed 5 ends at 61.9%. Full tables in `eval/rs29_monitor_analysis_output.txt`.
- **`piece_off_board` by edge square (i), last 50% of episodes.** Start-square edge vs non-edge rates: 18.6 vs 10.3%,
  23.9 vs 6.8%, 12.2 vs 4.8%, 37.4 vs 16.2%, 14.1 vs 6.6%, 33.9 vs 18.8% (seeds 0 to 5); by target square the rates are
  about equal (e.g. 13.8 vs 13.7%). 26 of the 62 allowed squares are edge squares (41.9%).
- **Held-but-not-lifted episodes (k).** Per-step rerun of the same 300 evaluation episodes per seed (`eval/rs29_trace.py`,
  checked to reproduce the eval exactly). Held-but-not-lifted episodes: 35, 108, 1, 83, 30, 132 for seeds 0 to 5. Median
  first-held step 266, 184, 391, 127, 210, 94; median steps held 2, 6, 2, 3, 4, 30; median per-episode gripper command
  while held 0.917, 0.843, 0.892, 0.935, 0.868, 0.958 (0 open, 1 closed). How a hold ended is classified by the next-step
  gripper command (cut-off 0.5, this read-out's choice): "slipped" (command >= 0.5) vs "gripper opening" (< 0.5). Per-episode
  rows in `eval/k_held_not_lifted_episodes.csv`. `piece_off_board` episodes (15, 61, 23, 73, 17, 98): ever held before
  leaving 0, 3, 0, 18, 4, 32; final-step phase was approach (never held) in 15, 58, 23, 55, 13, 66 of them.
- **Zero-shot generalisation of the D29 squash models.** D29 squash final models `_probe_cart_place29_ppotanh_seed{0..4}`
  (trained on e4->e5 only, 300-step budget from their manifests) on the 20 RS29 pairs, `eval_probe.py --pairs`,
  300 episodes, seed 20000, stochastic. Seeds 0, 2, 3, 4: 15/300 placed (5.0%, Wilson [3.1, 8.1]), all 15 on e4:e5 and
  nothing on any other pair (pairs with a placement 1/20; short band 15/90 = 16.7%, mid, long and diagonal 0); ever held
  15/300 each, median steps held 8, 9, 7, 12. Seed 1: 0/300 placed, held in 8/300 (median 1 step). Every held episode of
  every seed was on a short-band pair. Per-seed detail in `eval/rs29_zeroshot_analysis_output.txt`.

## 4. Files and provenance

| Item | Files in `eval/` | Produced by |
|---|---|---|
| Pre-registered evaluation | `final_seed{0..5}.json/.log` | `scripts/eval_probe.py _probe_cart_randsq29_ppotanh_seed$i --episodes 300 --seed 20000 --pairs <20 pairs>` |
| Checkpoint curve | `ck{1000000,2000000}_seed{0..5}.json/.log` | same with `--checkpoint N` |
| Control | `control_d29_seed3_e4e5.json` | `eval_probe.py _probe_cart_place29_ppotanh_seed3 --episodes 300 --seed 20000 --pairs e4:e5` |
| Zero-shot | `zeroshot_d29_seed{0..4}.json/.log`, `trace_zeroshot_d29_seed*.log` | `eval_probe.py _probe_cart_place29_ppotanh_seed$i ... --pairs <20 pairs>`; `rs29_trace.py` |
| Diagnostics 3a to 3d, k | `rs29_analysis.py`, `rs29_analysis_output.txt`, `k_held_not_lifted_episodes.csv`, `trace_final_seed*.log` | `rs29_trace.py` then `rs29_analysis.py` |
| 3e, h, i | `rs29_monitor_analysis.py`, `rs29_monitor_analysis_output.txt` | reads `runs/*/logs/monitor.monitor.csv` |
| Zero-shot analysis | `rs29_zeroshot_analysis.py`, `rs29_zeroshot_analysis_output.txt` | reads zero-shot eval and trace |

Per-step traces (11 files) are archived under `~/robochess_archive/rs29_eval_traces/` with SHA256SUMS and regeneration
commands, and `.gitignore` blocks committing them (`results/**/trace_*.json.gz`). The two `rs29_*analysis.py` scripts
that need them read them from `eval/`. No DECISIONS.md entry has been written yet.
