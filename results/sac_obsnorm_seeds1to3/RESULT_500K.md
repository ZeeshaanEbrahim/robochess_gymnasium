# SAC obs-norm T1 cell: fail-branch items, 500k vs 1M and training-log losses (written 16 Oct 2026)

The cell is 0 of 4 seeds passing (seeds 0 to 3, `RESULT.md` in this directory and `results/sac_obsnorm_seed0/RESULT.md`).
This file reports the two items the prereg (`results/sac_obsnorm_seed0/PREREGISTRATION.md` at `e0d8981`, "Fail" bullet of the
5 Oct superseding rule) requires before a next step is chosen: the `critic_loss` / `actor_loss` trends from the training
log (minus `actor_loss` as a rough Q proxy) and placement at 500k vs 1M. No threshold is changed and no next step is chosen
here; that is a separate dated decision.

## Placement, 500k checkpoint vs final model (1M)
`scripts/eval_probe.py <run> --checkpoint 500000 --episodes 300 --seed 20000` for the 500k column (the checkpoint's own
`rc_vecnormalize_500000_steps.pkl`, frozen; `obs_rms_identical_before_after` True for all), the final-model JSONs for the 1M
column. 300 stochastic episodes, eval seed 20000. Wilson 95%, z = 1.96, no continuity correction.

| Seed | 500k placed | 500k `terminated_reason` | 1M placed | 1M `terminated_reason` |
|---|---|---|---|---|
| 0 | 0/300 [0.00%, 1.26%] | time_limit 300 | 0/300 [0.00%, 1.26%] | time_limit 300 |
| 1 | 0/300 [0.00%, 1.26%] | time_limit 296, piece_off_board 4 | 0/300 [0.00%, 1.26%] | time_limit 300 |
| 2 | 0/300 [0.00%, 1.26%] | time_limit 299, bad_contact 1 | 0/300 [0.00%, 1.26%] | time_limit 300 |
| 3 | 0/300 [0.00%, 1.26%] | time_limit 300 | 0/300 [0.00%, 1.26%] | time_limit 300 |

Placement did not rise from 500k to 1M in any seed (0/300 at both).

Other 500k fields (descriptive; seed 0 from the local gitignored `runs/_eval_sac_obsnorm_seed0/ck500k_300.json`):

| Seed | grasp_ever | held | lifted >= 30 mm | carried within 3 cm | reached S2 |
|---|---|---|---|---|---|
| 0 | 300/300 | 300 | 300 | 297 | 0 |
| 1 | 300/300 | 300 | 300 | 0 | 0 |
| 2 | 0/300 | 300 | 0 | 0 | 0 |
| 3 | 299/300 | 299 | 298 | 296 | 0 |

## Training-log losses
Mean over a +-5k-step window of the logged rows (the 1M window is one-sided: the logs end at 999,863 / 999,171 / 999,208 /
999,182, so it holds 5 / 4 / 4 / 4 rows). Same method as `results/sac_obsnorm_seed0/RESULT.md`, whose seed 0 values it reproduces.

| Seed | Step | rows | critic_loss | actor_loss | -actor_loss | ent_coef | ent_coef_loss |
|---|---|---|---|---|---|---|---|
| 0 | 100k | 8 | 38.88 | -177.25 | 177.25 | 0.0906 | 0.1332 |
| 0 | 500k | 8 | 17.57 | -455.12 | 455.12 | 0.0853 | -0.1893 |
| 0 | 1M | 5 | 8.02 | -508.47 | 508.47 | 0.0534 | -0.0350 |
| 1 | 100k | 10 | 8.31 | -74.34 | 74.34 | 0.0380 | 0.1797 |
| 1 | 500k | 8 | 23.75 | -294.03 | 294.03 | 0.1247 | 0.0151 |
| 1 | 1M | 4 | 75.11 | -373.37 | 373.37 | 0.0740 | -0.1365 |
| 2 | 100k | 8 | 0.32 | -79.24 | 79.24 | 0.0189 | -0.1984 |
| 2 | 500k | 9 | 0.52 | -164.20 | 164.20 | 0.0327 | 0.0813 |
| 2 | 1M | 4 | 1.22 | -252.88 | 252.88 | 0.0689 | 0.1246 |
| 3 | 100k | 9 | 14.88 | -113.29 | 113.29 | 0.0579 | 0.1478 |
| 3 | 500k | 9 | 32.10 | -427.28 | 427.28 | 0.1007 | -0.0640 |
| 3 | 1M | 4 | 8.49 | -514.91 | 514.91 | 0.0465 | 0.0488 |

Whole-run `critic_loss` maxima (earlier extraction): seed 0 411.6 at step 200,869; seed 1 245.3 at 633,566; seed 2 15.2 at
267,484; seed 3 339.2 at 198,901.

## Not done
Seed 4, gSDE, and any other diagnostic were not run. Seed 2's `held` / `grasp_ever` pattern was not investigated.

## Files
`eval_500k/seed{1,2,3}_ck500k_300.json`, `eval_500k/seed{1,2,3}_ck500k_300.stdout.txt`, `eval_500k/summarize_500k.py` and its
output `eval_500k/summary_500k.txt`.
