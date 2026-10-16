# SAC obs-norm T1 cell, seeds 1 to 3: result (written 16 Oct 2026)

Evaluation of the final models of seeds 1, 2 and 3 under `results/sac_obsnorm_seed0/PREREGISTRATION.md` at `e0d8981`
(the 5 Oct superseding rule and the "Seeds 1 to 3, 16 Oct" section). The prereg wins over anything written here.
Seed 0 was evaluated earlier (0/300, `results/sac_obsnorm_seed0/RESULT.md` at `d6b06e0`).

## Protocol (as in the prereg; no threshold changed)
`scripts/eval_probe.py <run> --episodes 300 --seed 20000 --out <file>`: final model, 300 episodes, stochastic, eval
seed 20000, torch-seeded, frozen VecNormalize statistics (`final_vecnormalize.pkl`), the same command and arguments that
produced seed 0's official JSON. Per-seed gate: placement (env `is_success` at the last step) of 60/300 (20%) or more.
Wilson 95% interval with z = 1.96, no continuity correction.

Training runs: configs at `d33ebbc`, seeds 1 to 3, 1,000,000 steps, all `EXIT=0`. The run directories were compared
against the 16 Oct archive with `tar --compare` before evaluating: 0 differences.

## Result
| Seed | Placed | Wilson 95% | `terminated_reason` | Verdict (gate 60/300) |
|---|---|---|---|---|
| 0 (d6b06e0) | 0/300 | [0.00%, 1.26%] | time_limit 300 | FAIL |
| 1 | 0/300 | [0.00%, 1.26%] | time_limit 300 | FAIL |
| 2 | 0/300 | [0.00%, 1.26%] | time_limit 300 | FAIL |
| 3 | 0/300 | [0.00%, 1.26%] | time_limit 300 | FAIL |

**Cell: 0 of 4 seeds pass the per-seed gate (N = 4).** The prereg defines no cell-level threshold beyond "k seeds
passing out of N", so none is applied here. `obs_rms_identical_before_after` is True for all four.

Other fields from the same eval output (descriptive, no effect on the verdict):

| Seed | grasp_ever | held | lifted >= 30 mm | carried within 3 cm | reached S2 |
|---|---|---|---|---|---|
| 0 | 300/300 | 300/300 | 300/300 | 300/300 | 0/300 |
| 1 | 300/300 | 300/300 | 300/300 | 0/300 | 0/300 |
| 2 | 0/300 | 300/300 | 0/300 | 0/300 | 0/300 |
| 3 | 300/300 | 300/300 | 300/300 | 300/300 | 0/300 |

Seed 2 reports `held` 300/300 but `grasp_ever` 0/300 and `lifted_30mm` 0/300. This is recorded as observed and not
investigated here. `held` is the D27 condition (both pads on the piece, no piece-board contact); `grasp_ever` is the
env's `grasp_success` key.

## Not done
- No 500k-checkpoint evaluation, no training-log trend report and no per-step diagnostic for seeds 1 to 3. The prereg's
  fail branch asks for the `critic_loss` / `actor_loss` trends and placement at 500k vs 1M before the next step is
  chosen; that is left for a separate instruction.
- No seed 4.

## Files
`eval/seed{1,2,3}_final_300.json` (eval_probe output), `eval/seed{1,2,3}_final_300.stdout.txt`, `eval/summarize.py` (the
arithmetic above, reads the JSON only) and `eval/summary.txt` (its output; the seed 0 row was computed from the local,
gitignored `runs/_eval_sac_obsnorm_seed0/final_300.json`).
