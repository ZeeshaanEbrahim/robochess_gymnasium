# RS32: why seeds 0, 200 and 500 fell short (POST HOC, EXPLORATORY)

**Status.** Everything in this file was done after the results were known, in answer to the question "why did seeds 0, 200 and 500 fail". It is
exploratory. It does not change any rule verdict: under PREREGISTRATION.md rule 1, **0 of 6 seeds learn** and the claim is **not achieved**
(`../README.md`, unchanged). No pre-registered threshold, config or training code was touched. Nothing here was tested with a new training run, so
every mechanism below is a description of what the episodes show, not a demonstrated cause.

**Where the numbers come from.** Every number in this file comes from the scripts in this folder, and the tables in the appendix are their
unedited output (`make_diagnosis.py` pastes them in). Prose numbers are copied from those tables; the table letter is given in brackets.

| script | reads | output |
|---|---|---|
| `diag_eval_tables.py` | the committed eval JSONs (`../*_final.json`, `../zs_*`, `../det48_*`) | `diag_eval_tables_output.txt` (tables A to H) |
| `diag_training_tables.py` | `runs/` Monitor CSVs and checkpoint zips | `diag_training_tables_output.txt` (tables 1 to 4) |
| `diag_rollout.py` | runs `scripts/eval_probe.py` unmodified, `--episodes 480` (the first 10 episodes of every pair of each seed's eval stream), dumps per-step arrays | `ro_seedN_steps.json` (about 7 MB each, **not committed**) |
| `diag_rollout_tables.py` | those dumps | `diag_rollout_tables_output.txt` (tables R1 to R6) |
| `make_diagnosis.py` | `DIAGNOSIS_template.md` + the three outputs | this file |

Regenerate the dumps (about 8 minutes per wave of 3 seeds) with, from the repo root:
`python results/rs32_48_pairs/eval/diagnosis/diag_rollout.py _probe_cart_rs32D_ppotanh_seed<S> --episodes 480 --pairs <the 48 pairs of PREREGISTRATION.md, comma separated> --seed 20000 --out <dir>/ro_seed<S>.json`,
then `python diag_rollout_tables.py <dir>`.
The scope of this file is the items named in the request; I did not have an earlier message with other items.

## 0. Corrections to the first chat summary

1. The outcome table in my chat reply showed four of seven rows, so its columns did not sum to 480. The missing episodes are the "other" row (table R2),
   which is now explicit and sums to 480 for every seed. "Other" is the number of episodes that did not topple and ended at the time limit in place state S2 (the piece was released on the square, but env `is_success` was not reached): 6, 1, 7, 1, 6, 11 episodes for seeds 0, 100, 200, 300, 400, 500, giving "other" totals of 6, 1, 7, 3, 6, 11. The two further "other" episodes of seed 300 are one `bad_contact` end (state S0) and one `piece_off_board` end without tilt above 45 degrees (state S3). In the chat reply that `piece_off_board` end was counted as "lost at the time limit", so seed 300's "upright, lost" count is 8 here, not 9.
2. Seeds 100, 300 and 400 are **not** one group (section 2). Seed 400 was wrongly folded into "good seeds" on the stuck count in the chat reply.
3. Backward pairs: the chat reply gave "19/20, 16/20, 13/20 time out"; those were time-outs ending in state S3 only. All time-limit ends are 20, 20 and 18 of 20 (table R3). Also, only seeds 0 and 200 leave the piece on the wrong side of its start square (16 of 20 each); seed 500 mostly leaves it near its start (5 of 18 non-placed on the wrong side, median progress 0.10).

## 1. Where the gap is

The mean pooled placement of seeds 100, 300 and 400 is 27.8 points above the mean of seeds 0, 200 and 500 (table B; this grouping is by pooled
result only). By category (table B): forward/lateral neighbour pairs 14.1 points (28 pairs), long forward/lateral 8.6, backward neighbour 3.1 (2 pairs), long backward 2.1 (5 pairs).

Pair counts under a threshold (table C), per seed 0 / 100 / 200 / 300 / 400 / 500:
neighbour pairs below 10/50: 3 / 0 / 2 / 0 / 0 / 4; neighbour pairs below 25/50: 7 / 0 / 5 / 0 / 1 / 11; long pairs below 10/50: 9 / 4 / 15 / 3 / 5 / 6.
Seed 500's weak neighbour pairs are the pairs with a rank 7 or 8 target (c6:c7, c7:c8, d6:d7, d7:d8, e6:e7, e7:e8, f7:f8, e6:d7) plus c4:d4, e5:e4, c6:c5. Seed 0's are d7:d8, e7:e8, f6:f7, f7:f8, c4:d4, e5:e4, c6:c5.
The two backward neighbour pairs (e5:e4, c6:c5) are the sharpest split: 1/100, 89/100, 0/100, 83/100, 57/100, 8/100 placed in the full 2400 games for seeds 0, 100, 200, 300, 400, 500 (table D).

## 2. How episodes end (480-episode re-run)

All six re-runs reproduce the committed eval **episode for episode** (480/480 identical end reason, length, placed flag, start and target square, for every seed, including 100, 300 and 400; table R1).

Exclusive outcomes, sums to 480 per seed (table R2), seeds 0 / 100 / 200 / 300 / 400 / 500:

| outcome | 0 | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|---|
| placed | 218 | 337 | 199 | 348 | 352 | 213 |
| toppled (piece tilt above 45 deg at any step) | 144 | 133 | 181 | 121 | 57 | 131 |
| upright, still pinching at the time limit | 30 | 3 | 21 | 0 | 28 | 53 |
| upright, lost (S3) at the time limit | 82 | 6 | 72 | 8 | 37 | 72 |
| other (see section 0) | 6 | 1 | 7 | 3 | 6 | 11 |
| **sum** | 480 | 480 | 480 | 480 | 480 | 480 |
| upright and stuck (pinching + lost) | 112 | 9 | 93 | 8 | 65 | 125 |

"Toppled" is checked after "placed", so a few placed episodes (2 to 5 per seed, table R5) had tilted above 45 degrees earlier and still count as placed.

What the table shows, and what it does not:
- **Toppling does not separate the seeds.** Seeds 0, 100, 300 and 500 are at 121 to 144 toppled episodes, seed 200 is higher (181), seed 400 is much lower (57).
  Almost no toppled episode ends placed (2 to 5 per seed out of 61 to 186 toppled episodes, table R5). Most first-topple steps happen while the gripper is not pinching (S0 or S3), a median of 2 to 6 steps after the last pinch, with the pinch site a median of 20 to 29 mm from the piece (table R5). The data do not show how the contact tips the piece over.
- **Upright-and-stuck separates seeds 0, 200 and 500 (112, 93, 125) from seeds 100 and 300 (9, 8).** In those episodes the piece ends a median of 49 to 97 mm from the target (table R4, groups of at least 5 episodes) after 11.8 to 30.5 separate grasp phases per episode, against 5.4 to 11.0 in placed episodes (table R4).
- **The three seeds that placed best reached their results by different routes.**
  - Seeds 100 and 300: few stuck episodes (9, 8), about a quarter of episodes toppled (133, 121 of 480, mostly long pairs: 101 and 88 of the toppled episodes are on long pairs, table R5), and they place the backward neighbour pairs (89% and 83%, table D).
  - Seed 400: the **fewest toppled episodes (57)** but **65 upright-and-stuck episodes**, the highest long forward/lateral placement of all (63%, table A), and it places backward neighbour pairs only partly (57%) and long backward pairs almost never (2/250, table D). Its stuck count is closer to seeds 0, 200 and 500 than to seeds 100 and 300.
  So the stuck/topple split describes seeds 0, 200, 500 against seeds 100 and 300 only; seed 400 does not fit it.
- Seed 200 has 37 to 50 more toppled episodes than seeds 0 and 500 (181 vs 144 and 131), mostly on long pairs (88 of its 130 long forward/lateral episodes toppled, table R2b).

## 3. Backward pairs

Placed percentages come from the **full 2400-episode eval** (100 episodes per two backward neighbour pairs; table D). Time-out counts come from the **480-episode re-run, n = 20 episodes per seed** (10 per pair; table R3).

| seed | placed, full eval (n=100) | placed, re-run (n=20) | time-limit ends, re-run (n=20) | of which ended in S3 / S1 | non-placed that end on the wrong side of the start square |
|---|---|---|---|---|---|
| 0 | 1/100 | 0 | 20 | 19 / 1 | 16 of 20 |
| 100 | 89/100 | 20 | 0 | 0 / 0 | n/a |
| 200 | 0/100 | 0 | 20 | 16 / 4 | 16 of 20 |
| 300 | 83/100 | 16 | 3 | 3 / 0 | 0 of 4 |
| 400 | 57/100 | 12 | 8 | 4 / 3 | 0 of 8 |
| 500 | 8/100 | 2 | 18 | 13 / 4 | 5 of 18 |

## 4. Do placements come from sliding the piece or from lifting and carrying?

Among **placed** episodes, the share in which the piece lifted more than 10 mm at some step (full 2400 eval, table E; `max_lift_mm`):
seed 0 19.8% (213 of 1074), seed 100 10.2% (174 of 1704), seed 200 15.3% (157 of 1029), seed 300 8.2% (145 of 1773), seed 400 6.5% (112 of 1720), seed 500 16.7% (182 of 1091).
So 80% to 94% of placements never lift the piece more than 10 mm; the median per-episode maximum lift among placed episodes is 4.8 to 7.1 mm, and 0.0% to 0.5% exceed 30 mm.
At step level (480-episode re-run, table R6), 97.8% to 99.6% of the total along-target piece displacement in placed episodes happens at steps where the lift is at most 10 mm, and
38% to 68% of it happens at steps where both pads are on the piece (state S1).
Placements are therefore **predominantly low-lift movement of the piece along the board, not lift-and-carry**. These data cannot tell dragging while pinched from pushing with the pads;
the env's own `push_assisted` count (carry distance below 50% of the start-target distance) is 5% to 50% depending on the seed (5.4%, 34.8%, 12.2%, 49.6%, 5.5%, 12.3%, table E),
so even the low-lift behaviour is not the same across seeds.
Association only (table F): across all 48 pairs, episodes in which the piece lifted more than 10 mm placed 13% to 24% of the time, episodes that stayed low placed 65% to 97%; seeds 0, 200 and 500 lift above 10 mm in 32% to 36% of forward/lateral neighbour episodes, seeds 100, 300 and 400 in 11% to 18%.
Lift can be a consequence of what happens in an episode as well as a cause, so this is not shown to be a cause.

## 5. Training view (tables 1 to 4)

- In every seed the forward/lateral neighbour pairs get going first; backward neighbour pairs and long forward/lateral pairs follow, in an order that varies by seed, and long backward pairs come last or not at all (table 3, window rows).
  The seeds differ in when, and whether, each category gets going.
- Last 500k training steps, placement (table 3): all pairs 43%, 72%, 42%, 74%, 68%, 43% for seeds 0, 100, 200, 300, 400, 500. Forward/lateral neighbour pairs: 61%, 88%, 66%, 87%, 87%, 56%. Backward neighbour: 1%, 90%, 2%, 90%, 47%, 14%.
- In the last two windows seeds 0 and 500 are still rising on forward/lateral neighbour pairs (46% to 61%, and 42% to 56%); seed 200 is nearly flat there (64% to 66%) and low on long forward/lateral pairs throughout (13% in the last window). Seed 400 was near zero until about 3M steps and then rose fast (10% in 3-4M, 87% in the last window).
  The learning rate is linearly decayed to zero over training, so a seed that is still climbing near the end has little learning rate left. Whether more training would have closed the gap was **not tested**.
- `ever_held` (table 2) is 90% to 99% in the last 10% of training for every seed, so none of the seeds lacks the ability to get a hold of the piece.
- Policy std at the final model (table 1, mean over the four action dimensions): 0.726, 0.600, 0.748, 0.640, 0.573, 0.652 for seeds 0, 100, 200, 300, 400, 500; seed 200's dy std is 0.282 (others 0.513 to 0.655). The two highest means are seeds 200 and 0; no link to the outcome is shown.
  The 5M checkpoint and the final model give identical std values to four decimals.

## 6. Deterministic vs stochastic, and the eval vs training check

The deterministic and stochastic figures use **different units** (table G). The deterministic run has 2 episodes per pair, and the two episodes of every pair were identical in all six seeds, so it counts **pairs placed out of 48**
(17, 35, 17, 38, 32, 16 pairs for seeds 0, 100, 200, 300, 400, 500, i.e. 35.4%, 72.9%, 35.4%, 79.2%, 66.7%, 33.3%). The stochastic figure counts **episodes placed out of 2400** (44.8%, 71.0%, 42.9%, 73.9%, 71.7%, 45.5%); it equals the mean over the 48 pairs of the per-pair placement rate.
The two are not a like-for-like comparison of the same denominator, and a Wilson interval is only meaningful within its own unit (printed in table G). For seeds 0, 200 and 500 the deterministic pair rate is 35.4%, 35.4% and 33.3% (17, 17 and 16 of 48 pairs), against 44.8%, 42.9% and 45.5% of stochastic episodes; for seed 100 it is 72.9% against 71.0%, seed 300 79.2% against 73.9%, seed 400 66.7% against 71.7%.
The eval vs last-10%-training comparison is in `../section3_eval_output.txt` item g (unchanged): the Wilson intervals overlap for all six seeds.

## 7. Zero-shot overlap check

Table H lists the pairs actually present in each committed `zs_*.json`, against the 48 trained pairs: the 12 b/g-file pairs of zero-shot (a) overlap **none** of them; the 20 RS29 pairs of zero-shot (b) overlap **only e4:e5**, which is a trained pair, so the 19-pair figure (n = 285) in the README excludes it. Identical for all six seeds.

## 8. What this does not show

- The mechanism behind toppling and behind the re-grasp loops. The tables describe when they occur, not why.
- Whether seeds 0, 200 and 500 would converge with more training, a different learning-rate schedule, or a different pair mix. No training was run.
- Whether the backward-pair failures in seeds 0, 200 and 500 are an exploration problem or a representation problem. Seed 500 does not even move the piece the wrong way (section 0, point 3).
- The 480-episode re-run is 10 episodes per pair; per-category counts in the R tables are small (n = 20 for the backward neighbour pairs).
- The 45-degree topple threshold and the 10 mm lift threshold are my choices for this analysis, not pre-registered.

# Appendix: tables (verbatim script output)

## diag_eval_tables_output.txt

```
{{diag_eval_tables_output.txt}}
```

## diag_rollout_tables_output.txt

```
{{diag_rollout_tables_output.txt}}
```

## diag_training_tables_output.txt (tables 1 to 4)

```
{{diag_training_tables_output.txt}}
```
