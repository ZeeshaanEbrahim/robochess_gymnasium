# RS32: why seeds 0, 200 and 500 fell short (POST HOC, EXPLORATORY)

**Revision note.** First committed as `80abb93`; this version is a second commit (history not rewritten). It adds section 9 and the addendum tables (X1 to X6, appendix), adds the long-pairs-under-25/50 and "still rising" results to sections 1 and 5, adds the piece-height scale to section 4, and defines "S3 ending" and "S1" where they are used (sections 0, 2 and 3). The earlier numbers are unchanged.

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
| `diag_addendum_tables.py` | committed eval JSONs, Monitor CSVs, configs, run manifests, `assets/scene.xml`, PREREGISTRATION.md | `diag_addendum_tables_output.txt` (tables X1 to X6) |
| `make_diagnosis.py` | `DIAGNOSIS_template.md` + the four outputs | this file |

Regenerate the dumps (about 8 minutes per wave of 3 seeds) with, from the repo root:
`python results/rs32_48_pairs/eval/diagnosis/diag_rollout.py _probe_cart_rs32D_ppotanh_seed<S> --episodes 480 --pairs <the 48 pairs of PREREGISTRATION.md, comma separated> --seed 20000 --out <dir>/ro_seed<S>.json`,
then `python diag_rollout_tables.py <dir>`.
The scope of this file is the items named in the request; I did not have an earlier message with other items.

## 0. Corrections to the first chat summary

1. The outcome table in my chat reply showed four of seven rows, so its columns did not sum to 480. The missing episodes are the "other" row (table R2),
   which is now explicit and sums to 480 for every seed. "Other" is the number of episodes that did not topple and ended at the time limit in place state S2 (the piece was released on the square, but env `is_success` was not reached): 6, 1, 7, 1, 6, 11 episodes for seeds 0, 100, 200, 300, 400, 500, giving "other" totals of 6, 1, 7, 3, 6, 11. The two further "other" episodes of seed 300 are one `bad_contact` end (state S0) and one `piece_off_board` end without tilt above 45 degrees (state S3). In the chat reply that `piece_off_board` end was counted as "lost at the time limit", so seed 300's "upright, lost" count is 8 here, not 9.
2. Seeds 100, 300 and 400 are **not** one group (section 2). Seed 400 was wrongly folded into "good seeds" on the stuck count in the chat reply.
3. Backward pairs: the chat reply gave "19/20, 16/20, 13/20 time out"; those counted only the time-limit episodes whose last step was in place state S3, defined in section 2 (the piece had been held earlier, both pads are no longer on it, and it is not in the released-on-square state S2). All time-limit ends are 20, 20 and 18 of 20 (table R3). Also, only seeds 0 and 200 leave the piece on the wrong side of its start square (16 of 20 each); seed 500 mostly leaves it near its start (5 of 18 non-placed on the wrong side, median progress 0.10).

## 1. Where the gap is

The mean pooled placement of seeds 100, 300 and 400 is 27.8 points above the mean of seeds 0, 200 and 500 (table B; this grouping is by pooled
result only). By category (table B): forward/lateral neighbour pairs 14.1 points (28 pairs), long forward/lateral 8.6, backward neighbour 3.1 (2 pairs), long backward 2.1 (5 pairs).

Pair counts under a threshold (table C), per seed 0 / 100 / 200 / 300 / 400 / 500:
neighbour pairs below 10/50: 3 / 0 / 2 / 0 / 0 / 4; neighbour pairs below 25/50: 7 / 0 / 5 / 0 / 1 / 11; long pairs below 10/50: 9 / 4 / 15 / 3 / 5 / 6; **long pairs below 25/50: 17 / 10 / 18 / 8 / 6 / 17** (of the 18 long pairs; table C, listed pair by pair in table X4).
Seed 500's weak neighbour pairs are the pairs with a rank 7 or 8 target (c6:c7, c7:c8, d6:d7, d7:d8, e6:e7, e7:e8, f7:f8, e6:d7) plus c4:d4, e5:e4, c6:c5. Seed 0's are d7:d8, e7:e8, f6:f7, f7:f8, c4:d4, e5:e4, c6:c5.
The two backward neighbour pairs (e5:e4, c6:c5) are the sharpest split: 1/100, 89/100, 0/100, 83/100, 57/100, 8/100 placed in the full 2400 games for seeds 0, 100, 200, 300, 400, 500 (table D).

## 2. How episodes end (480-episode re-run)

All six re-runs reproduce the committed eval **episode for episode** (480/480 identical end reason, length, placed flag, start and target square, for every seed, including 100, 300 and 400; table R1).

Place states, as the environment defines them (`PlaceReward` in `robochess/tasks/t1_pick_and_place.py`): **S0** the piece has never been held; **S1** both pads are on the piece (it has been held and is pinched now); **S2** released on the square: the release was within 30 mm of the target and at most 15 mm above the board, both pads are off the piece, the piece is on the square and tilted less than 10 degrees; **S3 ("lost")** the piece was held earlier, both pads are not on it now, and it is not in S2. An "episode ending in S3" is an episode whose last step is in S3; "lost at the time limit" means the episode ended by the 300-step time limit in S3.

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

| seed | placed, full eval (n=100) | placed, re-run (n=20) | time-limit ends, re-run (n=20) | of which last step in S3 (lost, defined in section 2) / in S1 (both pads on the piece) | non-placed that end on the wrong side of the start square |
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
**Size of the lift against the piece.** The piece is 50 mm tall and 30 mm across the base, measured from the model (table X1: base cylinder 0 to 18 mm, stem to 34 mm, head to 50 mm above the piece origin, which is at its base). `max_lift_mm` is the rise of the piece **origin (its base)** above its resting height. Among placed episodes (table X2, full 2400 eval), seed 0 / 100 / 200 / 300 / 400 / 500:
median maximum lift 7.1 / 5.6 / 6.6 / 4.8 / 5.8 / 6.5 mm, 90th percentile 12.1 / 10.0 / 11.5 / 9.2 / 9.3 / 11.7 mm, largest 36.2 / 32.7 / 35.7 / 34.1 / 30.2 / 29.0 mm; placed episodes above 30 mm: 2 / 3 / 5 / 3 / 1 / 0; **above 50 mm: 0 in every seed**.
A median lift of 5 to 7 mm is about a tenth of the piece's height. To pass over a same-size piece on a neighbouring square the base would have to rise by more than 50 mm, and no placed episode did. This is for scale only: the scene holds a single piece (`piece_a`) and the six configs set `n_obstacles: 0`, so there is no neighbouring piece to clear in this task. A few failed episodes did exceed 50 mm (15, 4, 11, 1, 2, 3 episodes for seeds 0, 100, 200, 300, 400, 500; table X2).
**Pair e4:e5 only** (table X3, 50 episodes per seed): placed episodes with a maximum lift above 10 mm: 9 of 43 (21%), 2 of 44 (5%), 1 of 32 (3%), 3 of 41 (7%), 0 of 47 (0%), 4 of 46 (9%) for seeds 0 / 100 / 200 / 300 / 400 / 500 (counts are small).
Association only (table F): across all 48 pairs, episodes in which the piece lifted more than 10 mm placed 13% to 24% of the time, episodes that stayed low placed 65% to 97%; seeds 0, 200 and 500 lift above 10 mm in 32% to 36% of forward/lateral neighbour episodes, seeds 100, 300 and 400 in 11% to 18%.
Lift can be a consequence of what happens in an episode as well as a cause, so this is not shown to be a cause.

## 5. Training view (tables 1 to 4)

- In every seed the forward/lateral neighbour pairs get going first; backward neighbour pairs and long forward/lateral pairs follow, in an order that varies by seed, and long backward pairs come last or not at all (table 3, window rows).
  The seeds differ in when, and whether, each category gets going.
- Last 500k training steps, placement (table 3): all pairs 43%, 72%, 42%, 74%, 68%, 43% for seeds 0, 100, 200, 300, 400, 500. Forward/lateral neighbour pairs: 61%, 88%, 66%, 87%, 87%, 56%. Backward neighbour: 1%, 90%, 2%, 90%, 47%, 14%.
- In the last two windows seeds 0 and 500 are still rising on forward/lateral neighbour pairs (46% to 61%, and 42% to 56%); seed 200 is nearly flat there (64% to 66%) and low on long forward/lateral pairs throughout (13% in the last window). Seed 400 was near zero until about 3M steps and then rose fast (10% in 3-4M, 87% in the last window).
  The learning rate is linearly decayed to zero over training, so a seed that is still climbing near the end has little learning rate left. Whether more training would have closed the gap was **not tested**.
- **Was training placement still rising at the end?** (table X5; all 48 pairs pooled, 100k-step bins, nominal binomial uncertainty because the episodes come from one evolving policy.) Placement in 4.5-5.0M minus 4.0-4.5M, with an approximate 95% interval, for seeds 0 / 100 / 200 / 300 / 400 / 500:
  +10.1 [+7.1, +13.0], +5.5 [+3.2, +7.8], +2.2 [-0.7, +5.1], +4.5 [+2.2, +6.8], +18.4 [+15.7, +21.1], +13.0 [+10.1, +15.9] points. The weighted slope over the last ten bins (4.1 to 5.0M) is +2.00, +0.93, +0.45, +0.71, +4.13, +2.33 points per 100k steps; its interval excludes 0 for every seed except seed 200 (+0.45 +/- 0.51).
  Over the last five bins alone (4.6 to 5.0M) only seed 400 (+1.84 +/- 1.26) has an interval excluding 0. So by this nominal test, training placement was still rising over the final million steps for **five of six seeds, not only the failing ones** (seeds 100 and 300 rise more slowly, +0.9 and +0.7 per 100k), was not detectably changing for seed 200, and showed no detectable rise within the last 500k steps for all but seed 400 (those intervals are wide, about +/-1.1 to +/-1.5 points per 100k, so this is absence of evidence, not evidence of a plateau).
  The learning rate is 6.0e-5 at 4.0M, 3.0e-5 at 4.5M and 0 at 5.0M. Whether a seed would have kept rising with more training was not tested.
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
- The slopes in table X5 and the intervals on the window differences assume independent episodes, which they are not; they only say whether a rise is distinguishable from noise under that simplification.
- The 45-degree topple threshold and the 10 mm lift threshold are my choices for this analysis, not pre-registered.

## 9. Pair order check against PREREGISTRATION.md (table X6)

The pair order of PREREGISTRATION.md ("Training pairs ... in this order") was re-parsed here independently (whitespace tokens, not the regex used by the eval tooling) and compared, order and content, with every file that carries the 48 pairs: `EVAL_MANIFEST.json` (the pair list and the `--pairs` argument of all six eval commands), `EVAL_MANIFEST_section3.json` (pair list and the six deterministic commands), the episode order of all six `*_final.json` and all six `det48_*.json` (episode i is pair i mod 48), the six `configs/_probe_cart_rs32D_ppotanh_seed*.yaml` `train_pairs`, the six runs' `manifest.json` `config_resolved.train_pairs` (what each run actually used), and the first 48 rows of `expert_gate/TABLE.md`.
**All of them match; there is no mismatch** (41 checks, listed one by one in table X6). The pairs that occur in the six runs' Monitor CSVs are exactly the 48 pairs (2417 to 2664 episodes per pair across all six runs). Training draws a pair uniformly at random per episode, so the order of `train_pairs` does not affect training; it matters for the eval and the tables.

## 10. The three items asked for after the first version

| item | where it is in this file |
|---|---|
| long pairs under 25/50, per seed | section 1 (17 / 10 / 18 / 8 / 6 / 17), tables C and X4 |
| was training placement still rising at the end | section 5, table X5 |
| pair order matches PREREGISTRATION.md | section 9, table X6 |

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

## diag_addendum_tables_output.txt (tables X1 to X6)

```
{{diag_addendum_tables_output.txt}}
```
