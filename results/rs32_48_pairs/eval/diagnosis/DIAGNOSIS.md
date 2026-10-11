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
A. PLACEMENT BY PAIR CATEGORY (full 2400 episodes per seed, 50 per pair). back = target rank below start rank. k/n and %
category         pairs  n/seed           seed0         seed100         seed200         seed300         seed400         seed500
neigh_fwd/lat       28    1400     899/1400  64%    1216/1400  87%     937/1400  67%    1220/1400  87%    1250/1400  89%     837/1400  60%
neigh_back           2     100       1/100    1%      89/100   89%       0/100    0%      83/100   83%      57/100   57%       8/100    8%
long_fwd/lat        13     650     174/650   27%     339/650   52%      92/650   14%     380/650   58%     411/650   63%     246/650   38%
long_back            5     250       0/250    0%      60/250   24%       0/250    0%      90/250   36%       2/250    1%       0/250    0%

B. GAP DECOMPOSITION: mean pooled placement of seeds 100,300,400 minus mean of seeds 0,200,500, split by category (pts of the 48-pair pooled rate)
   (seeds 100, 300 and 400 are grouped here only by their pooled result; see section 2 of DIAGNOSIS.md: seed 400 differs from seeds 100 and 300)
   neigh_fwd/lat    pairs 28  failing-seed mean  63.6%  good-seed mean  87.8%  contribution  14.1 pts (50.6% of 27.8)
   neigh_back       pairs  2  failing-seed mean   3.0%  good-seed mean  76.3%  contribution   3.1 pts (11.0% of 27.8)
   long_fwd/lat     pairs 13  failing-seed mean  26.3%  good-seed mean  57.9%  contribution   8.6 pts (30.9% of 27.8)
   long_back        pairs  5  failing-seed mean   0.0%  good-seed mean  20.3%  contribution   2.1 pts ( 7.6% of 27.8)

C. NEIGHBOUR (30) PAIRS BELOW 10/50 AND BELOW 25/50, per seed (also the 18 long pairs and all 48)
seed            neigh <10/50  neigh <25/50  long <10/50  long <25/50  all48 <10  all48 <25
seed0                      3             7            9           17         12         24
seed100                    0             0            4           10          4         10
seed200                    2             5           15           18         17         23
seed300                    0             0            3            8          3          8
seed400                    0             1            5            6          5          7
seed500                    4            11            6           17         10         28
   neighbour pairs below 25/50, listed:
   seed0       d7:d8 22, e7:e8 18, f6:f7 21, f7:f8 6, c4:d4 13, e5:e4 0, c6:c5 1
   seed100     none
   seed200     f2:f3 23, c4:d4 23, f5:e5 10, e5:e4 0, c6:c5 0
   seed300     none
   seed400     e5:e4 16
   seed500     c6:c7 11, c7:c8 1, d6:d7 16, d7:d8 1, e6:e7 20, e7:e8 13, f7:f8 19, c4:d4 17, e6:d7 19, e5:e4 0, c6:c5 8

D. BACKWARD PAIRS, placed counts from the FULL 2400-episode eval (50 episodes per pair)
   neigh_back pairs: ['e5:e4', 'c6:c5']
   seed0        1/100 = 1.0%   per pair e5:e4 0 c6:c5 1
   seed100      89/100 = 89.0%   per pair e5:e4 43 c6:c5 46
   seed200      0/100 = 0.0%   per pair e5:e4 0 c6:c5 0
   seed300      83/100 = 83.0%   per pair e5:e4 42 c6:c5 41
   seed400      57/100 = 57.0%   per pair e5:e4 16 c6:c5 41
   seed500      8/100 = 8.0%   per pair e5:e4 0 c6:c5 8
   long_back pairs: ['d7:d4', 'g7:d4', 'b7:d5', 'h6:e4', 'g6:e3']
   seed0        0/250 = 0.0%   per pair d7:d4 0 g7:d4 0 b7:d5 0 h6:e4 0 g6:e3 0
   seed100      60/250 = 24.0%   per pair d7:d4 22 g7:d4 0 b7:d5 31 h6:e4 2 g6:e3 5
   seed200      0/250 = 0.0%   per pair d7:d4 0 g7:d4 0 b7:d5 0 h6:e4 0 g6:e3 0
   seed300      90/250 = 36.0%   per pair d7:d4 25 g7:d4 7 b7:d5 31 h6:e4 18 g6:e3 9
   seed400      2/250 = 0.8%   per pair d7:d4 0 g7:d4 0 b7:d5 2 h6:e4 0 g6:e3 0
   seed500      0/250 = 0.0%   per pair d7:d4 0 g7:d4 0 b7:d5 0 h6:e4 0 g6:e3 0

E. LIFT AMONG PLACED EPISODES (full 2400; max_lift_mm = eval_probe's per-episode maximum of the piece lift, mm)
   The question: do placements come from sliding the piece along the board (low lift) or from lifting and carrying?
seed          placed  placed with max_lift>10mm   >20mm   >30mm  median max_lift (placed)    p90  carry_mm<50% of dist (push-assisted, place_report)
seed0           1074           213 =  19.8%    1.5%    0.2%                       7.1   12.1            58/1074 = 5.4%
seed100         1704           174 =  10.2%    1.6%    0.2%                       5.6   10.0           593/1704 = 34.8%
seed200         1029           157 =  15.3%    2.6%    0.5%                       6.6   11.5           126/1029 = 12.2%
seed300         1773           145 =   8.2%    2.4%    0.2%                       4.8    9.2           880/1773 = 49.6%
seed400         1720           112 =   6.5%    1.0%    0.1%                       5.8    9.3            95/1720 = 5.5%
seed500         1091           182 =  16.7%    2.6%    0.0%                       6.5   11.7           134/1091 = 12.3%
   lift_30mm-style flags are not used (known unreliable); only max_lift_mm thresholds above.

F. OUTCOME BY WHETHER THE PIECE EVER LIFTED > 10 mm (association only; lift can be a consequence of what happens in the episode)
   all 48 pairs:
   seed0        episodes with max_lift>10mm 1089 (45.4%)  placed|lift>10  19.6%  placed|lift<=10  65.7%
   seed100      episodes with max_lift>10mm  826 (34.4%)  placed|lift>10  21.1%  placed|lift<=10  97.2%
   seed200      episodes with max_lift>10mm 1172 (48.8%)  placed|lift>10  13.4%  placed|lift<=10  71.0%
   seed300      episodes with max_lift>10mm  709 (29.5%)  placed|lift>10  20.5%  placed|lift<=10  96.3%
   seed400      episodes with max_lift>10mm  469 (19.5%)  placed|lift>10  23.9%  placed|lift<=10  83.3%
   seed500      episodes with max_lift>10mm 1013 (42.2%)  placed|lift>10  18.0%  placed|lift<=10  65.5%
   neighbour forward/lateral pairs only (28 pairs, 1400 episodes):
   seed0        episodes with max_lift>10mm  502 (35.9%)  placed|lift>10  30.3%  placed|lift<=10  83.2%
   seed100      episodes with max_lift>10mm  247 (17.6%)  placed|lift>10  30.4%  placed|lift<=10  99.0%
   seed200      episodes with max_lift>10mm  458 (32.7%)  placed|lift>10  28.2%  placed|lift<=10  85.8%
   seed300      episodes with max_lift>10mm  216 (15.4%)  placed|lift>10  24.5%  placed|lift<=10  98.6%
   seed400      episodes with max_lift>10mm  150 (10.7%)  placed|lift>10  30.0%  placed|lift<=10  96.4%
   seed500      episodes with max_lift>10mm  448 (32.0%)  placed|lift>10  24.3%  placed|lift<=10  76.5%

G. DETERMINISTIC vs STOCHASTIC (UNITS DIFFER). deterministic: 96 episodes = 2 per pair; the two episodes of every pair were identical in all six seeds,
   so its denominator is 48 PAIRS (a pair is placed or not). stochastic: 2400 EPISODES (50 per pair). Wilson 95% of each is over its own unit.
   seed0        deterministic: 17/48 pairs placed = 35.4% [23.4, 49.6]   stochastic: 1074/2400 episodes = 44.8% [42.8, 46.7] (= mean over the 48 pairs of per-pair placement 44.7%)
   seed100      deterministic: 35/48 pairs placed = 72.9% [59.0, 83.4]   stochastic: 1704/2400 episodes = 71.0% [69.2, 72.8] (= mean over the 48 pairs of per-pair placement 71.0%)
   seed200      deterministic: 17/48 pairs placed = 35.4% [23.4, 49.6]   stochastic: 1029/2400 episodes = 42.9% [40.9, 44.9] (= mean over the 48 pairs of per-pair placement 42.9%)
   seed300      deterministic: 38/48 pairs placed = 79.2% [65.7, 88.3]   stochastic: 1773/2400 episodes = 73.9% [72.1, 75.6] (= mean over the 48 pairs of per-pair placement 73.9%)
   seed400      deterministic: 32/48 pairs placed = 66.7% [52.5, 78.3]   stochastic: 1720/2400 episodes = 71.7% [69.8, 73.4] (= mean over the 48 pairs of per-pair placement 71.7%)
   seed500      deterministic: 16/48 pairs placed = 33.3% [21.7, 47.5]   stochastic: 1091/2400 episodes = 45.5% [43.5, 47.5] (= mean over the 48 pairs of per-pair placement 45.5%)

H. ZERO-SHOT OVERLAP CHECK (from the pairs actually present in the committed zs_*.json files, against the 48 trained pairs)
   seed0        zero-shot (a) b/g pairs: 12 distinct pairs, 300 episodes; overlap with the 48 trained pairs: none
   seed0        zero-shot (b) RS29 pairs: 20 distinct pairs, 300 episodes; overlap with the 48 trained pairs: ['e4:e5']
   seed100      zero-shot (a) b/g pairs: 12 distinct pairs, 300 episodes; overlap with the 48 trained pairs: none
   seed100      zero-shot (b) RS29 pairs: 20 distinct pairs, 300 episodes; overlap with the 48 trained pairs: ['e4:e5']
   seed200      zero-shot (a) b/g pairs: 12 distinct pairs, 300 episodes; overlap with the 48 trained pairs: none
   seed200      zero-shot (b) RS29 pairs: 20 distinct pairs, 300 episodes; overlap with the 48 trained pairs: ['e4:e5']
   seed300      zero-shot (a) b/g pairs: 12 distinct pairs, 300 episodes; overlap with the 48 trained pairs: none
   seed300      zero-shot (b) RS29 pairs: 20 distinct pairs, 300 episodes; overlap with the 48 trained pairs: ['e4:e5']
   seed400      zero-shot (a) b/g pairs: 12 distinct pairs, 300 episodes; overlap with the 48 trained pairs: none
   seed400      zero-shot (b) RS29 pairs: 20 distinct pairs, 300 episodes; overlap with the 48 trained pairs: ['e4:e5']
   seed500      zero-shot (a) b/g pairs: 12 distinct pairs, 300 episodes; overlap with the 48 trained pairs: none
   seed500      zero-shot (b) RS29 pairs: 20 distinct pairs, 300 episodes; overlap with the 48 trained pairs: ['e4:e5']
```

## diag_rollout_tables_output.txt

```
R1. REPRODUCTION of the committed eval by the diagnostic rollout (episode-for-episode: end reason, length, placed, start and target square)
   seed0      480/480 episodes identical
   seed100    480/480 episodes identical
   seed200    480/480 episodes identical
   seed300    480/480 episodes identical
   seed400    480/480 episodes identical
   seed500    480/480 episodes identical

R2. EXCLUSIVE EPISODE OUTCOMES, 480 episodes per seed (every row sums to 480; asserted). Evaluated in this order:
   placed = env is_success at the last step; toppled = piece tilt > 45 deg at ANY step of the episode (includes every piece_off_board end that tilted);
   upright, pinching/lost = not toppled, ended by time limit in place state S1 (both pads on the piece) / S3 (held once, now lost, not in S2);
   other = everything else: not toppled and (time limit in S2 [released, but not success] or in S0 [never held], or ended by bad_contact / piece_off_board without tilt > 45).
outcome                                         seed0  seed100  seed200  seed300  seed400  seed500
placed                                            218      337      199      348      352      213
toppled                                           144      133      181      121       57      131
upright, still pinching at time limit              30        3       21        0       28       53
upright, lost (S3) at time limit                   82        6       72        8       37       72
other                                               6        1        7        3        6       11
SUM                                               480      480      480      480      480      480
   composition of 'other' (reason / final place state S0..S3):
   seed0      {'time_limit/S2': 6}
   seed100    {'time_limit/S2': 1}
   seed200    {'time_limit/S2': 7}
   seed300    {'bad_contact/S0': 1, 'piece_off_board/S3': 1, 'time_limit/S2': 1}
   seed400    {'time_limit/S2': 6}
   seed500    {'time_limit/S2': 11}
   UPRIGHT-AND-STUCK = 'still pinching' + 'lost at time limit':
   seed0      30 + 82 = 112    toppled 144
   seed100    3 + 6 = 9    toppled 133
   seed200    21 + 72 = 93    toppled 181
   seed300    0 + 8 = 8    toppled 121
   seed400    28 + 37 = 65    toppled 57
   seed500    53 + 72 = 125    toppled 131

R2b. the same outcomes by pair category (episodes per category per seed: neigh_fwd/lat 280, neigh_back 20, long_fwd/lat 130, long_back 50)
-- neigh_fwd/lat
   placed                                         182      243      185      245      254      168
   toppled                                         56       33       64       31       18       46
   upright, still pinching at time limit           14        1        5        0        0       30
   upright, lost (S3) at time limit                24        2       25        4        8       28
   other                                            4        1        1        0        0        8
-- neigh_back
   placed                                           0       20        0       16       12        2
   toppled                                          0        0        5        4        2        1
   upright, still pinching at time limit            1        0        4        0        3        4
   upright, lost (S3) at time limit                19        0       11        0        2       12
   other                                            0        0        0        0        1        1
-- long_fwd/lat
   placed                                          36       61       14       72       86       43
   toppled                                         69       64       88       53       29       64
   upright, still pinching at time limit            6        2        7        0        5        9
   upright, lost (S3) at time limit                17        3       15        2        5       12
   other                                            2        0        6        3        5        2
-- long_back
   placed                                           0       13        0       15        0        0
   toppled                                         19       36       24       33        8       20
   upright, still pinching at time limit            9        0        5        0       20       10
   upright, lost (S3) at time limit                22        1       21        2       22       20
   other                                            0        0        0        0        0        0

R3. BACKWARD NEIGHBOUR PAIRS e5:e4 and c6:c5, n = 20 episodes per seed (10 per pair) from this 480-episode re-run; progress = piece displacement along start->target / pair distance at the last step (<0 = ended on the far side of its start square)
   seed0      n=20: placed 0, time-limit ends 20 (final state S1: 1, S3: 19), toppled 0; non-placed with progress<0: 16/20; median progress of non-placed -0.27
   seed100    n=20: placed 20, time-limit ends 0 (final state S1: 0, S3: 0), toppled 0; non-placed with progress<0: 0/0; median progress of non-placed nan
   seed200    n=20: placed 0, time-limit ends 20 (final state S1: 4, S3: 16), toppled 5; non-placed with progress<0: 16/20; median progress of non-placed -0.23
   seed300    n=20: placed 16, time-limit ends 3 (final state S1: 0, S3: 3), toppled 4; non-placed with progress<0: 0/4; median progress of non-placed 1.75
   seed400    n=20: placed 12, time-limit ends 8 (final state S1: 3, S3: 4), toppled 2; non-placed with progress<0: 0/8; median progress of non-placed 0.60
   seed500    n=20: placed 2, time-limit ends 18 (final state S1: 4, S3: 13), toppled 1; non-placed with progress<0: 5/18; median progress of non-placed 0.10

R4. WHERE THE PIECE ENDS IN THE UPRIGHT-AND-STUCK EPISODES (piece-target xy distance at the last step, mm; groups with n>=5)
   upright, still pinching at time limit       seed0 n=30  p25    40 p50    58 p75   159  share<30mm 0.17
   upright, still pinching at time limit     seed200 n=21  p25    58 p50    97 p75   133  share<30mm 0.05
   upright, still pinching at time limit     seed400 n=28  p25    49 p50    92 p75   120  share<30mm 0.04
   upright, still pinching at time limit     seed500 n=53  p25    35 p50    61 p75    79  share<30mm 0.17
   upright, lost (S3) at time limit            seed0 n=82  p25    57 p50    90 p75   124  share<30mm 0.04
   upright, lost (S3) at time limit          seed100 n=6   p25    45 p50    52 p75    71  share<30mm 0.00
   upright, lost (S3) at time limit          seed200 n=72  p25    62 p50    90 p75   139  share<30mm 0.00
   upright, lost (S3) at time limit          seed300 n=8   p25    35 p50    49 p75    81  share<30mm 0.00
   upright, lost (S3) at time limit          seed400 n=37  p25    43 p50    68 p75   108  share<30mm 0.05
   upright, lost (S3) at time limit          seed500 n=72  p25    59 p50    80 p75   121  share<30mm 0.01
   number of separate grasp phases (runs of place state S1) per episode:
   seed0      placed 11.0 per episode; upright-and-stuck 21.3 per episode (n=112)
   seed100    placed 5.6 per episode; upright-and-stuck 14.3 per episode (n=9)
   seed200    placed 9.1 per episode; upright-and-stuck 19.3 per episode (n=93)
   seed300    placed 5.4 per episode; upright-and-stuck 11.8 per episode (n=8)
   seed400    placed 8.9 per episode; upright-and-stuck 30.5 per episode (n=65)
   seed500    placed 9.7 per episode; upright-and-stuck 25.8 per episode (n=125)

R5. WHEN DOES THE PIECE TOPPLE (first step with tilt > 45 deg): place state at that step, and time since the last pinched step
   seed0      toppled episodes 148/480 (long pairs 90), placed after toppling 4; state at first topple step: S0 24, S1 19, S3 105; not pinched at topple: median steps since last pinch 4, pinch-site-to-piece distance median 21 mm, tool tilt median 0.1 deg
   seed100    toppled episodes 135/480 (long pairs 101), placed after toppling 2; state at first topple step: S0 64, S1 15, S3 56; not pinched at topple: median steps since last pinch 6, pinch-site-to-piece distance median 29 mm, tool tilt median 0.1 deg
   seed200    toppled episodes 186/480 (long pairs 115), placed after toppling 5; state at first topple step: S0 52, S1 10, S3 124; not pinched at topple: median steps since last pinch 6, pinch-site-to-piece distance median 26 mm, tool tilt median 0.1 deg
   seed300    toppled episodes 125/480 (long pairs 88), placed after toppling 4; state at first topple step: S0 39, S1 12, S3 74; not pinched at topple: median steps since last pinch 6, pinch-site-to-piece distance median 27 mm, tool tilt median 0.1 deg
   seed400    toppled episodes 61/480 (long pairs 41), placed after toppling 4; state at first topple step: S0 6, S1 23, S3 32; not pinched at topple: median steps since last pinch 2, pinch-site-to-piece distance median 20 mm, tool tilt median 0.1 deg
   seed500    toppled episodes 134/480 (long pairs 86), placed after toppling 3; state at first topple step: S0 22, S1 39, S3 73; not pinched at topple: median steps since last pinch 4, pinch-site-to-piece distance median 23 mm, tool tilt median 0.1 deg

R6. HOW PLACED EPISODES MOVE THE PIECE (placed episodes only; step level, 480-episode re-run). Along-target displacement D_t = (piece_t - piece_{t-1}) . u, u = unit vector start->target; shares of the SUM of D_t over steps
seed       placed  max_lift>10mm   share of D at steps with lift<=10mm  share of D at pinched steps  pinched steps with lift>10mm  share of D at pinched steps with lift<=10mm
seed0         218      47 = 21.6%                                 97.8%                        61.1%                          1.0%                                        59.4%
seed100       337      28 =  8.3%                                 99.0%                        51.4%                          1.0%                                        50.6%
seed200       199      29 = 14.6%                                 98.0%                        54.9%                          0.9%                                        54.2%
seed300       348      34 =  9.8%                                 99.2%                        38.0%                          1.0%                                        37.8%
seed400       352      18 =  5.1%                                 99.6%                        67.9%                          0.3%                                        67.5%
seed500       213      38 = 17.8%                                 99.3%                        65.1%                          0.9%                                        64.4%
   (lift = eval_probe's per-step piece lift in mm. A step with lift <= 10 mm can still have the piece pressed by the pads; this table does not tell dragging from pushing.)
```

## diag_training_tables_output.txt (tables 1 to 4)

```
1. POLICY STD = exp(log_std) per action dim (dx, dy, dz, gripper) and mean, from policy.pth in rc_{1,2,3,4,5}M_steps.zip and final_model.zip
rs32D_seed0
   1M ckpt         0.7833 0.7011 0.9780 0.9833   mean 0.8614
   2M ckpt         0.7193 0.6067 0.8793 0.9562   mean 0.7904
   3M ckpt         0.7393 0.5514 0.8637 0.8548   mean 0.7523
   4M ckpt         0.7525 0.5043 0.8722 0.8402   mean 0.7423
   5M ckpt         0.7413 0.5133 0.8214 0.8260   mean 0.7255
   final 5,005,312 0.7413 0.5133 0.8214 0.8260   mean 0.7255
rs32D_seed100
   1M ckpt         0.7201 0.8017 0.9394 0.9253   mean 0.8466
   2M ckpt         0.6657 0.6072 0.7879 0.7593   mean 0.7050
   3M ckpt         0.6106 0.5520 0.6139 0.6286   mean 0.6013
   4M ckpt         0.5641 0.5281 0.6753 0.6268   mean 0.5986
   5M ckpt         0.5965 0.5278 0.6929 0.5832   mean 0.6001
   final 5,005,312 0.5965 0.5278 0.6929 0.5832   mean 0.6001
rs32D_seed200
   1M ckpt         0.8070 0.7124 0.8347 1.0897   mean 0.8610
   2M ckpt         0.8773 0.4525 0.7164 1.1644   mean 0.8026
   3M ckpt         0.8848 0.3898 0.7607 1.1301   mean 0.7913
   4M ckpt         0.8029 0.2720 0.8262 1.1492   mean 0.7626
   5M ckpt         0.8084 0.2823 0.8267 1.0731   mean 0.7476
   final 5,005,312 0.8084 0.2823 0.8267 1.0731   mean 0.7476
rs32D_seed300
   1M ckpt         0.7899 0.6919 1.0114 1.0717   mean 0.8912
   2M ckpt         0.5992 0.6418 0.8658 0.8656   mean 0.7431
   3M ckpt         0.5097 0.5935 0.7262 0.9047   mean 0.6835
   4M ckpt         0.5326 0.5541 0.6740 0.8118   mean 0.6431
   5M ckpt         0.5901 0.5851 0.6573 0.7286   mean 0.6403
   final 5,005,312 0.5901 0.5851 0.6573 0.7286   mean 0.6403
rs32D_seed400
   1M ckpt         0.7385 0.6647 1.0175 0.7720   mean 0.7982
   2M ckpt         0.5624 0.6343 0.6974 0.6389   mean 0.6332
   3M ckpt         0.4792 0.6600 0.6804 0.6606   mean 0.6200
   4M ckpt         0.4277 0.6139 0.6548 0.6308   mean 0.5818
   5M ckpt         0.4595 0.5982 0.6304 0.6035   mean 0.5729
   final 5,005,312 0.4595 0.5982 0.6304 0.6035   mean 0.5729
rs32D_seed500
   1M ckpt         0.8258 0.7022 0.9044 0.9477   mean 0.8450
   2M ckpt         0.6577 0.6013 0.9141 0.7958   mean 0.7422
   3M ckpt         0.6754 0.6043 0.7505 0.7473   mean 0.6944
   4M ckpt         0.6459 0.6345 0.7558 0.6518   mean 0.6720
   5M ckpt         0.6124 0.6548 0.7220 0.6174   mean 0.6517
   final 5,005,312 0.6124 0.6548 0.7220 0.6174   mean 0.6517

2. EVER_HELD per 10% chunk of training (%; pooled over the 4 Monitor files; chunk = min((cum_l-1)*10//total_l, 9) per file)
seed          0-10  10-20  20-30  30-40  40-50  50-60  60-70  70-80  80-90  90-100
seed0          48.6   93.4   96.7   94.6   94.3   95.1   95.6   96.0   96.7   97.0
seed100        43.5   93.4   94.3   90.3   87.4   86.3   85.9   88.4   87.7   90.5
seed200        35.6   88.3   94.0   95.5   95.7   94.5   92.9   92.1   92.8   92.5
seed300        54.9   89.9   90.6   91.7   84.1   87.9   89.9   91.1   90.7   91.7
seed400        57.2   94.8   96.1   95.5   94.6   96.5   96.4   97.2   98.6   98.5
seed500        32.2   87.8   87.5   92.9   92.6   95.0   93.4   95.5   95.7   96.7

3. TRAINING PLACEMENT in the LAST 500k training steps (steps 4.5M to 5.0M; Monitor terminated_reason == success), pooled and by pair category; k/n and %
category                     seed0           seed100           seed200           seed300           seed400           seed500
all 48 pairs         925/2163   43%    2268/3135   72%     927/2212   42%    2227/3020   74%    1831/2687   68%     943/2189   43%
neigh_fwd/lat        772/1269   61%    1654/1883   88%     848/1279   66%    1553/1777   87%    1359/1564   87%     735/1311   56%
neigh_back             1/85      1%     107/119    90%       2/100     2%     113/126    90%      53/112    47%      13/93     14%
long_fwd/lat         152/584    26%     420/832    50%      77/603    13%     454/810    56%     415/707    59%     195/558    35%
long_back              0/225     0%      87/301    29%       0/230     0%     107/307    35%       4/304     1%       0/227     0%
neighbour 30         773/1354   57%    1761/2002   88%     850/1379   62%    1666/1903   88%    1412/1676   84%     748/1404   53%
long 18              152/809    19%     507/1133   45%      77/833     9%     561/1117   50%     419/1011   41%     195/785    25%

   the same for 1M-step windows (all seeds), by category: placement % (k/n omitted)
seed      category           0.0-1.0M   1.0-2.0M   2.0-3.0M   3.0-4.0M   4.0-4.5M   4.5-5.0M
seed0     neigh_fwd/lat           1.6        4.5        9.7       26.7       46.2       60.8
seed0     neigh_back              0.7        0.0        0.6        1.2        1.5        1.2
seed0     long_fwd/lat            0.1        0.9        1.4        7.6       15.8       26.0
seed0     long_back               0.0        0.0        0.0        0.0        0.0        0.0
seed100   neigh_fwd/lat           2.9       23.8       60.0       78.3       86.3       87.8
seed100   neigh_back              0.0        1.2        3.7       29.4       86.9       89.9
seed100   long_fwd/lat            0.5        4.7       23.5       34.0       42.1       50.5
seed100   long_back               0.0        0.2        0.0        3.3       21.0       28.9
seed200   neigh_fwd/lat           0.9        2.9       14.4       47.4       63.5       66.3
seed200   neigh_back              0.0        0.0        4.4        1.2        1.0        2.0
seed200   long_fwd/lat            0.0        0.0        1.0        6.0        9.3       12.8
seed200   long_back               0.0        0.0        0.0        0.0        0.0        0.0
seed300   neigh_fwd/lat           6.6       27.8       63.9       81.4       85.9       87.4
seed300   neigh_back              0.0        2.1       20.5       76.6       92.2       89.7
seed300   long_fwd/lat            0.1        5.0       20.0       36.8       48.6       56.0
seed300   long_back               0.0        0.0        3.4       15.0       20.9       34.9
seed400   neigh_fwd/lat           0.3        0.0        0.6       10.1       68.1       86.9
seed400   neigh_back              0.0        0.6        0.0        0.7        4.4       47.3
seed400   long_fwd/lat            0.0        0.0        0.0        1.2       35.2       58.7
seed400   long_back               0.0        0.0        0.0        0.0        0.0        1.3
seed500   neigh_fwd/lat           1.7        2.6        8.1       25.9       41.5       56.1
seed500   neigh_back              0.0        0.7        0.6        3.7       11.0       14.0
seed500   long_fwd/lat            0.1        0.2        1.4        8.0       22.0       34.9
seed500   long_back               0.0        0.0        0.0        0.0        0.4        0.0

4. TRAINING PLACEMENT PER PAIR in the last 500k steps (k/n), order of PREREGISTRATION.md
pair           seed0     seed100     seed200     seed300     seed400     seed500
c2:c3       35/45       52/72       31/45       65/71       41/53       27/50   
c3:c4       37/51       50/56       35/54       50/56       50/63       34/45   
c4:c5       46/53       63/73       17/31       57/64       52/56       30/43   
c5:c6       44/54       63/74       30/43       42/50       44/49       12/38   
c6:c7       35/48       70/78       36/55       55/64       48/60        7/50   
c7:c8       24/44       62/70       28/42       41/58       46/55        1/47   
d2:d3       38/52       67/76       31/55       69/74       47/56       21/39   
d3:d4       29/39       60/66       22/39       51/55       50/57       42/49   
d4:d5       40/50       58/61       45/59       64/72       50/52       41/52   
d5:d6       42/48       58/63       38/48       60/67       54/61       42/56   
d6:d7       31/52       52/56       37/46       55/64       50/63       14/48   
d7:d8        9/50       49/61       35/40       69/79       48/60        0/45   
e2:e3       22/34       56/65       27/50       62/66       39/53       33/43   
e3:e4       32/46       54/59       27/45       57/65       46/53       44/55   
e4:e5       33/37       61/67       23/35       58/63       50/54       36/45   
e5:e6       24/30       61/70       53/67       53/59       57/59       34/52   
e6:e7       20/38       47/54       31/38       61/67       45/47       20/44   
e7:e8       13/60       57/62       35/45       59/67       48/59        4/51   
f2:f3       20/40       63/84       25/55       44/59       37/43       43/55   
f3:f4       21/40       58/64       22/37       64/70       62/70       38/40   
f4:f5       33/45       55/73       28/47       44/48       57/61       47/54   
f5:f6       31/47       64/74       24/34       56/64       48/49       21/30   
f6:f7       13/35       70/79       39/44       49/56       54/58       18/41   
f7:f8        3/46       55/60       27/45       46/58       52/62        9/45   
c4:d4       11/47       65/68       32/51       61/63       51/60       23/45   
f5:e5       35/48       70/71       14/49       57/64       56/61       37/50   
d3:e4       28/51       62/72       28/38       43/56       41/49       38/49   
e6:d7       23/39       52/55       28/42       61/78       36/41       19/50   
e5:e4        1/43       54/60        1/54       50/56       14/51        0/53   
c6:c5        0/42       53/59        1/46       63/70       39/61       13/40   
b2:b4       24/42       42/71        5/44       45/61       23/54       17/45   
g3:g6        0/45       26/45        7/37       21/55       32/51        9/35   
d7:d4        0/48       34/71        0/47       28/64        0/72        0/42   
a5:c5       14/51       59/66       18/46       55/60       44/61       34/49   
h4:e4       18/54       17/61        0/47       44/68       23/51       10/46   
b6:e6       14/44       52/72        2/42       42/59       41/50       17/41   
a3:c5       18/47       27/45        5/38       46/68       33/51       17/44   
c3:f6        3/40       23/60        4/49       22/55       42/75       10/44   
h3:f5       12/41       27/58        7/51       25/65       43/58       12/42   
g7:d4        0/47        1/53        0/35        4/53        0/57        0/46   
b7:d5        0/43       47/70        0/47       33/57        3/52        0/39   
e2:g4        4/40       25/85        2/56       29/64       23/59       18/52   
a3:d5       16/36       47/73        8/49       36/62       26/41       13/33   
h6:e4        0/35        5/61        0/51       31/73        1/59        0/46   
c2:e5       10/49       19/63        6/51       36/64       33/70       14/39   
f3:d6        1/48        7/59        7/44       12/74       14/36        6/44   
b4:e6       18/47       49/74        6/49       41/55       38/50       18/44   
g6:e3        0/52        0/46        0/50       11/60        0/64        0/54
```

## diag_addendum_tables_output.txt (tables X1 to X6)

```
X1. PIECE GEOMETRY (measured from assets/scene.xml with MuJoCo: world z extent of piece_a's collision geoms above the piece origin, at the home keyframe)
   piece_a_base   z from   0.0 to  18.0 mm above the piece origin
   piece_a_stem   z from  18.0 to  34.0 mm above the piece origin
   piece_a_head   z from  30.0 to  50.0 mm above the piece origin
   piece height 50.0 mm (origin at the base); base diameter 30 mm; board square pitch 78 mm
   bodies named piece*: ['piece_a'] (the scene holds ONE piece); n_obstacles in the six configs: [0]
   `lift` (eval_probe max_lift_mm) = piece ORIGIN z minus its resting z at the home keyframe (pickup_geometry in t1_pick_and_place.py). It is the rise of the piece base.
   A piece of the same height standing on a neighbouring square would have its top 50 mm above the board, so a carried piece clears it only if its BASE is lifted by more than 50 mm
   (more with margin). This task has no neighbouring piece (one piece, n_obstacles 0); the comparison is for scale only.

X2. MAXIMUM PIECE LIFT (mm) PER EPISODE, PLACED EPISODES ONLY, full 2400-episode eval
seed         placed  median    p90    p99    max         >10mm   >30mm   >50mm
seed0          1074     7.1   12.1   21.3   36.2    213 =19.8%       2       0
seed100        1704     5.6   10.0   21.3   32.7    174 =10.2%       3       0
seed200        1029     6.6   11.5   26.0   35.7    157 =15.3%       5       0
seed300        1773     4.8    9.2   23.5   34.1    145 = 8.2%       3       0
seed400        1720     5.8    9.3   20.3   30.2    112 = 6.5%       1       0
seed500        1091     6.5   11.7   23.0   29.0    182 =16.7%       0       0
   same, FAILED episodes (for scale):
   seed0       n=1326  median  19.8  p90  26.2  max 125.9  >50mm: 15
   seed100     n=696   median  21.9  p90  28.1  max  78.3  >50mm: 4
   seed200     n=1371  median  21.0  p90  27.4  max  62.6  >50mm: 11
   seed300     n=627   median  21.8  p90  25.8  max  54.1  >50mm: 1
   seed400     n=680   median  10.7  p90  25.6  max  51.9  >50mm: 2
   seed500     n=1309  median  17.7  p90  25.2  max  66.2  >50mm: 3

X3. PAIR e4:e5 ONLY (50 episodes per seed in the full eval): placed episodes whose max piece lift exceeded 10 mm
seed          placed  lift>10mm   share  median max lift (placed)    p90    max
seed0          43/50          9     21%                       8.2   11.2   15.4
seed100        44/50          2      5%                       5.3    9.1   21.4
seed200        32/50          1      3%                       5.6    8.3   11.2
seed300        41/50          3      7%                       3.9    7.2   12.9
seed400        47/50          0      0%                       5.4    7.8    9.6
seed500        46/50          4      9%                       5.6    9.4   14.0

X4. LONG PAIRS (18) BELOW 25/50 AND BELOW 10/50, per seed (placed k/50 in the full eval)
seed          long <10/50  long <25/50   long pairs below 25/50 (k/50)
seed0                   9           17   b2:b4 22, g3:g6 2, d7:d4 0, a5:c5 9, h4:e4 14, b6:e6 16, a3:c5 20, c3:f6 10, h3:f5 12, g7:d4 0, b7:d5 0, e2:g4 7, a3:d5 18, h6:e4 0, c2:e5 13, f3:d6 6, g6:e3 0
seed100                 4           10   g3:g6 24, d7:d4 22, h4:e4 19, c3:f6 19, g7:d4 0, e2:g4 23, h6:e4 2, c2:e5 19, f3:d6 8, g6:e3 5
seed200                15           18   b2:b4 11, g3:g6 6, d7:d4 0, a5:c5 19, h4:e4 1, b6:e6 5, a3:c5 7, c3:f6 6, h3:f5 6, g7:d4 0, b7:d5 0, e2:g4 4, a3:d5 10, h6:e4 0, c2:e5 6, f3:d6 8, b4:e6 3, g6:e3 0
seed300                 3            8   g3:g6 24, c3:f6 21, h3:f5 20, g7:d4 7, h6:e4 18, c2:e5 24, f3:d6 9, g6:e3 9
seed400                 5            6   d7:d4 0, g7:d4 0, b7:d5 2, h6:e4 0, f3:d6 23, g6:e3 0
seed500                 6           17   b2:b4 13, g3:g6 8, d7:d4 0, h4:e4 21, b6:e6 18, a3:c5 21, c3:f6 12, h3:f5 23, g7:d4 0, b7:d5 0, e2:g4 20, a3:d5 22, h6:e4 0, c2:e5 16, f3:d6 11, b4:e6 24, g6:e3 0

X5. WAS TRAINING PLACEMENT STILL RISING AT THE END OF TRAINING?  (Monitor terminated_reason == success, pooled over the 4 files)
    100k bins as in ../section3_train_output.txt (25k steps of each file; bin j = training steps (j-1)*100k+1..j*100k); the partial bin after 5.0M is excluded.
    slope = weighted least-squares slope of the bin placement rates against bin index, in percentage points per 100k steps, with a nominal binomial SE
    (episodes of one evolving policy are not independent, so the SE is only nominal). 'rising' = slope - 1.96 SE > 0; 'falling' = slope + 1.96 SE < 0; else 'no detectable change'.
    diff = placement in 4.5-5.0M minus 4.0-4.5M (episode counts pooled), points, with an approximate 95% interval (unpooled normal).
seed                                                          last10 bins (4.1-5.0M) %
seed0                       25.6  30.6  31.3  35.8  39.1  41.5  41.1  42.3  45.6  42.9
seed100                     67.3  65.9  62.8  68.6  69.1  72.0  71.7  72.1  74.0  71.7
seed200                     41.6  37.9  37.9  39.4  40.4  41.6  38.7  41.5  40.6  45.8
seed300                     69.0  68.8  68.1  68.8  71.2  71.8  75.7  75.1  74.3  71.4
seed400                     25.7  39.2  54.0  59.8  62.5  64.1  65.1  69.9  70.3  70.7
seed500                     25.7  28.2  27.2  31.6  37.7  42.2  43.0  43.3  44.7  42.7

seed         4.0-4.5M  4.5-5.0M        diff pts [95%]   slope last10 (pts/100k)           slope last5   verdicts (diff / last10 slope / last5 slope)
seed0           32.6%     42.7%    +10.1 [+7.1,+13.0]         +2.00 +/- 0.50       +0.73 +/- 1.48   rising / rising / no detectable change
seed100         66.8%     72.3%     +5.5 [+3.2,+7.8]         +0.93 +/- 0.41       +0.16 +/- 1.12   rising / rising / no detectable change
seed200         39.4%     41.6%     +2.2 [-0.7,+5.1]         +0.45 +/- 0.51       +1.05 +/- 1.46   no detectable change / no detectable change / no detectable change
seed300         69.2%     73.7%     +4.5 [+2.2,+6.8]         +0.71 +/- 0.41       -0.23 +/- 1.15   rising / rising / no detectable change
seed400         49.7%     68.1%    +18.4 [+15.7,+21.1]         +4.13 +/- 0.46       +1.84 +/- 1.26   rising / rising / rising
seed500         30.2%     43.2%    +13.0 [+10.1,+15.9]         +2.33 +/- 0.50       +0.27 +/- 1.47   rising / rising / no detectable change
    Learning rate at 4.0M, 4.5M, 5.0M steps (linear decay 3e-4 to 0 over 5,000,000 steps): 6.0e-05, 3.0e-05, 0.0e+00

X6. PAIR ORDER CHECK against PREREGISTRATION.md (every file that carries the 48 pairs; order and content)
   MATCH    PREREGISTRATION.md 'Training pairs' line, parsed here as whitespace tokens vs rs32_lib.PAIRS (regex parse)
   MATCH    ../EVAL_MANIFEST.json  pairs
   MATCH    ../EVAL_MANIFEST.json  --pairs in the command of rs32D_seed0
   MATCH    ../EVAL_MANIFEST.json  --pairs in the command of rs32D_seed100
   MATCH    ../EVAL_MANIFEST.json  --pairs in the command of rs32D_seed200
   MATCH    ../EVAL_MANIFEST.json  --pairs in the command of rs32D_seed300
   MATCH    ../EVAL_MANIFEST.json  --pairs in the command of rs32D_seed400
   MATCH    ../EVAL_MANIFEST.json  --pairs in the command of rs32D_seed500
   MATCH    ../EVAL_MANIFEST_section3.json  pairs48
   MATCH    ../EVAL_MANIFEST_section3.json  --pairs in the command of det48_rs32D_seed0
   MATCH    ../EVAL_MANIFEST_section3.json  --pairs in the command of det48_rs32D_seed100
   MATCH    ../EVAL_MANIFEST_section3.json  --pairs in the command of det48_rs32D_seed200
   MATCH    ../EVAL_MANIFEST_section3.json  --pairs in the command of det48_rs32D_seed300
   MATCH    ../EVAL_MANIFEST_section3.json  --pairs in the command of det48_rs32D_seed400
   MATCH    ../EVAL_MANIFEST_section3.json  --pairs in the command of det48_rs32D_seed500
   MATCH    ../rs32D_seed0_final.json  episode i is pair ref[i mod 48] for all 2400 episodes
   MATCH    ../det48_rs32D_seed0.json  episode i is pair ref[i mod 48] for all 96 episodes
   MATCH    ../rs32D_seed100_final.json  episode i is pair ref[i mod 48] for all 2400 episodes
   MATCH    ../det48_rs32D_seed100.json  episode i is pair ref[i mod 48] for all 96 episodes
   MATCH    ../rs32D_seed200_final.json  episode i is pair ref[i mod 48] for all 2400 episodes
   MATCH    ../det48_rs32D_seed200.json  episode i is pair ref[i mod 48] for all 96 episodes
   MATCH    ../rs32D_seed300_final.json  episode i is pair ref[i mod 48] for all 2400 episodes
   MATCH    ../det48_rs32D_seed300.json  episode i is pair ref[i mod 48] for all 96 episodes
   MATCH    ../rs32D_seed400_final.json  episode i is pair ref[i mod 48] for all 2400 episodes
   MATCH    ../det48_rs32D_seed400.json  episode i is pair ref[i mod 48] for all 96 episodes
   MATCH    ../rs32D_seed500_final.json  episode i is pair ref[i mod 48] for all 2400 episodes
   MATCH    ../det48_rs32D_seed500.json  episode i is pair ref[i mod 48] for all 96 episodes
   MATCH    configs/_probe_cart_rs32D_ppotanh_seed0.yaml  train_pairs
   MATCH    runs/_probe_cart_rs32D_ppotanh_seed0/manifest.json  config_resolved.train_pairs (what the run used)
   MATCH    configs/_probe_cart_rs32D_ppotanh_seed100.yaml  train_pairs
   MATCH    runs/_probe_cart_rs32D_ppotanh_seed100/manifest.json  config_resolved.train_pairs (what the run used)
   MATCH    configs/_probe_cart_rs32D_ppotanh_seed200.yaml  train_pairs
   MATCH    runs/_probe_cart_rs32D_ppotanh_seed200/manifest.json  config_resolved.train_pairs (what the run used)
   MATCH    configs/_probe_cart_rs32D_ppotanh_seed300.yaml  train_pairs
   MATCH    runs/_probe_cart_rs32D_ppotanh_seed300/manifest.json  config_resolved.train_pairs (what the run used)
   MATCH    configs/_probe_cart_rs32D_ppotanh_seed400.yaml  train_pairs
   MATCH    runs/_probe_cart_rs32D_ppotanh_seed400/manifest.json  config_resolved.train_pairs (what the run used)
   MATCH    configs/_probe_cart_rs32D_ppotanh_seed500.yaml  train_pairs
   MATCH    runs/_probe_cart_rs32D_ppotanh_seed500/manifest.json  config_resolved.train_pairs (what the run used)
   MATCH    results/rs32_48_pairs/expert_gate/TABLE.md  first 48 rows
        (that table has 54 rows: 48 pairs + 6 reserves; reserves not in the 48: True)
   MATCH    the (start, target) pairs that occur in the six runs' Monitor CSVs are exactly the 48 pairs (48 distinct; episodes per pair min 2417 max 2664, all six runs pooled)
   Note: training draws one pair uniformly at random per episode, so the ORDER of train_pairs does not affect training; order matters for the eval (episode i uses pair i mod 48) and for the tables.
```
