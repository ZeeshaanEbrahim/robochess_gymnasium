# RS32 evaluation: PPO (D recipe), 48 start-target pairs, 6 seeds, 5M steps

Evaluation of the final models of `_probe_cart_rs32D_ppotanh_seed{0,100,200,300,400,500}` under `results/rs32_48_pairs/PREREGISTRATION.md`
(rules and thresholds unchanged; config, training code and pre-registration untouched). Numbers below are copied from the output files in this
folder, which are produced by the scripts in this folder from the eval JSONs. Reported, not interpreted.

## 1. Rule verdicts (mechanical)

Rule 1: a seed learns if pooled placement >= 1200/2400 AND >= 10/50 on every one of the 48 pairs.

| model | pooled | placement, Wilson 95% | pairs under 10/50 | pooled >= 1200 | no pair under 10/50 | Rule 1: learns | target tier: pooled >= 1920 |
|---|---|---|---|---|---|---|---|
| rs32D_seed0 | 1074/2400 | 44.8% [42.8, 46.7] | 12 | no | no | **no** | no |
| rs32D_seed100 | 1704/2400 | 71.0% [69.2, 72.8] | 4 | yes | no | **no** | no |
| rs32D_seed200 | 1029/2400 | 42.9% [40.9, 44.9] | 17 | no | no | **no** | no |
| rs32D_seed300 | 1773/2400 | 73.9% [72.1, 75.6] | 3 | yes | no | **no** | no |
| rs32D_seed400 | 1720/2400 | 71.7% [69.8, 73.4] | 5 | yes | no | **no** | no |
| rs32D_seed500 | 1091/2400 | 45.5% [43.5, 47.5] | 10 | no | no | **no** | no |

- Rule 2: seeds that learn = **0 of 6**.
- Rule 3: the claim "PPO with the D recipe learns 48 mixed pairs" holds if at least 4 of 6 seeds learn. 0 of 6 learn, so the claim is **not achieved**.
  Rule 3 asks to report which subset falls short; the pairs under 10/50 split as follows (`section2_subsets_output.txt`):
```
pairs under 10/50 split by subset (30 neighbour = first 30 pairs in PREREGISTRATION.md order, 18 longer = last 18)
  rs32D_seed0: neighbour 3/30 ['f7:f8', 'e5:e4', 'c6:c5']   long 9/18 ['g3:g6', 'd7:d4', 'a5:c5', 'g7:d4', 'b7:d5', 'e2:g4', 'h6:e4', 'f3:d6', 'g6:e3']
  rs32D_seed100: neighbour 0/30 []   long 4/18 ['g7:d4', 'h6:e4', 'f3:d6', 'g6:e3']
  rs32D_seed200: neighbour 2/30 ['e5:e4', 'c6:c5']   long 15/18 ['g3:g6', 'd7:d4', 'h4:e4', 'b6:e6', 'a3:c5', 'c3:f6', 'h3:f5', 'g7:d4', 'b7:d5', 'e2:g4', 'h6:e4', 'c2:e5', 'f3:d6', 'b4:e6', 'g6:e3']
  rs32D_seed300: neighbour 0/30 []   long 3/18 ['g7:d4', 'f3:d6', 'g6:e3']
  rs32D_seed400: neighbour 0/30 []   long 5/18 ['d7:d4', 'g7:d4', 'b7:d5', 'h6:e4', 'g6:e3']
  rs32D_seed500: neighbour 4/30 ['c7:c8', 'd7:d8', 'e5:e4', 'c6:c5']   long 6/18 ['g3:g6', 'd7:d4', 'g7:d4', 'b7:d5', 'h6:e4', 'g6:e3']
```
- Rule 4 (target tier, reported): pooled >= 1920/2400 on **0 of 6** seeds.

## 2. Provenance

- Git at start (EVAL_MANIFEST.json): HEAD `1767ca540ad090879866ccb6d6ca18954f870254`, branch `rs32`, tracked-dirty files 0; untracked: only `results/rs32_48_pairs/eval/` (this folder).
- Gate: all six `runs/_probe_cart_rs32D_ppotanh_seed*_train.log` contain EXIT=0; no python `scripts/train.py` process was running (the only `train.py` match was a `tmux new-session` wrapper of the rs31C run, ignored as instructed). No process was killed.
- Final model step counts (`num_timesteps` in `final_model.zip`): 0: 5,005,312, 100: 5,005,312, 200: 5,005,312, 300: 5,005,312, 400: 5,005,312, 500: 5,005,312 (expected 5,005,312: all as expected).
- Method as in RS31: `scripts/eval_probe.py` run unmodified through `rms_wrapper.py` (copied from `results/rs31_24_pairs/eval/`), stochastic policy, `--seed 20000`, `--episodes 2400` with the 48 pairs in PREREGISTRATION.md order (parsed from the file, 50 per pair, episode i uses pair i mod 48; asserted), final model + `final_vecnormalize.pkl` (frozen), placement = env `is_success`.
- obs_rms and ret_rms identical before and after for all six, and the pickles' sha256 unchanged (`section2_output.txt`). For the 18 Step 3 jobs `section3_eval.py` asserts the same before/after identity (the pickle sha256 check is stored in each `*_rms.json` but not asserted).
- Commands: `EVAL_MANIFEST.json` (`cmd` and `equivalent_eval_probe_cmd` per model), sha256 of models, VecNormalize files, `eval_probe.py` and the pre-registration; Step 3 in `EVAL_MANIFEST_section3.json`.
- Timing (2026-10-07, UTC+2), at most 3 eval processes at once, MemAvailable was 4.3 to 7.2 GB at the checks and never under 1.5 GB (no holds recorded in the main manifest):
- rs32D_seed0: 12:12:51 to 12:51:11, exit 0
- rs32D_seed100: 12:12:51 to 12:39:41, exit 0
- rs32D_seed200: 12:12:51 to 12:50:11, exit 0
- rs32D_seed300: 12:39:41 to 13:06:51, exit 0
- rs32D_seed400: 12:50:11 to 13:19:51, exit 0
- rs32D_seed500: 12:51:11 to 13:27:11, exit 0
- Step 3 evals: 15:34:13 to 16:01:14, 18 jobs, exits [0]. The session was paused between the two parts (laptop shut down) and resumed; no eval was running during the pause.
- Scripts: `run_evals.py` (Step 1), `rms_wrapper.py`, `section2_table.py`, `section2_subsets.py`, `section3_train.py`, `run_section3.py`, `section3_eval.py`, `rs32_lib.py`, `make_readme.py`.

## 3. Step 2: main results (`section2_output.txt`, verbatim)

Funnel columns: "lift over 10 mm" is eval_probe's `grasp_ever` (undercounts); held = both pads on piece and no piece-board contact; released = S2 reached; placed = env `is_success`.
`lifted_30mm` and `carried_within_3cm` are not used (known unreliable).

```
EVAL command (one per model; wrapper runs scripts/eval_probe.py unmodified, same argv):
  rs32D_seed0: /root/venvs/robochess/bin/python results/rs32_48_pairs/eval/rms_wrapper.py _probe_cart_rs32D_ppotanh_seed0 --episodes 2400 --pairs c2:c3,c3:c4,c4:c5,c5:c6,c6:c7,c7:c8,d2:d3,d3:d4,d4:d5,d5:d6,d6:d7,d7:d8,e2:e3,e3:e4,e4:e5,e5:e6,e6:e7,e7:e8,f2:f3,f3:f4,f4:f5,f5:f6,f6:f7,f7:f8,c4:d4,f5:e5,d3:e4,e6:d7,e5:e4,c6:c5,b2:b4,g3:g6,d7:d4,a5:c5,h4:e4,b6:e6,a3:c5,c3:f6,h3:f5,g7:d4,b7:d5,e2:g4,a3:d5,h6:e4,c2:e5,f3:d6,b4:e6,g6:e3 --seed 20000 --out results/rs32_48_pairs/eval/rs32D_seed0_final.json
  rs32D_seed100: /root/venvs/robochess/bin/python results/rs32_48_pairs/eval/rms_wrapper.py _probe_cart_rs32D_ppotanh_seed100 --episodes 2400 --pairs c2:c3,c3:c4,c4:c5,c5:c6,c6:c7,c7:c8,d2:d3,d3:d4,d4:d5,d5:d6,d6:d7,d7:d8,e2:e3,e3:e4,e4:e5,e5:e6,e6:e7,e7:e8,f2:f3,f3:f4,f4:f5,f5:f6,f6:f7,f7:f8,c4:d4,f5:e5,d3:e4,e6:d7,e5:e4,c6:c5,b2:b4,g3:g6,d7:d4,a5:c5,h4:e4,b6:e6,a3:c5,c3:f6,h3:f5,g7:d4,b7:d5,e2:g4,a3:d5,h6:e4,c2:e5,f3:d6,b4:e6,g6:e3 --seed 20000 --out results/rs32_48_pairs/eval/rs32D_seed100_final.json
  rs32D_seed200: /root/venvs/robochess/bin/python results/rs32_48_pairs/eval/rms_wrapper.py _probe_cart_rs32D_ppotanh_seed200 --episodes 2400 --pairs c2:c3,c3:c4,c4:c5,c5:c6,c6:c7,c7:c8,d2:d3,d3:d4,d4:d5,d5:d6,d6:d7,d7:d8,e2:e3,e3:e4,e4:e5,e5:e6,e6:e7,e7:e8,f2:f3,f3:f4,f4:f5,f5:f6,f6:f7,f7:f8,c4:d4,f5:e5,d3:e4,e6:d7,e5:e4,c6:c5,b2:b4,g3:g6,d7:d4,a5:c5,h4:e4,b6:e6,a3:c5,c3:f6,h3:f5,g7:d4,b7:d5,e2:g4,a3:d5,h6:e4,c2:e5,f3:d6,b4:e6,g6:e3 --seed 20000 --out results/rs32_48_pairs/eval/rs32D_seed200_final.json
  rs32D_seed300: /root/venvs/robochess/bin/python results/rs32_48_pairs/eval/rms_wrapper.py _probe_cart_rs32D_ppotanh_seed300 --episodes 2400 --pairs c2:c3,c3:c4,c4:c5,c5:c6,c6:c7,c7:c8,d2:d3,d3:d4,d4:d5,d5:d6,d6:d7,d7:d8,e2:e3,e3:e4,e4:e5,e5:e6,e6:e7,e7:e8,f2:f3,f3:f4,f4:f5,f5:f6,f6:f7,f7:f8,c4:d4,f5:e5,d3:e4,e6:d7,e5:e4,c6:c5,b2:b4,g3:g6,d7:d4,a5:c5,h4:e4,b6:e6,a3:c5,c3:f6,h3:f5,g7:d4,b7:d5,e2:g4,a3:d5,h6:e4,c2:e5,f3:d6,b4:e6,g6:e3 --seed 20000 --out results/rs32_48_pairs/eval/rs32D_seed300_final.json
  rs32D_seed400: /root/venvs/robochess/bin/python results/rs32_48_pairs/eval/rms_wrapper.py _probe_cart_rs32D_ppotanh_seed400 --episodes 2400 --pairs c2:c3,c3:c4,c4:c5,c5:c6,c6:c7,c7:c8,d2:d3,d3:d4,d4:d5,d5:d6,d6:d7,d7:d8,e2:e3,e3:e4,e4:e5,e5:e6,e6:e7,e7:e8,f2:f3,f3:f4,f4:f5,f5:f6,f6:f7,f7:f8,c4:d4,f5:e5,d3:e4,e6:d7,e5:e4,c6:c5,b2:b4,g3:g6,d7:d4,a5:c5,h4:e4,b6:e6,a3:c5,c3:f6,h3:f5,g7:d4,b7:d5,e2:g4,a3:d5,h6:e4,c2:e5,f3:d6,b4:e6,g6:e3 --seed 20000 --out results/rs32_48_pairs/eval/rs32D_seed400_final.json
  rs32D_seed500: /root/venvs/robochess/bin/python results/rs32_48_pairs/eval/rms_wrapper.py _probe_cart_rs32D_ppotanh_seed500 --episodes 2400 --pairs c2:c3,c3:c4,c4:c5,c5:c6,c6:c7,c7:c8,d2:d3,d3:d4,d4:d5,d5:d6,d6:d7,d7:d8,e2:e3,e3:e4,e4:e5,e5:e6,e6:e7,e7:e8,f2:f3,f3:f4,f4:f5,f5:f6,f6:f7,f7:f8,c4:d4,f5:e5,d3:e4,e6:d7,e5:e4,c6:c5,b2:b4,g3:g6,d7:d4,a5:c5,h4:e4,b6:e6,a3:c5,c3:f6,h3:f5,g7:d4,b7:d5,e2:g4,a3:d5,h6:e4,c2:e5,f3:d6,b4:e6,g6:e3 --seed 20000 --out results/rs32_48_pairs/eval/rs32D_seed500_final.json

Final model step counts (num_timesteps in final_model.zip; expected 5,005,312):
  rs32D_seed0: 5,005,312  as expected
  rs32D_seed100: 5,005,312  as expected
  rs32D_seed200: 5,005,312  as expected
  rs32D_seed300: 5,005,312  as expected
  rs32D_seed400: 5,005,312  as expected
  rs32D_seed500: 5,005,312  as expected

Exit status / times (manifest):
  rs32D_seed0: exit=0 start=2026-10-07T12:12:51+02:00 end=2026-10-07T12:51:11+02:00
  rs32D_seed100: exit=0 start=2026-10-07T12:12:51+02:00 end=2026-10-07T12:39:41+02:00
  rs32D_seed200: exit=0 start=2026-10-07T12:12:51+02:00 end=2026-10-07T12:50:11+02:00
  rs32D_seed300: exit=0 start=2026-10-07T12:39:41+02:00 end=2026-10-07T13:06:51+02:00
  rs32D_seed400: exit=0 start=2026-10-07T12:50:11+02:00 end=2026-10-07T13:19:51+02:00
  rs32D_seed500: exit=0 start=2026-10-07T12:51:11+02:00 end=2026-10-07T13:27:11+02:00

VecNormalize frozen check (before = pkl read before eval; after = loaded venv after rollout; keys: obs_rms/observation, ret_rms)
  rs32D_seed0: identical before/after (obs_rms + ret_rms, mean/var/count) = yes; eval_probe own obs_rms flag = True; pkl sha256 unchanged = True
  rs32D_seed100: identical before/after (obs_rms + ret_rms, mean/var/count) = yes; eval_probe own obs_rms flag = True; pkl sha256 unchanged = True
  rs32D_seed200: identical before/after (obs_rms + ret_rms, mean/var/count) = yes; eval_probe own obs_rms flag = True; pkl sha256 unchanged = True
  rs32D_seed300: identical before/after (obs_rms + ret_rms, mean/var/count) = yes; eval_probe own obs_rms flag = True; pkl sha256 unchanged = True
  rs32D_seed400: identical before/after (obs_rms + ret_rms, mean/var/count) = yes; eval_probe own obs_rms flag = True; pkl sha256 unchanged = True
  rs32D_seed500: identical before/after (obs_rms + ret_rms, mean/var/count) = yes; eval_probe own obs_rms flag = True; pkl sha256 unchanged = True

1. pooled placements k/2400 (Wilson 95%, %)
  rs32D_seed0: 1074/2400 = 44.8% [42.8,46.7]
  rs32D_seed100: 1704/2400 = 71.0% [69.2,72.8]
  rs32D_seed200: 1029/2400 = 42.9% [40.9,44.9]
  rs32D_seed300: 1773/2400 = 73.9% [72.1,75.6]
  rs32D_seed400: 1720/2400 = 71.7% [69.8,73.4]
  rs32D_seed500: 1091/2400 = 45.5% [43.5,47.5]

2. per pair k/50 [Wilson 95%, %], preregistration order
  pair                rs32D_seed0           rs32D_seed100           rs32D_seed200           rs32D_seed300           rs32D_seed400           rs32D_seed500
  c2:c3       31/50 [ 48.2, 74.1]     41/50 [ 69.2, 90.2]     34/50 [ 54.2, 79.2]     40/50 [ 67.0, 88.8]     38/50 [ 62.6, 85.7]     26/50 [ 38.5, 65.2]
  c3:c4       37/50 [ 60.4, 84.1]     47/50 [ 83.8, 97.9]     40/50 [ 67.0, 88.8]     45/50 [ 78.6, 95.7]     46/50 [ 81.2, 96.8]     41/50 [ 69.2, 90.2]
  c4:c5       43/50 [ 73.8, 93.0]     40/50 [ 67.0, 88.8]     27/50 [ 40.4, 67.0]     45/50 [ 78.6, 95.7]     48/50 [ 86.5, 98.9]     40/50 [ 67.0, 88.8]
  c5:c6       40/50 [ 67.0, 88.8]     43/50 [ 73.8, 93.0]     31/50 [ 48.2, 74.1]     47/50 [ 83.8, 97.9]     47/50 [ 83.8, 97.9]     29/50 [ 44.2, 70.6]
  c6:c7       35/50 [ 56.2, 80.9]     44/50 [ 76.2, 94.4]     37/50 [ 60.4, 84.1]     44/50 [ 76.2, 94.4]     46/50 [ 81.2, 96.8]     11/50 [ 12.8, 35.2]
  c7:c8       32/50 [ 50.1, 75.9]     44/50 [ 76.2, 94.4]     34/50 [ 54.2, 79.2]     48/50 [ 86.5, 98.9]     42/50 [ 71.5, 91.7]      1/50 [  0.4, 10.5]
  d2:d3       31/50 [ 48.2, 74.1]     48/50 [ 86.5, 98.9]     30/50 [ 46.2, 72.4]     46/50 [ 81.2, 96.8]     44/50 [ 76.2, 94.4]     34/50 [ 54.2, 79.2]
  d3:d4       36/50 [ 58.3, 82.5]     43/50 [ 73.8, 93.0]     38/50 [ 62.6, 85.7]     45/50 [ 78.6, 95.7]     46/50 [ 81.2, 96.8]     43/50 [ 73.8, 93.0]
  d4:d5       41/50 [ 69.2, 90.2]     47/50 [ 83.8, 97.9]     37/50 [ 60.4, 84.1]     46/50 [ 81.2, 96.8]     45/50 [ 78.6, 95.7]     48/50 [ 86.5, 98.9]
  d5:d6       39/50 [ 64.8, 87.2]     47/50 [ 83.8, 97.9]     38/50 [ 62.6, 85.7]     40/50 [ 67.0, 88.8]     46/50 [ 81.2, 96.8]     32/50 [ 50.1, 75.9]
  d6:d7       37/50 [ 60.4, 84.1]     48/50 [ 86.5, 98.9]     38/50 [ 62.6, 85.7]     47/50 [ 83.8, 97.9]     43/50 [ 73.8, 93.0]     16/50 [ 20.8, 45.8]
  d7:d8       22/50 [ 31.2, 57.7]     47/50 [ 83.8, 97.9]     33/50 [ 52.2, 77.6]     43/50 [ 73.8, 93.0]     43/50 [ 73.8, 93.0]      1/50 [  0.4, 10.5]
  e2:e3       27/50 [ 40.4, 67.0]     39/50 [ 64.8, 87.2]     25/50 [ 36.6, 63.4]     40/50 [ 67.0, 88.8]     42/50 [ 71.5, 91.7]     35/50 [ 56.2, 80.9]
  e3:e4       36/50 [ 58.3, 82.5]     42/50 [ 71.5, 91.7]     29/50 [ 44.2, 70.6]     45/50 [ 78.6, 95.7]     37/50 [ 60.4, 84.1]     42/50 [ 71.5, 91.7]
  e4:e5       43/50 [ 73.8, 93.0]     44/50 [ 76.2, 94.4]     32/50 [ 50.1, 75.9]     41/50 [ 69.2, 90.2]     47/50 [ 83.8, 97.9]     46/50 [ 81.2, 96.8]
  e5:e6       44/50 [ 76.2, 94.4]     41/50 [ 69.2, 90.2]     41/50 [ 69.2, 90.2]     44/50 [ 76.2, 94.4]     48/50 [ 86.5, 98.9]     33/50 [ 52.2, 77.6]
  e6:e7       36/50 [ 58.3, 82.5]     44/50 [ 76.2, 94.4]     39/50 [ 64.8, 87.2]     47/50 [ 83.8, 97.9]     45/50 [ 78.6, 95.7]     20/50 [ 27.6, 53.8]
  e7:e8       18/50 [ 24.1, 49.9]     46/50 [ 81.2, 96.8]     39/50 [ 64.8, 87.2]     44/50 [ 76.2, 94.4]     46/50 [ 81.2, 96.8]     13/50 [ 15.9, 39.6]
  f2:f3       25/50 [ 36.6, 63.4]     35/50 [ 56.2, 80.9]     23/50 [ 33.0, 59.6]     36/50 [ 58.3, 82.5]     47/50 [ 83.8, 97.9]     37/50 [ 60.4, 84.1]
  f3:f4       30/50 [ 46.2, 72.4]     42/50 [ 71.5, 91.7]     36/50 [ 58.3, 82.5]     42/50 [ 71.5, 91.7]     44/50 [ 76.2, 94.4]     42/50 [ 71.5, 91.7]
  f4:f5       41/50 [ 69.2, 90.2]     44/50 [ 76.2, 94.4]     40/50 [ 67.0, 88.8]     45/50 [ 78.6, 95.7]     43/50 [ 73.8, 93.0]     42/50 [ 71.5, 91.7]
  f5:f6       35/50 [ 56.2, 80.9]     40/50 [ 67.0, 88.8]     43/50 [ 73.8, 93.0]     41/50 [ 69.2, 90.2]     47/50 [ 83.8, 97.9]     41/50 [ 69.2, 90.2]
  f6:f7       21/50 [ 29.4, 55.8]     41/50 [ 69.2, 90.2]     40/50 [ 67.0, 88.8]     46/50 [ 81.2, 96.8]     48/50 [ 86.5, 98.9]     32/50 [ 50.1, 75.9]
  f7:f8        6/50 [  5.6, 23.8]     43/50 [ 73.8, 93.0]     36/50 [ 58.3, 82.5]     41/50 [ 69.2, 90.2]     46/50 [ 81.2, 96.8]     19/50 [ 25.9, 51.8]
  c4:d4       13/50 [ 15.9, 39.6]     46/50 [ 81.2, 96.8]     23/50 [ 33.0, 59.6]     45/50 [ 78.6, 95.7]     44/50 [ 76.2, 94.4]     17/50 [ 22.4, 47.8]
  f5:e5       35/50 [ 56.2, 80.9]     47/50 [ 83.8, 97.9]     10/50 [ 11.2, 33.0]     48/50 [ 86.5, 98.9]     50/50 [ 92.9,100.0]     44/50 [ 76.2, 94.4]
  d3:e4       32/50 [ 50.1, 75.9]     42/50 [ 71.5, 91.7]     32/50 [ 50.1, 75.9]     40/50 [ 67.0, 88.8]     42/50 [ 71.5, 91.7]     33/50 [ 52.2, 77.6]
  e6:d7       33/50 [ 52.2, 77.6]     41/50 [ 69.2, 90.2]     32/50 [ 50.1, 75.9]     39/50 [ 64.8, 87.2]     40/50 [ 67.0, 88.8]     19/50 [ 25.9, 51.8]
  e5:e4        0/50 [  0.0,  7.1]     43/50 [ 73.8, 93.0]      0/50 [  0.0,  7.1]     42/50 [ 71.5, 91.7]     16/50 [ 20.8, 45.8]      0/50 [  0.0,  7.1]
  c6:c5        1/50 [  0.4, 10.5]     46/50 [ 81.2, 96.8]      0/50 [  0.0,  7.1]     41/50 [ 69.2, 90.2]     41/50 [ 69.2, 90.2]      8/50 [  8.3, 28.5]
  b2:b4       22/50 [ 31.2, 57.7]     33/50 [ 52.2, 77.6]     11/50 [ 12.8, 35.2]     39/50 [ 64.8, 87.2]     30/50 [ 46.2, 72.4]     13/50 [ 15.9, 39.6]
  g3:g6        2/50 [  1.1, 13.5]     24/50 [ 34.8, 61.5]      6/50 [  5.6, 23.8]     24/50 [ 34.8, 61.5]     27/50 [ 40.4, 67.0]      8/50 [  8.3, 28.5]
  d7:d4        0/50 [  0.0,  7.1]     22/50 [ 31.2, 57.7]      0/50 [  0.0,  7.1]     25/50 [ 36.6, 63.4]      0/50 [  0.0,  7.1]      0/50 [  0.0,  7.1]
  a5:c5        9/50 [  9.8, 30.8]     44/50 [ 76.2, 94.4]     19/50 [ 25.9, 51.8]     44/50 [ 76.2, 94.4]     41/50 [ 69.2, 90.2]     37/50 [ 60.4, 84.1]
  h4:e4       14/50 [ 17.5, 41.7]     19/50 [ 25.9, 51.8]      1/50 [  0.4, 10.5]     32/50 [ 50.1, 75.9]     30/50 [ 46.2, 72.4]     21/50 [ 29.4, 55.8]
  b6:e6       16/50 [ 20.8, 45.8]     36/50 [ 58.3, 82.5]      5/50 [  4.3, 21.4]     35/50 [ 56.2, 80.9]     38/50 [ 62.6, 85.7]     18/50 [ 24.1, 49.9]
  a3:c5       20/50 [ 27.6, 53.8]     28/50 [ 42.3, 68.8]      7/50 [  7.0, 26.2]     37/50 [ 60.4, 84.1]     35/50 [ 56.2, 80.9]     21/50 [ 29.4, 55.8]
  c3:f6       10/50 [ 11.2, 33.0]     19/50 [ 25.9, 51.8]      6/50 [  5.6, 23.8]     21/50 [ 29.4, 55.8]     32/50 [ 50.1, 75.9]     12/50 [ 14.3, 37.4]
  h3:f5       12/50 [ 14.3, 37.4]     29/50 [ 44.2, 70.6]      6/50 [  5.6, 23.8]     20/50 [ 27.6, 53.8]     33/50 [ 52.2, 77.6]     23/50 [ 33.0, 59.6]
  g7:d4        0/50 [  0.0,  7.1]      0/50 [  0.0,  7.1]      0/50 [  0.0,  7.1]      7/50 [  7.0, 26.2]      0/50 [  0.0,  7.1]      0/50 [  0.0,  7.1]
  b7:d5        0/50 [  0.0,  7.1]     31/50 [ 48.2, 74.1]      0/50 [  0.0,  7.1]     31/50 [ 48.2, 74.1]      2/50 [  1.1, 13.5]      0/50 [  0.0,  7.1]
  e2:g4        7/50 [  7.0, 26.2]     23/50 [ 33.0, 59.6]      4/50 [  3.2, 18.8]     26/50 [ 38.5, 65.2]     26/50 [ 38.5, 65.2]     20/50 [ 27.6, 53.8]
  a3:d5       18/50 [ 24.1, 49.9]     29/50 [ 44.2, 70.6]     10/50 [ 11.2, 33.0]     35/50 [ 56.2, 80.9]     36/50 [ 58.3, 82.5]     22/50 [ 31.2, 57.7]
  h6:e4        0/50 [  0.0,  7.1]      2/50 [  1.1, 13.5]      0/50 [  0.0,  7.1]     18/50 [ 24.1, 49.9]      0/50 [  0.0,  7.1]      0/50 [  0.0,  7.1]
  c2:e5       13/50 [ 15.9, 39.6]     19/50 [ 25.9, 51.8]      6/50 [  5.6, 23.8]     24/50 [ 34.8, 61.5]     27/50 [ 40.4, 67.0]     16/50 [ 20.8, 45.8]
  f3:d6        6/50 [  5.6, 23.8]      8/50 [  8.3, 28.5]      8/50 [  8.3, 28.5]      9/50 [  9.8, 30.8]     23/50 [ 33.0, 59.6]     11/50 [ 12.8, 35.2]
  b4:e6       25/50 [ 36.6, 63.4]     28/50 [ 42.3, 68.8]      3/50 [  2.1, 16.2]     34/50 [ 54.2, 79.2]     33/50 [ 52.2, 77.6]     24/50 [ 34.8, 61.5]
  g6:e3        0/50 [  0.0,  7.1]      5/50 [  4.3, 21.4]      0/50 [  0.0,  7.1]      9/50 [  9.8, 30.8]      0/50 [  0.0,  7.1]      0/50 [  0.0,  7.1]

3. pairs below 10/50
  rs32D_seed0: 12 pairs: f7:f8 6/50, e5:e4 0/50, c6:c5 1/50, g3:g6 2/50, d7:d4 0/50, a5:c5 9/50, g7:d4 0/50, b7:d5 0/50, e2:g4 7/50, h6:e4 0/50, f3:d6 6/50, g6:e3 0/50
  rs32D_seed100: 4 pairs: g7:d4 0/50, h6:e4 2/50, f3:d6 8/50, g6:e3 5/50
  rs32D_seed200: 17 pairs: e5:e4 0/50, c6:c5 0/50, g3:g6 6/50, d7:d4 0/50, h4:e4 1/50, b6:e6 5/50, a3:c5 7/50, c3:f6 6/50, h3:f5 6/50, g7:d4 0/50, b7:d5 0/50, e2:g4 4/50, h6:e4 0/50, c2:e5 6/50, f3:d6 8/50, b4:e6 3/50, g6:e3 0/50
  rs32D_seed300: 3 pairs: g7:d4 7/50, f3:d6 9/50, g6:e3 9/50
  rs32D_seed400: 5 pairs: d7:d4 0/50, g7:d4 0/50, b7:d5 2/50, h6:e4 0/50, g6:e3 0/50
  rs32D_seed500: 10 pairs: c7:c8 1/50, d7:d8 1/50, e5:e4 0/50, c6:c5 8/50, g3:g6 8/50, d7:d4 0/50, g7:d4 0/50, b7:d5 0/50, h6:e4 0/50, g6:e3 0/50

4. terminated_reason counts (success endings vs placed checked equal by assertion)
  rs32D_seed0: {'bad_contact': 24, 'piece_off_board': 202, 'success': 1074, 'time_limit': 1100}
  rs32D_seed100: {'bad_contact': 16, 'piece_off_board': 254, 'success': 1704, 'time_limit': 426}
  rs32D_seed200: {'bad_contact': 30, 'piece_off_board': 292, 'success': 1029, 'time_limit': 1049}
  rs32D_seed300: {'bad_contact': 26, 'piece_off_board': 186, 'success': 1773, 'time_limit': 415}
  rs32D_seed400: {'bad_contact': 6, 'piece_off_board': 96, 'success': 1720, 'time_limit': 578}
  rs32D_seed500: {'bad_contact': 18, 'piece_off_board': 188, 'success': 1091, 'time_limit': 1103}

5. funnel counts /2400 (lift over 10 mm = eval_probe grasp_ever (undercounts); held = both pads on piece, no piece-board contact; released = S2 reached; placed = env is_success)
  rs32D_seed0: lift over 10 mm (undercounts) 765/2400  held 2322/2400  released 1347/2400  placed 1074/2400
  rs32D_seed100: lift over 10 mm (undercounts) 414/2400  held 2168/2400  released 1751/2400  placed 1704/2400
  rs32D_seed200: lift over 10 mm (undercounts) 686/2400  held 2231/2400  released 1194/2400  placed 1029/2400
  rs32D_seed300: lift over 10 mm (undercounts) 308/2400  held 2247/2400  released 1814/2400  placed 1773/2400
  rs32D_seed400: lift over 10 mm (undercounts) 342/2400  held 2375/2400  released 1841/2400  placed 1720/2400
  rs32D_seed500: lift over 10 mm (undercounts) 740/2400  held 2309/2400  released 1395/2400  placed 1091/2400

6. MECHANICAL RULE TABLE (PREREGISTRATION.md rules 1 to 4)
  model             pooled  pooled>=1200  pairs<10/50  RULE1 learns  target>=1920  failing conditions
  rs32D_seed0    1074/2400            no           12            no            no  pooled 1074<1200; 12 pair(s) <10/50: f7:f8, e5:e4, c6:c5, g3:g6, d7:d4, a5:c5, g7:d4, b7:d5, e2:g4, h6:e4, f3:d6, g6:e3
  rs32D_seed100  1704/2400           yes            4            no            no  4 pair(s) <10/50: g7:d4, h6:e4, f3:d6, g6:e3
  rs32D_seed200  1029/2400            no           17            no            no  pooled 1029<1200; 17 pair(s) <10/50: e5:e4, c6:c5, g3:g6, d7:d4, h4:e4, b6:e6, a3:c5, c3:f6, h3:f5, g7:d4, b7:d5, e2:g4, h6:e4, c2:e5, f3:d6, b4:e6, g6:e3
  rs32D_seed300  1773/2400           yes            3            no            no  3 pair(s) <10/50: g7:d4, f3:d6, g6:e3
  rs32D_seed400  1720/2400           yes            5            no            no  5 pair(s) <10/50: d7:d4, g7:d4, b7:d5, h6:e4, g6:e3
  rs32D_seed500  1091/2400            no           10            no            no  pooled 1091<1200; 10 pair(s) <10/50: c7:c8, d7:d8, e5:e4, c6:c5, g3:g6, d7:d4, g7:d4, b7:d5, h6:e4, g6:e3
  Rule 2: seeds that learn: 0 of 6 (none)
  Rule 3: claim 'PPO with the D recipe learns 48 mixed pairs' holds if >= 4 of 6 seeds learn: 0 of 6 -> NOT ACHIEVED
  Rule 4 (target tier, reported): pooled >= 1920/2400 per seed: rs32D_seed0 no, rs32D_seed100 no, rs32D_seed200 no, rs32D_seed300 no, rs32D_seed400 no, rs32D_seed500 no
```

## 4. Step 3 items a, e, f, g (`section3_eval_output.txt`, verbatim)

Item a (three subsets), e (zero-shot a and b; the 12 b/g pairs overlap none of the 48 trained pairs; of the 20 RS29 pairs only e4:e5 is a trained pair and is excluded from the 19-pair headline, n = 285), f (deterministic, 2 episodes per pair), g (eval vs training cross-check).
In item f all six seeds give identical trajectories for the two episodes of every pair (48/48), so the effective n is 48 pairs, not 96 episodes; the effective-n rows are the ones to use.
In item g the Wilson intervals were computed and compared in code (printed per line). Caveat on that comparison: the training episodes are not independent of each other (one evolving policy, 4 envs) and draw pairs at random rather than 50 each, so the training interval is only nominal.

```
Section-3 manifest exits: {'zs_near_rs32D_seed0': 0, 'zs_rs29pairs_rs32D_seed0': 0, 'zs_near_rs32D_seed100': 0, 'zs_rs29pairs_rs32D_seed100': 0, 'zs_near_rs32D_seed200': 0, 'zs_rs29pairs_rs32D_seed200': 0, 'zs_near_rs32D_seed300': 0, 'zs_rs29pairs_rs32D_seed300': 0, 'zs_near_rs32D_seed400': 0, 'zs_rs29pairs_rs32D_seed400': 0, 'zs_near_rs32D_seed500': 0, 'zs_rs29pairs_rs32D_seed500': 0, 'det48_rs32D_seed0': 0, 'det48_rs32D_seed100': 0, 'det48_rs32D_seed200': 0, 'det48_rs32D_seed300': 0, 'det48_rs32D_seed400': 0, 'det48_rs32D_seed500': 0}

ITEM a. success on three subsets (stochastic, final model, the 2400 Step-2 episodes; 50 per pair)
rs32D_seed0: all 48 1074/2400 = 44.8% [42.8,46.7]
   30 neighbour pairs                              900/1500 = 60.0% [57.5,62.5]
   18 long pairs (2 or 3 squares)                  174/900 = 19.3% [16.9,22.0]
   24 RS31 pairs (all neighbour, subset of the 30) 786/1200 = 65.5% [62.8,68.1]
   6 neighbour pairs not in RS31 (extra)           114/300 = 38.0% [32.7,43.6]
rs32D_seed100: all 48 1704/2400 = 71.0% [69.2,72.8]
   30 neighbour pairs                              1305/1500 = 87.0% [85.2,88.6]
   18 long pairs (2 or 3 squares)                  399/900 = 44.3% [41.1,47.6]
   24 RS31 pairs (all neighbour, subset of the 30) 1040/1200 = 86.7% [84.6,88.5]
   6 neighbour pairs not in RS31 (extra)           265/300 = 88.3% [84.2,91.5]
rs32D_seed200: all 48 1029/2400 = 42.9% [40.9,44.9]
   30 neighbour pairs                              937/1500 = 62.5% [60.0,64.9]
   18 long pairs (2 or 3 squares)                  92/900 = 10.2% [8.4,12.4]
   24 RS31 pairs (all neighbour, subset of the 30) 840/1200 = 70.0% [67.3,72.5]
   6 neighbour pairs not in RS31 (extra)           97/300 = 32.3% [27.3,37.8]
rs32D_seed300: all 48 1773/2400 = 73.9% [72.1,75.6]
   30 neighbour pairs                              1303/1500 = 86.9% [85.1,88.5]
   18 long pairs (2 or 3 squares)                  470/900 = 52.2% [49.0,55.5]
   24 RS31 pairs (all neighbour, subset of the 30) 1048/1200 = 87.3% [85.3,89.1]
   6 neighbour pairs not in RS31 (extra)           255/300 = 85.0% [80.5,88.6]
rs32D_seed400: all 48 1720/2400 = 71.7% [69.8,73.4]
   30 neighbour pairs                              1307/1500 = 87.1% [85.3,88.7]
   18 long pairs (2 or 3 squares)                  413/900 = 45.9% [42.7,49.2]
   24 RS31 pairs (all neighbour, subset of the 30) 1074/1200 = 89.5% [87.6,91.1]
   6 neighbour pairs not in RS31 (extra)           233/300 = 77.7% [72.6,82.0]
rs32D_seed500: all 48 1091/2400 = 45.5% [43.5,47.5]
   30 neighbour pairs                              845/1500 = 56.3% [53.8,58.8]
   18 long pairs (2 or 3 squares)                  246/900 = 27.3% [24.5,30.3]
   24 RS31 pairs (all neighbour, subset of the 30) 724/1200 = 60.3% [57.5,63.1]
   6 neighbour pairs not in RS31 (extra)           121/300 = 40.3% [34.9,46.0]

ITEM e(a). zero-shot: 12 pairs on files b and g, 300 episodes, 25 per pair, stochastic, seed 20000
   overlap with the 48 trained pairs: [] (asserted empty; the 12 pairs: b2:b3,b3:b4,b4:b5,b5:b6,b6:b7,b7:b8,g2:g3,g3:g4,g4:g5,g5:g6,g6:g7,g7:g8)
rs32D_seed0: pooled 135/300 = 45.0% [39.5,50.7]   rms identical before/after: yes
   funnel: lift over 10 mm (undercounts) 101/300  held 296/300  released 200/300  placed 135/300   reasons {'bad_contact': 4, 'piece_off_board': 56, 'success': 135, 'time_limit': 105}
   per pair k/25: b2:b3 12  b3:b4 16  b4:b5 18  b5:b6 21  b6:b7 17  b7:b8 6  g2:g3 9  g3:g4 14  g4:g5 7  g5:g6 13  g6:g7 2  g7:g8 0
rs32D_seed100: pooled 224/300 = 74.7% [69.5,79.3]   rms identical before/after: yes
   funnel: lift over 10 mm (undercounts) 44/300  held 288/300  released 240/300  placed 224/300   reasons {'bad_contact': 1, 'piece_off_board': 27, 'success': 224, 'time_limit': 48}
   per pair k/25: b2:b3 12  b3:b4 23  b4:b5 20  b5:b6 21  b6:b7 19  b7:b8 19  g2:g3 15  g3:g4 18  g4:g5 19  g5:g6 20  g6:g7 18  g7:g8 20
rs32D_seed200: pooled 169/300 = 56.3% [50.7,61.8]   rms identical before/after: yes
   funnel: lift over 10 mm (undercounts) 82/300  held 292/300  released 209/300  placed 169/300   reasons {'piece_off_board': 38, 'success': 169, 'time_limit': 93}
   per pair k/25: b2:b3 11  b3:b4 13  b4:b5 14  b5:b6 17  b6:b7 15  b7:b8 13  g2:g3 7  g3:g4 13  g4:g5 13  g5:g6 20  g6:g7 19  g7:g8 14
rs32D_seed300: pooled 242/300 = 80.7% [75.8,84.7]   rms identical before/after: yes
   funnel: lift over 10 mm (undercounts) 29/300  held 287/300  released 257/300  placed 242/300   reasons {'piece_off_board': 27, 'success': 242, 'time_limit': 31}
   per pair k/25: b2:b3 18  b3:b4 20  b4:b5 22  b5:b6 21  b6:b7 22  b7:b8 17  g2:g3 16  g3:g4 23  g4:g5 19  g5:g6 23  g6:g7 23  g7:g8 18
rs32D_seed400: pooled 231/300 = 77.0% [71.9,81.4]   rms identical before/after: yes
   funnel: lift over 10 mm (undercounts) 46/300  held 300/300  released 269/300  placed 231/300   reasons {'piece_off_board': 18, 'success': 231, 'time_limit': 51}
   per pair k/25: b2:b3 21  b3:b4 17  b4:b5 20  b5:b6 15  b6:b7 16  b7:b8 21  g2:g3 17  g3:g4 19  g4:g5 21  g5:g6 21  g6:g7 20  g7:g8 23
rs32D_seed500: pooled 157/300 = 52.3% [46.7,57.9]   rms identical before/after: yes
   funnel: lift over 10 mm (undercounts) 95/300  held 293/300  released 194/300  placed 157/300   reasons {'bad_contact': 1, 'piece_off_board': 36, 'success': 157, 'time_limit': 106}
   per pair k/25: b2:b3 8  b3:b4 17  b4:b5 13  b5:b6 9  b6:b7 4  b7:b8 0  g2:g3 22  g3:g4 16  g4:g5 21  g5:g6 18  g6:g7 18  g7:g8 11

ITEM e(b). zero-shot: the 20 RS29 pairs, 300 episodes, 15 per pair, stochastic, seed 20000
   e4:e5,c1:c2,d2:e2,a6:a5,e8:d7,d4:e5,f3:c6,h5:d3,g5:b3,c1:f6,h1:h8,h8:h1,a4:h4,b7:g2,b2:g7,f1:c8,a1:h8,h8:a1,h1:a8,a8:h1
   overlap with the 48 trained pairs: ['e4:e5'] (asserted; e4:e5 is a training pair) -> excluded from the headline; 19 untrained pairs, n = 285
rs32D_seed0: *** 19 UNTRAINED pairs, n = 285: 24/285 = 8.4% [5.7,12.2] ***   (all 20 incl. e4:e5: 37/300 = 12.3% [9.1,16.5]; e4:e5 alone 13/15)
   pairs with >=1 placement: 4/20 {'e4:e5 (TRAINING pair)': 13, 'c1:c2': 11, 'e8:d7': 3, 'd4:e5': 10}   rms identical before/after: yes
   funnel (all 20 pairs; includes e4:e5): lift over 10 mm (undercounts) 54/300  held 154/300  released 44/300  placed 37/300   reasons {'bad_contact': 18, 'piece_off_board': 60, 'success': 37, 'time_limit': 185}
   bands, 19 untrained: short 24/75 = 32.0% [22.5,43.2]; mid 0/60 = 0.0% [0.0,6.0]; long 0/90 = 0.0% [0.0,4.1]; diagonal 0/60 = 0.0% [0.0,6.0]
   per pair k/15: e4:e5 (TRAINING pair) 13 c1:c2 11 d2:e2 0 a6:a5 0 e8:d7 3 d4:e5 10 f3:c6 0 h5:d3 0 g5:b3 0 c1:f6 0 h1:h8 0 h8:h1 0 a4:h4 0 b7:g2 0 b2:g7 0 f1:c8 0 a1:h8 0 h8:a1 0 h1:a8 0 a8:h1 0
rs32D_seed100: *** 19 UNTRAINED pairs, n = 285: 43/285 = 15.1% [11.4,19.7] ***   (all 20 incl. e4:e5: 56/300 = 18.7% [14.7,23.5]; e4:e5 alone 13/15)
   pairs with >=1 placement: 6/20 {'e4:e5 (TRAINING pair)': 13, 'c1:c2': 8, 'd2:e2': 1, 'a6:a5': 10, 'e8:d7': 12, 'd4:e5': 12}   rms identical before/after: yes
   funnel (all 20 pairs; includes e4:e5): lift over 10 mm (undercounts) 31/300  held 130/300  released 60/300  placed 56/300   reasons {'bad_contact': 39, 'piece_off_board': 75, 'success': 56, 'time_limit': 130}
   bands, 19 untrained: short 43/75 = 57.3% [46.1,67.9]; mid 0/60 = 0.0% [0.0,6.0]; long 0/90 = 0.0% [0.0,4.1]; diagonal 0/60 = 0.0% [0.0,6.0]
   per pair k/15: e4:e5 (TRAINING pair) 13 c1:c2 8 d2:e2 1 a6:a5 10 e8:d7 12 d4:e5 12 f3:c6 0 h5:d3 0 g5:b3 0 c1:f6 0 h1:h8 0 h8:h1 0 a4:h4 0 b7:g2 0 b2:g7 0 f1:c8 0 a1:h8 0 h8:a1 0 h1:a8 0 a8:h1 0
rs32D_seed200: *** 19 UNTRAINED pairs, n = 285: 24/285 = 8.4% [5.7,12.2] ***   (all 20 incl. e4:e5: 38/300 = 12.7% [9.4,16.9]; e4:e5 alone 14/15)
   pairs with >=1 placement: 5/20 {'e4:e5 (TRAINING pair)': 14, 'c1:c2': 10, 'd2:e2': 2, 'a6:a5': 3, 'd4:e5': 9}   rms identical before/after: yes
   funnel (all 20 pairs; includes e4:e5): lift over 10 mm (undercounts) 38/300  held 139/300  released 45/300  placed 38/300   reasons {'bad_contact': 26, 'piece_off_board': 119, 'success': 38, 'time_limit': 117}
   bands, 19 untrained: short 24/75 = 32.0% [22.5,43.2]; mid 0/60 = 0.0% [0.0,6.0]; long 0/90 = 0.0% [0.0,4.1]; diagonal 0/60 = 0.0% [0.0,6.0]
   per pair k/15: e4:e5 (TRAINING pair) 14 c1:c2 10 d2:e2 2 a6:a5 3 e8:d7 0 d4:e5 9 f3:c6 0 h5:d3 0 g5:b3 0 c1:f6 0 h1:h8 0 h8:h1 0 a4:h4 0 b7:g2 0 b2:g7 0 f1:c8 0 a1:h8 0 h8:a1 0 h1:a8 0 a8:h1 0
rs32D_seed300: *** 19 UNTRAINED pairs, n = 285: 38/285 = 13.3% [9.9,17.8] ***   (all 20 incl. e4:e5: 52/300 = 17.3% [13.5,22.0]; e4:e5 alone 14/15)
   pairs with >=1 placement: 9/20 {'e4:e5 (TRAINING pair)': 14, 'c1:c2': 1, 'd2:e2': 9, 'a6:a5': 9, 'e8:d7': 3, 'd4:e5': 12, 'h5:d3': 1, 'c1:f6': 2, 'a4:h4': 1}   rms identical before/after: yes
   funnel (all 20 pairs; includes e4:e5): lift over 10 mm (undercounts) 28/300  held 129/300  released 55/300  placed 52/300   reasons {'bad_contact': 55, 'piece_off_board': 46, 'success': 52, 'time_limit': 147}
   bands, 19 untrained: short 34/75 = 45.3% [34.6,56.6]; mid 3/60 = 5.0% [1.7,13.7]; long 1/90 = 1.1% [0.2,6.0]; diagonal 0/60 = 0.0% [0.0,6.0]
   per pair k/15: e4:e5 (TRAINING pair) 14 c1:c2 1 d2:e2 9 a6:a5 9 e8:d7 3 d4:e5 12 f3:c6 0 h5:d3 1 g5:b3 0 c1:f6 2 h1:h8 0 h8:h1 0 a4:h4 1 b7:g2 0 b2:g7 0 f1:c8 0 a1:h8 0 h8:a1 0 h1:a8 0 a8:h1 0
rs32D_seed400: *** 19 UNTRAINED pairs, n = 285: 39/285 = 13.7% [10.2,18.2] ***   (all 20 incl. e4:e5: 53/300 = 17.7% [13.8,22.4]; e4:e5 alone 14/15)
   pairs with >=1 placement: 8/20 {'e4:e5 (TRAINING pair)': 14, 'c1:c2': 8, 'd2:e2': 5, 'a6:a5': 4, 'd4:e5': 15, 'f3:c6': 4, 'c1:f6': 1, 'h1:h8': 2}   rms identical before/after: yes
   funnel (all 20 pairs; includes e4:e5): lift over 10 mm (undercounts) 28/300  held 148/300  released 70/300  placed 53/300   reasons {'bad_contact': 32, 'piece_off_board': 78, 'success': 53, 'time_limit': 137}
   bands, 19 untrained: short 32/75 = 42.7% [32.1,53.9]; mid 5/60 = 8.3% [3.6,18.1]; long 2/90 = 2.2% [0.6,7.7]; diagonal 0/60 = 0.0% [0.0,6.0]
   per pair k/15: e4:e5 (TRAINING pair) 14 c1:c2 8 d2:e2 5 a6:a5 4 e8:d7 0 d4:e5 15 f3:c6 4 h5:d3 0 g5:b3 0 c1:f6 1 h1:h8 2 h8:h1 0 a4:h4 0 b7:g2 0 b2:g7 0 f1:c8 0 a1:h8 0 h8:a1 0 h1:a8 0 a8:h1 0
rs32D_seed500: *** 19 UNTRAINED pairs, n = 285: 24/285 = 8.4% [5.7,12.2] ***   (all 20 incl. e4:e5: 37/300 = 12.3% [9.1,16.5]; e4:e5 alone 13/15)
   pairs with >=1 placement: 4/20 {'e4:e5 (TRAINING pair)': 13, 'c1:c2': 10, 'd2:e2': 1, 'd4:e5': 13}   rms identical before/after: yes
   funnel (all 20 pairs; includes e4:e5): lift over 10 mm (undercounts) 53/300  held 155/300  released 44/300  placed 37/300   reasons {'bad_contact': 32, 'piece_off_board': 123, 'success': 37, 'time_limit': 108}
   bands, 19 untrained: short 24/75 = 32.0% [22.5,43.2]; mid 0/60 = 0.0% [0.0,6.0]; long 0/90 = 0.0% [0.0,4.1]; diagonal 0/60 = 0.0% [0.0,6.0]
   per pair k/15: e4:e5 (TRAINING pair) 13 c1:c2 10 d2:e2 1 a6:a5 0 e8:d7 0 d4:e5 13 f3:c6 0 h5:d3 0 g5:b3 0 c1:f6 0 h1:h8 0 h8:h1 0 a4:h4 0 b7:g2 0 b2:g7 0 f1:c8 0 a1:h8 0 h8:a1 0 h1:a8 0 a8:h1 0

ITEM f. deterministic vs stochastic on the 48 pairs: 96 deterministic episodes = 2 per pair, seed 20000 (reporting choice carried over from RS31)
rs32D_seed0: deterministic 34/96 = 35.4% [26.6,45.4] [96-episode interval assumes independent episodes; NOT valid if the two episodes of a pair are identical]   stochastic (Step 2, 2400 ep) 1074/2400 = 44.8% [42.8,46.7]
   per-pair placed count (0, 1 or 2): pairs with 0: 31, with 1: 0, with 2: 17
   two episodes of a pair: identical per-step trajectory records in 48/48 pairs; identical whole episode records in 48/48 pairs
   all 48 pairs give identical results across their 2 episodes -> effective n = 48 pairs: 17/48 = 35.4% [23.4,49.6]  (pairs placed / 48 pairs)
   deterministic funnel: lift over 10 mm (undercounts) 30/96  held 96/96  released 48/96  placed 34/96   reasons {'piece_off_board': 4, 'success': 34, 'time_limit': 58}
   per pair placed (0/1/2): c2:c3 0 c3:c4 0 c4:c5 2 c5:c6 0 c6:c7 2 c7:c8 0 d2:d3 0 d3:d4 2 d4:d5 0 d5:d6 2 d6:d7 0 d7:d8 0 e2:e3 2 e3:e4 0 e4:e5 2 e5:e6 2 e6:e7 2 e7:e8 2 f2:f3 0 f3:f4 0 f4:f5 0 f5:f6 0 f6:f7 2 f7:f8 0 c4:d4 0 f5:e5 0 d3:e4 2 e6:d7 0 e5:e4 0 c6:c5 0 b2:b4 0 g3:g6 0 d7:d4 0 a5:c5 0 h4:e4 0 b6:e6 2 a3:c5 2 c3:f6 2 h3:f5 2 g7:d4 0 b7:d5 0 e2:g4 0 a3:d5 0 h6:e4 0 c2:e5 2 f3:d6 2 b4:e6 0 g6:e3 0
rs32D_seed100: deterministic 70/96 = 72.9% [63.3,80.8] [96-episode interval assumes independent episodes; NOT valid if the two episodes of a pair are identical]   stochastic (Step 2, 2400 ep) 1704/2400 = 71.0% [69.2,72.8]
   per-pair placed count (0, 1 or 2): pairs with 0: 13, with 1: 0, with 2: 35
   two episodes of a pair: identical per-step trajectory records in 48/48 pairs; identical whole episode records in 48/48 pairs
   all 48 pairs give identical results across their 2 episodes -> effective n = 48 pairs: 35/48 = 72.9% [59.0,83.4]  (pairs placed / 48 pairs)
   deterministic funnel: lift over 10 mm (undercounts) 14/96  held 90/96  released 70/96  placed 70/96   reasons {'bad_contact': 4, 'piece_off_board': 10, 'success': 70, 'time_limit': 12}
   per pair placed (0/1/2): c2:c3 2 c3:c4 2 c4:c5 2 c5:c6 2 c6:c7 2 c7:c8 2 d2:d3 2 d3:d4 2 d4:d5 2 d5:d6 2 d6:d7 2 d7:d8 2 e2:e3 2 e3:e4 2 e4:e5 2 e5:e6 2 e6:e7 2 e7:e8 2 f2:f3 2 f3:f4 2 f4:f5 2 f5:f6 2 f6:f7 2 f7:f8 2 c4:d4 2 f5:e5 2 d3:e4 2 e6:d7 2 e5:e4 2 c6:c5 2 b2:b4 0 g3:g6 2 d7:d4 0 a5:c5 0 h4:e4 0 b6:e6 2 a3:c5 2 c3:f6 0 h3:f5 0 g7:d4 0 b7:d5 0 e2:g4 0 a3:d5 0 h6:e4 2 c2:e5 0 f3:d6 0 b4:e6 2 g6:e3 0
rs32D_seed200: deterministic 34/96 = 35.4% [26.6,45.4] [96-episode interval assumes independent episodes; NOT valid if the two episodes of a pair are identical]   stochastic (Step 2, 2400 ep) 1029/2400 = 42.9% [40.9,44.9]
   per-pair placed count (0, 1 or 2): pairs with 0: 31, with 1: 0, with 2: 17
   two episodes of a pair: identical per-step trajectory records in 48/48 pairs; identical whole episode records in 48/48 pairs
   all 48 pairs give identical results across their 2 episodes -> effective n = 48 pairs: 17/48 = 35.4% [23.4,49.6]  (pairs placed / 48 pairs)
   deterministic funnel: lift over 10 mm (undercounts) 36/96  held 90/96  released 40/96  placed 34/96   reasons {'bad_contact': 2, 'piece_off_board': 12, 'success': 34, 'time_limit': 48}
   per pair placed (0/1/2): c2:c3 0 c3:c4 2 c4:c5 2 c5:c6 2 c6:c7 2 c7:c8 0 d2:d3 2 d3:d4 2 d4:d5 0 d5:d6 0 d6:d7 0 d7:d8 2 e2:e3 2 e3:e4 0 e4:e5 2 e5:e6 2 e6:e7 2 e7:e8 2 f2:f3 0 f3:f4 0 f4:f5 2 f5:f6 2 f6:f7 2 f7:f8 2 c4:d4 0 f5:e5 0 d3:e4 0 e6:d7 0 e5:e4 0 c6:c5 0 b2:b4 0 g3:g6 0 d7:d4 0 a5:c5 0 h4:e4 0 b6:e6 0 a3:c5 0 c3:f6 0 h3:f5 0 g7:d4 0 b7:d5 0 e2:g4 0 a3:d5 0 h6:e4 0 c2:e5 0 f3:d6 0 b4:e6 2 g6:e3 0
rs32D_seed300: deterministic 76/96 = 79.2% [70.0,86.1] [96-episode interval assumes independent episodes; NOT valid if the two episodes of a pair are identical]   stochastic (Step 2, 2400 ep) 1773/2400 = 73.9% [72.1,75.6]
   per-pair placed count (0, 1 or 2): pairs with 0: 10, with 1: 0, with 2: 38
   two episodes of a pair: identical per-step trajectory records in 48/48 pairs; identical whole episode records in 48/48 pairs
   all 48 pairs give identical results across their 2 episodes -> effective n = 48 pairs: 38/48 = 79.2% [65.7,88.3]  (pairs placed / 48 pairs)
   deterministic funnel: lift over 10 mm (undercounts) 4/96  held 90/96  released 76/96  placed 76/96   reasons {'piece_off_board': 4, 'success': 76, 'time_limit': 16}
   per pair placed (0/1/2): c2:c3 2 c3:c4 2 c4:c5 2 c5:c6 2 c6:c7 2 c7:c8 2 d2:d3 2 d3:d4 2 d4:d5 2 d5:d6 0 d6:d7 2 d7:d8 2 e2:e3 0 e3:e4 2 e4:e5 2 e5:e6 2 e6:e7 2 e7:e8 2 f2:f3 2 f3:f4 2 f4:f5 2 f5:f6 2 f6:f7 2 f7:f8 2 c4:d4 2 f5:e5 2 d3:e4 2 e6:d7 2 e5:e4 2 c6:c5 0 b2:b4 2 g3:g6 0 d7:d4 2 a5:c5 2 h4:e4 2 b6:e6 2 a3:c5 2 c3:f6 2 h3:f5 0 g7:d4 0 b7:d5 2 e2:g4 0 a3:d5 2 h6:e4 2 c2:e5 2 f3:d6 0 b4:e6 0 g6:e3 0
rs32D_seed400: deterministic 64/96 = 66.7% [56.8,75.3] [96-episode interval assumes independent episodes; NOT valid if the two episodes of a pair are identical]   stochastic (Step 2, 2400 ep) 1720/2400 = 71.7% [69.8,73.4]
   per-pair placed count (0, 1 or 2): pairs with 0: 16, with 1: 0, with 2: 32
   two episodes of a pair: identical per-step trajectory records in 48/48 pairs; identical whole episode records in 48/48 pairs
   all 48 pairs give identical results across their 2 episodes -> effective n = 48 pairs: 32/48 = 66.7% [52.5,78.3]  (pairs placed / 48 pairs)
   deterministic funnel: lift over 10 mm (undercounts) 16/96  held 96/96  released 68/96  placed 64/96   reasons {'bad_contact': 2, 'success': 64, 'time_limit': 30}
   per pair placed (0/1/2): c2:c3 2 c3:c4 2 c4:c5 2 c5:c6 2 c6:c7 2 c7:c8 2 d2:d3 0 d3:d4 2 d4:d5 2 d5:d6 2 d6:d7 2 d7:d8 0 e2:e3 2 e3:e4 2 e4:e5 2 e5:e6 2 e6:e7 2 e7:e8 2 f2:f3 0 f3:f4 2 f4:f5 2 f5:f6 0 f6:f7 2 f7:f8 2 c4:d4 2 f5:e5 2 d3:e4 2 e6:d7 0 e5:e4 2 c6:c5 2 b2:b4 0 g3:g6 2 d7:d4 0 a5:c5 2 h4:e4 2 b6:e6 2 a3:c5 0 c3:f6 0 h3:f5 2 g7:d4 0 b7:d5 2 e2:g4 0 a3:d5 0 h6:e4 0 c2:e5 0 f3:d6 0 b4:e6 2 g6:e3 0
rs32D_seed500: deterministic 32/96 = 33.3% [24.7,43.2] [96-episode interval assumes independent episodes; NOT valid if the two episodes of a pair are identical]   stochastic (Step 2, 2400 ep) 1091/2400 = 45.5% [43.5,47.5]
   per-pair placed count (0, 1 or 2): pairs with 0: 32, with 1: 0, with 2: 16
   two episodes of a pair: identical per-step trajectory records in 48/48 pairs; identical whole episode records in 48/48 pairs
   all 48 pairs give identical results across their 2 episodes -> effective n = 48 pairs: 16/48 = 33.3% [21.7,47.5]  (pairs placed / 48 pairs)
   deterministic funnel: lift over 10 mm (undercounts) 24/96  held 96/96  released 44/96  placed 32/96   reasons {'bad_contact': 2, 'piece_off_board': 4, 'success': 32, 'time_limit': 58}
   per pair placed (0/1/2): c2:c3 2 c3:c4 0 c4:c5 2 c5:c6 2 c6:c7 0 c7:c8 0 d2:d3 0 d3:d4 0 d4:d5 0 d5:d6 0 d6:d7 0 d7:d8 0 e2:e3 2 e3:e4 2 e4:e5 2 e5:e6 2 e6:e7 0 e7:e8 0 f2:f3 0 f3:f4 0 f4:f5 0 f5:f6 2 f6:f7 0 f7:f8 0 c4:d4 0 f5:e5 2 d3:e4 0 e6:d7 0 e5:e4 0 c6:c5 2 b2:b4 0 g3:g6 2 d7:d4 0 a5:c5 2 h4:e4 2 b6:e6 0 a3:c5 2 c3:f6 0 h3:f5 2 g7:d4 0 b7:d5 0 e2:g4 0 a3:d5 0 h6:e4 0 c2:e5 2 f3:d6 0 b4:e6 0 g6:e3 0

ITEM g. eval vs training cross-check (Wilson 95%, z=1.96, no continuity correction); gap = eval - training, percentage points; overlap computed from the printed intervals
(1) PRIMARY: eval pooled placement (Step 2, k/2400) vs training placement in the LAST 10% chunk of training (Monitor terminated_reason == success, chunk 9, pooled over the 4 files)
(2) supplementary (RS31 convention): eval placement vs the last 100k training steps (last 25k steps of each file)
(3) supplementary (RS31 convention): eval held (place.held /2400) vs ever_held in the last 10% chunk
rs32D_seed0
  (1) placed, last 10% chunk: eval 1074/2400 = 44.8% [42.8,46.7]   training 917/2143 = 42.8% [40.7,44.9]   gap +2.0 pts; Wilson intervals OVERLAP (eval [42.77,46.75] vs training [40.71,44.90])
  (2) placed, last 100k steps: eval 1074/2400 = 44.8% [42.8,46.7]   training 187/434 = 43.1% [38.5,47.8]   gap +1.7 pts; Wilson intervals OVERLAP (eval [42.77,46.75] vs training [38.51,47.79])
  (3) held,   last 10% chunk: eval 2322/2400 = 96.8% [96.0,97.4]   training 2079/2143 = 97.0% [96.2,97.7]   gap -0.3 pts; Wilson intervals OVERLAP (eval [95.96,97.39] vs training [96.20,97.65])
rs32D_seed100
  (1) placed, last 10% chunk: eval 1704/2400 = 71.0% [69.2,72.8]   training 2249/3108 = 72.4% [70.8,73.9]   gap -1.4 pts; Wilson intervals OVERLAP (eval [69.15,72.78] vs training [70.76,73.91])
  (2) placed, last 100k steps: eval 1704/2400 = 71.0% [69.2,72.8]   training 456/633 = 72.0% [68.4,75.4]   gap -1.0 pts; Wilson intervals OVERLAP (eval [69.15,72.78] vs training [68.42,75.39])
  (3) held,   last 10% chunk: eval 2168/2400 = 90.3% [89.1,91.5]   training 2813/3108 = 90.5% [89.4,91.5]   gap -0.2 pts; Wilson intervals OVERLAP (eval [89.09,91.45] vs training [89.43,91.49])
rs32D_seed200
  (1) placed, last 10% chunk: eval 1029/2400 = 42.9% [40.9,44.9]   training 921/2193 = 42.0% [39.9,44.1]   gap +0.9 pts; Wilson intervals OVERLAP (eval [40.91,44.86] vs training [39.95,44.08])
  (2) placed, last 100k steps: eval 1029/2400 = 42.9% [40.9,44.9]   training 210/448 = 46.9% [42.3,51.5]   gap -4.0 pts; Wilson intervals OVERLAP (eval [40.91,44.86] vs training [42.30,51.50])
  (3) held,   last 10% chunk: eval 2231/2400 = 93.0% [91.9,93.9]   training 2029/2193 = 92.5% [91.3,93.5]   gap +0.4 pts; Wilson intervals OVERLAP (eval [91.86,93.91] vs training [91.34,93.55])
rs32D_seed300
  (1) placed, last 10% chunk: eval 1773/2400 = 73.9% [72.1,75.6]   training 2205/2992 = 73.7% [72.1,75.2]   gap +0.2 pts; Wilson intervals OVERLAP (eval [72.08,75.59] vs training [72.09,75.24])
  (2) placed, last 100k steps: eval 1773/2400 = 73.9% [72.1,75.6]   training 427/593 = 72.0% [68.3,75.5]   gap +1.9 pts; Wilson intervals OVERLAP (eval [72.08,75.59] vs training [68.26,75.47])
  (3) held,   last 10% chunk: eval 2247/2400 = 93.6% [92.6,94.5]   training 2744/2992 = 91.7% [90.7,92.6]   gap +1.9 pts; Wilson intervals OVERLAP (eval [92.58,94.53] vs training [90.67,92.65])
rs32D_seed400
  (1) placed, last 10% chunk: eval 1720/2400 = 71.7% [69.8,73.4]   training 1820/2664 = 68.3% [66.5,70.1]   gap +3.3 pts; Wilson intervals OVERLAP (eval [69.83,73.43] vs training [66.53,70.06])
  (2) placed, last 100k steps: eval 1720/2400 = 71.7% [69.8,73.4]   training 388/544 = 71.3% [67.4,75.0]   gap +0.3 pts; Wilson intervals OVERLAP (eval [69.83,73.43] vs training [67.38,74.96])
  (3) held,   last 10% chunk: eval 2375/2400 = 99.0% [98.5,99.3]   training 2625/2664 = 98.5% [98.0,98.9]   gap +0.4 pts; Wilson intervals OVERLAP (eval [98.47,99.29] vs training [98.01,98.93])
rs32D_seed500
  (1) placed, last 10% chunk: eval 1091/2400 = 45.5% [43.5,47.5]   training 936/2169 = 43.2% [41.1,45.2]   gap +2.3 pts; Wilson intervals OVERLAP (eval [43.48,47.46] vs training [41.08,45.25])
  (2) placed, last 100k steps: eval 1091/2400 = 45.5% [43.5,47.5]   training 184/438 = 42.0% [37.5,46.7]   gap +3.4 pts; Wilson intervals OVERLAP (eval [43.48,47.46] vs training [37.48,46.68])
  (3) held,   last 10% chunk: eval 2309/2400 = 96.2% [95.4,96.9]   training 2097/2169 = 96.7% [95.8,97.4]   gap -0.5 pts; Wilson intervals OVERLAP (eval [95.37,96.90] vs training [95.84,97.36])
```

## 5. Step 3 items b, c, d (`section3_train_output.txt`, verbatim)

b: policy std = exp(log_std) per action dim (dx, dy, dz, gripper) from `policy.pth` in the 1M, 2M, 3M, 4M and 5M checkpoints and in `final_model.zip` (5,005,312 steps).
c: `ever_held` per 10% chunk of training (Monitor CSVs; chunk = min((cum_l - 1)*10 // total_l, 9) per file, counts pooled over the four files).
d: training placement (Monitor `terminated_reason == success`) per 100k training steps = 25k steps of each of the four files, pooled (RS31 convention).

```
ITEM b. policy std = exp(log_std) per action dim (dx, dy, dz, gripper) and their mean
rs32D_seed0
   1000000 (checkpoint)   : 0.7833 0.7011 0.9780 0.9833   mean 0.8614
   2000000 (checkpoint)   : 0.7193 0.6067 0.8793 0.9562   mean 0.7904
   3000000 (checkpoint)   : 0.7393 0.5514 0.8637 0.8548   mean 0.7523
   4000000 (checkpoint)   : 0.7525 0.5043 0.8722 0.8402   mean 0.7423
   5000000 (checkpoint)   : 0.7413 0.5133 0.8214 0.8260   mean 0.7255
  final_model (5,005,312) : 0.7413 0.5133 0.8214 0.8260   mean 0.7255
rs32D_seed100
   1000000 (checkpoint)   : 0.7201 0.8017 0.9394 0.9253   mean 0.8466
   2000000 (checkpoint)   : 0.6657 0.6072 0.7879 0.7593   mean 0.7050
   3000000 (checkpoint)   : 0.6106 0.5520 0.6139 0.6286   mean 0.6013
   4000000 (checkpoint)   : 0.5641 0.5281 0.6753 0.6268   mean 0.5986
   5000000 (checkpoint)   : 0.5965 0.5278 0.6929 0.5832   mean 0.6001
  final_model (5,005,312) : 0.5965 0.5278 0.6929 0.5832   mean 0.6001
rs32D_seed200
   1000000 (checkpoint)   : 0.8070 0.7124 0.8347 1.0897   mean 0.8610
   2000000 (checkpoint)   : 0.8773 0.4525 0.7164 1.1644   mean 0.8026
   3000000 (checkpoint)   : 0.8848 0.3898 0.7607 1.1301   mean 0.7913
   4000000 (checkpoint)   : 0.8029 0.2720 0.8262 1.1492   mean 0.7626
   5000000 (checkpoint)   : 0.8084 0.2823 0.8267 1.0731   mean 0.7476
  final_model (5,005,312) : 0.8084 0.2823 0.8267 1.0731   mean 0.7476
rs32D_seed300
   1000000 (checkpoint)   : 0.7899 0.6919 1.0114 1.0717   mean 0.8912
   2000000 (checkpoint)   : 0.5992 0.6418 0.8658 0.8656   mean 0.7431
   3000000 (checkpoint)   : 0.5097 0.5935 0.7262 0.9047   mean 0.6835
   4000000 (checkpoint)   : 0.5326 0.5541 0.6740 0.8118   mean 0.6431
   5000000 (checkpoint)   : 0.5901 0.5851 0.6573 0.7286   mean 0.6403
  final_model (5,005,312) : 0.5901 0.5851 0.6573 0.7286   mean 0.6403
rs32D_seed400
   1000000 (checkpoint)   : 0.7385 0.6647 1.0175 0.7720   mean 0.7982
   2000000 (checkpoint)   : 0.5624 0.6343 0.6974 0.6389   mean 0.6332
   3000000 (checkpoint)   : 0.4792 0.6600 0.6804 0.6606   mean 0.6200
   4000000 (checkpoint)   : 0.4277 0.6139 0.6548 0.6308   mean 0.5818
   5000000 (checkpoint)   : 0.4595 0.5982 0.6304 0.6035   mean 0.5729
  final_model (5,005,312) : 0.4595 0.5982 0.6304 0.6035   mean 0.5729
rs32D_seed500
   1000000 (checkpoint)   : 0.8258 0.7022 0.9044 0.9477   mean 0.8450
   2000000 (checkpoint)   : 0.6577 0.6013 0.9141 0.7958   mean 0.7422
   3000000 (checkpoint)   : 0.6754 0.6043 0.7505 0.7473   mean 0.6944
   4000000 (checkpoint)   : 0.6459 0.6345 0.7558 0.6518   mean 0.6720
   5000000 (checkpoint)   : 0.6124 0.6548 0.7220 0.6174   mean 0.6517
  final_model (5,005,312) : 0.6124 0.6548 0.7220 0.6174   mean 0.6517

ITEM c. Monitor ever_held per 10% chunk of training (%; pooled over the four files)
rs32D_seed0: files ['monitor_0.monitor.csv', 'monitor_1.monitor.csv', 'monitor_2.monitor.csv', 'monitor_3.monitor.csv'] steps per file [1251320, 1251295, 1251203, 1251120] (sum 5,004,938) episodes 18486
   ever_held %:  48.6  93.4  96.7  94.6  94.3  95.1  95.6  96.0  96.7  97.0   | n eps: 1770 1762 1773 1780 1762 1800 1812 1883 2001 2143
rs32D_seed100: files ['monitor_0.monitor.csv', 'monitor_1.monitor.csv', 'monitor_2.monitor.csv', 'monitor_3.monitor.csv'] steps per file [1251278, 1251144, 1251315, 1251173] (sum 5,004,910) episodes 22845
   ever_held %:  43.5  93.4  94.3  90.3  87.4  86.3  85.9  88.4  87.7  90.5   | n eps: 1766 1762 1821 1924 2075 2354 2467 2681 2887 3108
rs32D_seed200: files ['monitor_0.monitor.csv', 'monitor_1.monitor.csv', 'monitor_2.monitor.csv', 'monitor_3.monitor.csv'] steps per file [1251141, 1251280, 1251321, 1251320] (sum 5,005,062) episodes 19105
   ever_held %:  35.6  88.3  94.0  95.5  95.7  94.5  92.9  92.1  92.8  92.5   | n eps: 1760 1772 1770 1769 1804 1830 1963 2086 2158 2193
rs32D_seed300: files ['monitor_0.monitor.csv', 'monitor_1.monitor.csv', 'monitor_2.monitor.csv', 'monitor_3.monitor.csv'] steps per file [1251264, 1251250, 1251285, 1251297] (sum 5,005,096) episodes 22711
   ever_held %:  54.9  89.9  90.6  91.7  84.1  87.9  89.9  91.1  90.7  91.7   | n eps: 1791 1859 1877 1905 2053 2276 2449 2671 2838 2992
rs32D_seed400: files ['monitor_0.monitor.csv', 'monitor_1.monitor.csv', 'monitor_2.monitor.csv', 'monitor_3.monitor.csv'] steps per file [1251289, 1251311, 1251127, 1251283] (sum 5,005,010) episodes 18926
   ever_held %:  57.2  94.8  96.1  95.5  94.6  96.5  96.4  97.2  98.6  98.5   | n eps: 1752 1777 1740 1744 1749 1717 1727 1792 2264 2664
rs32D_seed500: files ['monitor_0.monitor.csv', 'monitor_1.monitor.csv', 'monitor_2.monitor.csv', 'monitor_3.monitor.csv'] steps per file [1251229, 1251264, 1251184, 1251312] (sum 5,004,989) episodes 18614
   ever_held %:  32.2  87.8  87.5  92.9  92.6  95.0  93.4  95.5  95.7  96.7   | n eps: 1752 1809 1796 1768 1758 1784 1821 1943 2014 2169

ITEM d. rolling training placement per 100k training steps; k/n eps = % (bin j = training steps (j-1)*100k+1 .. j*100k; 25k steps of each of the 4 files)
 bin(100k)      rs32D_seed0    rs32D_seed100    rs32D_seed200    rs32D_seed300    rs32D_seed400    rs32D_seed500
    100000       0.0% 0/336       0.0% 0/332       0.0% 0/335       0.0% 0/337       0.0% 0/334       0.0% 0/332
    200000       0.0% 0/346       0.0% 0/352       0.0% 0/356       0.3% 1/358       0.0% 0/349       0.3% 1/355
    300000       0.0% 0/362       0.8% 3/357       0.6% 2/355       0.8% 3/364       0.3% 1/358       0.3% 1/351
    400000       1.1% 4/365       0.3% 1/369       0.6% 2/358       1.4% 5/363       0.3% 1/355       0.0% 0/356
    500000       2.2% 8/359       2.0% 7/355       0.6% 2/353      3.8% 14/368       0.3% 1/354       0.0% 0/357
    600000       1.7% 6/357       1.4% 5/354       0.3% 1/357      4.6% 17/372       0.3% 1/356       1.4% 5/363
    700000       0.8% 3/353       2.3% 8/351       0.8% 3/359      5.4% 20/368       0.3% 1/356       0.3% 1/359
    800000       0.6% 2/351      3.1% 11/351       1.4% 5/352      8.3% 31/375       0.3% 1/355       1.4% 5/357
    900000       1.4% 5/353      5.1% 18/353       0.6% 2/356      8.8% 33/375       0.0% 0/354       2.5% 9/362
   1000000       1.7% 6/347      3.2% 11/349       0.3% 1/348      3.8% 14/366       0.0% 0/355      3.8% 14/365
   1100000      2.8% 10/356      6.7% 24/359       1.4% 5/351      6.6% 25/377       0.0% 0/347       2.5% 9/359
   1200000       2.3% 8/351      8.6% 31/360      2.9% 10/350     10.0% 38/379       0.0% 0/348       1.9% 7/368
   1300000      3.1% 11/350      6.8% 25/366       0.6% 2/358     12.9% 48/372       0.0% 0/344       0.8% 3/354
   1400000      2.8% 10/358      8.1% 29/358       1.4% 5/357     15.9% 60/378       0.3% 1/348       0.8% 3/355
   1500000      3.4% 12/354     14.6% 55/376       1.4% 5/352     15.2% 56/369       0.0% 0/351       1.7% 6/360
   1600000       2.0% 7/358     18.1% 69/381       1.7% 6/353     15.8% 58/367       0.0% 0/346       1.4% 5/351
   1700000       2.6% 9/352     16.8% 64/380       0.9% 3/350     19.3% 72/373       0.0% 0/349       1.4% 5/360
   1800000      3.4% 12/356     18.7% 71/379      2.8% 10/357     24.6% 94/382       0.0% 0/348       1.1% 4/353
   1900000      3.4% 12/358    28.2% 112/397      2.8% 10/353     24.9% 97/389       0.0% 0/348       2.6% 9/349
   2000000      3.1% 11/355     21.7% 84/387       0.8% 3/354    31.0% 122/394       0.0% 0/351       2.0% 7/353
   2100000       1.1% 4/353    27.8% 111/400      4.5% 16/359    37.7% 152/403       0.0% 0/354       2.3% 8/352
   2200000       0.6% 2/352    30.7% 125/407      3.9% 14/359    34.6% 141/408       0.0% 0/353       2.5% 9/355
   2300000       1.4% 5/348    36.0% 150/417      6.5% 23/355    30.5% 122/400       0.0% 0/349       2.3% 8/346
   2400000       2.0% 7/354    35.8% 150/419      7.7% 28/364    34.5% 142/412       0.0% 0/345      4.9% 17/348
   2500000      5.9% 21/355    37.7% 161/427      9.3% 34/364    40.7% 173/425       0.0% 0/345      6.8% 24/355
   2600000      7.8% 28/359    46.3% 205/443      6.1% 22/359    38.3% 163/426       0.9% 3/344      4.8% 17/353
   2700000      8.6% 31/361    44.8% 201/449      9.1% 33/361    46.0% 205/446       0.3% 1/340      4.3% 15/352
   2800000     11.3% 41/363    48.2% 224/465     11.5% 42/365    52.5% 243/463       0.0% 0/349      6.2% 22/355
   2900000     12.1% 43/356    51.9% 255/491     15.8% 59/373    55.9% 261/467       0.9% 3/343      7.6% 27/354
   3000000      8.7% 31/357    54.6% 275/504     14.1% 52/368    54.1% 255/471       1.2% 4/343      9.0% 33/366
   3100000     11.3% 41/363    48.0% 234/488     20.5% 78/380    53.7% 251/467       1.5% 5/341      8.6% 31/361
   3200000     12.7% 46/363    48.1% 238/495     18.0% 69/384    56.5% 270/478       1.2% 4/347      7.3% 26/355
   3300000     15.8% 58/366    50.0% 247/494     20.6% 81/394    53.1% 252/475       2.0% 7/345      8.9% 32/359
   3400000     13.1% 47/359    46.4% 223/481    28.1% 112/399    62.6% 316/505       2.3% 8/343     16.4% 62/377
   3500000     12.7% 46/362    58.7% 297/506    30.8% 124/403    64.9% 337/519       2.3% 8/347     14.1% 52/369
   3600000     15.8% 58/368    57.3% 290/506    31.6% 128/405    62.9% 322/512      5.9% 21/353     16.2% 60/371
   3700000     21.6% 80/371    63.9% 349/546    34.1% 140/411    66.4% 352/530      7.2% 25/348     20.5% 78/380
   3800000     23.9% 90/376    60.5% 317/524    36.1% 151/418    68.0% 366/538      8.5% 30/354     23.6% 91/386
   3900000     23.1% 88/381    59.8% 320/535    38.4% 164/427    65.5% 349/533     10.5% 38/362    27.0% 107/397
   4000000     25.8% 99/383    66.3% 374/564    35.4% 151/426    66.4% 368/554     19.6% 73/373    29.0% 117/404
   4100000     25.6% 97/379    67.3% 381/566    41.6% 182/438    69.0% 382/554     25.7% 99/385     25.7% 99/385
   4200000    30.6% 123/402    65.9% 377/572    37.9% 159/419    68.8% 382/555    39.2% 160/408    28.2% 113/401
   4300000    31.3% 123/393    62.8% 354/564    37.9% 162/428    68.1% 388/570    54.0% 249/461    27.2% 109/401
   4400000    35.8% 148/413    68.6% 404/589    39.4% 169/429    68.8% 392/570    59.8% 294/492    31.6% 129/408
   4500000    39.1% 160/409    69.1% 409/592    40.4% 178/441    71.2% 416/584    62.5% 318/509    37.7% 157/416
   4600000    41.5% 175/422    72.0% 437/607    41.6% 182/438    71.8% 412/574    64.1% 336/524    42.2% 180/427
   4700000    41.1% 171/416    71.7% 443/618    38.7% 167/432    75.7% 464/613    65.1% 336/516    43.0% 184/428
   4800000    42.3% 182/430    72.1% 446/619    41.5% 181/436    75.1% 456/607    69.9% 374/535    43.3% 184/425
   4900000    45.6% 201/441    74.0% 464/627    40.6% 176/434    74.3% 449/604    70.3% 381/542    44.7% 201/450
   5000000    42.9% 183/427    71.7% 450/628    45.8% 204/445    71.4% 419/587    70.7% 382/540    42.7% 186/436
     5.0M+      48.1% 13/27      77.8% 28/36      63.0% 17/27      77.1% 27/35      73.3% 22/30       34.8% 8/23   (partial bin: the last ~5k training steps after 5,000,000; Monitor files hold 5,004,9xx-5,005,0xx steps, the rest of the 5,005,312 is unfinished episodes)
```

## 6. Deviations from the prompt, and what could not be verified

- Zero-shot (b): 300 episodes cannot be split over 19 pairs, so the run uses the 20 RS29 pairs (15 each) and the headline excludes e4:e5 afterwards (n = 285), as in RS31. The all-20 figure is also printed.
- Deterministic evaluation: 96 episodes (2 per pair) with `--deterministic`.
- Item g: the primary comparison is the last 10% chunk of training (as asked). The last-100k-steps placement and the held-rate comparisons are RS31-style extras, labelled as such.
- Item d: the 5.0M+ row is a partial bin. The Monitor files together hold 5,004,9xx to 5,005,0xx steps (each seed's total is printed in item c), not 5,005,312, because unfinished episodes at the end are not logged.
- Item b: the 5M checkpoint and the final model give identical std values (printed to 4 decimals); I did not check whether the two zip files are otherwise identical.
- "Report after 10 minutes": `eval_probe.py` prints no progress, so episodes per minute could not be measured. The estimate given then (24 to 51 min per eval) came from the RS31 timings (47 to 101 episodes/min per process). Observed: 27 to 38 min per eval, i.e. 63 to 90 episodes/min.
- The Step 3 runner was held back for a few hours by a placeholder manifest (it was deleted before the run); the `EVAL_MANIFEST_section3.json` here is the real one.
- Not verified: the label "lift over 10 mm" for `grasp_ever` is taken from RS31, I did not re-derive it from `eval_probe.py` here. I did not compare the wrapper's output with a plain `eval_probe.py` run for RS32 (RS31 did so for one model).
- Not verified: whether the eval and training placement are comparable beyond the check in item g (same env, stochastic policy, same pair set; the eval uses a fixed seed 20000, training used its own).
- The 24 RS31 pairs were parsed from `results/rs31_24_pairs/PREREGISTRATION.md` on branch `rs31` (`git show`); they are the first 24 of the 48.
- Nothing was pushed. Only files under `results/rs32_48_pairs/eval/` are committed.

