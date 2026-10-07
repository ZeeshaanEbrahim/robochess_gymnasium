#!/usr/bin/env python
"""Builds README.md from the *_output.txt files and the manifests. Tables come from the data, not from retyping."""
import json
from rs32_lib import *
man = json.load(open(EV / "EVAL_MANIFEST.json")); m3 = json.load(open(EV / "EVAL_MANIFEST_section3.json"))
rd = lambda f: (EV / f).read_text().rstrip()
rows = []
for name in NAMES:
    d = json.load(open(EV / f"{name}_final.json")); E = d["episodes"]; pr = d["place_report"]["rates"]
    import collections; k = collections.Counter()
    for e in E: k[f'{e["start_square"]}:{e["target_square"]}'] += e["place"]["placed"]
    tot = sum(k.values()); lo, hi = wilson(tot, 2400); nf = sum(k[p] < 10 for p in PAIRS)
    rows.append(f"| {name} | {tot}/2400 | {100*tot/2400:.1f}% [{lo:.1f}, {hi:.1f}] | {nf} | {'yes' if tot >= 1200 else 'no'} | {'yes' if nf == 0 else 'no'} | **{'yes' if tot >= 1200 and nf == 0 else 'no'}** | {'yes' if tot >= 1920 else 'no'} |")
L = sum("**yes**" in r for r in rows)
t = lambda e: f'{e["start"][11:19]} to {e["end"][11:19]}'
timing = "\n".join(f"- {e['label']}: {t(e)}, exit {e['exit']}" for e in man["evals"])
timing3 = f"{m3['start'][11:19]} to {m3['end'][11:19]}, {len(m3['evals'])} jobs, exits {sorted(set(e['exit'] for e in m3['evals']))}"
txt = f"""# RS32 evaluation: PPO (D recipe), 48 start-target pairs, 6 seeds, 5M steps

Evaluation of the final models of `_probe_cart_rs32D_ppotanh_seed{{0,100,200,300,400,500}}` under `results/rs32_48_pairs/PREREGISTRATION.md`
(rules and thresholds unchanged; config, training code and pre-registration untouched). Numbers below are copied from the output files in this
folder, which are produced by the scripts in this folder from the eval JSONs. Reported, not interpreted.

## 1. Rule verdicts (mechanical)

Rule 1: a seed learns if pooled placement >= 1200/2400 AND >= 10/50 on every one of the 48 pairs.

| model | pooled | placement, Wilson 95% | pairs under 10/50 | pooled >= 1200 | no pair under 10/50 | Rule 1: learns | target tier: pooled >= 1920 |
|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

- Rule 2: seeds that learn = **{L} of 6**.
- Rule 3: the claim "PPO with the D recipe learns 48 mixed pairs" holds if at least 4 of 6 seeds learn. {L} of 6 learn, so the claim is **not achieved**.
  Rule 3 asks to report which subset falls short; the pairs under 10/50 split as follows (`section2_subsets_output.txt`):
```
{rd('section2_subsets_output.txt')}
```
- Rule 4 (target tier, reported): pooled >= 1920/2400 on **0 of 6** seeds.

## 2. Provenance

- Git at start (EVAL_MANIFEST.json): HEAD `{man['git_start']['head']}`, branch `{man['git_start']['branch']}`, tracked-dirty files {man['git_start']['tracked_dirty_count']}; untracked: only `results/rs32_48_pairs/eval/` (this folder).
- Gate: all six `runs/_probe_cart_rs32D_ppotanh_seed*_train.log` contain EXIT=0; no python `scripts/train.py` process was running (the only `train.py` match was a `tmux new-session` wrapper of the rs31C run, ignored as instructed). No process was killed.
- Final model step counts (`num_timesteps` in `final_model.zip`): {', '.join(f'{k.split("seed")[-1]}: {v:,}' for k, v in man['final_num_timesteps'].items())} (expected 5,005,312: all as expected).
- Method as in RS31: `scripts/eval_probe.py` run unmodified through `rms_wrapper.py` (copied from `results/rs31_24_pairs/eval/`), stochastic policy, `--seed 20000`, `--episodes 2400` with the 48 pairs in PREREGISTRATION.md order (parsed from the file, 50 per pair, episode i uses pair i mod 48; asserted), final model + `final_vecnormalize.pkl` (frozen), placement = env `is_success`.
- obs_rms and ret_rms identical before and after for all six, and the pickles' sha256 unchanged (`section2_output.txt`). For the 18 Step 3 jobs `section3_eval.py` asserts the same before/after identity (the pickle sha256 check is stored in each `*_rms.json` but not asserted).
- Commands: `EVAL_MANIFEST.json` (`cmd` and `equivalent_eval_probe_cmd` per model), sha256 of models, VecNormalize files, `eval_probe.py` and the pre-registration; Step 3 in `EVAL_MANIFEST_section3.json`.
- Timing (2026-10-07, UTC+2), at most 3 eval processes at once, MemAvailable was 4.3 to 7.2 GB at the checks and never under 1.5 GB (no holds recorded in the main manifest):
{timing}
- Step 3 evals: {timing3}. The session was paused between the two parts (laptop shut down) and resumed; no eval was running during the pause.
- Scripts: `run_evals.py` (Step 1), `rms_wrapper.py`, `section2_table.py`, `section2_subsets.py`, `section3_train.py`, `run_section3.py`, `section3_eval.py`, `rs32_lib.py`, `make_readme.py`.

## 3. Step 2: main results (`section2_output.txt`, verbatim)

Funnel columns: "lift over 10 mm" is eval_probe's `grasp_ever` (undercounts); held = both pads on piece and no piece-board contact; released = S2 reached; placed = env `is_success`.
`lifted_30mm` and `carried_within_3cm` are not used (known unreliable).

```
{rd('section2_output.txt')}
```

## 4. Step 3 items a, e, f, g (`section3_eval_output.txt`, verbatim)

Item a (three subsets), e (zero-shot a and b; the 12 b/g pairs overlap none of the 48 trained pairs; of the 20 RS29 pairs only e4:e5 is a trained pair and is excluded from the 19-pair headline, n = 285), f (deterministic, 2 episodes per pair), g (eval vs training cross-check).
In item f all six seeds give identical trajectories for the two episodes of every pair (48/48), so the effective n is 48 pairs, not 96 episodes; the effective-n rows are the ones to use.
In item g the Wilson intervals were computed and compared in code (printed per line). Caveat on that comparison: the training episodes are not independent of each other (one evolving policy, 4 envs) and draw pairs at random rather than 50 each, so the training interval is only nominal.

```
{rd('section3_eval_output.txt')}
```

## 5. Step 3 items b, c, d (`section3_train_output.txt`, verbatim)

b: policy std = exp(log_std) per action dim (dx, dy, dz, gripper) from `policy.pth` in the 1M, 2M, 3M, 4M and 5M checkpoints and in `final_model.zip` (5,005,312 steps).
c: `ever_held` per 10% chunk of training (Monitor CSVs; chunk = min((cum_l - 1)*10 // total_l, 9) per file, counts pooled over the four files).
d: training placement (Monitor `terminated_reason == success`) per 100k training steps = 25k steps of each of the four files, pooled (RS31 convention).

```
{rd('section3_train_output.txt')}
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
"""
(EV / "README.md").write_text(txt + "\n")
print(len(txt.splitlines()), "lines")
