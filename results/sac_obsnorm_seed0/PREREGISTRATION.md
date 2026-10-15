# SAC observation normalisation, seed 0: pre-registration (written 4 Oct 2026, before any training)

> **The rule in the section "Pre-registered rule (4 Oct, SUPERSEDED)" below is SUPERSEDED** by the section
> "Superseding rule, 5 Oct, written before any 1M result exists" at the end of this file. The 4 Oct text is kept
> unchanged for the record. The Question and Training sections still apply.


## Question
Does D29 SAC (e4->e5, posture gate + carry potential) place when its observations are
normalised as PPO's are (`sac_obs_normalize: true`)? The flag exists for parity with PPO,
so the PPO vs SAC comparison is not confounded by one learner seeing raw inputs. It is not
expected to fix the D29 SAC failure. n = 1 seed, so the outcome decides the next step; it is
not a general claim about SAC.

## Training (fixed)
- Branch sac-obsnorm (flag implemented in 9450774), plus the commit that adds this file and the config.
- Config `configs/_probe_cart_place30_sac_obsnorm_seed0.yaml` = the D29 SAC config
  (`_probe_cart_place29_sac_seed0`) with `sac_obs_normalize: true`; only the flag and
  `run_name` differ (the `notes` string is unchanged from D29).
- Normalisation: observation-only VecNormalize, `norm_obs_keys` ["observation"],
  `clip_obs` 10, `gamma` 0.99, `norm_reward` False. `desired_goal` and `achieved_goal` stay raw.
- 1,000,000 steps, seed 0, checkpoints every 100k, one run, no reruns. train.py has no resume.
- D29 SAC baselines (run `_probe_cart_place29_sac_seed0`, 300 stochastic episodes, seed 20000):
  placement 0/300, per-episode min d_xy median 37.9 mm, S1 lift median 110.7 mm.

## Pre-registered rule (4 Oct, SUPERSEDED)
Verbatim from the decision of 4 Oct 2026:

* Eval: final model and the 500k checkpoint, 300 stochastic episodes, eval seed 20000, torch-seeded, frozen VecNormalize loaded via `uses_vecnormalize`. Report placement (env success), grasp, lift >= 30 mm, per-episode min d_xy, S1 lift (median, p10, p90) and S0 to S3 occupancy, against the D29 SAC baselines (placement 0/300, min d_xy median 37.9 mm, S1 lift median 110.7 mm).
* Placement >= 10% at the final model: obs-norm on becomes the SAC configuration for factorial tiers 0 and 1.
* Placement > 0 and < 10%, and higher at 1M than at 500k: extend to 2M before judging.
* Placement 0: obs-norm alone is insufficient. The reward is not changed (it is shared with PPO). The next step is chosen from the diagnostics, in a separate decision.

## Superseding rule, 5 Oct, written before any 1M result exists
This section replaces the 4 Oct rule above. Only a 20k-step smoke run exists at the time of writing; no 1M
result does.

- **Run:** 1 seed (0), 1,000,000 steps, e4->e5 only (D30, the fixed move for T1), SAC with
  `sac_obs_normalize` true, D29 settings otherwise. D29 is the 27 Sep comparison where PPO reached about 92 to
  99% on e4->e5 and SAC placed 0/300.
- **Gate:** evaluate the FINAL model on 300 episodes, stochastic, eval seed 20000, with frozen VecNormalize
  statistics (`final_vecnormalize.pkl`; `training` False, `norm_reward` False), via `scripts/eval_probe.py`.
  **Pass = placement of 60/300 (20%) or more.** The D29 SAC baseline is 0/300. This gate does not change after
  results.
- **Pass** means the observation-normalisation fix works and SAC enters the factorial. Reading tiers,
  descriptive only: 60 to 239 out of 300 means a documented weaker SAC cell; 240 or more (80%) meets the
  development target.
- **Descriptive reporting, no effect on pass or fail:** Wilson 95% interval, `terminated_reason` counts, grasp,
  lift of 30 mm or more, min d_xy, S1 lift, S0 to S3 occupancy. Reported for the final model and for the 500k
  checkpoint (same 300 episodes and seed, using that checkpoint's own VecNormalize pickle).
- **Fail** (below the gate): first report the `critic_loss` and `actor_loss` trends from the training log
  (minus `actor_loss` is a rough Q proxy) and placement at 500k vs 1M. D17 (SAC critic divergence, likely raw
  reward scale, which observation normalisation does not address) is a competing cause. Then the next step is
  chosen in a separate dated decision. Candidates: finite-horizon bootstrapping (after about step 149, holding
  on target looks better than placing), D17, and a 2M-step run if placement rises from 500k to 1M.
- **Replaced from the 4 Oct text:** the 30/300 (10%) gate, the 500k-checkpoint gate, and the automatic 2M
  branch. The 500k checkpoint is now descriptive only, and 2M is a candidate for a later decision, not automatic.

## Interruption note, 6 Oct
- The first launch (SHA f20916f, tmux session sac_obsnorm_1m, started 00:02:32 SAST on 6 Oct) stopped at about
  250,000 steps at about 02:09 SAST, when its process tree was killed: no EXIT line, no traceback, no OOM.
  PID 1 of the WSL distribution started at 02:10:36 SAST, one minute after the stop.
- No model from that run was evaluated. Only its training-log step counts, fps and loss lines were viewed; no
  evaluation result of any kind was seen. Its outputs are kept, renamed, as
  `runs/_probe_cart_place30_sac_obsnorm_seed0_aborted_250k` (and the matching `_aborted_250k_train.log`) and
  are not evaluated.
- The relaunch uses the identical config, seed 0 and code. Its diff from f20916f is this documentation change
  only. Every rule above is unchanged, and the gate (the Superseding rule, 5 Oct) applies to the final model of
  the relaunch.

## Seeds 1 to 3, 16 Oct
- Written 16 Oct 2026, after the seed-0 result (`RESULT.md`, d6b06e0: 0/300 against the 60/300 gate) and before any
  seed 1 to 3 result exists.
- These are the factorial T1 SAC seeds: configs `configs/_probe_cart_place30_sac_obsnorm_seed{1,2,3}.yaml`, identical
  to the seed-0 config (`sac_obs_normalize` true, e4->e5, 1,000,000 steps, checkpoints every 100k) except `seed` and
  `run_name`.
- Same protocol as above: final model, 300 episodes, stochastic, eval seed 20000, frozen VecNormalize statistics
  (`final_vecnormalize.pkl`; `training` False, `norm_reward` False), via `scripts/eval_probe.py`.
- Per-seed gate: 60/300 (20%) placement. No threshold changes.
- The cell is reported as k seeds passing out of N, whatever the outcome.
