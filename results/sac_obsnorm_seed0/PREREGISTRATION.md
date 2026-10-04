# SAC observation normalisation, seed 0: pre-registration (written 4 Oct 2026, before any training)

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

## Pre-registered rule
Verbatim from the decision of 4 Oct 2026:

* Eval: final model and the 500k checkpoint, 300 stochastic episodes, eval seed 20000, torch-seeded, frozen VecNormalize loaded via `uses_vecnormalize`. Report placement (env success), grasp, lift >= 30 mm, per-episode min d_xy, S1 lift (median, p10, p90) and S0 to S3 occupancy, against the D29 SAC baselines (placement 0/300, min d_xy median 37.9 mm, S1 lift median 110.7 mm).
* Placement >= 10% at the final model: obs-norm on becomes the SAC configuration for factorial tiers 0 and 1.
* Placement > 0 and < 10%, and higher at 1M than at 500k: extend to 2M before judging.
* Placement 0: obs-norm alone is insufficient. The reward is not changed (it is shared with PPO). The next step is chosen from the diagnostics, in a separate decision.
