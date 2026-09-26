# Cartesian pick-up probes: evaluation

Runs: `runs/_probe_cart_pickup_{ppo_seed0,ppo_seed1,ppo_seed2,sac_seed0}` (commit `7175f7e`):
`action_mode: cartesian`, `pickup_mode` + `pickup_obs` on, e4->e5, 300-step episodes, 1M steps.
PPO uses VecNormalize (loaded frozen for evaluation: `training=False`, `norm_reward=False`;
`obs_rms` identical before and after in every evaluation); SAC does not.

Definitions: D18 `grasp_success` = both pads on the piece AND no piece-board contact.
D21 `pickup_success` = `grasp_success` with the piece lifted >= 30 mm and held for 5 consecutive
steps (latched for the episode). g = piece position + 0.03725 m in z.

Produced by `scripts/eval_probe.py` (stochastic, seed 20000; 300 episodes at `final_model`, 100 at
the 100k / 200k / 500k checkpoints; 20 deterministic episodes at `final_model`) and aggregated by
`scripts/eval_probe_agg.py`; `agg.out` is its full output. Files: `<run>_{ck100000,ck200000,
ck500000,final,det}.json`. `ppo_seed2_final.json` keeps per-step trajectories for its first 5 of
300 episodes only (file size); its summaries cover all 300.

## Headline (Section 2): stochastic rates, 95% Wilson intervals

| Run | Model | `grasp_success` ever | `pickup_success` |
|---|---|---|---|
| ppo_seed0 | 100k / 200k / 500k | 19% / 15% / 100% | 0% / 0% / 100% |
| | **final (n=300)** | 300/300 = 100% [98.7, 100] | 299/300 = 99.7% [98.1, 99.9] |
| ppo_seed1 | 100k / 200k / 500k | 22% / 11% / 100% | 0% / 0% / 0% |
| | **final** | 299/300 = 99.7% [98.1, 99.9] | 299/300 = 99.7% [98.1, 99.9] |
| ppo_seed2 | 100k / 200k / 500k | 5% / 18% / 2% | 0% / 2% / 0% |
| | **final** | 300/300 = 100% [98.7, 100] | **1/300 = 0.3% [0.1, 1.9]** |
| sac_seed0 | 100k / 200k / 500k | 5% / 45% / 100% | 4% / 7% / 100% |
| | **final** | 300/300 = 100% [98.7, 100] | 300/300 = 100% [98.7, 100] |

Deterministic, 20 episodes at `final_model`: trajectories identical across the 20 in all four
runs; `pickup_success` 20/20 for ppo_seed0, ppo_seed1, sac_seed0 and 0/20 for ppo_seed2.

ppo_seed2 grasps (both pads on, piece off the board) but holds the piece at about 13-22 mm
(median max lift 20.8 mm), below the 30 mm bar. ppo_seed1 at 500k shows the same pattern
(`grasp_success` 100%, `pickup_success` 0%, median max lift 15.7 mm).

## Decision table (Section 3, report only)

| Run | final `grasp_success` | final `pickup_success` | `grasp_success` 500k -> final | Fisher p |
|---|---|---|---|---|
| ppo_seed0 | 300/300 = 100% [98.7, 100] | 299/300 = 99.7% [98.1, 99.9] | 100/100 -> 300/300 | 1 |
| ppo_seed1 | 299/300 = 99.7% [98.1, 99.9] | 299/300 = 99.7% [98.1, 99.9] | 100/100 -> 299/300 | 1 |
| ppo_seed2 | 300/300 = 100% [98.7, 100] | 1/300 = 0.3% [0.1, 1.9] | 2/100 -> 300/300 | 2.0e-92 |
| sac_seed0 (not in the rule) | 300/300 = 100% [98.7, 100] | 300/300 = 100% [98.7, 100] | 100/100 -> 300/300 | 1 |

Rule 1 (`grasp_success` >= 10% on at least 2 of 3 PPO seeds at `final_model`): **met** (3 of 3).

## Notes
- After `pickup_success` fires the piece stays lifted (>= 30 mm) to the end of the episode in the
  median case: 224/224 (ppo_seed0), 218/218 (ppo_seed1), 231/231 (sac_seed0) post-success steps.
- `ik_rejects` is 0 in every evaluation. `gripper_self_contact` steps appear (e.g. ppo_seed1 final:
  296 steps in 152 of 300 episodes) without ending episodes.
- Tilt in `agg.out` is tool tilt (pinch-site z-axis against straight down) unless labelled piece tilt.
