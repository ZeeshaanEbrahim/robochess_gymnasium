# D32: T2 (two fixed obstacles) against a T1 control, PPO with the D29 reward and the D31 recipe (pre-registered before any run)

Question: does PPO with the D29 configuration and the D31 recipe still place the piece on e4:e5 when two obstacles stand on d4 and f4, compared with the same recipe without obstacles?

## Cell
- Move: e4:e5 (D30), start_square e4, target_square e5, no train_pairs.
- T2: task t2, n_obstacles 2, obstacle squares fixed to d4 and f4 (the file-adjacent squares of e4, which the random T2 placement excludes by default, D15). T1 control: task t1, no obstacles.
- action_mode cartesian, max_episode_steps 300, algorithm ppo, reward_type shaped, state observation.
- D29 configuration: place_mode, place_obs, pickup_obs, place_posture_gate, place_carry_potential, ppo_mean_squash, vecnormalize_clip_reward 1000.
- D31 recipe: n_envs 4, ppo_lr_schedule linear, ppo_ent_coef 0.01.
- 3,000,000 steps. Seeds 0, 100, 200, 300, 400, for both arms (10 runs).
- Observation: T1 control 39 entries (21 base, 15 pickup, 3 place). T2 47 entries (21 base, 8 obstacle at 21:29, 15 pickup at 29:44, 3 place at 44:47). The two arms therefore differ in observation size.

## Evaluation (fixed)
- Final model only, 200 stochastic episodes per seed, eval seed 20000, torch-seeded, frozen VecNormalize statistics (obs_rms and ret_rms identical before and after), scripts/eval_probe.py as in RS32, fixed pair e4:e5.
- Success: info terminated_reason "success". For T2 this is is_success with no obstacle disturbed. T1 control: the same reason.
- Secondary (reported): placement only, i.e. terminated_reason "success" or "success_obstacles_disturbed", for T2.

## Decision rule (fixed; thresholds not changed after any result)
1. A seed learns if success is >= 100 of 200 episodes.
2. Result for an arm = number of seeds that learn, out of 5.
3. The claim "PPO with the D29 configuration and the D31 recipe learns the T2 cell" holds if at least 3 of 5 T2 seeds learn. The same rule is applied to the T1 control.

## Reported, not gated
- Share of placed episodes whose maximum piece lift exceeded 10 mm and exceeded 50 mm, for T1 and T2 (max_lift_mm per episode).
- For each T2 episode that is placed but not a success (success_obstacles_disturbed), and each T2 episode where the latch fired, whether the disturbance latch fired through tilt (obstacle tilt above 0.26 rad) or through displacement (obstacle xy displacement above 5 mm): the cause at the first step where obstacles_ever_disturbed is true; displacement if it exceeds 5 mm, tilt if only the tilt exceeds 0.26 rad, both if both. The current scripts/eval_probe.py does not record obstacle displacement or tilt; a recorder will be added to the evaluation script (not to the environment or reward) in its own commit before any T2 evaluation is run, and that commit will be named in the results.
- Pooled success over the five seeds for each arm, with Wilson 95% intervals.

## Prediction (not a gate)
Pooled T2 success within 10 percentage points of the pooled T1 control. Reasoning: in the Cartesian controller with the locked top-down tool, measured in the rs32 worktree at grasp height, the first robot-obstacle contact with an obstacle on d4 or f4 occurs after 15.5 mm (d4, -x) and 15.8 mm (f4, +x) of lateral hand travel, and the latch fires at about 25 mm of hand travel (24.9 and 25.2 mm), through tilt (obstacle tilt above 0.26 rad) while the obstacle has moved less than 1 mm. RS32 pinch offsets along x relative to the piece, at the first step both pads touched the piece, had median 0.37 mm (n 283, p90 2.56 mm, maximum 28.26 mm; deterministic 48-pair evaluation pooled over six seeds, piece-relative, results/rs32_48_pairs/eval/det48_rs32D_seed*.json; the stochastic evaluations did not record it).

## Limitations
- The e5 end is unconstrained: nothing stands on the target side, so the obstacle test applies on the lateral sides of the start square only.
- Fixed squares make the 8 obstacle observation entries constant within the cell, so the policy cannot learn to read them.
- A dragging policy can pass: RS32 found that 80% to 94% of placements never lifted the piece more than 10 mm (results/rs32_48_pairs/eval/diagnosis/DIAGNOSIS.md), and a low drag keeps the hand on the e4 file.
- No penalty exists for a brush: disturbing an obstacle only withholds the success lump (1020); the first-S2 bonus and the S2 per-step reward are still paid.
- D15 applies: T2 as configured tests clutter, not avoidance, and the file-adjacent squares are kept out of the random placement as a precaution rather than a measured requirement.
- The T1 control and the T2 runs come from different SHAs (the control from branch recipe, the T2 runs from branch t2-d29), so the two arms are not a same-commit comparison. The T2 code change is intended not to alter the T1 path; that is checked, not assumed, by the protections listed in the T2 commit.
- Five seeds per arm; a 3-of-5 rule is coarse. No attribution within the D31 bundle is claimed.
