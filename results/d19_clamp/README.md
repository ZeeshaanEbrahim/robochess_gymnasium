# D19 windup clamp: 100k-step sanity re-run (15 Sep 2026)

Run: `runs/_sanity_t1_ppo_shaped_state_delta_clamp_seed0` (config `configs/_sanity_t1_ppo_shaped_state_delta_clamp.yaml`,
SHA cd39ec2, clean tree). Identical to `results/delta_sanity/` (T1, PPO, shaped, state, seed 0, delta 0.1 rad/step,
checkpoints every 20k) except that delta targets are clamped to within 2 x forcerange/kp of the actual joint
(0.15 rad pan/lift/elbow, 0.112 rad wrists). Not a success-rate result.

- `clamp_ck*.json`: `scripts/eval_final.py --episodes 50 --eval-seed 10000` (same episodes as `results/delta_sanity/ck*.json`).
  Grasps use the pad-contact grasp_success of 93b4ee6 (evaluated before the board-contact term was added).
- `absolute_pilot_ck100000.json`: same eval on `runs/t1_ppo_shaped_state_seed0_seed0` at 100k.
- `saturation_*.txt`: `results/pilot_diagnosis/s4/saturation_probe.py` (6 start squares x 100 steps), run with the repo
  root taken from `$REPO`. The unclamped checkpoints were probed from a worktree at b434fac (pre-clamp code), so the clamp
  never acted on them; its 20k and 100k outputs reproduce `results/delta_sanity/` exactly.
- `table.md`: `python results/d19_clamp/compare.py`.

## Result

See `table.md`. Summary:

- Windup removed: median |target - actual| holds at 5-7 deg on every joint at every checkpoint (unclamped: lift 58-100 deg by 40k-100k).
- Saturation no longer rises over training (clamped 100k: 0.17/0.23/0.29/0.20/0.08/0.00 vs unclamped 0.53/0.66/0.36/0.36/0.28/0.00)
  but is not low: 20-42% on pan/lift/elbow/w1 at some checkpoints, and higher than unclamped at 20k on elbow (0.42 vs 0.01) and
  wrist_1 (0.38 vs 0.00). Expected from the margin choice: a 2x margin permits errors between 1x (the saturation point) and 2x.
- Approach: median min pinch-to-piece 0.427 / 0.421 / 0.318 / 0.258 / 0.266 m -- nearly but not strictly monotonic (+8 mm at 100k).
  Paired 20k->100k: -0.147 m, closer in 47/50, Wilcoxon p=8.5e-9. Paired vs unclamped at 100k: -0.052 m, closer in 31/50, p=0.015.
- 0 real grasps, 0 successes at every checkpoint in all three runs.
