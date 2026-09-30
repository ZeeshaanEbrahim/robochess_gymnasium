# RS29: random-squares pre-registration (written 30 Sep 2026, before any training)

## Question
Does PPO with mean squash (D29, ppo_mean_squash), with the D29 reward unchanged, learn
T1 pick-and-place on random start and target squares?

Decision references: D26 = excluded_squares (d1, e1 excluded because the arm
self-collides there). D28 = place-mode reward. D29 = posture-gated grasp, carry
potential (k = 50 per metre, gamma 0.99), PPO mean squash adopted on 4/5 vs 3/5 seeds
at e4->e5.

## Training (fixed)
- Branch random-squares, base 8cf881b plus this commit.
- Configs configs/_probe_cart_randsq29_ppotanh_seed{0..5}.yaml: the D29 squash config
  with start/target unset (uniform random ordered pair, start != target),
  excluded_squares [d1, e1], max_episode_steps 400, total_timesteps 3,000,000,
  checkpoints every 100k, seeds 0 to 5, run concurrently.
- Budget 400 follows a rule fixed before the expert sweep: 300 only if every expert
  pair finishes within 200 steps; 8 of 20 exceeded 200 (max 266).
- No extension and no reruns. train.py has no resume.

## Evaluation (fixed)
- Final model only. scripts/eval_probe.py --pairs with the 20 pairs below,
  --episodes 300 (15 per pair), stochastic, eval seed 20000, torch-seeded, frozen
  VecNormalize.
- The list is fixed, not held out: training samples all 3782 ordered pairs.
- Pairs and distance bands (all solved by the scripted expert within 266 steps):
  short (<=110 mm): e4:e5, c1:c2, d2:e2, a6:a5, e8:d7, d4:e5
  mid (330-455 mm): f3:c6, h5:d3, g5:b3, c1:f6
  long (546-594 mm): h1:h8, h8:h1, a4:h4, b7:g2, b2:g7, f1:c8
  diagonal (772 mm): a1:h8, h8:a1, h1:a8, a8:h1
- Placement = env is_success (D27 release rules plus F2 settle criteria).

## Decision rules (fixed; thresholds are not changed after results)
1. A seed learns random squares if its placement rate over the 300 episodes is
   >= 20% AND it places at least once on >= 10 of the 20 pairs. (The second condition
   excludes a policy that solves only the 6 short pairs, which could reach 30%.)
2. RS29 passes if >= 4 of 6 seeds learn. Pass: T1 is complete for the benchmark
   config; proceed to T2 and T3. Fail: a diagnosis-only prompt before any change.
Rationale for 6 seeds and >= 4: about 1 in 5 seeds got stuck in D29 (2 of 10, 95%
interval roughly 6% to 51%). If the method works, the chance stuck seeds alone fail
the rule is about 10% at a 20% stuck rate and 26% at 30%.

## Reported, not gated
- Placement per pair (Wilson 95% intervals) and per band; pairs starting < 0.4 m vs
  >= 0.4 m (closeness term is 0 beyond 0.4 m; 29.2% of training pairs start there).
- Grasp, lift, carry-within-3 cm and release rates.
- Stuck-seed classification with the D29 failure taxonomy.
- Truncated-in-S2 tripwire (>= 10%), shaping total per episode.
- Training-log placement by band (Monitor carries start_square, target_square).
- e4->e5 anchor vs D29 squash (92-99% on passing seeds); confounded by budget 400 vs
  300 and multi-pair training.
- Checkpoint curve at 1M and 2M, same protocol.

## Known caveats
- The stuck-seed rate is uncertain (n = 10).
- Carry potential shaping does not change which policy is optimal (its discounted idle
  total is 96-98% of the carry gain); the incentive to carry comes from the placement
  reward.
- Episode return is not comparable across pairs (idle shaping is 0.5 x d_xy per step).
- The scripted expert releases more than 5 mm above rest on all 14 non-rank-1 starts
  (legal under D27, not investigated).
- SAC is not part of RS29.
