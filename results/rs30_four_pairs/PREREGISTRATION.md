# RS30: four-pair training, entropy bonus off/on (pre-registered 4 Oct 2026)

## Question
Can PPO with mean squash (D29 config) learn several fixed moves from different start
squares, and does an entropy bonus change that? Motivation: RS29 (random squares) failed
0/6; its policy std fell below 0.1 on all dimensions by about 1.1M steps in the seeds
examined, at the same time grasping peaked and then declined (a timing coincidence in
2 seeds, not tested as a cause).

## Training (fixed)
- Training pairs, uniform per episode: e4:e5, d4:d5, f4:f5, e3:e4 (scripted expert
  145-156 steps each).
- Arm A: D29 squash config (SB3 PPO defaults, n_envs 1, VecNormalize obs and reward,
  reward clip 1000, 300-step episodes) with train_pairs. ent_coef 0.0.
- Arm B: identical to A plus ent_coef 0.01 (single untuned value).
- 2,000,000 steps, seeds 0-3 per arm, checkpoints every 100k, run concurrently.
  No extension, no reruns.

## Evaluation (fixed)
- Final model only. scripts/eval_probe.py --pairs e4:e5,d4:d5,f4:f5,e3:e4,
  --episodes 300 (75 per pair), stochastic, eval seed 20000, torch-seeded, frozen
  VecNormalize. Placement = env is_success.

## Decision rules (fixed; thresholds not changed after results)
1. A seed learns four pairs if its placement rate is >= 20% on EACH of the 4 training
   pairs (75 episodes each).
2. An arm passes if >= 3 of its 4 seeds learn.
3. Interpretation fixed in advance:
   - A passes: several start squares are learnable with the D29 config.
   - A fails, B passes: the entropy bonus is needed at this budget; it becomes the
     candidate for any wider follow-up.
   - Both fail: learning more than one start square is not achieved at this budget;
     the generalisation track stops and is documented.
   - Both pass: the entropy bonus is not necessary at four pairs.
4. D30 (fixed-move T1 benchmark) stands unless a change is decided before 12 Oct.

## Reported, not gated
- Per-pair placement with Wilson 95% intervals, per seed and arm.
- Held-out near pairs d5:d6 and c4:c5 (zero-shot, 75 episodes each) and the 20 RS29
  pairs (zero-shot, 300 episodes), same protocol.
- Policy std (exp log_std) at 100k, 300k, 500k, 1M, 2M per seed; Monitor ever_held
  per 10% chunk.
- e4:e5 placement vs D29 squash (92.3, 0, 96.0, 99.3, 97.3).

## Known caveats
- 4 seeds per arm: with about 1 in 5 seeds stuck, a working arm fails the rule about
  18% of the time by chance.
- All training moves are one rank forward; direction is not varied.
- ent_coef 0.01 is one untuned value.
