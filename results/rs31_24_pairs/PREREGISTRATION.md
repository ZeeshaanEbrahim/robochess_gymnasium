# RS31: PPO on 24 start squares, entropy bonus 0.01 (pre-registered before any run)
Question: can PPO with ent_coef 0.01 learn the D29 place task from 24 start squares?
Training pairs (24, uniform per episode): c2:c3 c3:c4 c4:c5 c5:c6 c6:c7 c7:c8 d2:d3 d3:d4 d4:d5 d5:d6 d6:d7 d7:d8 e2:e3 e3:e4 e4:e5 e5:e6 e6:e7 e7:e8 f2:f3 f3:f4 f4:f5 f5:f6 f6:f7 f7:f8
Arms: C = RS30-B recipe (ent_coef 0.01, n_envs 1, constant lr 3e-4). D = C + n_envs 4 + linear lr decay to 0. D is a bundle; no attribution to either change is claimed. Seeds 0 and 1 per arm. 3,000,000 steps, 300-step episodes, SB3 defaults otherwise (n_steps 2048 per env, batch 64).
Evaluation (fixed): final model only. scripts/eval_probe.py --pairs <the 24 pairs in the order above> --episodes 1200 (50 per pair), stochastic, eval seed 20000, torch-seeded, frozen VecNormalize. Placement = env is_success.
Decision rules (fixed; thresholds not changed after results):
1. A seed learns if pooled placement is >= 50% AND placement is >= 20% (10 of 50) on EVERY one of the 24 pairs.
2. Arm result = number of seeds that learn (0, 1 or 2). No pass/fail threshold (n = 2).
3. D replaces C as the multi-start recipe only if D learns on more seeds than C. Ties keep C.
4. If no seed in either arm learns: 24 pairs at 3M steps is not achieved; document it; the 4-pair RS30 result remains the multi-start claim.
5. Target tier, reported: pooled placement >= 80%.
Reported, not gated: per-pair placement with Wilson 95% intervals; number of pairs under 20%; policy std at 500k, 1M, 2M, 3M; ever_held per 10% chunk of training (Monitor CSV); training-time rolling placement per 100k steps; zero-shot held-out: (a) 12 pairs on files b and g: b2:b3 b3:b4 b4:b5 b5:b6 b6:b7 b7:b8 g2:g3 g3:g4 g4:g5 g5:g6 g6:g7 g7:g8 (25 per pair, 300 episodes); (b) the 20 RS29 pairs (300 episodes); deterministic vs stochastic placement on the 24 pairs.
