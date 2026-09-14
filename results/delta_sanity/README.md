# Delta action mode: 100k-step sanity run (15 Sep 2026)

Run: `runs/_sanity_t1_ppo_shaped_state_delta_seed0` (config `configs/_sanity_t1_ppo_shaped_state_delta.yaml`,
SHA 6cbabdd, clean tree). T1, PPO, shaped reward, state-vector, seed 0, delta mode, max_joint_delta 0.1.
Checkpoints every 20k steps. Not a success-rate result.

`ck*.json`: `scripts/eval_final.py`, 50 deterministic episodes per checkpoint, eval seed 10000 (same
50 start/target squares at every checkpoint and in the absolute-mode reference).
`saturation_*.txt`: `results/pilot_diagnosis/s4/saturation_probe.py`, 6 start squares x 100 steps.

## Approach (min pinch-to-piece distance, m; no-motion reset baseline median 0.482)

| ckpt | median | p10 | closest | eps below reset-5cm | eps < 10 cm | real grasps | success |
|---|---|---|---|---|---|---|---|
| 20k | 0.480 | 0.431 | 0.418 | 0/50 | 0/50 | 0 | 0 |
| 40k | 0.300 | 0.167 | 0.118 | 50/50 | 0/50 | 0 | 0 |
| 60k | 0.393 | 0.312 | 0.206 | 32/50 | 0/50 | 0 | 0 |
| 80k | 0.355 | 0.176 | 0.106 | 42/50 | 0/50 | 0 | 0 |
| 100k | 0.360 | 0.151 | 0.086 | 42/50 | 3/50 | 0 | 0 |

Paired 20k -> 100k: median -0.121 m, closer in 50/50 episodes, Wilcoxon p = 7.6e-10. Not monotonic
(closest median at 40k). Absolute-mode pilot, same cell and seed, at 100k: median 0.446, closest 0.292.

## Actuator force saturation (fraction of steps at forcerange; pan / lift / elbow / w1 / w2 / w3)

| model | saturated | median command-vs-actual (deg) |
|---|---|---|
| delta @20k | 0.17 / 0.17 / 0.01 / 0.00 / 0.00 / 0.00 | 7.5 / 5.1 / 3.4 / 6.7 / 3.9 / 10.9 |
| delta @100k | 0.53 / 0.66 / 0.36 / 0.36 / 0.28 / 0.00 | 13.2 / 100.2 / 5.9 / 18.2 / 16.7 / 10.0 |
| absolute @100k (pilot) | 0.79 / 0.87 / 0.89 / 0.65 / 0.58 / 0.03 | 37.4 / 120.9 / 29.0 / 51.7 / 38.2 / 3.0 |
| absolute @500k (pilot) | 0.93 / 0.98 / 0.88 / 0.95 / 0.93 / 0.35 | 109 / 264 / 29 / 219 / 139 / 48 |

Saturation is lower than absolute mode at the same budget on every joint except wrist_3,
but rises from 20k to 100k, and shoulder_lift's commanded target runs a median 100 deg ahead of the
actual joint at 100k without the joint being near its limit: the running delta target has wound up
ahead of a joint that is not following it. The commanded target is not in the observation.
