# Pilot-matrix diagnosis (15 Sep 2026)

Diagnosis only. No reward wrapper, penalty, hyperparameter or config was changed.
Models evaluated are the pilot-matrix runs in `runs/` (git SHAs e32990b for PPO,
3d60075 for SAC; see each run's manifest.json).

| Dir | Section | Produced by |
|---|---|---|
| `s1_verify_*.json` | 1: extended eval fields, 10 episodes | `scripts/eval_final.py` |
| `s2/` | 2: checkpoint sequence, 7 stable shaped cells x 5 checkpoints x 50 episodes; `_reset_baseline_pinch_to_piece.json` is the no-motion baseline | `scripts/eval_final.py --checkpoint` |
| `s3/` | 3: T1 PPO seed-0 shaped vs sparse final models, 50 episodes, shared eval seed 20000 | `scripts/eval_final.py --eval-seed 20000` |
| `s4/` | 4: tracking error a trained policy experiences; actuator saturation | `scripts/measure_tracking_error.py`, `s4/saturation_probe.py` |
| `s5/` | 5: D14 guard trigger states with per-substep replay | `s5/solver_instability_probe.py` |

## Summary

| Question | Measurement | Result |
|---|---|---|
| Does the gripper approach the piece over training? (7 stable shaped cells) | median min pinch-to-piece distance, 100k to 500k checkpoints, vs a no-motion reset baseline | 0.38-0.47 m at every checkpoint vs 0.48-0.50 m at reset. Paired 100k to 500k: decreases in 1 cell (-0.029 m), flat in 4, increases in 1 (+0.092 m), ambiguous in 1 |
| Grasps | `grasp_success` ever true, 50 episodes per checkpoint | 50 flagged episodes, 0 genuine: 49/50 never had the pinch site within 5 cm of the piece; contact replay shows toppled pieces touching only the board, so a knocked-over piece next to a closed gripper satisfies the check |
| Shaped vs sparse approach (T1 PPO seed 0) | min pinch-to-piece distance; min pinch height over the board and anywhere | piece distance: no difference (0.399 vs 0.408 m, p=0.80). Height while over the board: no significant difference (0.268 vs 0.188 m, p=0.13). Height anywhere: sparse reaches far lower OFF the board (median -0.227 m vs +0.203 m, p=2.7e-16), consistent with its table/pedestal/floor bad contacts |
| D7: same uncompensated actuators? | runtime actuator dump on the env built from the run manifest | Yes: gaintype fixed, biastype affine, kp 2000/500, gravcomp 0; action reaches data.ctrl unmodified |
| D7: steady-state droop at near vs far square | commanded vs actual joints and pinch position | Not measurable with these policies: gripper stays 0.20-0.42 m above the board, mostly outside its y span, and never quasi-static (0 of 600 steps). What the policy does experience: median command-vs-actual error 21-295 deg; big-joint actuators force-saturated on 88-99% of steps; lift/elbow actions at +-1 (target +-360 deg). D7 droop (~0.6-1.5 deg) is not the dominant effect |
| D14 trigger states (t1_ppo_sparse_seed0) | per-substep replay of each trigger | 4 events (3/100 eval seed 10000, 1/50 seed 20000), 3.9-18.8 m/s, 101-590 rad/s, 0.25-3.7 J. In every event an arm or gripper link was in contact with the resting piece while it was also on the board; the spike began with the piece pressed between them in 3 of 4. Penetrations 1-33 mm, normal forces up to 1.3 kN |

Hypothesis supported by the data: the policies do not approach the piece. They do not
approach and then fail to grasp.
