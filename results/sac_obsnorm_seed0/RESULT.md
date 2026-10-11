# SAC observation normalisation, seed 0: result (written 11 Oct 2026)

**Gate not met (0/300 vs 60/300).**

This records the result against `PREREGISTRATION.md` (the 5 Oct superseding rule). Descriptive only: no cause is
concluded here. The next step is chosen in a separate dated decision.

## Run
- Config `configs/_probe_cart_place30_sac_obsnorm_seed0.yaml` (D29 SAC config, `sac_obs_normalize: true`), seed 0,
  1,000,000 steps, e4->e5, checkpoints every 100k with VecNormalize pickles.
- Relaunch at SHA `fb1c0d8` (clean tracked tree). The first launch (`f20916f`) was killed at about 250,000 steps; see
  the interruption note in the prereg. The relaunch's 100k and 200k checkpoint parameter digests are EQUAL to the
  aborted run's (below).
- `EXIT=0`, final logged step 999,863, wall clock 7:43:46, peak RSS 1,485,272 KB. PID 1 start time
  (`Tue Oct 6 02:10:36 2026`) was the same before and after the run.

## Evaluation
`scripts/eval_probe.py`, 300 episodes, stochastic, eval seed 20000, torch-seeded, frozen VecNormalize
(`final_vecnormalize.pkl` for the final model, `rc_vecnormalize_500000_steps.pkl` for the 500k checkpoint).
`obs_rms_identical_before_after` is True for both. Placement is env `is_success` at the last step.

| | Final model | 500k checkpoint | D29 SAC baseline (re-evaluated, like-for-like, raw obs) |
|---|---|---|---|
| Placement | **0/300**, Wilson 95% [0.00%, 1.26%] | 0/300, [0.00%, 1.26%] | 0/300, [0.00%, 1.26%] |
| `terminated_reason` | time_limit 300 | time_limit 300 | time_limit 300 |
| Grasp (`grasp_ever`) | 300/300 | 300/300 | 300/300 |
| Lift >= 30 mm | 300/300 | 300/300 | 300/300 |
| Carried within 3 cm | 300/300 | 297/300 | 0/300 |
| Reached S2 ("released") | 0/300 | 0/300 | 0/300 |
| Per-episode min d_xy (mm) | median 0.2 (p10 0.1, p90 0.4, max 2.2) | median 3.0 (p10 0.8, p90 7.1, max 78.0) | median 37.2 (p10 33.8, p90 42.4, max 48.5) |
| S1 lift (mm), pooled S1 steps | median 344.6 (p10 166.5, p90 364.7) | median 376.1 (p10 164.5, p90 382.8) | median 110.3 (p10 83.9, p90 118.5) |
| S0 / S1 / S2 / S3 occupancy (90,000 steps) | 18.37% / 81.62% / 0.00% / 0.01% | 18.89% / 81.09% / 0.00% / 0.02% | 18.37% / 81.63% / 0.00% / 0.00% |

- Baseline figures quoted in the prereg and earlier reports (D29 SAC, 300 stochastic episodes, seed 20000): placement
  0/300, per-episode min d_xy median 37.9 mm, S1 lift median 110.7 mm. The like-for-like re-evaluation above ran the
  D29 SAC `final_model.zip` (read from the main repo's `runs/`, output written only to this worktree's
  `runs/_eval_sac_d29_baseline/`) through the current code with the identical protocol and no VecNormalize.
- First S1 step: step 54 to 56 in every final-model episode; no episode failed to reach S1. No episode had `is_success`
  at any step.
- S1 d_xy (mm), pooled S1 steps: final median 2.7 (p10 1.0, p90 61.2); 500k median 7.9 (p10 4.3, p90 65.8); D29
  baseline median 45.9 (p10 39.0, p90 88.8).
- `min_d_mm` in the official eval JSON is pinch-to-grasp-point 3D distance, not the piece-to-target `d_xy` above:
  final median 3.6 mm, 500k median 1.3 mm.
- Method: per-step `d_xy`, state and lift come from scratchpad copies of `eval_probe.py` that also write per-step
  fields (no repo code changed). Their per-episode summaries and `place_report` are identical to the official
  `eval_probe.py` JSON for the final model and the 500k checkpoint. Outputs are in `runs/_eval_sac_obsnorm_seed0/`
  and `runs/_eval_sac_d29_baseline/` (gitignored).

## Training log (`progress.csv`)
Mean over a +-5k-step window (about 8 or 9 logged rows each; the 1M window is one-sided, 5 rows).

| Step | critic_loss | actor_loss | ent_coef | ent_coef_loss |
|---|---|---|---|---|
| 100k | 38.88 | -177.25 | 0.0906 | 0.1332 |
| 250k | 26.10 | -344.23 | 0.1350 | 0.1656 |
| 500k | 17.57 | -455.12 | 0.0853 | -0.1893 |
| 750k | 7.51 | -491.96 | 0.0588 | -0.1976 |
| 1M | 8.02 | -508.47 | 0.0534 | -0.0350 |

Over the whole run `critic_loss` peaked at 411.59 at step 200,869 and ended at 11.42.

## Checkpoint digests, relaunch vs aborted run
`state_digest` = sha256 over `policy.state_dict` tensors and `log_ent_coef` (function from `scripts/repro_check.py`).
- 100k: EQUAL (`79bb1f9a00b7e1e40f894c003eb2ddfad98f3bdb4ffebaba959b117a7c09a67f`).
- 200k: EQUAL (`6b3de637c6a6a0d98c3f4c1350485eecc9bdf53be6474a437b65c28fc5619814`).

## Diagnostics (final model, 300 episodes, seed 20000; descriptive)

### 1. Truncation (code read, plus one measurement on the real stack)
- `RoboChessEnv.step` returns `terminated=False, truncated=True` at `elapsed_steps` 300 (measured: reason None). No
  wrapper changes either flag.
- `DummyVecEnv.step_wait` sets `info["TimeLimit.truncated"] = truncated and not terminated`, which is True here
  (measured), and stores `info["terminal_observation"]` before the reset.
- `handle_timeout_termination` is True for the replay buffer (`DictReplayBuffer` default; the training run builds it
  with no override). The buffer stores `timeouts` from `TimeLimit.truncated` (`buffers.py:641-642`) and returns
  `dones * (1 - timeouts)` at sampling (`buffers.py:688-690`). Measured on the 300th transition: `dones` 1.0,
  `timeouts` 1.0, sampled `(1 - done)` factor 1.0.
- Critic target at the last step of an episode: `r + (1 - done) * gamma * (min Q_target(s', a') - ent_coef * log pi)`
  (`sac.py`, target block), with `done` zeroed by the timeout, so it bootstraps from the stored terminal observation
  (which differs from the next episode's reset observation; max abs diff 1.67 in the measurement). The 39-entry
  observation contains no step counter.
- Causes of episode end (`robochess_env.py`, step; precedence in this order): `solver_instability` (truncated, not
  terminated), `success` (terminated; so success ends the episode), `piece_off_board` (terminated), `bad_contact`
  (terminated), time limit at `max_episode_steps` 300 (truncated). `PENALISED_TERMINATIONS` is `bad_contact` and
  `piece_off_board`.

### 2. Actions over S1 steps (73,458 steps)
Raw policy output = the action passed to `env.step` (stochastic sample, already in [-1, 1]). Executed = the
displacement actually sent to IK, `(target after box and radial clamps - measured pinch) / 0.02 m`, from a wrap of
`_solve_cartesian_ik` on the instance (no repo edit).

| | p10 | p50 | p90 |
|---|---|---|---|
| raw dx | -0.516 | -0.021 | 0.542 |
| raw dy | -0.576 | 0.068 | 0.633 |
| raw dz | -0.581 | 0.156 | 0.817 |
| raw gripper | 0.919 | 0.974 | 0.994 |
| executed dx | -0.516 | -0.021 | 0.542 |
| executed dy | -0.576 | 0.068 | 0.633 |
| executed dz | -0.581 | 0.148 | 0.813 |
| executed gripper ctrl (0 open .. 255 closed) | 244.703 | 251.644 | 254.197 |

- Fraction of S1 steps with raw |dz| > 0.95: 0.0160 (dz > +0.95: 0.0155; dz < -0.95: 0.0005). Executed |dz| > 0.95:
  0.0159.
- Pinch z in S1: median 1.1469 m, p90 1.1684 m, max 1.1869 m, against the box ceiling 1.1822 m (median 35.3 mm below,
  p90 13.8 mm below; 0.0189 of S1 steps within 5 mm of the ceiling).
- Fraction of S1 steps where the box clip changed the target: 0.0217 (all of it the z ceiling; z floor 0.0000). The
  radial and inner limits changed it on 0.0000 of S1 steps. IK rejected on 0.0000.
- Gripper command (-1 open, +1 closed; ctrl 0 to 255) in S1: 0.7631 of steps above 0.95, 0.2369 between -0.95 and
  0.95, 0.0000 below -0.95; mean 0.962. Gripper joint qpos in S1: median 0.648 (p10 0.648, p90 0.650).
- Raw dz by episode-step bucket in S1 (p10 / p50 / p90): steps 50-99 0.383 / 0.779 / 0.941; 100-149 -0.637 / -0.006 /
  0.642; 150-199 -0.622 / 0.006 / 0.647; 200-249 -0.622 / 0.004 / 0.645; 250-299 -0.611 / 0.005 / 0.643.

### 3. Reward per S1 step
Env reward (n = 73,458): mean 5.7978, p10 5.0961, p50 5.9009, p90 6.0093 (min 2.8508, max 10.0985). **100 x mean =
579.78**, next to minus `actor_loss` at 1M (window mean) = **508.47**. S1 means by component: `state_reward` 5.7353,
`shaping` 0.0217. All-steps mean reward 4.8201; S0 steps 0.4777.

### 4. Critic counterfactual
100 S1 states (25 episodes sampled at each of steps 100, 150, 200 and 250; S1 at the step and the step before),
observations normalised with `final_vecnormalize.pkl` first (no entry at the +-10 clip; normalised range -2.78 to
1.87). Mean of min(Q1, Q2):

| Action | all 100 | t=100 | t=150 | t=200 | t=250 |
|---|---|---|---|---|---|
| policy (deterministic) | 565.69 | 564.89 | 565.80 | 565.36 | 566.69 |
| same, dz = -1 | 564.93 | 564.67 | 564.83 | 564.31 | 565.90 |
| same, dz = +1 | 565.65 | 564.91 | 565.72 | 565.26 | 566.69 |
| same, gripper open (a3 = -1) | 461.22 | 465.10 | 460.06 | 458.93 | 460.78 |
| same, dz = -1 and open | 455.55 | 475.78 | 446.88 | 452.12 | 447.40 |

Share of states where Q(variant) > Q(policy action): dz = -1 **15.0%** (t=100 20%, t=150 16%, t=200 16%, t=250 8%);
dz = +1 50.0%; open 0.0%; dz = -1 and open 0.0%. Q(dz=-1) - Q(policy): mean -0.759, median -0.433, p10 -1.986,
p90 0.073. Q(open) - Q(policy): mean -104.469. Deterministic policy action at these states: mean dz 0.085 (p10 / p50 /
p90 -0.454 / 0.108 / 0.581). Median |Q1 - Q2| at the policy action 0.363 (max 2.677).

### 5. State definitions (`PlaceReward` state labels, as `place_eval.py` consumes them as `st`)
- S0: not `ever_held` (the pads have never both been on the piece with no piece-board contact).
- S1: `ever_held` and `pinched` (both pads touching the piece; board contact is allowed).
- S2: `ever_held`, not pinched, `release_ok` (at the last pinched step the piece was within `goal_tolerance` 0.03 m of
  the goal in XY and its base at most `PLACEMENT_RELEASE_MAX_HEIGHT` 0.015 m above the board), `piece_on_square`
  (XY distance to the goal < 0.03 m) and piece tilt < `PLACE_S2_TILT_DEG` 10 deg.
- S3: `ever_held`, not pinched, not S2.
- "Reached S2" in the tables above is `place_eval.py`'s `released`: `any(row.st == "S2")` over the episode.
- "Lift >= 30 mm" is a step with `st == S1`, pinched and `lift >= 0.03 m`. "Carried within 3 cm" is a later S1 step with
  `d_xy < 0.03 m`. "Held" is the D27 condition (both pads on the piece, no piece-board contact).

### 6. D29 baseline re-evaluation
Done: see the baseline column in the evaluation table. Read access to the main repo's `runs/_probe_cart_place29_sac_seed0`
only; every output is under `runs/_eval_sac_d29_baseline/` in this worktree; no file in the main repo's run directory
was modified.
