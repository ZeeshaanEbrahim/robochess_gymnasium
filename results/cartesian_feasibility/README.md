# Cost and reliability of a per-step mink IK solve (20 Sep 2026)

Measurements only. No recommendation, no implementation, no env/config/default change.

Context: evaluating a Cartesian end-effector action space, where the policy commands dx, dy, dz
of the pinch site plus a gripper value and `mink` solves IK every macro step.

> **Correction, 20 Sep 2026.** An earlier version of this file reported ~5 ms per solve, 11 solver
> iterations regardless of displacement, and a 62% training-throughput cost. Those figures were
> wrong, and the fault was in the measurement harness, not the solver. It built the env in DELTA
> mode and drove it with `ScriptedExpert._drive_to`, which issues ABSOLUTE joint targets; delta mode
> reinterpreted them as per-step deltas, so the arm drifted 557 mm from the intended pose and ended
> 96 deg away from the fixed top-down `tool_rotation` that `solve_ik()` always demands. Every
> "1 mm displacement" solve was therefore really "re-orient the tool by 96 deg, then move" -- which
> is exactly why cost looked flat from 1 mm to 20 mm. Corrected figures below: **~0.4 ms per solve,
> one iteration, about 10% throughput cost.** Section 4b's sweep was also mislabelled as
> "warm-started from the previous solution" when `solve_ik()` reseeds from the env's qpos every
> call and the env was never advanced; it has been re-run with the seed genuinely advancing, and
> its conclusion is unchanged.

Scripts: `measure_ik_cost.py` (section 3), `measure_ik_reliability.py` (section 4). JSON output:
`ik_cost.json`, `ik_reliability.json`. Run from the repo root with the venv active.

## 3a. How solve_ik() is configured

`mink.solve_ik` with solver `daqp`, `dt = 0.02`, damping `1e-3`, wrapped in a loop with
`IK_MAX_ITERS = 400` as a **cap**, and tolerances `IK_POS_TOL = 1e-3 m` and `IK_ORI_TOL = 1e-2 rad`
checked at the top of every iteration, returning as soon as both hold. It is a convergence test
with a cap, not a fixed iteration budget.

## 3a (cont). One solve: small displacement, arm seated at a valid top-down solution

Arm seated AT an IK solution (qpos set directly, then `mj_forward`), each solve re-seated at its own
solution so the chain matches what a Cartesian action space would produce. 200 solves per size.

| displacement | median | p95 | max | iterations median / max | residual (median) | failures |
|---|---|---|---|---|---|---|
| 1 mm | 0.056 ms | 0.507 ms | 0.799 ms | 0 / 1 | 0.999 mm | 0/200 |
| 5 mm | 0.406 ms | 0.664 ms | 0.806 ms | 1 / 1 | 0.069 mm | 0/200 |
| 20 mm | 0.451 ms | 1.154 ms | 184.123 ms | 1 / 400 | 0.452 mm | 3/200 |

A genuine small displacement converges in **one** iteration.

## 3b. Error after a truncated number of iterations (warm-started 5 mm, 50 solves)

| iterations | median pos err | p95 pos err |
|---|---|---|
| 1 | 0.0675 mm | 0.0944 mm |
| 2 | 0.0675 mm | 0.0944 mm |
| 3 | 0.0675 mm | 0.0944 mm |
| 5 | 0.0675 mm | 0.0944 mm |

Identical from 1 onward because the solve is already finished. There is nothing to buy past the
first iteration. (`converged` reads 0/50 at a cap of 1 purely as a counting artefact: the flag is
set by the re-check at the top of the NEXT iteration, so a cap of 1 cannot set it even when the
error is already 0.04 mm. Verified directly.)

## 3c. Cost and projected throughput

Env macro step: median 1.235 ms, mean 1.381 ms -> 724 steps/s env only. Training loop measured at
264 steps/s on the e4->e5 1M cell (env + PPO).

| basis | solve ms | env-only steps/s | training steps/s | 1M steps |
|---|---|---|---|---|
| no IK (today) | -- | 724 | 264 | 1.05 h |
| 1 mm | 0.056 | 696 | 260 | 1.1 h |
| 5 mm | 0.406 | 559 | 238 | 1.2 h |
| 20 mm | 0.451 | 546 | 236 | 1.2 h |

About **10%** of training throughput, and the solve is roughly **0.3x** the physics step it sits on.

### Two caveats

* **Sub-1 mm commands are exact no-ops.** `IK_POS_TOL` is 1e-3 m, so a 0.5 mm request is already
  inside tolerance: `solve_ik()` returns the seed configuration byte-identically (joint change
  0.00e+00 rad) and the pinch does not move. A Cartesian action space whose per-step displacement
  can fall below 1 mm would have a dead zone at the bottom of its range.
* **The 20 mm chain has a tail.** 3 of 200 solves hit the 400-iteration cap, one taking 184 ms --
  400x the median. Rare, but a per-step solve inside a training loop would meet it occasionally.

## 4a. Small-displacement convergence across the board

30 points (files a, c, e, g, h x ranks 1, 2, 4, 5, 7, 8; all four corners included) at `GRASP_Z`,
each solved in +x, -x, +y, -y, +z, -z with a 5 mm displacement, arm seated at the square first.

| region | converged |
|---|---|
| files a-b, ranks 1-2 | 12/12 |
| files a-b, ranks 3-6 | 12/12 |
| files a-b, ranks 7-8 | 12/12 |
| files c-f, ranks 1-2 | 24/24 |
| files c-f, ranks 3-6 | 24/24 |
| files c-f, ranks 7-8 | 24/24 |
| files g-h, ranks 1-2 | 24/24 |
| files g-h, ranks 3-6 | 24/24 |
| files g-h, ranks 7-8 | 24/24 |
| **total** | **180/180 (100%)** |

Zero failures, zero solves hitting the iteration cap, in every region including the corners.

## 4b. Continuity along a genuinely warm-started sweep

a4 -> h4 at grasp height, 20 mm increments, 28 solves, the arm seated at each solution before the
next solve. 0 failures.

| joint | largest single-step change (rad) |
|---|---|
| shoulder_pan | 0.0430 |
| shoulder_lift | 0.0146 |
| elbow | 0.0288 |
| wrist_1 | 0.0142 |
| wrist_2 | 0.0000 |
| wrist_3 | 0.0430 |

Largest change on any joint: **0.0430 rad**. Discontinuities above the 0.2 rad flag threshold:
**0**. For scale, that is under half the delta action space's 0.1 rad per-step cap. (The earlier
mislabelled version of this sweep gave 0.0429 rad and the same zero flips.)

## 4c. What solve_ik() does with an unreachable request

| request | result |
|---|---|
| +2 m in x (far outside reach) | raises `IKFailure`; env `qpos` unchanged; the solver's internal `mink.Configuration` has moved 1.571 rad |
| +0.5 m in y | raises `IKFailure`; same pattern, internal configuration moved 1.571 rad |
| -1 m in z (through the table) | **returns a pose, no exception**; 2.016 rad from the previous joint configuration |

Two distinct behaviours. Out-of-reach targets raise, leaving the env untouched but the solver's own
configuration displaced ~1.57 rad, so a subsequent solve seeded from it would begin from a pose that
is not where the arm is. A target unreachable because geometry is in the way rather than because it
is out of reach converges to a pose passing through the table, 2.016 rad from the previous one --
47x the largest step in the 4b sweep and 20x the delta cap.
