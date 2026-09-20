# Cost and reliability of a per-step mink IK solve (20 Sep 2026)

Measurements only. No recommendation, no implementation, no env/config/default change.

Context: evaluating a Cartesian end-effector action space, where the policy commands dx, dy, dz
of the pinch site plus a gripper value and `mink` solves IK every macro step. These numbers say
what that would cost and how reliably today's `solve_ik()` behaves under that usage pattern.

Scripts: `measure_ik_cost.py` (section 3), `measure_ik_reliability.py` (section 4). JSON output:
`ik_cost.json`, `ik_reliability.json`. Both take the repo root from their own path; run from the
repo root with the venv active.

## 3a. One solve_ik() call: small displacement, warm-started from the current configuration

Arm parked above the piece at e4, 200 solves per size, random directions, `solve_ik()` as it stands.

| displacement | median | p95 | max | iterations median / max | failures |
|---|---|---|---|---|---|
| 1 mm | 5.087 ms | 6.855 ms | 7.736 ms | 11 / 11 | 0/200 |
| 5 mm | 5.115 ms | 6.932 ms | 9.001 ms | 11 / 11 | 0/200 |
| 20 mm | 5.272 ms | 7.206 ms | 11.614 ms | 11 / 12 | 0/200 |

Cost is flat in displacement size: 11 iterations either way, ~0.46 ms per iteration. A repeat run
(n=100) gave medians of 5.09-5.31 ms.

## 3b. One env.step() macro step, as configured (frame_skip 20)

1200 steps: median **1.337 ms**, p95 2.051 ms, mean 1.469 ms -> 681 steps/s for the env alone.
Repeat (600 steps): median 1.222 ms, mean 1.323 ms.

## 3c. Ratio, and projected throughput with an IK solve on every step

| basis | IK : env step | projected |
|---|---|---|
| env alone (681 steps/s now) | 3.46x (1 mm), 3.48x (5 mm), 3.59x (20 mm) | 148-153 steps/s, 78% slower |
| full training loop (measured 319 steps/s on the cell running now, includes PPO) | -- | 119-122 steps/s, 62% slower |

At the training level 1M steps goes from ~0.9 h to ~2.3 h on this machine.

CPU contention: 12 cores; the concurrent training run is pinned to `num_threads: 1` and occupies
one. Load average was 0.87-1.11 during these measurements, and a repeat 30 s later agreed within
4%, so contention is not materially affecting the figures. The training run's own throughput dipped
319 -> 300 fps while measuring, i.e. contention in the other direction.

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

Zero failures and zero solves hitting the 400-iteration cap, in every region including the corners.

## 4b. Continuity along a warm-started sweep

a4 -> h4 at grasp height, 20 mm increments, 28 solves, each warm-started from the previous
solution. 0 failures.

| joint | largest single-step change (rad) |
|---|---|
| shoulder_pan | 0.0429 |
| shoulder_lift | 0.0146 |
| elbow | 0.0288 |
| wrist_1 | 0.0141 |
| wrist_2 | 0.0000 |
| wrist_3 | 0.0429 |

Largest change on any joint: **0.0429 rad** (shoulder_pan and wrist_3). Discontinuities above the
0.2 rad flag threshold: **0**. For scale, 0.0429 rad is under half the delta action space's
0.1 rad per-step cap.

## 4c. What solve_ik() does with an unreachable request

| request | result |
|---|---|
| +2 m in x (far outside reach) | raises `IKFailure`; env `qpos` unchanged; the solver's internal `mink.Configuration` has moved 1.571 rad |
| +0.5 m in y | raises `IKFailure`; same pattern, internal configuration moved 1.571 rad |
| -1 m in z (through the table) | **returns a pose, no exception**; 2.016 rad from the previous joint configuration |

Two distinct behaviours. Out-of-reach targets raise, leaving the env untouched but the solver's own
configuration displaced ~1.57 rad, so a subsequent warm-started solve begins from a pose that is not
where the arm is. A target unreachable because geometry is in the way rather than because it is out
of reach converges to a pose passing through the table, and that pose is 2.016 rad from the previous
one -- 47x the largest step in the 4b sweep and 20x the delta cap.
