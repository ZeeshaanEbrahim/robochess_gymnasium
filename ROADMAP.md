# Roadmap gates

Phase-gate definitions for this project, as recorded in-repo. The canonical
roadmap lives outside this repository (same as the environment setup
document referenced in `README.md`); this file exists because there is no
in-repo location to point to otherwise, and Gate M1's criteria are worth
tracking against evidence that already lives here. Update this file when a
gate's wording changes upstream, and note the date it changed.

## Gate M1 (end of P1)

> Scripted expert solves single-piece pick-and-place reliably on an empty
> board. T1 and T2 as formal task classes, including T2's obstacle handling,
> are built and validated in P2 under Gate M2, not P1. Gymnasium checker
> clean; profiling table produced; training budget revised in writing (see
> D4 in DECISIONS.md).
>
> — as stated 2026-08-24, superseding any earlier, less precise phrasing of
> what P1 validates.

**Status: satisfied.**

| criterion | evidence |
| --- | --- |
| scripted expert solves single-piece pick-and-place reliably on an empty board | `scripts/scripted_expert.py`, 50/50 episodes against the corrected success criterion (piece on-square, resting, gripper released — see D3's amendment in `DECISIONS.md`) |
| Gymnasium checker clean | `tests/test_env_basic.py::test_check_env_passes_with_zero_warnings` — `gymnasium.utils.env_checker.check_env` passes with zero warnings |
| profiling table produced | `profiling/p1_throughput.csv`, narrative in `profiling/README.md` |
| training budget revised in writing | D4 in `DECISIONS.md` |

### What Gate M1 explicitly does NOT cover

This is the amendment this entry exists to record. Earlier framing in this
project's discussion conflated "the scripted expert works" with "the task
classes are validated," which is not the same claim. Gate M1 validates
**environment mechanics** on the single easiest case:

- single piece, not multiple
- empty board, no obstacles
- one fixed pick-and-place waypoint sequence, not a general task class

**T1 and T2 as formal, general task classes — including T2's obstacle
handling — are P2 work, validated under Gate M2.** Nothing in this repo
should be read as claiming T1/T2 are done because the P1 scripted expert
succeeds; the scripted expert deliberately never encounters an obstacle, a
second piece, or a randomized goal (see D3's note that P1 uses a fixed
adjacent-square goal, with randomization deferred to P2's task classes).

## Gate M2 (end of P2)

Not yet defined in this repository. Record it here once it exists, in the
same evidence-linked format as Gate M1 above.

### T1 status (in progress, 2026-08-25)

Not a gate claim -- Gate M2 needs all five task classes. Recording where T1
stands so a future session doesn't have to re-derive it from git history.

Built: `robochess/tasks/t1_pick_and_place.py` (`T1PickAndPlaceTask`,
`ShapedPickAndPlaceReward`, `SparsePickAndPlaceReward`), composed as
Gymnasium wrappers over `RoboChessEnv` rather than a subclass, per the
precedent the roadmap set for T2-T5. `robochess/tasks/squares.py` gives
algebraic square naming (78 mm pitch, D1) shared across the task family.

| check | result |
| --- | --- |
| unit tests (task + wrapper files) | 30 new tests, 99% line coverage (`robochess/tasks/*`) |
| `check_env` on the wrapped T1 env | passes; only the two expected wrapper-detection warnings, no space-contract violation |
| scripted-expert regression, P1-identical conditions (fixed near-board squares, e.g. e4->e5) | 100%, matches P1 |
| scripted-expert regression, default random square sampling (all 64 squares) | 64% (32/50) -- see D7, not a T1 bug, root cause unresolved |
| HER + SAC smoke test, 8k steps | completes cleanly, no crash/NaN; 0/100 episodes successful in this short a run (expected -- not a convergence check, see D7's caveat and the roadmap's own note that short runs shouldn't be over-interpreted) |
| full test suite under `requirements-ci.txt` | 48/48 pass (18 P1 + 30 T1) |

**Open item before this can be called done:** D7 in `DECISIONS.md` --
default random square sampling surfaces a square-dependent grasp-reliability
problem on the far half of the board, unrelated to T1's own code. A
three-pass diagnostic effort (2026-08-25) instrumented per-timestep joint
state, contact force, and end-effector orientation to isolate it: ruled out
wrist tilt, joint-limit proximity, singularity, yaw drift, and IK
path-dependence as the trigger; found a tangential (shear) contact-force
imbalance at first pad contact as the most timing-precise anomaly, with the
second-touching pad usually bearing the dominant force in catastrophic
cases (3/3 tested) -- but confirmed this isn't sufficient on its own (one
partial-outcome square shows the identical signature without going
catastrophic) and that peak force magnitude does not separate outcomes as
a threshold (the lowest-magnitude case in the whole diagnosed set is
catastrophic). Diagnosis is closed by decision, not because the mechanism
is fully understood -- see D7's "Diagnostic passes 1-3" section for the
full breakdown. **Still needs a decision** (narrow the default reachable
set, accept and document, or something else) before P2's training runs or
P3's pilot matrix lean on uniform square sampling; that decision and any
mitigation design are separate, upcoming work.
