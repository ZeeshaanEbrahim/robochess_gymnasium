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
| scripted-expert regression, default random square sampling (all 64 squares) | 64% (32/50) originally -- root-caused and fixed as D7 (`8139b27`); CLI regression now 46/50 (92%) with the residual traced to D8 |
| HER + SAC smoke test, 8k steps | completes cleanly, no crash/NaN; 0/100 episodes successful in this short a run (expected -- not a convergence check; short runs shouldn't be over-interpreted) |
| full test suite under `requirements-ci.txt` | 48/48 pass (18 P1 + 30 T1) |

**Open item before this can be called done:** D8 in `DECISIONS.md` -- D7
(the far-half grasp-reliability problem) is resolved (`8139b27`, 2026-08-27):
root cause was a lack of gravity compensation on the UR5e's position
actuators, producing a reach-dependent steady-state tracking error; fixed
via task-space residual compensation on the `descend` waypoint plus a
`GRASP_Z` recalibration to a validated head-only grasp (0.79925). Validated
18/18 on the diagnosed sample and 46/50 (92%) on the CLI regression, up
from the 64% baseline, with no catastrophic `piece_off_board` failures
remaining -- see D7's "Resolution" section in `DECISIONS.md` for the full
account, including four earlier fix attempts that failed for confirmed,
distinct reasons.

That same validation scan surfaced a **second, separate** failure
population, diagnosed as D8 (2026-08-27, not yet fixed): the head-only
grasp D7 committed to is rotationally underconstrained, and the instant the
gripper opens at `release`, the piece spins violently (12.8-30.3 rad/s) --
universally, across passing and failing squares alike, not only the ones
that end up failing. Whether the spin resolves inside tolerance (pass), or
the piece tumbles-and-resettles shifted sideways, or lands perched in a
still-spinning metastable state above the board, is chaotic; no
discriminating variable (wrist orientation, spin magnitude, reach) was
found despite checking. See D8 in `DECISIONS.md` for the full diagnostic
account. **Fix design is separate, upcoming work, not attempted as part of
either D7 or D8's diagnosis** -- this task class isn't done until it lands.
