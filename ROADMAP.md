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
| scripted-expert regression, default random square sampling (all 64 squares) | 64% (32/50) originally -- root-caused and fixed as D7 (`8139b27`); 46/50 (92%) with the residual traced to D8; D8/D9 now resolved (Candidate 1 + is_success piece-settle fix) -- CLI regression 50/50 (100%) |
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
population, diagnosed as D8: the head-only grasp D7 committed to is
rotationally underconstrained, and the instant the gripper opens at
`release`, the piece spins violently (12.8-30.3 rad/s) -- universally,
across passing and failing squares alike, not only the ones that end up
failing. See D8 in `DECISIONS.md` for the full diagnostic account.

A follow-on issue surfaced while investigating D8's release instability:
D9 in `DECISIONS.md` -- the production `is_success` check had no
velocity/settle requirement on the piece itself, and could register success
while the piece was still spinning (observed up to 17.7 rad/s).

**Both D8 and D9 are now RESOLVED (2026-08-27, fix-design session).** D9
was fixed at the source (`RoboChessEnv._is_success()`/`_build_info()` now
require the piece's own angular speed to stay below 0.05 rad/s for 3
consecutive steps, tracked as new per-episode state, reset every
`reset()`). D8 was fixed as Candidate 1: `release` ramps the gripper open
command gradually instead of jumping, gated on the piece genuinely
settling rather than the arm/gripper settle-check every other waypoint
uses -- validated as a real release-dynamics effect (not a time-budget or
success-check-timing artifact) via a paired A/B/C comparison, then at full
64-square-board scale across 3 targets: 98.9% (187/189), zero regressions
vs. the old release. Real CLI regression (`--task t1 --episodes 50`):
**100% (50/50)**, up from the historical 46/50. See D8's "Resolution" and
D9's "Resolution" sections in `DECISIONS.md` for the full account,
including the single-environment-build caveat on these exact figures.

Residual failures at full-board scale, unrelated to D8/D9 (they occur
before release is ever reached and are identical under old and new release),
are logged separately in `DECISIONS.md`. Diagnostic characterization showed
they do not share a mechanism, so they are split:

- **D10** (h1->h8, and its reverse h8->h1) is **root-caused, with a fix
  implemented but disabled**: `traverse` is commanded as a single joint-space
  target, and the FK of a joint-space interpolation between two
  widely-separated poses is an arc that leaves the board footprint even
  though both endpoints sit 78 mm inside it. The planned path overshoots the
  XY bound by 25-27 mm; the executed path overshoots by 4.05 mm, so actuator
  lag *reduces* the excursion rather than causing it. Grip and piece dynamics
  are unremarkable throughout. Walking the traverse in task space instead
  removes the excursion on both pairs (+4.05 -> -74.65 mm, +4.36 -> -75.82 mm)
  and takes the sweep to 311/315, but it breaks four episodes that passed at
  baseline, so it ships behind `TRAVERSE_TASK_SPACE = False` pending D11.
  **D10 is not resolved** — the default path still fails both pairs.
- **D11** (pad load goes strongly asymmetric during `place` and the piece is
  squeezed out of a still-closed gripper; 4 known instances, peak pad force
  229.8-504.1 N, aftermath ranging from a 151.86 m/s ejection to the piece
  merely ending up on the wrong square) remains **open and not root-caused**,
  and is now known to be *chaotic under upstream perturbation*: changing the
  traverse resamples which squares fail, at a roughly constant rate, without
  changing what `place` inherits in the piece's own velocity. That is what
  gates D10's fix.

Neither blocks calling D8/D9 done. The sweep behind these figures is 5
targets x 63 starts = 315 episodes, 6 failures (98.1%), reproducible via
`scripts/fullboard_validation.py --targets e5,a1,h8,a8,h1`. Widening it from
3 targets to 5 also killed the "reach extremity" hypothesis the original
combined entry carried: `a8`, the far corner at ~99% of rated reach, is clean
at 0/63, while the two *near* corners account for 5 of the 6 failures.
