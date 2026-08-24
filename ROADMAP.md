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
