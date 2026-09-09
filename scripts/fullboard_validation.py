#!/usr/bin/env python3
"""Full-board scripted-expert validation: every start square against a fixed
set of target squares, under either release implementation.

This is D8 "Section 5 Part D". RECONSTRUCTION NOTE: the original harness was
ad hoc and never checked in (as were D8s own diagnostic drivers -- see D8 in
DECISIONS.md). This file was rebuilt from the prose specification recorded in
DECISIONS.md and re-validated against the figures recorded there:

    --release new   187/189 (98.9%)   condition (iii), Candidate 1
    --release old   149/189 (78.8%)   condition (ii), old instant release

Both reproduce exactly, including the identity of the two residual failures
(c2->a1 failing in "place", h1->h8 failing in "traverse") and their step
counts. It is committed so the numbers stop depending on code that only ever
existed in a scratch directory.

Read-only with respect to the environment and the expert: the old-release
condition is produced by subclassing at runtime, never by editing
scripts/scripted_expert.py.

Determinism: with jitter=0 the env reset is fully deterministic and the start
and target squares are forced per episode, so the seed does not influence the
outcome. Seeds are therefore fixed at 0.

Usage:
    python scripts/fullboard_validation.py                    # current code
    python scripts/fullboard_validation.py --release old      # pre-D8 control
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Same convention as scripts/scripted_expert.py: repo root for the package,
# plus this directory so the expert module itself is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripted_expert as se  # noqa: E402
from robochess.envs import RoboChessEnv  # noqa: E402
from robochess.tasks import T1PickAndPlaceTask  # noqa: E402
from robochess.tasks.squares import ALL_SQUARES  # noqa: E402

# The "release" budget before the D8 fix raised it to 120. Restored only for
# the old-release control condition.
OLD_RELEASE_BUDGET = 35


class OldReleaseExpert(se.ScriptedExpert):
    """The expert as it behaved before D8 Candidate 1.

    Two overrides, both confined to this class:
      1. _drive_to is forced to gripper_ctrl_fn=None and piece_settle=False,
         so "release" commands the open ctrl in one jump and completes on the
         ordinary arm/gripper settle check rather than the piece-settle gate.
      2. The "release" waypoint budget is restored to its pre-D8 value.
    """

    def _drive_to(self, *args, **kwargs):
        kwargs["gripper_ctrl_fn"] = None
        kwargs["piece_settle"] = False
        return super()._drive_to(*args, **kwargs)

    def waypoints(self, piece_xy, goal_xy):
        out = []
        for label, pos, grip, budget in super().waypoints(piece_xy, goal_xy):
            out.append((label, pos, grip, OLD_RELEASE_BUDGET if label == "release" else budget))
        return out


def run(targets, budget, release, progress_every=20):
    env = RoboChessEnv(max_episode_steps=budget)
    task = T1PickAndPlaceTask(env)
    expert = (se.ScriptedExpert if release == "new" else OldReleaseExpert)(task)

    total = len(targets) * (len(ALL_SQUARES) - 1)
    results = []
    started = time.time()
    for target in targets:
        for start in ALL_SQUARES:
            if start == target:
                continue
            task.start_square = start
            task.target_square = target
            rec = expert.run_episode(seed=0)
            results.append({
                "target": target,
                "start": start,
                "success": bool(rec["success"]),
                "outcome": rec["outcome"],
                "failed_phase": rec["failed_phase"],
                "steps": int(rec["steps"]),
                "final_distance_mm": round(rec["final_distance"] * 1000.0, 2),
            })
            if progress_every and len(results) % progress_every == 0:
                ok = sum(r["success"] for r in results)
                print("  %d/%d ok=%d (%.0fs)" % (len(results), total, ok, time.time() - started),
                      flush=True)
    return results, time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", choices=["new", "old"], default="new",
                        help="new is the current Candidate-1 ramped release; "
                             "old restores the pre-D8 instant release as a control.")
    parser.add_argument("--targets", default="e5,a1,h8",
                        help="comma-separated target squares (default: the three "
                             "recorded in DECISIONS.md).")
    parser.add_argument("--budget", type=int, default=350,
                        help="max_episode_steps (default: 350, as recorded).")
    parser.add_argument("--out", default=None, help="optional JSON path for per-episode records.")
    parser.add_argument("--quiet", action="store_true", help="suppress progress lines.")
    args = parser.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    for t in targets:
        if t not in ALL_SQUARES:
            parser.error("not a valid square: %r" % t)

    results, elapsed = run(targets, args.budget, args.release,
                           progress_every=0 if args.quiet else 20)

    ok = sum(r["success"] for r in results)
    n = len(results)
    print("")
    print("release=%s targets=%s budget=%d" % (args.release, ",".join(targets), args.budget))
    print("RESULT: %d/%d (%.1f%%)  elapsed=%.0fs" % (ok, n, 100.0 * ok / n, elapsed))

    failures = [r for r in results if not r["success"]]
    if failures:
        print("failures (%d):" % len(failures))
        for r in failures:
            print("  %s->%s  outcome=%-16s phase=%-9s steps=%3d  dist=%8.1f mm"
                  % (r["start"], r["target"], r["outcome"], r["failed_phase"],
                     r["steps"], r["final_distance_mm"]))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
