#!/usr/bin/env python3
"""Full-board scripted-expert validation for T2, the obstacle task.

Mirrors scripts/fullboard_validation.py's structure (every start square against
a fixed set of targets, jitter=0, seed fixed at 0 so the reset is
deterministic), with three differences:

  * the env is built on T2's scene (assets/scene_t2.xml, D13), never T1's;
  * `--obstacles N` sets how many of the 4 pool slots are placed;
  * each record carries the obstacle outcome as well as the placement one,
    because T2 introduces a failure mode T1 does not have.

T2's control is `--obstacles 0` ON THIS SCENE, not T1's figure. Per D13, the
pool bodies perturb the constraint solve whether or not obstacles are placed,
so T1's 309/315 is a number for a different model and comparing to it would be
a category error.

The scripted expert is the T1 expert unchanged: it knows nothing about
obstacles and makes no attempt to avoid them. That is deliberate here -- it
measures how often the existing waypoint sequence disturbs a bystander, which
is the quantity T2's design questions turn on.

Usage:
    python scripts/fullboard_validation_t2.py --obstacles 0      # T2's control
    python scripts/fullboard_validation_t2.py --obstacles 2      # default config
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripted_expert as se  # noqa: E402
from robochess.tasks import T2ObstacleTask, make_t2_env  # noqa: E402
from robochess.tasks.squares import ALL_SQUARES  # noqa: E402


def run(targets, budget, n_obstacles, progress_every=20):
    env = make_t2_env(max_episode_steps=budget)
    task = T2ObstacleTask(env, n_obstacles=n_obstacles)
    expert = se.ScriptedExpert(task)

    total = len(targets) * (len(ALL_SQUARES) - 1)
    results = []
    started = time.time()
    for target in targets:
        for start in ALL_SQUARES:
            if start == target:
                continue
            task.start_square, task.target_square = start, target
            rec = expert.run_episode(seed=0)
            # The expert's record does not carry T2's fields; read them off the
            # task's last info, which reflects the terminal step.
            info = task._augment_info(env._build_info(terminated_reason=None))
            results.append({
                "target": target,
                "start": start,
                "success": bool(rec["success"]),
                "outcome": rec["outcome"],
                "failed_phase": rec["failed_phase"],
                "steps": int(rec["steps"]),
                "final_distance_mm": round(rec["final_distance"] * 1000.0, 2),
                "obstacle_squares": list(task.current_obstacle_squares),
                "obstacles_undisturbed": bool(info["obstacles_undisturbed"]),
                "n_disturbed": int(info["unintended_displacement_count"]),
                "max_obstacle_displacement_mm": round(
                    float(info["max_obstacle_displacement_mm"]), 2),
            })
            if progress_every and len(results) % progress_every == 0:
                ok = sum(r["success"] for r in results)
                print("  %d/%d ok=%d (%.0fs)" % (len(results), total, ok,
                                                 time.time() - started), flush=True)
    return results, time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", default="e5,a1,h8,a8,h1",
                        help="comma-separated target squares (default: the five swept "
                             "for D10/D11).")
    parser.add_argument("--obstacles", type=int, default=2,
                        help="how many of the 4 pool slots to place (default: 2). "
                             "0 is T2's control.")
    parser.add_argument("--budget", type=int, default=350)
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    for t in targets:
        if t not in ALL_SQUARES:
            parser.error("not a valid square: %r" % t)

    results, elapsed = run(targets, args.budget, args.obstacles,
                           progress_every=0 if args.quiet else 20)

    ok = sum(r["success"] for r in results)
    n = len(results)
    disturbed = [r for r in results if not r["obstacles_undisturbed"]]
    print("")
    print("scene=t2 targets=%s obstacles=%d budget=%d"
          % (",".join(targets), args.obstacles, args.budget))
    print("RESULT: %d/%d (%.1f%%)  elapsed=%.0fs" % (ok, n, 100.0 * ok / n, elapsed))
    print("OBSTACLE TERM fired: %d/%d (%.1f%%)" % (len(disturbed), n, 100.0 * len(disturbed) / n))

    if results:
        print("outcomes: %s" % dict(Counter(r["outcome"] for r in results)))

    failures = [r for r in results if not r["success"]]
    if failures:
        print("failures (%d):" % len(failures))
        for r in failures:
            print("  %s->%s  outcome=%-28s phase=%-9s steps=%3d  dist=%8.1f mm  "
                  "obstacles=%s disturbed=%d maxdisp=%.1f mm"
                  % (r["start"], r["target"], r["outcome"], r["failed_phase"],
                     r["steps"], r["final_distance_mm"], ",".join(r["obstacle_squares"]),
                     r["n_disturbed"], r["max_obstacle_displacement_mm"]))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
