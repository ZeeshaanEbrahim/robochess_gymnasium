"""Section 3: wall-clock cost of one mink IK solve vs one env macro step.

The case that matters for a Cartesian action space is a SMALL displacement from
the arm's current pose, warm-started from the current joint configuration -- not
the scripted expert's large waypoint jumps. solve_ik() is used as it stands.

    python results/cartesian_feasibility/measure_ik_cost.py [--out x.json]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import mink  # noqa: E402
import scripted_expert as se  # noqa: E402
from robochess.envs import RoboChessEnv  # noqa: E402
from robochess.tasks import T1PickAndPlaceTask  # noqa: E402


def _iters_for(ex, target):
    """Re-run solve_ik's own loop, counting iterations (solve_ik returns none)."""
    t = np.asarray(target, float).copy(); t[2] += ex.tool_length
    ex.task.set_target(mink.SE3.from_rotation_and_translation(ex.tool_rotation, t))
    ex.configuration.update(ex.base.data.qpos.copy())
    for i in range(se.IK_MAX_ITERS):
        err = ex.task.compute_error(ex.configuration)
        if np.linalg.norm(err[:3]) < se.IK_POS_TOL and np.linalg.norm(err[3:]) < se.IK_ORI_TOL:
            return i
        vel = mink.solve_ik(ex.configuration, [ex.task], se.IK_DT, se.IK_SOLVER, damping=1e-3)
        masked = np.zeros_like(vel); masked[ex._arm_dofs] = vel[ex._arm_dofs]
        ex.configuration.integrate_inplace(masked, se.IK_DT)
    return se.IK_MAX_ITERS


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=None)
    p.add_argument("--n", type=int, default=200); p.add_argument("--steps", type=int, default=1200)
    a = p.parse_args()

    env = RoboChessEnv(action_mode="delta", max_episode_steps=10 ** 6)
    task = T1PickAndPlaceTask(env, start_square="e4", target_square="e5")
    ex = se.ScriptedExpert(task)
    task.reset(seed=0)
    # put the arm somewhere representative: just above the piece, as a policy would be
    piece = env._achieved_goal().astype(float)
    for _ in range(2):
        q = ex.solve_ik(np.array([piece[0], piece[1], se.CLEAR_Z]))
        for _ in ex._drive_to(q, se.GRIPPER_OPEN_CTRL, 25):
            pass

    rng = np.random.default_rng(0)
    out = {"ik": {}}
    for mm in (1, 5, 20):
        times, iters, failures = [], [], 0
        for _ in range(a.n):
            here = ex.pinch_pos()
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            target = here + d * (mm / 1000.0)
            t0 = time.perf_counter()
            try:
                ex.solve_ik(target)
            except se.IKFailure:
                failures += 1
            times.append((time.perf_counter() - t0) * 1000.0)
            iters.append(_iters_for(ex, target))
        t = np.array(times)
        out["ik"][f"{mm}mm"] = dict(n=a.n, median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)),
                                    max_ms=float(t.max()), mean_ms=float(t.mean()),
                                    median_iters=float(np.median(iters)), max_iters=int(max(iters)),
                                    failures=failures)
        print(f"IK {mm:2d} mm: median {np.median(t):6.3f} ms, p95 {np.percentile(t,95):6.3f}, max {t.max():7.3f}, "
              f"iters median {np.median(iters):.0f} max {max(iters)}, failures {failures}/{a.n}")

    # one env macro step, as configured
    task.reset(seed=0)
    act = env.home_action()
    for _ in range(50):
        task.step(act)
    times = []
    for i in range(a.steps):
        t0 = time.perf_counter(); _, _, te, tr, _ = task.step(act)
        times.append((time.perf_counter() - t0) * 1000.0)
        if te or tr:
            task.reset(seed=i)
    t = np.array(times)
    out["env_step"] = dict(n=a.steps, median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)),
                           max_ms=float(t.max()), mean_ms=float(t.mean()))
    print(f"env.step(): median {np.median(t):.3f} ms, p95 {np.percentile(t,95):.3f}, mean {t.mean():.3f} "
          f"-> {1000/t.mean():.0f} steps/s single env")
    env.close()

    step_ms = out["env_step"]["mean_ms"]
    for k, v in out["ik"].items():
        ratio = v["median_ms"] / step_ms
        proj = 1000.0 / (step_ms + v["median_ms"])
        out["ik"][k].update(ratio_to_env_step=ratio, projected_steps_per_sec=proj)
        print(f"  {k}: IK/step ratio {ratio:.2f}x -> {proj:.0f} steps/s if solved every step "
              f"(vs {1000/step_ms:.0f} now, {100*(1-proj/(1000/step_ms)):.0f}% slower)")
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
