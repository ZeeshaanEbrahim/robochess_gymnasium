"""Sections 3a-3c: cost of one solve_ik() call, and what its iterations buy.

CORRECTED METHODOLOGY (20 Sep 2026). The first version of this script drove the
arm with ScriptedExpert._drive_to on a DELTA-mode env, which reinterprets the
expert's absolute joint targets as per-step deltas; the arm ended 557 mm from the
intended pose and 96 deg away from solve_ik()'s fixed top-down tool_rotation. Every
"small displacement" solve was therefore really "re-orient the tool by 96 deg and
move", which is why cost looked flat at ~5 ms / 11 iterations from 1 mm to 20 mm.
Here the arm is seated AT a valid top-down IK solution (qpos set directly, then
mj_forward), and each solve is re-seated at its own solution, which is the
sequence a Cartesian action space would actually produce. solve_ik() is unmodified.

    python results/cartesian_feasibility/measure_ik_cost.py --out ik_cost.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import mink      # noqa: E402
import mujoco    # noqa: E402
import scripted_expert as se  # noqa: E402
from robochess.envs import RoboChessEnv  # noqa: E402
from robochess.tasks import T1PickAndPlaceTask  # noqa: E402


def seat(ex, env, target):
    """Put the arm exactly at the IK solution for `target` (no physics, no drift)."""
    q = ex.solve_ik(np.asarray(target, float))
    env.data.qpos[:6] = q
    mujoco.mj_forward(env.model, env.data)
    return q


def count_iters(ex, env, target, cap=None):
    """solve_ik()'s own loop, instrumented: (iterations_used, pos_err_mm, ori_err, converged)."""
    t = np.asarray(target, float).copy(); t[2] += ex.tool_length
    ex.task.set_target(mink.SE3.from_rotation_and_translation(ex.tool_rotation, t))
    ex.configuration.update(env.data.qpos.copy())
    cap = se.IK_MAX_ITERS if cap is None else cap
    for i in range(cap):
        err = ex.task.compute_error(ex.configuration)
        if np.linalg.norm(err[:3]) < se.IK_POS_TOL and np.linalg.norm(err[3:]) < se.IK_ORI_TOL:
            return i, np.linalg.norm(err[:3]) * 1000, float(np.linalg.norm(err[3:])), True
        vel = mink.solve_ik(ex.configuration, [ex.task], se.IK_DT, se.IK_SOLVER, damping=1e-3)
        m = np.zeros_like(vel); m[ex._arm_dofs] = vel[ex._arm_dofs]
        ex.configuration.integrate_inplace(m, se.IK_DT)
    err = ex.task.compute_error(ex.configuration)
    return cap, np.linalg.norm(err[:3]) * 1000, float(np.linalg.norm(err[3:])), False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200); p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--out", type=Path, default=None); a = p.parse_args()

    env = RoboChessEnv(max_episode_steps=10 ** 6)
    task = T1PickAndPlaceTask(env, start_square="e4", target_square="e5")
    ex = se.ScriptedExpert(task); task.reset(seed=0)
    piece = env._achieved_goal().astype(float)
    start = np.array([piece[0], piece[1], se.GRASP_Z])
    seat(ex, env, start)
    out = {"method": "arm seated at a valid top-down IK solution; each solve re-seated at its own solution",
           "config": dict(max_iters=se.IK_MAX_ITERS, pos_tol_m=se.IK_POS_TOL, ori_tol_rad=se.IK_ORI_TOL,
                          dt=se.IK_DT, solver=se.IK_SOLVER, damping=1e-3, early_exit_on_convergence=True),
           "ik": {}}

    rng = np.random.default_rng(0)
    print("| displacement | median ms | p95 ms | max ms | iterations median/max | residual mm (median) | failures |")
    print("|---|---|---|---|---|---|---|")
    for mm in (1, 5, 20):
        seat(ex, env, start)
        times, iters, res, fails = [], [], [], 0
        for _ in range(a.n):
            here = ex.pinch_pos().copy()
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            target = here + d * (mm / 1000.0)
            t0 = time.perf_counter()
            try:
                q = ex.solve_ik(target)
            except se.IKFailure:
                fails += 1; q = None
            times.append((time.perf_counter() - t0) * 1000)
            it, pe, _, _ = count_iters(ex, env, target)
            iters.append(it); res.append(pe)
            if q is not None:                       # walk the chain, as a policy would
                env.data.qpos[:6] = q; mujoco.mj_forward(env.model, env.data)
        t = np.array(times)
        out["ik"][f"{mm}mm"] = dict(n=a.n, median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)),
                                    max_ms=float(t.max()), mean_ms=float(t.mean()),
                                    median_iters=float(np.median(iters)), max_iters=int(max(iters)),
                                    median_residual_mm=float(np.median(res)), failures=fails)
        print(f"| {mm} mm | {np.median(t):.3f} | {np.percentile(t,95):.3f} | {t.max():.3f} | "
              f"{np.median(iters):.0f} / {max(iters)} | {np.median(res):.4f} | {fails}/{a.n} |")

    # 3b: error after a truncated number of iterations, warm-started 5 mm
    seat(ex, env, start)
    targets = []
    for _ in range(max(50, a.n // 4)):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        targets.append(ex.pinch_pos() + d * 0.005)
    print("\n| iterations | median pos err | p95 pos err | converged |")
    print("|---|---|---|---|")
    out["truncated"] = {}
    for n in (1, 2, 3, 5):
        errs, conv = [], 0
        for tg in targets:
            _, pe, _, ok = count_iters(ex, env, tg, cap=n)
            errs.append(pe); conv += int(ok)
        e = np.array(errs)
        out["truncated"][str(n)] = dict(median_pos_err_mm=float(np.median(e)),
                                        p95_pos_err_mm=float(np.percentile(e, 95)),
                                        converged=conv, n=len(targets))
        print(f"| {n} | {np.median(e):.4f} mm | {np.percentile(e,95):.4f} mm | {conv}/{len(targets)} |")

    # 3b timing per truncated budget
    for n in out["truncated"]:
        tms = []
        for tg in targets:
            t0 = time.perf_counter(); count_iters(ex, env, tg, cap=int(n))
            tms.append((time.perf_counter() - t0) * 1000)
        out["truncated"][n]["median_ms"] = float(np.median(tms))

    # env macro step
    task.reset(seed=0); act = env.home_action()
    for _ in range(50):
        task.step(act)
    tms = []
    for i in range(a.steps):
        t0 = time.perf_counter(); _, _, te, tr, _ = task.step(act)
        tms.append((time.perf_counter() - t0) * 1000)
        if te or tr:
            task.reset(seed=i)
    t = np.array(tms)
    out["env_step"] = dict(n=a.steps, median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)),
                           mean_ms=float(t.mean()))
    print(f"\nenv.step(): median {np.median(t):.3f} ms, mean {t.mean():.3f} -> {1000/t.mean():.0f} steps/s env only")

    step_ms, train_sps = out["env_step"]["mean_ms"], 264.0
    print(f"\n| basis | solve ms | env-only steps/s | training steps/s | 1M steps |")
    print("|---|---|---|---|---|")
    for k, v in list(out["ik"].items()) + [(f"{n} iters", v) for n, v in out["truncated"].items()]:
        ms = v["median_ms"]
        env_only = 1000.0 / (step_ms + ms); train = 1.0 / (1.0 / train_sps + ms / 1000.0)
        v.update(env_only_steps_per_sec=env_only, training_steps_per_sec=train)
        print(f"| {k} | {ms:.3f} | {env_only:.0f} | {train:.0f} | {1e6/train/3600:.1f} h |")
    env.close()
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
