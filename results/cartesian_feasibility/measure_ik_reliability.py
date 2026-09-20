"""Section 4: does solve_ik() converge, and stay continuous, across the board?

4a: small-displacement solves in each of +-x, +-y, +-z at a grid of board points.
4b: a straight sweep in 20 mm increments, each solve warm-started from the last.
4c: what solve_ik() does with an unreachable request.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import mink  # noqa: E402
import mujoco  # noqa: E402
import scripted_expert as se  # noqa: E402
from robochess.envs import RoboChessEnv  # noqa: E402
from robochess.tasks import T1PickAndPlaceTask  # noqa: E402
from robochess.tasks.squares import ALL_SQUARES, square_to_local_xy  # noqa: E402

STEP_M = 0.005          # "small displacement"
DISCONTINUITY = 0.2     # rad, flagged as a possible configuration flip


def region(square):
    file_, rank = square[0], int(square[1])
    return ("a-b" if file_ <= "b" else "g-h" if file_ >= "g" else "c-f") + "/" + \
           ("rank 1-2" if rank <= 2 else "rank 7-8" if rank >= 7 else "rank 3-6")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=None); a = p.parse_args()
    env = RoboChessEnv(action_mode="delta", max_episode_steps=10 ** 6)
    task = T1PickAndPlaceTask(env, start_square="e4", target_square="e5")
    ex = se.ScriptedExpert(task); task.reset(seed=0)
    board_xy = env.model.body_pos[env._body_id("board")][:2]
    out = {}

    # ---- 4a -------------------------------------------------------------
    # 5 files x 6 ranks = 30 points spanning the board, corners included explicitly
    squares = ["a1", "a8", "h1", "h8"] + [s for s in ALL_SQUARES
                                          if s[0] in "acegh" and int(s[1]) in (1, 2, 4, 5, 7, 8)]
    squares = list(dict.fromkeys(squares))
    dirs = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0),
            "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
    rows, per_region = [], {}
    for sq in squares:
        xy = square_to_local_xy(sq) + board_xy
        home = np.array([xy[0], xy[1], se.GRASP_Z])
        # seed the arm at this square first, so each solve is a genuinely small move
        try:
            q = ex.solve_ik(home)
            ex.base.data.qpos[:6] = q; mujoco.mj_forward(ex.model, ex.base.data)
            seated = True
        except se.IKFailure:
            seated = False
        for name, d in dirs.items():
            target = home + np.array(d) * STEP_M
            ok, err_mm = True, None
            try:
                qs = ex.solve_ik(target)
                ex.configuration.update(np.concatenate([qs, ex.base.data.qpos[6:]]))
                err_mm = float(np.linalg.norm(ex.task.compute_error(ex.configuration)[:3])) * 1000
            except se.IKFailure:
                ok = False
            rows.append(dict(square=sq, direction=name, converged=ok, seated=seated,
                             region=region(sq), residual_mm=err_mm))
            r = per_region.setdefault(region(sq), [0, 0])
            r[0] += 1; r[1] += int(not ok)
    n = len(rows); fails = sum(not r["converged"] for r in rows)
    out["4a"] = dict(points=len(squares), attempts=n, converged=n - fails, failed=fails,
                     converged_fraction=(n - fails) / n, per_region=per_region,
                     failures=[r for r in rows if not r["converged"]], step_m=STEP_M)
    print(f"4a: {len(squares)} board points x 6 directions = {n} solves at grasp height, step {STEP_M*1000:.0f} mm")
    print(f"    converged {n-fails}/{n} ({100*(n-fails)/n:.1f}%), failed/hit the {se.IK_MAX_ITERS}-iteration cap {fails}")
    for reg, (tot, bad) in sorted(per_region.items()):
        print(f"      {reg:16s}: {tot-bad}/{tot} converged")

    # ---- 4b -------------------------------------------------------------
    task.reset(seed=0)
    start = square_to_local_xy("a4") + board_xy
    end = square_to_local_xy("h4") + board_xy
    n_steps = int(np.linalg.norm(end - start) / 0.020)
    qs, xs, failed = [], [], []
    ex.solve_ik(np.array([start[0], start[1], se.GRASP_Z]))
    for i in range(n_steps + 1):
        t = start + (end - start) * (i / n_steps)
        target = np.array([t[0], t[1], se.GRASP_Z])
        try:
            q = ex.solve_ik(target)          # warm-started: configuration carries over
            qs.append(q.copy()); xs.append(i * 0.020)
        except se.IKFailure:
            failed.append(i)
    Q = np.array(qs); steps = np.abs(np.diff(Q, axis=0))
    J = ("pan", "lift", "elbow", "w1", "w2", "w3")
    worst = np.unravel_index(steps.argmax(), steps.shape)
    disc = [(int(i), J[j], float(steps[i, j])) for i, j in zip(*np.where(steps > DISCONTINUITY))]
    out["4b"] = dict(from_square="a4", to_square="h4", increment_m=0.020, solves=len(qs),
                     failed_indices=failed, max_step_rad=float(steps.max()),
                     max_step_joint=J[worst[1]], max_step_at_index=int(worst[0]),
                     per_joint_max={J[j]: float(steps[:, j].max()) for j in range(6)},
                     discontinuities_over_0_2_rad=disc)
    print(f"\n4b: a4 -> h4 at grasp height, {len(qs)} solves in 20 mm increments, {len(failed)} failures")
    print(f"    largest single-step joint change: {steps.max():.4f} rad on {J[worst[1]]} (step {worst[0]})")
    print("    per joint max: " + " ".join(f"{J[j]}{steps[:, j].max():.4f}" for j in range(6)))
    print(f"    discontinuities > {DISCONTINUITY} rad: {len(disc)}")

    # ---- 4c -------------------------------------------------------------
    task.reset(seed=0)
    here = ex.pinch_pos().copy()
    q_before = ex.base.data.qpos[:6].copy()
    cases = {}
    for label, target in (("+2 m in x (far outside reach)", here + [2.0, 0, 0]),
                          ("-1 m in z (through the table)", here + [0, 0, -1.0]),
                          ("0.5 m in y", here + [0, 0.5, 0])):
        try:
            q = ex.solve_ik(np.asarray(target, float))
            cases[label] = dict(raised=False, returned_q=q.tolist(),
                                max_joint_change_rad=float(np.abs(q - q_before).max()))
        except se.IKFailure as exc:
            # what has happened to the solver's internal configuration?
            q_after = ex.configuration.q[:6].copy()
            cases[label] = dict(raised=True, exception=str(exc)[:160],
                                internal_config_moved_rad=float(np.abs(q_after - q_before).max()),
                                env_qpos_changed=bool(np.any(ex.base.data.qpos[:6] != q_before)))
    out["4c"] = cases
    print("\n4c: unreachable requests")
    for k, v in cases.items():
        if v["raised"]:
            print(f"    {k}: raised IKFailure; env qpos changed: {v['env_qpos_changed']}; "
                  f"solver's internal configuration moved {v['internal_config_moved_rad']:.3f} rad")
        else:
            print(f"    {k}: returned a pose, max joint change {v['max_joint_change_rad']:.3f} rad")
    env.close()
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
