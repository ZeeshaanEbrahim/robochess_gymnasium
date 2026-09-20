"""Section 3: at the closest approach of a deterministic episode, could it just close?

3a: geometry at the closest step. 3b: from that exact state, force the gripper shut
and report pad contact and converged aperture. Nothing is modified in the env.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_s = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(_s); _s.loader.exec_module(train)


def main() -> int:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import mujoco

    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path); p.add_argument("--checkpoint", type=int, default=None)
    p.add_argument("--out", type=Path, default=None); a = p.parse_args()

    cfg = json.loads((a.run_dir / "manifest.json").read_text())["config_resolved"]
    train.apply_thread_pinning(cfg.get("num_threads", 1))
    venv = DummyVecEnv([lambda: Monitor(train.build_env(cfg))])
    ck = a.checkpoint
    mp = a.run_dir / ("final_model.zip" if ck is None else f"checkpoints/rc_{ck}_steps.zip")
    vn = a.run_dir / ("final_vecnormalize.pkl" if ck is None else f"checkpoints/rc_vecnormalize_{ck}_steps.pkl")
    venv = VecNormalize.load(str(vn), venv); venv.training = False; venv.norm_reward = False
    model = PPO.load(str(mp), env=venv, device="cpu")
    inner = venv.venv.envs[0]
    base = inner.unwrapped
    task = inner.env
    while not hasattr(task, "_piece_contacts"):
        task = task.env
    sid = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_SITE, "ur_gripper_pinch")

    venv.seed(10_000); obs = venv.reset()
    best = None
    states = []
    for step in range(cfg["max_episode_steps"]):
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, infos = venv.step(act)
        pinch = base.data.site_xpos[sid].copy(); piece = base._achieved_goal().copy()
        d3 = float(np.linalg.norm(pinch - piece))
        states.append((d3, step, base.data.qpos.copy(), base.data.qvel.copy(),
                       base.data.ctrl.copy(), pinch.copy(), piece.copy()))
        if best is None or d3 < best[0]:
            best = states[-1]
        if done[0]:
            break
    d3, step, qpos, qvel, ctrl, pinch, piece = best
    # grasp band: the piece's head, which GRASP_Z targets (see scripted_expert.GRASP_Z)
    sys.path.insert(0, str(REPO / "scripts"))
    import scripted_expert as se  # noqa: E402
    out = dict(run=a.run_dir.name, checkpoint=ck, closest_step=step,
               pinch_xyz=[float(x) for x in pinch], piece_xyz=[float(x) for x in piece],
               pinch_to_piece_3d_m=d3,
               pinch_to_piece_xy_m=float(np.linalg.norm(pinch[:2] - piece[:2])),
               pinch_z_m=float(pinch[2]), grasp_band_z_m=float(se.GRASP_Z),
               pinch_above_grasp_band_m=float(pinch[2] - se.GRASP_Z),
               pinch_above_piece_origin_m=float(pinch[2] - piece[2]),
               gripper_aperture_rad=float(qpos[base._grip_qadr]))
    print("3a. closest approach, deterministic episode")
    for k in ("closest_step", "pinch_to_piece_3d_m", "pinch_to_piece_xy_m", "pinch_z_m",
              "grasp_band_z_m", "pinch_above_grasp_band_m", "pinch_above_piece_origin_m",
              "gripper_aperture_rad"):
        print(f"    {k:28s} {out[k]:.4f}" if isinstance(out[k], float) else f"    {k:28s} {out[k]}")

    # 3b: restore that exact state and close the gripper, holding the arm still
    base.data.qpos[:] = qpos; base.data.qvel[:] = qvel; base.data.ctrl[:] = ctrl
    mujoco.mj_forward(base.model, base.data)
    base.data.ctrl[6] = base._ctrl_high[6]            # command fully closed
    rows = []
    for i in range(60):
        mujoco.mj_step(base.model, base.data, nstep=base.frame_skip)
        left, right, board = task._piece_contacts()
        rows.append((i, left, right, board, float(base.data.qpos[base._grip_qadr])))
    left, right, board, ap = rows[-1][1], rows[-1][2], rows[-1][3], rows[-1][4]
    ever_both = any(l and r for _, l, r, _, _ in rows)
    out.update(forced_close=dict(both_pads_final=bool(left and right), left_final=bool(left),
                                 right_final=bool(right), board_contact_final=bool(board),
                                 final_aperture_rad=ap, both_pads_at_any_point=bool(ever_both),
                                 first_both_pads_step=next((i for i, l, r, _, _ in rows if l and r), None),
                                 piece_moved_mm=float(1000 * np.linalg.norm(
                                     base._achieved_goal()[:2] - piece[:2]))))
    print("\n3b. forced gripper close from that exact state, arm held")
    for k, v in out["forced_close"].items():
        print(f"    {k:28s} {v}")
    venv.close()
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
