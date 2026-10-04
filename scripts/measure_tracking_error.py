#!/usr/bin/env python
"""D7 follow-on: tracking error a TRAINED policy experiences on the real actuators.

Read-only diagnosis. Builds the env exactly as scripts/train.py does (from the
run's manifest), loads a saved checkpoint, and for chosen start squares records
every macro-step:

  commanded arm joint targets   data.ctrl[:6] after RoboChessEnv._scale_action
  actual arm joints / velocity  data.qpos[:6], data.qvel[:6]
  commanded pinch position      FK of the commanded joints (scratch MjData)
  actual pinch position         data.site_xpos[ur_gripper_pinch]

Two measurements per episode:
  (A) EXPERIENCED  -- every step of the policy's own rollout, and the
      quasi-static subset (max |qvel| < 0.01 rad/s and the command changed by
      < 1e-3 rad since the previous step), where remaining error is droop,
      not lag.
  (B) SETTLED      -- at step --hold-at the policy's current action is held
      fixed for --hold-steps macro-steps; the error once the arm has come to
      rest is the true steady-state error for a pose the policy itself chose.

    python scripts/measure_tracking_error.py runs/<run> --checkpoint 500000 --out x.json
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(train)

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--checkpoint", type=int, required=True)
    p.add_argument("--near", nargs="+", default=["a1", "d1", "h1"])
    p.add_argument("--far", nargs="+", default=["a8", "d8", "h8"])
    p.add_argument("--hold-at", type=int, default=50)
    p.add_argument("--hold-steps", type=int, default=75)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    cfg = json.loads((a.run_dir / "manifest.json").read_text())["config_resolved"]
    train.apply_thread_pinning(1)
    import mujoco, numpy as np
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from robochess.tasks.t1_pick_and_place import _pinch_site_id

    env = train.build_env(cfg)                      # same stack as training
    base, task = env.unwrapped, env.env             # task = T1PickAndPlaceTask (or T2)
    m, d = base.model, base.data
    pin = _pinch_site_id(m)
    scratch = mujoco.MjData(m)

    # Runtime confirmation of the actuators this env drives (not from code reading).
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(6)]
    actuators = [dict(name=names[i], gaintype=int(m.actuator_gaintype[i]), biastype=int(m.actuator_biastype[i]),
                      kp=float(m.actuator_gainprm[i][0]), biasprm=[float(x) for x in m.actuator_biasprm[i][:3]])
                 for i in range(6)]
    gravcomp_max = float(m.body_gravcomp.max())

    Algo = PPO if cfg["algorithm"] == "ppo" else SAC
    model = Algo.load(str(a.run_dir / f"checkpoints/rc_{a.checkpoint}_steps.zip"), device="cpu")
    vn = None
    if train.uses_vecnormalize(cfg):
        vn = VecNormalize.load(str(a.run_dir / f"checkpoints/rc_vecnormalize_{a.checkpoint}_steps.pkl"),
                               DummyVecEnv([lambda: Monitor(train.build_env(cfg))]))
        vn.training = False

    def act(obs):
        o = {k: v[None] for k, v in obs.items()}
        if vn is not None:
            o = vn.normalize_obs(o)
        return model.predict(o, deterministic=True)[0][0]

    def fk_pinch(q_arm):
        scratch.qpos[:] = d.qpos
        scratch.qpos[:6] = q_arm
        mujoco.mj_kinematics(m, scratch)
        return scratch.site_xpos[pin].copy()

    def sample():
        cmd = d.ctrl[:6].copy()
        return dict(cmd=cmd, q=d.qpos[:6].copy(), qd=d.qvel[:6].copy(),
                    pinch=d.site_xpos[pin].copy(), pinch_cmd=fk_pinch(cmd))

    def summarise(samples):
        if not samples:
            return None
        jerr = np.degrees(np.array([s["q"] - s["cmd"] for s in samples]))
        terr = np.array([s["pinch"] - s["pinch_cmd"] for s in samples]) * 1000.0
        return dict(
            n=len(samples),
            joint_err_deg_median={j: float(np.median(jerr[:, k])) for k, j in enumerate(JOINTS)},
            joint_abs_err_deg_p90={j: float(np.percentile(np.abs(jerr[:, k]), 90)) for k, j in enumerate(JOINTS)},
            pinch_err_mm_median_xyz=[float(x) for x in np.median(terr, axis=0)],
            pinch_err_mm_norm_median=float(np.median(np.linalg.norm(terr, axis=1))),
            pinch_err_mm_norm_p90=float(np.percentile(np.linalg.norm(terr, axis=1), 90)),
            pinch_actual_xyz_median=[float(x) for x in np.median([s["pinch"] for s in samples], axis=0)],
        )

    board_c = base.data.xpos[base._body_id("board")][:2].copy()
    results = dict(run=a.run_dir.name, checkpoint=a.checkpoint, algorithm=cfg["algorithm"],
                   actuators=actuators, body_gravcomp_max=gravcomp_max, squares={})
    for group, squares in (("near", a.near), ("far", a.far)):
        for sq in squares:
            task.start_square, task.target_square = sq, ("d4" if group == "near" else "d5")
            obs, info = env.reset(seed=0)
            assert info["start_square"] == sq
            piece_xyz = base._achieved_goal().copy()
            experienced, quasi = [], []
            prev_cmd = None
            for t in range(a.hold_at):
                obs, r, term, trunc, info = env.step(act(obs))
                s = sample(); experienced.append(s)
                if prev_cmd is not None and np.max(np.abs(s["qd"])) < 0.01 and np.max(np.abs(s["cmd"] - prev_cmd)) < 1e-3:
                    quasi.append(s)
                prev_cmd = s["cmd"]
                if term or trunc:
                    break
            held = None
            if not (term or trunc):
                hold_action = act(obs)
                trace = []
                for t in range(a.hold_steps):
                    obs, r, term, trunc, info = env.step(hold_action)
                    s = sample(); trace.append(s)
                    if term or trunc:
                        break
                settled = [s for s in trace[-10:] if np.max(np.abs(s["qd"])) < 0.01]
                held = dict(summary=summarise(settled), settled_samples=len(settled),
                            final_max_abs_qvel=float(np.max(np.abs(trace[-1]["qd"]))),
                            ended_early=bool(term or trunc))
            results["squares"][sq] = dict(
                group=group, piece_xyz=[float(x) for x in piece_xyz],
                piece_y_from_board_centre=float(piece_xyz[1] - board_c[1]),
                episode_ended_before_hold=bool(held is None),
                experienced_all=summarise(experienced), experienced_quasi_static=summarise(quasi),
                settled_hold=held)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
