"""Section 2: stochastic rollouts of a checkpoint, with penalty-firing detail.

Deterministic eval with fixed squares gives N identical episodes and cannot show
how often the failure penalty fires. This samples from the policy instead.
"""
from __future__ import annotations
import argparse, collections, importlib.util, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_s = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(_s); _s.loader.exec_module(train)
from robochess.tasks.t1_pick_and_place import PENALISED_TERMINATIONS, failure_penalty_magnitude  # noqa: E402


def main() -> int:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import mujoco

    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path); p.add_argument("--checkpoint", type=int, default=None)
    p.add_argument("--episodes", type=int, default=100); p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    cfg = json.loads((a.run_dir / "manifest.json").read_text())["config_resolved"]
    train.apply_thread_pinning(cfg.get("num_threads", 1))
    venv = DummyVecEnv([lambda: Monitor(train.build_env(cfg))])
    ck = a.checkpoint
    mp = a.run_dir / ("final_model.zip" if ck is None else f"checkpoints/rc_{ck}_steps.zip")
    vn = a.run_dir / ("final_vecnormalize.pkl" if ck is None else f"checkpoints/rc_vecnormalize_{ck}_steps.pkl")
    venv = VecNormalize.load(str(vn), venv); venv.training = False; venv.norm_reward = False
    model = PPO.load(str(mp), env=venv, device="cpu")
    base = venv.venv.envs[0].unwrapped
    sid = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_SITE, "ur_gripper_pinch")

    venv.seed(a.seed)
    obs = venv.reset()
    eps, ret, step, pen_events = [], 0.0, 0, []
    while len(eps) < a.episodes:
        act, _ = model.predict(obs, deterministic=False)      # STOCHASTIC
        pinch = base.data.site_xpos[sid].copy()
        piece = base._achieved_goal().copy()
        obs, r, done, infos = venv.step(act)
        ret += float(r[0]); step += 1
        if done[0]:
            i = infos[0]
            reason = i.get("terminated_reason") or ("time_limit" if i.get("TimeLimit.truncated") else "other")
            penalised = reason in PENALISED_TERMINATIONS
            if penalised:
                pen_events.append(dict(episode=len(eps), step=step, reason=reason,
                                       pinch_xyz=[round(float(x), 4) for x in pinch],
                                       pinch_height_above_board=round(float(pinch[2] - base._board_surface_z), 4),
                                       pinch_to_piece_m=round(float(np.linalg.norm(pinch - piece)), 4),
                                       piece_moved_mm=round(1000 * float(np.linalg.norm(
                                           piece[:2] - base._desired_goal[:2])), 1),
                                       gripper_qpos=round(float(base.data.qpos[base._grip_qadr]), 3),
                                       episode_return=round(ret, 2)))
            eps.append(dict(ret=ret, length=step, reason=reason, penalised=penalised,
                            success=bool(i["is_success"])))
            ret, step = 0.0, 0
    venv.close()

    rs = np.array([e["ret"] for e in eps]); pen = np.array([e["penalised"] for e in eps])
    reasons = collections.Counter(e["reason"] for e in eps)
    out = dict(run=a.run_dir.name, checkpoint=ck, episodes=len(eps), stochastic=True,
               max_episode_steps=cfg["max_episode_steps"], reasons=dict(reasons),
               mean_return=float(rs.mean()), worst_return=float(rs.min()), best_return=float(rs.max()),
               penalised_episodes=int(pen.sum()),
               mean_return_unpenalised=float(rs[~pen].mean()) if (~pen).any() else None,
               mean_return_penalised=float(rs[pen].mean()) if pen.any() else None,
               mean_length=float(np.mean([e["length"] for e in eps])),
               successes=int(sum(e["success"] for e in eps)),
               failure_penalty=failure_penalty_magnitude(cfg["max_episode_steps"], 0.3),
               penalty_events=pen_events)
    print(json.dumps({k: v for k, v in out.items() if k != "penalty_events"}, indent=1))
    for ev in pen_events[:10]:
        print("   ", ev)
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
