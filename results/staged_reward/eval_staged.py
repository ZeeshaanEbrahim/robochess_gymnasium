"""Checkpoint eval reporting the staged reward's own two milestones separately.

scripts/eval_final.py records grasp_success, which (since the board-contact term)
already implies the piece is clear of the board -- so it cannot distinguish the
staged reward's grasp_bonus condition (both pads on the piece) from its lift_bonus
condition (that, AND clear of the board). This records both, plus approach.

    python results/staged_reward/eval_staged.py <run_dir> --checkpoint N --episodes 50 --out x.json
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

import gymnasium as gym
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(train)


class Milestones(gym.Wrapper):
    """Per-episode: closest approach, pads-on-piece ever, pads-and-clear ever."""

    def __init__(self, env):
        super().__init__(env)
        import mujoco
        base = env.unwrapped
        self._sid = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_SITE, "ur_gripper_pinch")
        node = env
        while not hasattr(node, "_piece_contacts"):
            node = node.env
        self._task = node

    def _sample(self):
        base = self.unwrapped
        d = float(np.linalg.norm(base.data.site_xpos[self._sid] - base._achieved_goal()))
        self._min_d = min(self._min_d, d)
        left, right, board = self._task._piece_contacts()
        if left and right:
            self._pads = True
            if not board:
                self._lift = True

    def reset(self, **kw):
        out = self.env.reset(**kw)
        self._min_d, self._pads, self._lift, self._grasp = np.inf, False, False, False
        self._sample()
        return out

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._sample()
        self._grasp = self._grasp or bool(info.get("grasp_success", False))
        info = dict(info)
        info.update(ep_min_d=self._min_d, ep_pads=self._pads, ep_lift=self._lift, ep_grasp=self._grasp)
        return obs, r, term, trunc, info


def main() -> int:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--checkpoint", type=int, default=None)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--eval-seed", type=int, default=10_000)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    cfg = json.loads((a.run_dir / "manifest.json").read_text())["config_resolved"]
    train.apply_thread_pinning(cfg.get("num_threads", 1))
    venv = DummyVecEnv([lambda: Monitor(Milestones(train.build_env(cfg)))])
    ck = a.checkpoint
    model_path = a.run_dir / ("final_model.zip" if ck is None else f"checkpoints/rc_{ck}_steps.zip")
    if cfg["algorithm"] == "ppo":
        vn = a.run_dir / ("final_vecnormalize.pkl" if ck is None else
                          f"checkpoints/rc_vecnormalize_{ck}_steps.pkl")
        venv = VecNormalize.load(str(vn), venv); venv.training = False; venv.norm_reward = False
    model = (PPO if cfg["algorithm"] == "ppo" else SAC).load(str(model_path), env=venv, device="cpu")

    venv.seed(a.eval_seed)
    obs = venv.reset()
    eps, ret = [], 0.0
    while len(eps) < a.episodes:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, infos = venv.step(act)
        ret += float(r[0])
        if done[0]:
            i = infos[0]
            eps.append(dict(success=bool(i["is_success"]), ret=ret, length=int(i["elapsed_steps"]),
                            reason=i.get("terminated_reason") or ("time_limit" if i.get("TimeLimit.truncated") else "other"),
                            min_gripper_to_piece_m=float(i["ep_min_d"]), pads_ever=bool(i["ep_pads"]),
                            lift_ever=bool(i["ep_lift"]), grasp_ever=bool(i["ep_grasp"]),
                            final_distance_to_goal_m=float(i["distance_to_goal"]),
                            start=i.get("start_square"), target=i.get("target_square")))
            ret = 0.0
    venv.close()

    d = np.array([e["min_gripper_to_piece_m"] for e in eps])
    out = dict(run=a.run_dir.name, checkpoint=ck, eval_seed=a.eval_seed, n=len(eps),
               max_episode_steps=cfg["max_episode_steps"],
               shaped_staged=cfg.get("shaped_staged", False),
               success=float(np.mean([e["success"] for e in eps])),
               pads_count=int(sum(e["pads_ever"] for e in eps)),
               lift_count=int(sum(e["lift_ever"] for e in eps)),
               grasp_count=int(sum(e["grasp_ever"] for e in eps)),
               median_min_d=float(np.median(d)), min_d=float(d.min()),
               eps_under_10cm=int((d < 0.10).sum()),
               mean_return=float(np.mean([e["ret"] for e in eps])),
               mean_len=float(np.mean([e["length"] for e in eps])),
               episodes=eps)
    print(json.dumps({k: v for k, v in out.items() if k != "episodes"}))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
