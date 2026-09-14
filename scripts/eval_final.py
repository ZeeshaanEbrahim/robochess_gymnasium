#!/usr/bin/env python
"""Deterministic evaluation of a trained run's final model or a saved checkpoint.

Per episode it records what the training logs cannot:
  * success, return, length, terminated_reason (incl. D14 solver_instability)
  * min_gripper_to_piece_m      closest the pinch site ever got to the piece
  * min_pinch_height_m          lowest the pinch site got above the board-surface z
                                (anywhere; can be negative off the board's footprint)
  * min_pinch_height_while_over_board_m
                                same, restricted to steps with the pinch site over
                                the board slab's XY footprint (None if never over it)
  * final_distance_to_goal_m    piece-to-goal distance on the last step
  * grasp_ever                  whether info["grasp_success"] was ever True

The env is rebuilt from the run's own manifest (config_resolved), so the eval
matches what was trained. PPO reloads the VecNormalize statistics saved with
the evaluated model, frozen (training=False, norm_reward=False).

Geometry is read INSIDE the env on every step by `ApproachMetrics`, a
pass-through wrapper. It cannot be read from outside the VecEnv: DummyVecEnv
auto-resets on done, so state read after the terminating step already belongs
to the next episode. The wrapper does not change observations, rewards or
actions.

Eval episodes are seeded from --eval-seed, so evaluating several checkpoints of
one run with the same seed replays the same start/target squares.

    python scripts/eval_final.py runs/t1_ppo_shaped_state_seed0_seed0 --episodes 10
    python scripts/eval_final.py runs/<run> --checkpoint 300000 --episodes 50 --out x.json
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)   # imports nothing heavy at module scope


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--checkpoint", type=int, default=None,
                   help="evaluate checkpoints/rc_<N>_steps.zip instead of final_model.zip")
    p.add_argument("--eval-seed", type=int, default=None,
                   help="default: 10000 + the run's training seed")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    man = json.loads((a.run_dir / "manifest.json").read_text())
    cfg = man["config_resolved"]
    train.apply_thread_pinning(1)   # before torch/numpy/mujoco are imported

    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from robochess.tasks.t1_pick_and_place import _pinch_site_id

    class ApproachMetrics(gym.Wrapper):
        """Pass-through: adds running per-episode geometry to info, nothing else."""

        def __init__(self, env):
            super().__init__(env)
            self._base = env.unwrapped
            self._pinch = _pinch_site_id(self._base.model)

        def _sample(self):
            b = self._base
            pinch = b.data.site_xpos[self._pinch]
            d = float(np.linalg.norm(pinch - b._achieved_goal()))
            h = float(pinch[2] - b._board_surface_z)
            self._min_d = min(self._min_d, d)
            # "Height above the board surface" only means that when the pinch
            # site is over the board slab's XY footprint.
            c = b.data.xpos[b._body_id("board")][:2]
            half = b.model.geom_size[b._geom_id("board_slab")][:2]
            over = bool(np.all(np.abs(pinch[:2] - c) <= half))
            if over:
                self._min_h_ob = min(self._min_h_ob, h)
            if h < self._min_h:
                self._min_h = h
                self._min_h_over_board = over

        def reset(self, **kw):
            out = self.env.reset(**kw)
            self._min_d, self._min_h, self._grasp = np.inf, np.inf, False
            self._min_h_over_board = None
            self._min_h_ob = np.inf
            self._sample()
            return out

        def step(self, action):
            obs, r, term, trunc, info = self.env.step(action)
            self._sample()
            self._grasp = self._grasp or bool(info.get("grasp_success", False))
            info = dict(info)
            info["ep_min_gripper_to_piece_m"] = self._min_d
            info["ep_min_pinch_height_m"] = self._min_h
            info["ep_min_pinch_height_over_board"] = self._min_h_over_board
            info["ep_min_pinch_height_while_over_board_m"] = (
                None if np.isinf(self._min_h_ob) else self._min_h_ob)
            info["ep_grasp_ever"] = self._grasp
            return obs, r, term, trunc, info

    venv = DummyVecEnv([lambda: Monitor(ApproachMetrics(train.build_env(cfg)))])
    ck = a.checkpoint
    model_path = a.run_dir / ("final_model.zip" if ck is None else f"checkpoints/rc_{ck}_steps.zip")
    if cfg["algorithm"] == "ppo":
        vn = a.run_dir / ("final_vecnormalize.pkl" if ck is None else f"checkpoints/rc_vecnormalize_{ck}_steps.pkl")
        venv = VecNormalize.load(str(vn), venv)
        venv.training = False
        venv.norm_reward = False
    Algo = PPO if cfg["algorithm"] == "ppo" else SAC
    # env must be attached: SB3 refuses to load a HerReplayBuffer model without one
    model = Algo.load(str(model_path), env=venv, device="cpu")

    eval_seed = a.eval_seed if a.eval_seed is not None else 10_000 + cfg["seed"]
    venv.seed(eval_seed)
    obs = venv.reset()
    episodes, ep_ret = [], 0.0
    while len(episodes) < a.episodes:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, infos = venv.step(act)
        ep_ret += float(r[0])
        if done[0]:
            i = infos[0]
            reason = i.get("terminated_reason") or ("time_limit" if i.get("TimeLimit.truncated") else "other")
            episodes.append(dict(
                success=bool(i["is_success"]), ret=ep_ret, length=int(i["elapsed_steps"]), reason=reason,
                start=i.get("start_square"), target=i.get("target_square"),
                min_gripper_to_piece_m=i["ep_min_gripper_to_piece_m"],
                min_pinch_height_m=i["ep_min_pinch_height_m"],
                min_pinch_height_over_board=i["ep_min_pinch_height_over_board"],
                min_pinch_height_while_over_board_m=i["ep_min_pinch_height_while_over_board_m"],
                final_distance_to_goal_m=float(i["distance_to_goal"]),
                grasp_ever=bool(i["ep_grasp_ever"]),
            ))
            ep_ret = 0.0
    venv.close()

    col = lambda k: np.array([e[k] for e in episodes], dtype=float)
    dmin = col("min_gripper_to_piece_m")
    out = dict(
        run=a.run_dir.name, model=str(model_path.relative_to(a.run_dir)), checkpoint=ck,
        eval_seed=eval_seed, n=len(episodes),
        success=float(col("success").mean()), grasp_ever_count=int(col("grasp_ever").sum()),
        mean_return=float(col("ret").mean()), mean_len=float(col("length").mean()),
        reasons=dict(collections.Counter(e["reason"] for e in episodes)),
        min_gripper_to_piece_m=dict(median=float(np.median(dmin)), p10=float(np.percentile(dmin, 10)),
                                    p90=float(np.percentile(dmin, 90)), min=float(dmin.min())),
        min_pinch_height_m=dict(median=float(np.median(col("min_pinch_height_m"))),
                                min=float(col("min_pinch_height_m").min())),
        final_distance_to_goal_m_median=float(np.median(col("final_distance_to_goal_m"))),
        episodes=episodes,
    )
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "episodes"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
