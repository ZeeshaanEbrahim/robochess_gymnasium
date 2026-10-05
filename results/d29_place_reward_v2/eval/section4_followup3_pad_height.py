#!/usr/bin/env python
"""D29 follow-up (read-only, 20-min timebox): median pad-contact height on the
piece while held, mm above the piece's own origin (base). Reuses eval_probe.py's
own geometry formula: pinch = data.site_xpos[pinch_site]; piece = achieved_goal()
(body xpos, at rest == the piece's base in world z); height_above_base =
(pinch_z - piece_z) * 1000.

150-episode stochastic rerun (seed 20000, torch-seeded, frozen VecNormalize for
PPO) -- fewer than the usual 300 to stay inside the timebox; only a pooled
median over held steps is needed, not per-episode structure.

Does not modify tracked files or runs/. Reads runs/, writes only --out.
"""
import argparse, importlib.util, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

import copy
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robochess.policies import SquashedMultiInputActorCriticPolicy  # noqa: F401
from robochess.tasks.t1_pick_and_place import _pinch_site_id

EVAL_SEED = 20000
N_EPISODES = 150

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--episodes", type=int, default=N_EPISODES)
p.add_argument("--runs-dir", type=Path, default=REPO / "runs")
a = p.parse_args()
N_EPISODES = a.episodes

run_dir = a.runs_dir / a.run
cfg = json.loads((run_dir / "manifest.json").read_text())["config_resolved"]
algo = cfg["algorithm"]
train.apply_thread_pinning(1)


class Rec(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.b = env.unwrapped
        self.sid = _pinch_site_id(self.b.model)
        self.episodes = []
        self.cur = []

    def reset(self, **kw):
        out = self.env.reset(**kw)
        self.cur = []
        return out

    def step(self, act):
        obs, r, term, trunc, info = self.env.step(act)
        t = info.get("place_terms", {})
        pinch_z = float(self.b.data.site_xpos[self.sid][2])
        piece_z = float(self.b._achieved_goal()[2])
        row = dict(held=bool(t.get("held", False)), pinched=bool(t.get("pinched", False)),
                   height_above_base_mm=(pinch_z - piece_z) * 1000.0)
        self.cur.append(row)
        if term or trunc:
            self.episodes.append(self.cur)
        return obs, r, term, trunc, info


holder = {}
def mk():
    w = Rec(train.build_env(cfg))
    holder["w"] = w
    return Monitor(w)


venv = DummyVecEnv([mk])
mp = run_dir / "final_model.zip"
if algo == "ppo":
    vn = run_dir / "final_vecnormalize.pkl"
    venv = VecNormalize.load(str(vn), venv)
    venv.training = False
    venv.norm_reward = False

model = (PPO if algo == "ppo" else SAC).load(str(mp), env=venv, device="cpu")
set_random_seed(EVAL_SEED)
model.set_random_seed(EVAL_SEED)
venv.seed(EVAL_SEED)
obs = venv.reset()
w = holder["w"]

while len(w.episodes) < N_EPISODES:
    act, _ = model.predict(obs, deterministic=False)
    obs, r, done, infos = venv.step(act)
venv.close()

episodes = w.episodes[:N_EPISODES]
held_heights = [row["height_above_base_mm"] for ep in episodes for row in ep if row["held"]]
pinched_heights = [row["height_above_base_mm"] for ep in episodes for row in ep if row["pinched"]]
out = dict(run=a.run, n_episodes=len(episodes),
           n_held=len(held_heights),
           median_height_above_base_mm_held=float(np.median(held_heights)) if held_heights else None,
           n_pinched=len(pinched_heights),
           median_height_above_base_mm_pinched=float(np.median(pinched_heights)) if pinched_heights else None)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out, indent=2))
print(a.run, out, flush=True)
