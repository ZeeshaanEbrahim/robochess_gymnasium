#!/usr/bin/env python
"""Section 1c: SAC actor std at a typical held state. Read-only."""
import importlib.util, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

import gymnasium as gym, numpy as np, torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

run_dir = REPO / "runs" / "_probe_cart_place_sac_seed0"
cfg = json.loads((run_dir / "manifest.json").read_text())["config_resolved"]
train.apply_thread_pinning(1)


class Rec(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.held_obs = []

    def step(self, act):
        obs, r, term, trunc, info = self.env.step(act)
        t = info.get("place_terms", {})
        if t.get("held", False) and len(self.held_obs) < 20:
            self.held_obs.append(obs)
        return obs, r, term, trunc, info


holder = {}


def mk():
    w = Rec(train.build_env(cfg))
    holder["w"] = w
    return Monitor(w)


venv = DummyVecEnv([mk])
model = SAC.load(str(run_dir / "final_model.zip"), env=venv, device="cpu")
venv.seed(20000)
obs = venv.reset()
w = holder["w"]
steps = 0
while len(w.held_obs) < 20 and steps < 300 * 30:
    act, _ = model.predict(obs, deterministic=False)
    obs, r, done, infos = venv.step(act)
    steps += 1

print(f"collected {len(w.held_obs)} held-state observations over {steps} steps")

actor = model.actor
stds = []
for o in w.held_obs:
    obs_t, _ = actor.obs_to_tensor(o)
    with th.no_grad():
        mean_actions, log_std, kwargs = actor.get_action_dist_params(obs_t)
    stds.append(th.exp(log_std).cpu().numpy().reshape(-1).tolist())
stds = np.array(stds)
print("per-dim std across held obs: min", stds.min(axis=0).tolist())
print("per-dim std across held obs: median", np.median(stds, axis=0).tolist())
print("per-dim std across held obs: max", stds.max(axis=0).tolist())
