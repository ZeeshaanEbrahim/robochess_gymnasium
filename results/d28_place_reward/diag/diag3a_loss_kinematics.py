#!/usr/bin/env python
"""Section 3 addendum (read-only diagnosis): losses per 100 held steps, and
piece/gripper motion in the 5 steps before each loss. Same 50-episode
stochastic rerun (seed 20000, final_model, frozen VecNormalize) as Section 2.
Does not modify tracked files.
"""
import importlib.util, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

import gymnasium as gym, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SEEDS = ["ppo_seed0", "ppo_seed1", "ppo_seed2"]
N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 50
EVAL_SEED = 20000
CARTESIAN_STEP_MM = 20.0


class Rec(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.b = env.unwrapped
        self.episodes = []
        self.cur = []

    def reset(self, **kw):
        out = self.env.reset(**kw)
        self.cur = []
        return out

    def step(self, act):
        obs, r, term, trunc, info = self.env.step(act)
        t = info.get("place_terms", {})
        piece_speed = float(np.linalg.norm(self.b.data.qvel[self.b._piece_vadr:self.b._piece_vadr + 3]))
        a = np.asarray(act).reshape(-1)
        cart_mag = float(np.linalg.norm(a[:3]))
        self.cur.append(dict(st=info["place_state"], held=bool(t.get("held", False)),
                             piece_speed=piece_speed, cart_mag=cart_mag,
                             cart_mag_mm=cart_mag * CARTESIAN_STEP_MM))
        if term or trunc:
            self.episodes.append(self.cur)
            self.cur = []
        return obs, r, term, trunc, info


def run(seed_name, n_episodes, seed=EVAL_SEED):
    run_dir = REPO / "runs" / f"_probe_cart_place_{seed_name}"
    cfg = json.loads((run_dir / "manifest.json").read_text())["config_resolved"]
    train.apply_thread_pinning(1)
    holder = {}

    def mk():
        w = Rec(train.build_env(cfg))
        holder["w"] = w
        return Monitor(w)

    venv = DummyVecEnv([mk])
    vn = run_dir / "final_vecnormalize.pkl"
    venv = VecNormalize.load(str(vn), venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(str(run_dir / "final_model.zip"), env=venv, device="cpu")
    venv.seed(seed)
    obs = venv.reset()
    w = holder["w"]
    while len(w.episodes) < n_episodes:
        act, _ = model.predict(obs, deterministic=False)
        obs, r, done, infos = venv.step(act)
    venv.close()
    return w.episodes[:n_episodes]


results = {}
for s in SEEDS:
    eps = run(s, N_EPISODES)
    total_held_steps = 0
    losses = []
    for ei, e in enumerate(eps):
        held_idx = [k for k, r in enumerate(e) if r["held"]]
        total_held_steps += len(held_idx)
        if held_idx and e[-1]["st"] == "S3":
            k = held_idx[-1]  # last held step before the piece stays lost
            window = e[max(0, k - 4):k + 1]  # the 5 steps up to and including the last held step
            losses.append(dict(
                episode=ei, loss_step=k, episode_len=len(e), window_len=len(window),
                piece_speed=[round(r["piece_speed"], 5) for r in window],
                cart_mag_mm=[round(r["cart_mag_mm"], 3) for r in window],
            ))
    results[s] = dict(
        n_episodes=len(eps), total_held_steps=total_held_steps, n_losses=len(losses),
        losses_per_100_held_steps=(len(losses) / total_held_steps * 100 if total_held_steps else None),
        losses=losses,
    )

print(json.dumps(results, indent=2))
