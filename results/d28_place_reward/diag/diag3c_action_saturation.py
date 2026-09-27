#!/usr/bin/env python
"""Section 3c (read-only diagnosis): action saturation.
50-episode stochastic rerun (seed 20000), same procedure as diag2/diag3a.
Does not modify tracked files.
"""
import importlib.util, json, math, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

import gymnasium as gym, numpy as np
from scipy.stats import norm as _norm
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 50
EVAL_SEED = 20000
FORK_STEP = 150


def frac_outside(mean, std):
    """Analytic P(sample outside [-1, 1]) under N(mean, std), per dim (vectorised)."""
    lo = _norm.cdf(-1.0, loc=mean, scale=std)
    hi = 1.0 - _norm.cdf(1.0, loc=mean, scale=std)
    return lo + hi


class Rec(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episodes = []
        self.cur = []

    def reset(self, **kw):
        out = self.env.reset(**kw)
        self.cur = []
        return out

    def step(self, act):
        obs, r, term, trunc, info = self.env.step(act)
        t = info.get("place_terms", {})
        self.cur.append(dict(held=bool(t.get("held", False))))
        if term or trunc:
            self.episodes.append(self.cur)
            self.cur = []
        return obs, r, term, trunc, info


def run(seed_name, checkpoint, n_episodes, capture_step150=False):
    run_dir = REPO / "runs" / f"_probe_cart_place_{seed_name}"
    cfg = json.loads((run_dir / "manifest.json").read_text())["config_resolved"]
    train.apply_thread_pinning(1)
    holder = {}

    def mk():
        w = Rec(train.build_env(cfg))
        holder["w"] = w
        return Monitor(w)

    venv = DummyVecEnv([mk])
    ck = checkpoint
    mp = run_dir / ("final_model.zip" if ck is None else f"checkpoints/rc_{ck}_steps.zip")
    vn = run_dir / ("final_vecnormalize.pkl" if ck is None else f"checkpoints/rc_vecnormalize_{ck}_steps.pkl")
    venv = VecNormalize.load(str(vn), venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(str(mp), env=venv, device="cpu")
    venv.seed(EVAL_SEED)
    obs = venv.reset()
    w = holder["w"]

    per_step_means, per_step_stds, per_step_held = [], [], []
    step150_records = []
    step_in_episode = 0
    while len(w.episodes) < n_episodes:
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_tensor)
        mean = dist.distribution.mean.detach().cpu().numpy().reshape(-1)
        std = dist.distribution.stddev.detach().cpu().numpy().reshape(-1)
        step_in_episode += 1
        if capture_step150 and step_in_episode == FORK_STEP:
            step150_records.append(dict(mean=mean.tolist(), std=std.tolist()))
        act, _ = model.predict(obs, deterministic=False)
        obs, r, done, infos = venv.step(act)
        if len(w.episodes) > 0 and w.episodes[-1] and len(w.episodes[-1]) >= 1:
            pass  # held flag read below after we know this step's outcome
        # the step we just took belongs to the CURRENT (possibly just-finished) episode;
        # its `held` flag is the last row appended to w.cur before it was flushed, or to
        # w.episodes[-1] if this step ended the episode.
        last_row = (w.episodes[-1][-1] if done[0] else w.cur[-1])
        per_step_means.append(mean)
        per_step_stds.append(std)
        per_step_held.append(bool(last_row["held"]))
        if done[0]:
            step_in_episode = 0
    venv.close()

    means = np.array(per_step_means)
    stds = np.array(per_step_stds)
    held_mask = np.array(per_step_held)
    held_means = means[held_mask]
    held_stds = stds[held_mask]
    fo = np.array([frac_outside(held_means[i], held_stds[i]) for i in range(len(held_means))]) if len(held_means) else np.zeros((0, 4))

    out = dict(
        n_episodes=n_episodes, n_held_steps=int(held_mask.sum()),
        mean_min=held_means.min(axis=0).tolist() if len(held_means) else None,
        mean_median=np.median(held_means, axis=0).tolist() if len(held_means) else None,
        mean_max=held_means.max(axis=0).tolist() if len(held_means) else None,
        std=held_stds[0].tolist() if len(held_stds) else None,
        frac_outside_mean=fo.mean(axis=0).tolist() if len(fo) else None,
    )
    if capture_step150:
        out["step150_records"] = step150_records
    return out


results = {}
results["ppo_seed0_final"] = run("ppo_seed0", None, N_EPISODES)
results["ppo_seed1_final"] = run("ppo_seed1", None, N_EPISODES, capture_step150=True)
results["ppo_seed2_final"] = run("ppo_seed2", None, N_EPISODES)
results["ppo_seed1_rc300000"] = run("ppo_seed1", 300000, N_EPISODES)
results["ppo_seed1_rc500000"] = run("ppo_seed1", 500000, N_EPISODES)

print(json.dumps(results, indent=2))
