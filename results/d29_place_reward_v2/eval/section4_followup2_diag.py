#!/usr/bin/env python
"""D29 section 4 follow-ups, round 2 (read-only diagnosis): gripper qpos and
pre-tanh mean, for the gripper-conversion / release-window / tanh-saturation
checks. Same 300-episode stochastic protocol (seed 20000, torch-seeded, frozen
VecNormalize for PPO) as section4_followup_diag.py; adds two fields per step
that script didn't capture:
  - qpos: the gripper driver joint's actual qpos (data.qpos[grip_qadr])
  - mean_pretanh: action_net's raw output before ppo_mean_squash's tanh
    (captured via a forward hook on model.policy.action_net; for the
    non-squash policy this is identical to `mean`, since no tanh is applied
    there -- SquashedMultiInputActorCriticPolicy is the only subclass that
    diverges, see robochess/policies.py)

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
import torch as th
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robochess.policies import SquashedMultiInputActorCriticPolicy  # noqa: F401

EVAL_SEED = 20000
N_EPISODES = 300

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

pending = {}


class Rec(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.b = env.unwrapped
        self.episodes = []
        self.cur = []
        self.start_xy = None

    def reset(self, **kw):
        out = self.env.reset(**kw)
        self.cur = []
        self.start_xy = [float(v) for v in self.b._achieved_goal()[:2]]
        return out

    def step(self, act):
        obs, r, term, trunc, info = self.env.step(act)
        t = info.get("place_terms", {})
        row = dict(
            state=t.get("state"), d_xy=float(t.get("d_xy", np.nan)),
            tilt=float(t.get("piece_tilt_deg", np.nan)), held=bool(t.get("held", False)),
            pinched=bool(t.get("pinched", False)),
            action=pending["action"], mean=pending["mean"], mean_pretanh=pending["mean_pretanh"],
            std=pending["std"], qpos=float(self.b.data.qpos[self.b._grip_qadr]),
            is_success=bool(info.get("is_success", False)),
            terminated_reason=info.get("terminated_reason"),
        )
        self.cur.append(row)
        if term or trunc:
            self.episodes.append(dict(rows=self.cur))
        return obs, r, term, trunc, info


holder = {}
def mk():
    w = Rec(train.build_env(cfg))
    holder["w"] = w
    return Monitor(w)


venv = DummyVecEnv([mk])
mp = run_dir / "final_model.zip"
rms_same = None
if algo == "ppo":
    vn = run_dir / "final_vecnormalize.pkl"
    venv = VecNormalize.load(str(vn), venv)
    venv.training = False
    venv.norm_reward = False
    snap = lambda: {k: (copy.deepcopy(np.asarray(v.mean)), copy.deepcopy(np.asarray(v.var)), float(v.count))
                    for k, v in (venv.obs_rms.items() if isinstance(venv.obs_rms, dict) else [("obs", venv.obs_rms)])}
    rms_before = snap()

model = (PPO if algo == "ppo" else SAC).load(str(mp), env=venv, device="cpu")

pretanh_holder = {}
if algo == "ppo":
    orig_forward = model.policy.action_net.forward
    def hooked(x):
        out = orig_forward(x)
        pretanh_holder["v"] = out.detach().cpu().numpy().copy()
        return out
    model.policy.action_net.forward = hooked

set_random_seed(EVAL_SEED)
model.set_random_seed(EVAL_SEED)
venv.seed(EVAL_SEED)
obs = venv.reset()
w = holder["w"]

while len(w.episodes) < N_EPISODES:
    if algo == "ppo":
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_tensor)  # triggers the action_net hook
        mean = dist.distribution.mean.detach().cpu().numpy().reshape(-1)
        std = dist.distribution.stddev.detach().cpu().numpy().reshape(-1)
        mean_pretanh = pretanh_holder["v"].reshape(-1)
    else:
        obs_t, _ = model.actor.obs_to_tensor(obs)
        with th.no_grad():
            mean_actions, log_std, _kw = model.actor.get_action_dist_params(obs_t)
        mean = mean_actions.detach().cpu().numpy().reshape(-1)  # pre-tanh (SAC has no separate "mean" concept)
        std = th.exp(log_std).detach().cpu().numpy().reshape(-1)
        mean_pretanh = mean
    act, _ = model.predict(obs, deterministic=False)
    pending["action"] = [float(x) for x in act[0]]
    pending["mean"] = mean.tolist()
    pending["mean_pretanh"] = mean_pretanh.tolist()
    pending["std"] = std.tolist()
    obs, r, done, infos = venv.step(act)
if algo == "ppo":
    rms_after = snap()
    rms_same = all(np.array_equal(rms_before[k][0], rms_after[k][0]) and np.array_equal(rms_before[k][1], rms_after[k][1])
                   and rms_before[k][2] == rms_after[k][2] for k in rms_before)
venv.close()

episodes = w.episodes[:N_EPISODES]
a.out.parent.mkdir(parents=True, exist_ok=True)
out = dict(run=a.run, algo=algo, seed=EVAL_SEED, n_episodes=len(episodes), rms_same=rms_same, episodes=episodes)
a.out.write_text(json.dumps(out))
placed = sum(1 for e in episodes if e["rows"][-1]["is_success"])
print(a.run, "n", len(episodes), "rms_same", rms_same, "placed", placed, flush=True)
