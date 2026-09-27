#!/usr/bin/env python
"""Section 1 (read-only diagnosis): exploration state.
Does not modify tracked files. Reads runs/*, writes only its own stdout / --out.
"""
import argparse, importlib.util, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
from stable_baselines3 import PPO, SAC
from tensorboard.backend.event_processing import event_accumulator

RUNS_DIR = REPO / "runs"
PPO_SEEDS = ["ppo_seed0", "ppo_seed1", "ppo_seed2"]
CKPTS = [100000, 300000, 500000, None]  # None = final_model


def model_path(run, ck):
    d = RUNS_DIR / f"_probe_cart_place_{run}"
    return d / ("final_model.zip" if ck is None else f"checkpoints/rc_{ck}_steps.zip")


def action_std(run, ck):
    mp = model_path(run, ck)
    model = PPO.load(str(mp), device="cpu")
    log_std = model.policy.log_std.detach().cpu().numpy()
    return np.exp(log_std).tolist()


def tb_scalar(run, tag, every=100000, tmax=1_100_000):
    logdir = RUNS_DIR / f"_probe_cart_place_{run}" / "logs"
    ea = event_accumulator.EventAccumulator(str(logdir))
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    vals = np.array([e.value for e in events])
    out = []
    for target in range(every, tmax + 1, every):
        idx = int(np.argmin(np.abs(steps - target)))
        out.append(dict(target_step=target, actual_step=int(steps[idx]), value=float(vals[idx])))
    return out


results = dict(ppo_action_std={}, ppo_entropy_loss={}, sac_ent_coef=None, sac_actor_std_at_held_state=None)

for run in PPO_SEEDS:
    results["ppo_action_std"][run] = {}
    for ck in CKPTS:
        label = "final" if ck is None else f"rc_{ck}"
        results["ppo_action_std"][run][label] = action_std(run, ck)
    results["ppo_entropy_loss"][run] = tb_scalar(run, "train/entropy_loss")

results["sac_ent_coef"] = tb_scalar("sac_seed0", "train/ent_coef", tmax=1_000_000)

print(json.dumps(results, indent=2))
