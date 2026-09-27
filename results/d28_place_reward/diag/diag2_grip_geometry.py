#!/usr/bin/env python
"""Section 2 (read-only diagnosis): grip geometry. Fresh 50-episode stochastic
rerun (seed 20000) per PPO seed, final_model, frozen VecNormalize.
Does not modify tracked files; writes only its own stdout/--out.
"""
import argparse, importlib.util, json, sys, copy
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

import gymnasium as gym, numpy as np, statistics as st
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

SEEDS = ["ppo_seed0", "ppo_seed1", "ppo_seed2"]
N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 50
EVAL_SEED = 20000


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

    def _pad_contact_heights_mm(self):
        m, d = self.b.model, self.b.data
        piece = self.b._piece_bid
        piece_z = float(self.b._achieved_goal()[2])
        left_z = right_z = None
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if piece == b1:
                other = b2
            elif piece == b2:
                other = b1
            else:
                continue
            if other == self.b._left_pad_bid:
                left_z = float(c.pos[2])
            elif other == self.b._right_pad_bid:
                right_z = float(c.pos[2])
        left_mm = None if left_z is None else (left_z - piece_z) * 1e3
        right_mm = None if right_z is None else (right_z - piece_z) * 1e3
        return left_mm, right_mm

    def step(self, act):
        obs, r, term, trunc, info = self.env.step(act)
        t = info.get("place_terms", {})
        left_mm, right_mm = self._pad_contact_heights_mm()
        row = dict(
            st=info["place_state"],
            held=bool(t.get("held", False)),
            pinched=bool(t.get("pinched", False)),
            ap=float(self.b.data.qpos[self.b._grip_qadr]),
            tilt_deg=float(self.b._piece_tilt_deg()),
            pad_left_mm=left_mm, pad_right_mm=right_mm,
        )
        self.cur.append(row)
        if term or trunc:
            self.episodes.append(self.cur)
            self.cur = []
        return obs, r, term, trunc, info


def run(seed_name, n_episodes=N_EPISODES, seed=EVAL_SEED):
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
    eps = run(s)
    first_held = []  # (qpos, pad_height_mm, tilt_deg)
    held_carry_qpos = []
    n_held_episodes = 0
    n_held_lost_s3 = 0
    for e in eps:
        held_idx = [k for k, r in enumerate(e) if r["held"]]
        if held_idx:
            n_held_episodes += 1
            k0 = held_idx[0]
            r0 = e[k0]
            pads = [x for x in (r0["pad_left_mm"], r0["pad_right_mm"]) if x is not None]
            pad_h = sum(pads) / len(pads) if pads else None
            first_held.append(dict(qpos=r0["ap"], pad_height_mm=pad_h, tilt_deg=r0["tilt_deg"]))
            if e[-1]["st"] == "S3":
                n_held_lost_s3 += 1
            for k in held_idx:
                held_carry_qpos.append(e[k]["ap"])
    results[s] = dict(
        n_episodes=len(eps), n_held_episodes=n_held_episodes, n_held_lost_s3=n_held_lost_s3,
        first_held_qpos=[x["qpos"] for x in first_held],
        first_held_pad_height_mm=[x["pad_height_mm"] for x in first_held if x["pad_height_mm"] is not None],
        first_held_tilt_deg=[x["tilt_deg"] for x in first_held],
        held_carry_qpos=held_carry_qpos,
    )

print(json.dumps(results, indent=2))
