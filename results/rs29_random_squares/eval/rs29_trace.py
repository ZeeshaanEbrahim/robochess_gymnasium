#!/usr/bin/env python
"""RS29 (EXPLORATORY / diagnostic, not pre-registered): per-step trace rerun of the pre-registered eval.

Same protocol as scripts/eval_probe.py --pairs (final model, pairs cycled in order, stochastic,
seed 20000, torch-seeded, frozen VecNormalize), same RNG call order, so episode i is the same episode as
episode i of results/rs29_random_squares/eval/final_seed<k>.json (checked by the caller). Adds what that
output does not keep: per step place_state, d_xy, lift, piece tilt, held/pinched, gripper command and
aperture, shaping, ever_held. Reads runs/, writes only --out (gzip json).

Derived from results/d29_place_reward_v2/eval/section4_full_diag.py (per-step capture) and
scripts/eval_probe.py (--pairs injection).
"""
import argparse, copy, gzip, importlib.util, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
import gymnasium as gym, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robochess.policies import SquashedMultiInputActorCriticPolicy  # noqa: F401

PAIRS = "e4:e5,c1:c2,d2:e2,a6:a5,e8:d7,d4:e5,f3:c6,h5:d3,g5:b3,c1:f6,h1:h8,h8:h1,a4:h4,b7:g2,b2:g7,f1:c8,a1:h8,h8:a1,h1:a8,a8:h1"
p = argparse.ArgumentParser()
p.add_argument("run"); p.add_argument("--out", type=Path, required=True)
p.add_argument("--episodes", type=int, default=300); p.add_argument("--seed", type=int, default=20000)
p.add_argument("--pairs", default=PAIRS); p.add_argument("--checkpoint", type=int, default=None)
a = p.parse_args()
pairs = [tuple(x.split(":")) for x in a.pairs.split(",")]
assert a.episodes % len(pairs) == 0
run_dir = REPO / "runs" / a.run
cfg = json.loads((run_dir / "manifest.json").read_text())["config_resolved"]
train.apply_thread_pinning(1)

class Cycle(gym.Wrapper):
    def __init__(self, env): super().__init__(env); self.n = 0
    def reset(self, **kw):
        s, t = pairs[self.n % len(pairs)]; self.n += 1
        kw["options"] = dict(kw.get("options") or {}, start_square=s, target_square=t)
        return self.env.reset(**kw)

venv = DummyVecEnv([lambda: Monitor(Cycle(train.build_env(cfg)))])
ck = a.checkpoint
vn = run_dir / ("final_vecnormalize.pkl" if ck is None else f"checkpoints/rc_vecnormalize_{ck}_steps.pkl")
mp = run_dir / ("final_model.zip" if ck is None else f"checkpoints/rc_{ck}_steps.zip")
venv = VecNormalize.load(str(vn), venv); venv.training = False; venv.norm_reward = False
snap = lambda: {k: (copy.deepcopy(np.asarray(v.mean)), copy.deepcopy(np.asarray(v.var)), float(v.count)) for k, v in
                (venv.obs_rms.items() if isinstance(venv.obs_rms, dict) else [("obs", venv.obs_rms)])}
before = snap()
model = PPO.load(str(mp), env=venv, device="cpu")
set_random_seed(a.seed); model.set_random_seed(a.seed); venv.seed(a.seed); obs = venv.reset()
base = venv.venv.envs[0].unwrapped
episodes, cur = [], []
while len(episodes) < a.episodes:
    act, _ = model.predict(obs, deterministic=False)
    obs, r, done, infos = venv.step(act); info = infos[0]; t = info["place_terms"]
    cur.append([t["state"], round(float(t["d_xy"]), 5), round(float(t["lift"]), 5), round(float(t.get("piece_tilt_deg", 0.0)), 2),
                int(bool(t["held"])), int(bool(t["pinched"])), round(float((act[0][3] + 1) * 0.5), 3),
                round(float(base.data.qpos[base._grip_qadr]), 4), round(float(t["shaping"]), 5), int(bool(info.get("ever_held", False))),
                round(float(r[0]), 4)])
    if done[0]:
        episodes.append(dict(pair=":".join(pairs[(len(episodes)) % len(pairs)]), reason=info.get("terminated_reason") or "time_limit",
                             success=bool(info.get("is_success", False)), rows=cur)); cur = []
after = snap()
same = all(np.array_equal(before[k][0], after[k][0]) and np.array_equal(before[k][1], after[k][1]) and before[k][2] == after[k][2] for k in before)
venv.close()
cols = ["state", "d_xy", "lift", "tilt", "held", "pinched", "grip_cmd", "grip_qpos", "shaping", "ever_held", "reward"]
a.out.parent.mkdir(parents=True, exist_ok=True)
with gzip.open(a.out, "wt") as f:
    json.dump(dict(run=a.run, checkpoint=ck, seed=a.seed, cols=cols, obs_rms_identical=same, episodes=episodes), f)
print(a.run, ck, "episodes", len(episodes), "rms_same", same, flush=True)
