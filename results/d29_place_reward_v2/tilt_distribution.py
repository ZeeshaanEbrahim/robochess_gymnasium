"""Report-only: tilt distribution over all S1 steps for a D28 PPO place run.
Same eval settings as acceptance rule 1 (300-episode gate) but 50 episodes,
seed 20000, stochastic, torch-seeded, frozen VecNormalize. Standalone process
(builds a real SB3 model).

Usage: python tilt_distribution.py RUN_NAME OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("train", str(REPO / "scripts" / "train.py"))
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

import numpy as np  # noqa: E402
from robochess.tasks.t1_pick_and_place import place_posture_gate_value  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

RUN_NAME = sys.argv[1]
RUN_DIR = REPO / "runs" / RUN_NAME
SEED = 20000
N_EPISODES = 50
GATE_LO, GATE_HI = 10.0, 45.0

cfg = json.loads((RUN_DIR / "manifest.json").read_text())["config_resolved"]
venv = DummyVecEnv([lambda: Monitor(train.build_env(cfg))])
venv = VecNormalize.load(str(RUN_DIR / "final_vecnormalize.pkl"), venv)
venv.training = False
venv.norm_reward = False
model = PPO.load(str(RUN_DIR / "final_model.zip"), env=venv, device="cpu")
set_random_seed(SEED)
model.set_random_seed(SEED)
venv.seed(SEED)
obs = venv.reset()

s1_tilts = []
n_episodes = 0
while n_episodes < N_EPISODES:
    act, _ = model.predict(obs, deterministic=False)
    obs, r, done, infos = venv.step(act)
    t = infos[0].get("place_terms")
    if t is not None and t.get("state") == "S1":
        s1_tilts.append(float(t["piece_tilt_deg"]))
    if done[0]:
        n_episodes += 1

arr = np.asarray(s1_tilts, dtype=float)
gates = np.asarray([place_posture_gate_value(x, GATE_LO, GATE_HI) for x in arr])
out = dict(
    run=RUN_NAME, n_episodes=N_EPISODES, n_s1_steps=int(arr.size),
    median_tilt_deg=float(np.median(arr)) if arr.size else None,
    p90_tilt_deg=float(np.percentile(arr, 90)) if arr.size else None,
    max_tilt_deg=float(arr.max()) if arr.size else None,
    frac_tilt_gt_10=float(np.mean(arr > 10.0)) if arr.size else None,
    frac_tilt_gt_20=float(np.mean(arr > 20.0)) if arr.size else None,
    mean_gate_value=float(gates.mean()) if arr.size else None,
)
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(RUN_NAME, out, flush=True)
