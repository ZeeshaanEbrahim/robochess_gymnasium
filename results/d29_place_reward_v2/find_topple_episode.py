"""D29 toppled-hold cheat, step (1): search the D28 SAC seed0 model's own real
episodes (stochastic, torch-seeded, seed 20000) for one that is pinched with
tilt > 45 deg at step t=150. Must run as its own process (never imported into
acceptance.py's process): it builds a real SB3 model, which segfaults torch
intermittently when built alongside acceptance.py's many RoboChessEnv
instances in one process (DECISIONS D28).

Usage: python find_topple_episode.py OUT.json
Output committed as topple_actions.json (found on the very first of up to 20
scanned episodes -- episode_index 0, t150 tilt 90.61 deg).

Writes JSON: {"found": bool, "episode_index": int|None, "actions": [[...], ...],
"t150": {...} | None, "pose150": {...} | None, "episodes_scanned": int}.
`actions` is the raw action array model.predict returned at each step of the
found episode, 1..150 only; `acceptance.py`'s toppled_hold cheat replays it
directly through a D29-flagged env (no model needed for that part).
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
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

RUN_DIR = REPO / "runs" / "_probe_cart_place_sac_seed0"
SEED = 20000
MAX_EPISODES = 20
TARGET_T = 150
TILT_THRESHOLD = 45.0

cfg = json.loads((RUN_DIR / "manifest.json").read_text())["config_resolved"]
venv = DummyVecEnv([lambda: Monitor(train.build_env(cfg))])
model = SAC.load(str(RUN_DIR / "final_model.zip"), env=venv, device="cpu")
set_random_seed(SEED)
model.set_random_seed(SEED)
venv.seed(SEED)
obs = venv.reset()

found = None
episodes_scanned = 0
cur_actions: list = []
cur_terms: list = []

while episodes_scanned < MAX_EPISODES and found is None:
    act, _ = model.predict(obs, deterministic=False)
    obs, r, done, infos = venv.step(act)
    info = infos[0]
    cur_actions.append(np.asarray(act[0]).tolist())
    t = info.get("place_terms")
    cur_terms.append(t)
    if t is not None and t.get("t") == TARGET_T:
        if t.get("pinched") and t.get("piece_tilt_deg", 0.0) > TILT_THRESHOLD:
            base = venv.envs[0].unwrapped
            pose = dict(piece_xyz=base._achieved_goal().astype(float).tolist(),
                       piece_quat=base.data.qpos[base._piece_qadr + 3:base._piece_qadr + 7].tolist())
            found = dict(episode_index=episodes_scanned, actions=list(cur_actions),
                        t150=t, pose150=pose)
    if done[0]:
        episodes_scanned += 1
        cur_actions = []
        cur_terms = []

out = dict(found=found is not None, episode_index=(found or {}).get("episode_index"),
           actions=(found or {}).get("actions"), t150=(found or {}).get("t150"),
           pose150=(found or {}).get("pose150"), episodes_scanned=episodes_scanned)
Path(sys.argv[1]).write_text(json.dumps(out))
print("found:", out["found"], "episodes_scanned:", episodes_scanned,
      "t150:", out["t150"], flush=True)
