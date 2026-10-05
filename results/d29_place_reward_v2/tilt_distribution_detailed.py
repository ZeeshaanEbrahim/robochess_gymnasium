"""Report-only: split S1-step tilt distribution by episode outcome (placed vs
not) and by phase (before the first loss of pinch vs after a re-grip), for a
D28 PPO place run. Same settings as the earlier aggregate run: seed 20000,
50 episodes, stochastic, torch-seeded, frozen VecNormalize -- rerun because
that run only saved aggregate stats, not per-step data.

Usage: python tilt_distribution_detailed.py RUN_NAME OUT.json
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

episodes = []
cur = []
n_episodes = 0
while n_episodes < N_EPISODES:
    act, _ = model.predict(obs, deterministic=False)
    obs, r, done, infos = venv.step(act)
    info = infos[0]
    t = info.get("place_terms")
    cur.append(dict(state=t["state"], pinched=bool(t["pinched"]), tilt=float(t["piece_tilt_deg"]),
                    place_state=info.get("place_state")))
    if done[0]:
        episodes.append(dict(steps=cur, is_success=bool(info.get("is_success"))))
        cur = []
        n_episodes += 1

# tag every S1 step with its phase (before_first_loss / after_loss) and its episode's outcome
rows = []
for ep in episodes:
    steps = ep["steps"]
    loss_count = 0
    prev_pinched = False
    release_tilt = None
    max_s1_tilt = None
    for s in steps:
        if prev_pinched and not s["pinched"]:
            loss_count += 1
        if s["state"] == "S1":
            phase = "before_first_loss" if loss_count == 0 else "after_loss"
            rows.append(dict(placed=ep["is_success"], phase=phase, tilt=s["tilt"]))
            max_s1_tilt = s["tilt"] if max_s1_tilt is None else max(max_s1_tilt, s["tilt"])
        if s["place_state"] == "S2" and release_tilt is None:
            release_tilt = s["tilt"]
        prev_pinched = s["pinched"]
    ep["max_s1_tilt"] = max_s1_tilt
    ep["release_tilt"] = release_tilt


def cell_stats(tilts):
    arr = np.asarray(tilts, dtype=float)
    if arr.size == 0:
        return dict(n=0, median=None, p90=None, max=None, frac_gt_10=None, frac_gt_27p5=None, mean_gate=None)
    gates = np.asarray([place_posture_gate_value(x, GATE_LO, GATE_HI) for x in arr])
    return dict(n=int(arr.size), median=float(np.median(arr)), p90=float(np.percentile(arr, 90)),
               max=float(arr.max()), frac_gt_10=float(np.mean(arr > 10.0)),
               frac_gt_27p5=float(np.mean(arr > 27.5)), mean_gate=float(gates.mean()))


cells = {}
for placed in (True, False):
    for phase in ("before_first_loss", "after_loss"):
        tilts = [r["tilt"] for r in rows if r["placed"] == placed and r["phase"] == phase]
        cells[f"placed={placed}/{phase}"] = cell_stats(tilts)

placed_eps = [ep for ep in episodes if ep["is_success"]]
placed_detail = dict(
    n_placed_episodes=len(placed_eps),
    max_s1_tilt_per_episode=[round(ep["max_s1_tilt"], 3) if ep["max_s1_tilt"] is not None else None for ep in placed_eps],
    release_tilt_per_episode=[round(ep["release_tilt"], 3) if ep["release_tilt"] is not None else None for ep in placed_eps],
)

out = dict(run=RUN_NAME, n_episodes=N_EPISODES, n_placed=len(placed_eps), n_not_placed=N_EPISODES - len(placed_eps),
           cells=cells, placed_detail=placed_detail)
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(RUN_NAME, "n_placed", out["n_placed"], "n_not_placed", out["n_not_placed"])
for k, v in cells.items():
    print(" ", k, v)
print(" placed episodes max_s1_tilt:", placed_detail["max_s1_tilt_per_episode"])
print(" placed episodes release_tilt:", placed_detail["release_tilt_per_episode"])
