#!/usr/bin/env python
"""D29 section 4 (read-only diagnosis): full per-step diagnostics for one run.

300-episode stochastic rerun (seed 20000, torch-seeded, frozen VecNormalize for
PPO), same protocol as eval_probe.py / DECISIONS.md D29's pre-registered rules.
Captures per-step place_terms (state, d_xy, lift, piece_tilt_deg, held, pinched,
state_reward, shaping), the raw action, and (PPO only) the policy's mean/std used
to sample that action, then aggregates:
  - fraction of pooled steps in each place_state (S0-S3)
  - S1-step medians: lift (mm), d_xy (mm), piece_tilt (deg), state_reward, shaping
  - median first-held step (episodes with a held step only)
  - saturation (PPO only): fraction of held steps with analytic P(sample outside
    [-1,1]) > 0.9 on any of dx/dy/dz, using the policy's mean and std at that step
  - fraction of held steps with normalized gripper command < 0.9
  - toppled-hold at step 150 (pinched and tilt > 45 deg at the 150th step of the
    episode; False for episodes shorter than 150 steps)
  - shaping total per episode (median), next to the idle baseline for a piece
    that never moves: (1 - gamma) * k * d_xy per step, d_xy = 0.078 m (e4->e5)

Does not modify tracked files or runs/. Reads runs/, writes only --out.
"""
import argparse, importlib.util, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

import numpy as np
import torch as th
from scipy.stats import norm as _norm
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robochess.policies import SquashedMultiInputActorCriticPolicy  # noqa: F401

EVAL_SEED = 20000
N_EPISODES = 300
FORK_STEP = 150
GAMMA = 0.99
K = 50.0
E4_E5_DXY_M = 0.078

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--runs-dir", type=Path, default=REPO / "runs")
p.add_argument("--episodes", type=int, default=N_EPISODES)
a = p.parse_args()
N_EPISODES = a.episodes

run_dir = a.runs_dir / a.run
cfg = json.loads((run_dir / "manifest.json").read_text())["config_resolved"]
algo = cfg["algorithm"]
train.apply_thread_pinning(1)


def frac_outside(mean, std):
    lo = _norm.cdf(-1.0, loc=mean, scale=std)
    hi = 1.0 - _norm.cdf(1.0, loc=mean, scale=std)
    return lo + hi


venv = DummyVecEnv([lambda: Monitor(train.build_env(cfg))])
mp = run_dir / "final_model.zip"
rms_same = None
if algo == "ppo":
    vn = run_dir / "final_vecnormalize.pkl"
    venv = VecNormalize.load(str(vn), venv)
    venv.training = False
    venv.norm_reward = False
    import copy
    snap = lambda: {k: (copy.deepcopy(np.asarray(v.mean)), copy.deepcopy(np.asarray(v.var)), float(v.count))
                    for k, v in (venv.obs_rms.items() if isinstance(venv.obs_rms, dict) else [("obs", venv.obs_rms)])}
    rms_before = snap()

model = (PPO if algo == "ppo" else SAC).load(str(mp), env=venv, device="cpu")
set_random_seed(EVAL_SEED)
model.set_random_seed(EVAL_SEED)
venv.seed(EVAL_SEED)
obs = venv.reset()

episodes = []
cur = []
while len(episodes) < N_EPISODES:
    if algo == "ppo":
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_tensor)
        mean = dist.distribution.mean.detach().cpu().numpy().reshape(-1)
        std = dist.distribution.stddev.detach().cpu().numpy().reshape(-1)
    else:
        obs_t, _ = model.actor.obs_to_tensor(obs)
        with th.no_grad():
            mean_actions, log_std, _kw = model.actor.get_action_dist_params(obs_t)
        mean = mean_actions.detach().cpu().numpy().reshape(-1)  # pre-tanh
        std = th.exp(log_std).detach().cpu().numpy().reshape(-1)  # pre-tanh
    act, _ = model.predict(obs, deterministic=False)
    obs, r, done, infos = venv.step(act)
    info = infos[0]
    t = info.get("place_terms", {})
    row = dict(
        state=t.get("state"), d_xy=float(t.get("d_xy", np.nan)), lift=float(t.get("lift", np.nan)),
        tilt=float(t.get("piece_tilt_deg", np.nan)), held=bool(t.get("held", False)),
        pinched=bool(t.get("pinched", False)), state_reward=float(t.get("state_reward", np.nan)),
        shaping=float(t.get("shaping", 0.0)), action=[float(x) for x in act[0]],
        mean=mean.tolist(), std=std.tolist(),
    )
    cur.append(row)
    if done[0]:
        episodes.append(cur)
        cur = []
if algo == "ppo":
    rms_after = snap()
    rms_same = all(np.array_equal(rms_before[k][0], rms_after[k][0]) and np.array_equal(rms_before[k][1], rms_after[k][1])
                   and rms_before[k][2] == rms_after[k][2] for k in rms_before)
venv.close()

# ---- pooled state fractions ----
all_rows = [row for ep in episodes for row in ep]
n_total = len(all_rows)
state_counts = {s: sum(1 for row in all_rows if row["state"] == s) for s in ("S0", "S1", "S2", "S3")}
state_fraction = {s: c / n_total for s, c in state_counts.items()}

# ---- S1 medians ----
s1_rows = [row for row in all_rows if row["state"] == "S1"]
def med(key, rows, scale=1.0):
    vals = [row[key] * scale for row in rows if np.isfinite(row[key])]
    return float(np.median(vals)) if vals else None

s1 = dict(
    n_steps=len(s1_rows),
    median_lift_mm=med("lift", s1_rows, 1e3),
    median_dxy_mm=med("d_xy", s1_rows, 1e3),
    median_tilt_deg=med("tilt", s1_rows, 1.0),
    median_state_reward=med("state_reward", s1_rows, 1.0),
    median_shaping=med("shaping", s1_rows, 1.0),
)

# ---- first-held step (median over episodes that ever held) ----
first_held_steps = []
for ep in episodes:
    idx = next((k for k, row in enumerate(ep) if row["held"]), None)
    if idx is not None:
        first_held_steps.append(idx + 1)
first_held = dict(
    n_with_held=len(first_held_steps), n_without_held=len(episodes) - len(first_held_steps),
    median_first_held_step=(float(np.median(first_held_steps)) if first_held_steps else None),
)

# ---- held-step diagnostics: saturation (PPO only), gripper command ----
held_rows = [row for row in all_rows if row["held"]]
n_held = len(held_rows)
gripper_cmd_lt_09 = None
if n_held:
    cmds = [(row["action"][3] + 1.0) * 0.5 for row in held_rows]
    gripper_cmd_lt_09 = dict(k=sum(1 for c in cmds if c < 0.9), n=n_held,
                              fraction=sum(1 for c in cmds if c < 0.9) / n_held)

saturation = None
if algo == "ppo" and n_held:
    means = np.array([row["mean"][:3] for row in held_rows])
    stds = np.array([row["std"][:3] for row in held_rows])
    fo = frac_outside(means, stds)  # (n_held, 3), per-dim P(outside [-1,1])
    any_gt_09 = (fo > 0.9).any(axis=1)
    saturation = dict(
        n_held_steps=n_held,
        frac_any_dim_gt_0p9=float(any_gt_09.mean()),
        frac_dx_gt_0p9=float((fo[:, 0] > 0.9).mean()),
        frac_dy_gt_0p9=float((fo[:, 1] > 0.9).mean()),
        frac_dz_gt_0p9=float((fo[:, 2] > 0.9).mean()),
    )
elif algo == "sac":
    saturation = "N/A: SAC's actor is a squashed (tanh) Gaussian -- action = tanh(sample) is always in (-1, 1), " \
                 "there is no literal clip boundary the way there is for PPO's clipped Gaussian; not computed."

# ---- toppled-hold at step 150 ----
toppled = []
for ep in episodes:
    if len(ep) >= FORK_STEP:
        row = ep[FORK_STEP - 1]
        toppled.append(bool(row["pinched"] and row["tilt"] > 45.0))
    else:
        toppled.append(False)
toppled_hold = dict(k=sum(toppled), n=len(episodes), fraction=sum(toppled) / len(episodes))

# ---- shaping total per episode (median) vs idle baseline ----
shaping_totals = [sum(row["shaping"] for row in ep) for ep in episodes]
idle_baseline_per_step = (1.0 - GAMMA) * K * E4_E5_DXY_M
out = dict(
    run=a.run, algo=algo, seed=EVAL_SEED, n_episodes=len(episodes), n_steps_total=n_total,
    rms_same=rms_same,
    state_fraction=state_fraction, state_counts=state_counts,
    s1=s1, first_held=first_held,
    gripper_cmd_lt_0p9_while_held=gripper_cmd_lt_09,
    saturation=saturation,
    toppled_hold_step150=toppled_hold,
    shaping_total_per_episode_median=float(np.median(shaping_totals)),
    idle_baseline_per_step=idle_baseline_per_step,
    idle_baseline_episode_total_300steps=idle_baseline_per_step * 300,
)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out, indent=2))
print(a.run, "n", len(episodes), "rms_same", rms_same,
      "state_fraction", state_fraction, "toppled_hold", toppled_hold["fraction"], flush=True)
