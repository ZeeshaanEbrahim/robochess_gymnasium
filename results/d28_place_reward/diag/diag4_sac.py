#!/usr/bin/env python
"""Section 4 (read-only diagnosis): SAC.
- Pre-tanh actor mean/std at held steps, 50-episode stochastic rerun (seed 20000).
- Counterfactual from 10 stuck states (step 150): move horizontally above e5,
  descend at the expert's descend speed, then open.
Does not modify tracked files. No VecNormalize (SAC's config carries none).
"""
import copy, importlib.util, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
import gate_expert_cartesian as gate

import gymnasium as gym, mujoco, numpy as np, torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 50
N_STUCK_EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 10
EVAL_SEED = 20000
FORK_STEP = 150
CARTESIAN_STEP = 0.020
DESCEND_TOL_M = 0.002
HORIZ_TOL_M = 0.020  # "above e5": horizontal alignment tolerance before descending

run_dir = REPO / "runs" / "_probe_cart_place_sac_seed0"
cfg = json.loads((run_dir / "manifest.json").read_text())["config_resolved"]
train.apply_thread_pinning(1)


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
        self.cur.append(dict(held=bool(t.get("held", False))))
        if term or trunc:
            self.episodes.append(self.cur)
            self.cur = []
        return obs, r, term, trunc, info


holder = {}


def mk():
    w = Rec(train.build_env(cfg))
    holder["w"] = w
    return Monitor(w)


venv = DummyVecEnv([mk])
model = SAC.load(str(run_dir / "final_model.zip"), env=venv, device="cpu")
venv.seed(EVAL_SEED)
obs = venv.reset()
w = holder["w"]

actor = model.actor
per_step_means, per_step_stds, per_step_held = [], [], []
step_in_episode = 0
while len(w.episodes) < N_EPISODES:
    obs_t, _ = actor.obs_to_tensor(obs)
    with th.no_grad():
        mean_actions, log_std, kwargs = actor.get_action_dist_params(obs_t)
    mean = mean_actions.detach().cpu().numpy().reshape(-1)
    std = th.exp(log_std).detach().cpu().numpy().reshape(-1)
    step_in_episode += 1
    act, _ = model.predict(obs, deterministic=False)
    obs, r, done, infos = venv.step(act)
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

pretanh = dict(
    n_held_steps=int(held_mask.sum()),
    mean_min=held_means.min(axis=0).tolist(),
    mean_median=np.median(held_means, axis=0).tolist(),
    mean_max=held_means.max(axis=0).tolist(),
    std_min=held_stds.min(axis=0).tolist(),
    std_median=np.median(held_stds, axis=0).tolist(),
    std_max=held_stds.max(axis=0).tolist(),
)
print("PRETANH_RESULT " + json.dumps(pretanh), file=sys.stderr, flush=True)

# ---------------------------------------------------------------- counterfactual
BASE_ATTRS = ("_arm_target", "_ik_rejects", "_elapsed_steps", "_piece_ang_settled_streak",
             "_piece_lin_settled_streak", "_piece_upright_streak", "_ever_held",
             "_release_record", "_contact_flags")
PLACE_ATTRS = ("_t", "_held_once", "_lift_streak", "_lift_bonus_paid", "_s2_entered",
              "_last_state", "_last_terms")


def snapshot(base, place):
    return dict(data=copy.deepcopy(base.data),
               base={a: copy.deepcopy(getattr(base, a)) for a in BASE_ATTRS},
               place={a: copy.deepcopy(getattr(place, a)) for a in PLACE_ATTRS})


def restore(base, place, snap):
    base.data = copy.deepcopy(snap["data"])
    mujoco.mj_forward(base.model, base.data)
    for a, v in snap["base"].items():
        setattr(base, a, copy.deepcopy(v))
    for a, v in snap["place"].items():
        setattr(place, a, copy.deepcopy(v))


holder2 = {}


def mk2():
    w2 = train.build_env(cfg)
    holder2["place"] = w2
    holder2["base"] = w2.unwrapped
    return Monitor(w2)


venv2 = DummyVecEnv([mk2])
model2 = SAC.load(str(run_dir / "final_model.zip"), env=venv2, device="cpu")
venv2.seed(EVAL_SEED)
obs2 = venv2.reset()
base = holder2["base"]
place = holder2["place"]

stuck_snapshots = []
episode_idx = 0
step_in_episode2 = 0
while len(stuck_snapshots) < N_STUCK_EPISODES:
    act2, _ = model2.predict(obs2, deterministic=False)
    step_in_episode2 += 1
    if step_in_episode2 == FORK_STEP:
        stuck_snapshots.append(snapshot(base, place))
    obs2, r2, done2, infos2 = venv2.step(act2)
    if done2[0]:
        episode_idx += 1
        step_in_episode2 = 0
print(f"collected {len(stuck_snapshots)} SAC stuck snapshots at step {FORK_STEP}", file=sys.stderr, flush=True)


def unit_xy_toward_goal(base):
    piece_xy = base._achieved_goal()[:2].astype(float)
    goal_xy = base._desired_goal[:2].astype(float)
    d = goal_xy - piece_xy
    dist = float(np.linalg.norm(d))
    return (d / dist if dist > 1e-9 else np.zeros(2)), dist


def scripted_step(base, phase, gripper_cmd):
    from robochess.tasks.t1_pick_and_place import pickup_geometry, _pinch_site_id
    sid = _pinch_site_id(base.model)
    geom = pickup_geometry(base, sid, float(place.dz))
    lift_m = float(geom["lift"])
    direction, dist = unit_xy_toward_goal(base)
    a = np.zeros(4, dtype=np.float64)
    if phase == "move_horizontal":
        step_m = gate.braked_step_m(dist, gate.BRAKE_FRACTION)
        a[:2] = direction * min(step_m / CARTESIAN_STEP, 1.0)
        a[2] = float(np.clip((0.0 - 0.0) / CARTESIAN_STEP, -1.0, 1.0))  # hold current height
        arrived = dist < HORIZ_TOL_M
    elif phase == "descend":
        err = 0.0 - lift_m  # target lift 0 (board rest height)
        step_m = gate.braked_step_m(abs(err), gate.BRAKE_FRACTION_DESCEND)
        a[2] = float(np.clip(np.sign(err) * step_m / CARTESIAN_STEP, -1.0, 1.0))
        a[:2] = direction * min(gate.braked_step_m(dist, gate.BRAKE_FRACTION_DESCEND) / CARTESIAN_STEP, 1.0)
        arrived = lift_m < DESCEND_TOL_M
    elif phase == "open":
        arrived = None
    else:
        raise ValueError(phase)
    a[3] = -1.0 if phase == "open" else gripper_cmd
    return a, lift_m, dist, arrived


def run_variant(snap, n_steps=300 - FORK_STEP):
    restore(base, place, snap)
    rows = []
    phase = "move_horizontal"
    gripper_cmd = 1.0  # keep closed while moving/descending; opened explicitly in the "open" phase
    for i in range(n_steps):
        a, lift_m, dist, arrived = scripted_step(base, phase, gripper_cmd)
        obs3, r3, term, trunc, info = place.step(a)
        t = info.get("place_terms", {})
        rows.append(dict(step=i, st=info["place_state"], held=bool(t.get("held", False)),
                         pinched=bool(t.get("pinched", False)), dxy_mm=float(info["xy_distance_to_goal"]) * 1e3,
                         lift_mm=lift_m * 1e3, reward=float(r3), succ=bool(info.get("is_success", False))))
        if phase == "move_horizontal" and arrived:
            phase = "descend"
        elif phase == "descend" and arrived:
            phase = "open"
        if term or trunc:
            break
    return rows


results = []
for ei in range(N_STUCK_EPISODES):
    rows = run_variant(stuck_snapshots[ei])
    held_flags = [r["held"] for r in rows]
    pinched_flags = [r["pinched"] for r in rows]
    loss_step = None
    was_held = False
    for k, h in enumerate(held_flags):
        if h:
            was_held = True
        elif was_held and not h:
            loss_step = k
            break
    results.append(dict(
        episode=ei, n_steps=len(rows), final_pinched=pinched_flags[-1], final_held=held_flags[-1],
        ever_success=any(r["succ"] for r in rows), loss_step=loss_step,
        final_dxy_mm=rows[-1]["dxy_mm"], final_lift_mm=rows[-1]["lift_mm"],
        reward_path=[dict(step=r["step"], reward=round(r["reward"], 4)) for r in rows
                    if r["step"] == 0 or r["step"] % 10 == 0 or r["step"] == len(rows) - 1],
    ))

print(json.dumps(dict(pretanh=pretanh, counterfactual=results), indent=2))
