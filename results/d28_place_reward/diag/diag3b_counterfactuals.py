#!/usr/bin/env python
"""Section 3 (read-only diagnosis): counterfactuals from seed 1's stuck state.

10 seed-1 stochastic episodes (seed 20000, final_model, frozen VecNormalize),
run the trained policy to step 150, then fork the live MuJoCo state (deepcopy
of `data` plus every stateful python attribute on the base env and the
PlaceReward wrapper) into four scripted continuations per episode:
  a: lift 20 mm (absolute, piece height above board rest) then hold
  b: lift 20 mm then move toward e5 at the Cartesian gate expert's braked
     carry-speed law (scripts/gate_expert_cartesian.py: braked_step_m,
     BRAKE_FRACTION=0.4), gripper held at the policy's last commanded value
  c: move toward e5 at the same braked rate without lifting first
  b_closed: variant b with the gripper command forced to +1 (fully closed)

Does not modify tracked files.
"""
import copy, importlib.util, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
import gate_expert_cartesian as gate  # unmodified; reused only for braked_step_m/BRAKE_FRACTION

import gymnasium as gym, mujoco, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

N_STUCK_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
FORK_STEP = 150
EVAL_SEED = 20000
TARGET_LIFT_M = 0.020
LIFT_TOL_M = 0.002
LIFT_PHASE_CAP = 60
CARTESIAN_STEP = 0.020  # DEFAULT_CARTESIAN_STEP, robochess/envs/robochess_env.py

BASE_ATTRS = ("_arm_target", "_ik_rejects", "_elapsed_steps", "_piece_ang_settled_streak",
             "_piece_lin_settled_streak", "_piece_upright_streak", "_ever_held",
             "_release_record", "_contact_flags")
PLACE_ATTRS = ("_t", "_held_once", "_lift_streak", "_lift_bonus_paid", "_s2_entered",
              "_last_state", "_last_terms")


def snapshot(base, place):
    return dict(
        data=copy.deepcopy(base.data),
        base={a: copy.deepcopy(getattr(base, a)) for a in BASE_ATTRS},
        place={a: copy.deepcopy(getattr(place, a)) for a in PLACE_ATTRS},
    )


def restore(base, place, snap):
    base.data = copy.deepcopy(snap["data"])
    mujoco.mj_forward(base.model, base.data)
    for a, v in snap["base"].items():
        setattr(base, a, copy.deepcopy(v))
    for a, v in snap["place"].items():
        setattr(place, a, copy.deepcopy(v))


run_dir = REPO / "runs" / "_probe_cart_place_ppo_seed1"
cfg = json.loads((run_dir / "manifest.json").read_text())["config_resolved"]
train.apply_thread_pinning(1)

holder = {}


def mk():
    w = train.build_env(cfg)          # PlaceReward(...) instance
    holder["place"] = w
    holder["base"] = w.unwrapped
    return Monitor(w)


venv = DummyVecEnv([mk])
vn = run_dir / "final_vecnormalize.pkl"
venv = VecNormalize.load(str(vn), venv)
venv.training = False
venv.norm_reward = False
model = PPO.load(str(run_dir / "final_model.zip"), env=venv, device="cpu")
venv.seed(EVAL_SEED)
obs = venv.reset()
base = holder["base"]
place = holder["place"]

stuck_snapshots = []
last_gripper_cmd = []
episode_idx = 0
step_in_episode = 0
while len(stuck_snapshots) < N_STUCK_EPISODES:
    act, _ = model.predict(obs, deterministic=False)
    step_in_episode += 1
    if step_in_episode == FORK_STEP:
        stuck_snapshots.append(snapshot(base, place))
        last_gripper_cmd.append(float(np.asarray(act).reshape(-1)[3]))
    obs, r, done, infos = venv.step(act)
    if done[0]:
        episode_idx += 1
        step_in_episode = 0
print(f"collected {len(stuck_snapshots)} stuck snapshots at step {FORK_STEP}", file=sys.stderr, flush=True)


def unit_xy_toward_goal(base):
    piece_xy = base._achieved_goal()[:2].astype(float)
    goal_xy = base._desired_goal[:2].astype(float)
    d = goal_xy - piece_xy
    dist = float(np.linalg.norm(d))
    return (d / dist if dist > 1e-9 else np.zeros(2)), dist


def scripted_step(base, phase, lift_target_reached_m, gripper_cmd):
    """One (dx, dy, dz, gripper) action, normalised to [-1, 1]."""
    from robochess.tasks.t1_pick_and_place import pickup_geometry, _pinch_site_id
    sid = _pinch_site_id(base.model)
    dz_pos = float(place.dz)
    geom = pickup_geometry(base, sid, dz_pos)
    lift_m = float(geom["lift"])
    a = np.zeros(4, dtype=np.float64)
    if phase == "lift":
        err = TARGET_LIFT_M - lift_m
        a[2] = float(np.clip(err / CARTESIAN_STEP, -1.0, 1.0))
        arrived = abs(err) < LIFT_TOL_M
    elif phase == "hold":
        err = lift_target_reached_m - lift_m
        a[2] = float(np.clip(err / CARTESIAN_STEP, -1.0, 1.0))
        arrived = None
    elif phase == "hold_height_move_xy":
        err = lift_target_reached_m - lift_m
        a[2] = float(np.clip(err / CARTESIAN_STEP, -1.0, 1.0))
        direction, dist = unit_xy_toward_goal(base)
        step_m = gate.braked_step_m(dist, gate.BRAKE_FRACTION)
        a[:2] = direction * min(step_m / CARTESIAN_STEP, 1.0)
        arrived = dist < 0.03
    elif phase == "no_lift_move_xy":
        a[2] = 0.0
        direction, dist = unit_xy_toward_goal(base)
        step_m = gate.braked_step_m(dist, gate.BRAKE_FRACTION)
        a[:2] = direction * min(step_m / CARTESIAN_STEP, 1.0)
        arrived = dist < 0.03
    else:
        raise ValueError(phase)
    a[3] = gripper_cmd
    return a, lift_m, arrived


def run_variant(snap, gripper_cmd_default, variant, n_steps=300 - FORK_STEP):
    restore(base, place, snap)
    rows = []
    phase = "lift" if variant in ("a", "b", "b_closed") else "no_lift_move_xy"
    lift_reached = None
    for i in range(n_steps):
        gcmd = 1.0 if variant == "b_closed" else gripper_cmd_default
        a, lift_m, arrived = scripted_step(base, phase, lift_reached, gcmd)
        obs2, r, term, trunc, info = place.step(a)
        t = info.get("place_terms", {})
        rows.append(dict(step=i, st=info["place_state"], held=bool(t.get("held", False)),
                         pinched=bool(t.get("pinched", False)), dxy_mm=float(info["xy_distance_to_goal"]) * 1e3,
                         lift_mm=lift_m * 1e3, reward=float(r)))
        if phase == "lift" and arrived:
            lift_reached = lift_m
            phase = "hold" if variant == "a" else "hold_height_move_xy"
        if term or trunc:
            break
    return rows


VARIANTS = ["a", "b", "c", "b_closed"]
results = {}
for vi, v in enumerate(VARIANTS):
    results[v] = []
    for ei in range(N_STUCK_EPISODES):
        rows = run_variant(stuck_snapshots[ei], last_gripper_cmd[ei], v)
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
        results[v].append(dict(
            episode=ei, n_steps=len(rows),
            final_pinched=pinched_flags[-1], final_held=held_flags[-1],
            ever_within_3cm=any(r["dxy_mm"] < 30.0 for r in rows),
            loss_step=loss_step,
            reward_path=[dict(step=r["step"], reward=round(r["reward"], 4)) for r in rows
                        if r["step"] == 0 or r["step"] % 10 == 0 or r["step"] == len(rows) - 1],
            final_dxy_mm=rows[-1]["dxy_mm"], final_lift_mm=rows[-1]["lift_mm"],
        ))

print(json.dumps(results, indent=2))
