#!/usr/bin/env python
"""Section 4c (read-only diagnosis): mechanism of the SAC ~20mm plateau.
Reuses the exact scripted counterfactual from Section 4b (move horizontal ->
descend -> open) but instruments every step of the descend phase, plus the
step-150 grip geometry, plus seed 2's release-step geometry for comparison.
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
import gate_expert_cartesian as gate

import gymnasium as gym, mujoco, numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

N_STUCK_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
EVAL_SEED = 20000
FORK_STEP = 150
CARTESIAN_STEP = 0.020
DESCEND_TOL_M = 0.002
HORIZ_TOL_M = 0.020
CARTESIAN_Z_FLOOR_ABOVE_BOARD = 0.0315


def pad_contact_height_mm(base):
    m, d = base.model, base.data
    piece = base._piece_bid
    piece_z = float(base._achieved_goal()[2])
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
        if other == base._left_pad_bid:
            left_z = float(c.pos[2])
        elif other == base._right_pad_bid:
            right_z = float(c.pos[2])
    pads = [z for z in (left_z, right_z) if z is not None]
    return ((sum(pads) / len(pads) - piece_z) * 1e3) if pads else None


# ------------------------------------------------------------- SAC part a & b
run_dir = REPO / "runs" / "_probe_cart_place_sac_seed0"
cfg = json.loads((run_dir / "manifest.json").read_text())["config_resolved"]
train.apply_thread_pinning(1)

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


holder = {}


def mk():
    w = train.build_env(cfg)
    holder["place"] = w
    holder["base"] = w.unwrapped
    return Monitor(w)


venv = DummyVecEnv([mk])
model = SAC.load(str(run_dir / "final_model.zip"), env=venv, device="cpu")
venv.seed(EVAL_SEED)
obs = venv.reset()
base = holder["base"]
place = holder["place"]

stuck_snapshots = []
step150_geometry = []
step_in_episode = 0
while len(stuck_snapshots) < N_STUCK_EPISODES:
    act, _ = model.predict(obs, deterministic=False)
    step_in_episode += 1
    if step_in_episode == FORK_STEP:
        stuck_snapshots.append(snapshot(base, place))
        step150_geometry.append(dict(
            pad_height_mm=pad_contact_height_mm(base),
            qpos=float(base.data.qpos[base._grip_qadr]),
            tilt_deg=float(base._piece_tilt_deg()),
        ))
    obs, r, done, infos = venv.step(act)
    if done[0]:
        step_in_episode = 0
print(f"collected {len(stuck_snapshots)} SAC stuck snapshots", file=sys.stderr, flush=True)


def unit_xy_toward_goal(base):
    piece_xy = base._achieved_goal()[:2].astype(float)
    goal_xy = base._desired_goal[:2].astype(float)
    d = goal_xy - piece_xy
    dist = float(np.linalg.norm(d))
    return (d / dist if dist > 1e-9 else np.zeros(2)), dist


def commanded_and_actual_pinch_z(base, dz_action_component):
    """Replicates _scale_action's cartesian branch z-handling, read-only."""
    pinch_now = base.data.site_xpos[base._pinch_sid].copy()
    commanded_z = float(pinch_now[2] + dz_action_component * base.cartesian_step)
    clipped_z = float(np.clip(commanded_z, base._cartesian_low[2], base._cartesian_high[2]))
    return commanded_z, clipped_z


def scripted_step_instrumented(base, place, phase, gripper_cmd, trace):
    from robochess.tasks.t1_pick_and_place import pickup_geometry
    geom = pickup_geometry(base, base._pinch_sid, float(place.dz))
    lift_m = float(geom["lift"])
    direction, dist = unit_xy_toward_goal(base)
    a = np.zeros(4, dtype=np.float64)
    if phase == "move_horizontal":
        step_m = gate.braked_step_m(dist, gate.BRAKE_FRACTION)
        a[:2] = direction * min(step_m / CARTESIAN_STEP, 1.0)
        a[2] = 0.0
        arrived = dist < HORIZ_TOL_M
    elif phase == "descend":
        err = 0.0 - lift_m
        step_m = gate.braked_step_m(abs(err), gate.BRAKE_FRACTION_DESCEND)
        dz_component = float(np.clip(np.sign(err) * step_m / CARTESIAN_STEP, -1.0, 1.0))
        a[2] = dz_component
        a[:2] = direction * min(gate.braked_step_m(dist, gate.BRAKE_FRACTION_DESCEND) / CARTESIAN_STEP, 1.0)
        arrived = lift_m < DESCEND_TOL_M
        cmd_z, act_z_after_clip = commanded_and_actual_pinch_z(base, dz_component)
        pinch_now_z = float(base.data.site_xpos[base._pinch_sid][2])
        board_z = base._board_surface_z
        trace.append(dict(
            commanded_pinch_z_above_board_mm=(cmd_z - board_z) * 1e3,
            clip_target_z_above_board_mm=(act_z_after_clip - board_z) * 1e3,
            clamp_active=bool(cmd_z < base._cartesian_low[2] - 1e-9),
            actual_pinch_z_above_board_mm=(pinch_now_z - board_z) * 1e3,
            piece_base_height_above_board_mm=float(base._piece_height_above_board()) * 1e3,
            pad_height_mm=pad_contact_height_mm(base),
            tilt_deg=float(base._piece_tilt_deg()),
        ))
    elif phase == "open":
        arrived = None
    else:
        raise ValueError(phase)
    a[3] = -1.0 if phase == "open" else gripper_cmd
    return a, lift_m, arrived


def run_variant(snap, n_steps=300 - FORK_STEP):
    restore(base, place, snap)
    trace = []
    phase = "move_horizontal"
    for i in range(n_steps):
        a, lift_m, arrived = scripted_step_instrumented(base, place, phase, 1.0, trace)
        obs3, r3, term, trunc, info = place.step(a)
        if phase == "move_horizontal" and arrived:
            phase = "descend"
        elif phase == "descend" and arrived:
            phase = "open"
        if term or trunc:
            break
    return trace


descend_traces = []
for ei in range(N_STUCK_EPISODES):
    descend_traces.append(run_variant(stuck_snapshots[ei]))

# ------------------------------------------------------------- seed2 part c
run_dir2 = REPO / "runs" / "_probe_cart_place_ppo_seed2"
cfg2 = json.loads((run_dir2 / "manifest.json").read_text())["config_resolved"]


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
        self.cur.append(dict(held=bool(t.get("held", False)), succ=bool(info.get("is_success", False)),
                             pad_height_mm=pad_contact_height_mm(self.b),
                             pinch_z_above_board_mm=(float(self.b.data.site_xpos[self.b._pinch_sid][2]) - self.b._board_surface_z) * 1e3))
        if term or trunc:
            self.episodes.append(self.cur)
            self.cur = []
        return obs, r, term, trunc, info


holder2 = {}


def mk2():
    w2 = Rec(train.build_env(cfg2))
    holder2["w"] = w2
    return Monitor(w2)


venv2 = DummyVecEnv([mk2])
venv2 = VecNormalize.load(str(run_dir2 / "final_vecnormalize.pkl"), venv2)
venv2.training = False
venv2.norm_reward = False
model2 = PPO.load(str(run_dir2 / "final_model.zip"), env=venv2, device="cpu")
venv2.seed(EVAL_SEED)
obs2 = venv2.reset()
w2 = holder2["w"]
target_successes = 20
max_episodes = 60
while len(w2.episodes) < max_episodes and sum(e[-1]["succ"] for e in w2.episodes) < target_successes:
    act2, _ = model2.predict(obs2, deterministic=False)
    obs2, r2, done2, infos2 = venv2.step(act2)
venv2.close()

seed2_release = []
for e in w2.episodes:
    if not e[-1]["succ"]:
        continue
    held_idx = [k for k, r in enumerate(e) if r["held"]]
    if held_idx:
        r = e[held_idx[-1]]
        seed2_release.append(dict(pinch_z_above_board_mm=r["pinch_z_above_board_mm"], pad_height_mm=r["pad_height_mm"]))
    if len(seed2_release) >= target_successes:
        break

print(json.dumps(dict(
    step150_geometry=step150_geometry,
    descend_traces=descend_traces,
    seed2_release_step=seed2_release,
), indent=2))
