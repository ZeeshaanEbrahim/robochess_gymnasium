#!/usr/bin/env python
"""Evaluate a pick-up probe (a run under runs/) and record per-episode geometry.

Measures, per episode, over a stochastic (default) or deterministic rollout of
final_model.zip or a checkpoint, seeded with --seed and built from the run's own
manifest config:
  * D18 `grasp_success`: both pads on the piece AND no piece-board contact
    (the env's `grasp_success` info key, ever true in the episode);
  * D21 `pickup_success`: `grasp_success` with the piece lifted >= 30 mm and held
    for 5 consecutive steps, latched for the episode (PickupReward's info key);
  * closest pinch-to-g distance, with g = piece position + 0.03725 m in z, and the
    tool and piece tilt at that step; any two-pad contact; lifted / held steps.
PPO loads its VecNormalize statistics frozen (training=False, norm_reward=False);
the output records whether obs_rms was identical before and after the rollout.
Derived from scripts/eval_final.py (deterministic only, no g, no pickup metrics).

    python scripts/eval_probe.py _probe_cart_pickup_ppo_seed0 --episodes 300 --out x.json
    python scripts/eval_probe.py <run> --checkpoint 500000 --episodes 100 --out x.json
    python scripts/eval_probe.py <run> --episodes 20 --deterministic --out x.json

Aggregate the outputs with scripts/eval_probe_agg.py. Reads runs/, writes only --out.
"""
import argparse, collections, importlib.util, json, sys, copy
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)

p = argparse.ArgumentParser()
p.add_argument("run"); p.add_argument("--checkpoint", type=int, default=None)
p.add_argument("--episodes", type=int, default=100); p.add_argument("--seed", type=int, default=20000)
p.add_argument("--deterministic", action="store_true"); p.add_argument("--steps", action="store_true")
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
run_dir = REPO / "runs" / a.run
cfg = json.loads((run_dir / "manifest.json").read_text())["config_resolved"]
train.apply_thread_pinning(1)

import gymnasium as gym, numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robochess.tasks.t1_pick_and_place import _pinch_site_id, find_contacts_owner

class Rec(gym.Wrapper):
    """Pass-through; records per-step geometry, harvests a finished episode on done."""
    def __init__(self, env):
        super().__init__(env); self.b = env.unwrapped; self.sid = _pinch_site_id(self.b.model)
        self.owner = find_contacts_owner(self.env, "need _piece_contacts"); self.episodes = []; self.cur = []
        self.dz = None
    def _geo(self):
        b = self.b; pinch = b.data.site_xpos[self.sid].copy(); piece = b._achieved_goal().astype(float)
        g = piece + np.array([0, 0, 0.03725]); rel = pinch - g
        zax = b.data.site_xmat[self.sid].reshape(3, 3)[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(-zax[2], -1, 1))))
        l, r, bd = self.owner._piece_contacts()
        return rel, tilt, bool(l), bool(r), bool(bd), float(b.data.qpos[b._grip_qadr]), float(b._piece_tilt_deg()), float(piece[2] - b._board_surface_z)
    def reset(self, **kw):
        out = self.env.reset(**kw); self.cur = []; return out
    def step(self, act):
        obs, r, term, trunc, info = self.env.step(act)
        rel, tilt, l, rr, bd, ap, ptilt, ph = self._geo()
        t = info.get("pickup_terms", {})
        self.cur.append(dict(d=float(np.linalg.norm(rel)), rel=[round(float(x) * 1e3, 2) for x in rel], tilt=tilt, l=l, r=rr, board=bd,
                             ap=ap, ptilt=ptilt, ph=ph, held=bool(t.get("held", False)), pinched=bool(t.get("pinched", False)),
                             grasp=bool(info.get("grasp_success", False)), lift=float(t.get("lift", 0.0)),
                             ps=bool(info.get("pickup_success", False)), gsc=bool(info.get("gripper_self_contact", False))))
        if term or trunc:
            self.episodes.append(dict(rows=self.cur, reason=info.get("terminated_reason") or "time_limit",
                                      ik=int(info.get("ik_rejects", 0)), n=int(info["elapsed_steps"]))); self.cur = []
        return obs, r, term, trunc, info

holder = {}
def mk():
    w = Rec(train.build_env(cfg)); holder["w"] = w; return Monitor(w)
venv = DummyVecEnv([mk])
ck = a.checkpoint
mp = run_dir / ("final_model.zip" if ck is None else f"checkpoints/rc_{ck}_steps.zip")
rms_before = rms_after = None
if cfg["algorithm"] == "ppo":
    vn = run_dir / ("final_vecnormalize.pkl" if ck is None else f"checkpoints/rc_vecnormalize_{ck}_steps.pkl")
    venv = VecNormalize.load(str(vn), venv); venv.training = False; venv.norm_reward = False
    snap = lambda: {k: (copy.deepcopy(np.asarray(v.mean)), copy.deepcopy(np.asarray(v.var)), float(v.count)) for k, v in
                    (venv.obs_rms.items() if isinstance(venv.obs_rms, dict) else [("obs", venv.obs_rms)])}
    rms_before = snap()
model = (PPO if cfg["algorithm"] == "ppo" else SAC).load(str(mp), env=venv, device="cpu")
venv.seed(a.seed); obs = venv.reset(); w = holder["w"]
while len(w.episodes) < a.episodes:
    act, _ = model.predict(obs, deterministic=a.deterministic)
    obs, r, done, infos = venv.step(act)
if rms_before is not None:
    rms_after = snap()
    same = all(np.array_equal(rms_before[k][0], rms_after[k][0]) and np.array_equal(rms_before[k][1], rms_after[k][1])
               and rms_before[k][2] == rms_after[k][2] for k in rms_before)
else:
    same = None
venv.close()

eps = w.episodes[:a.episodes]
summ = []
for e in eps:
    R = e["rows"]; d = np.array([x["d"] for x in R]); i = int(d.argmin())
    lifted = [x["lift"] >= 0.03 for x in R]; ps_first = next((k for k, x in enumerate(R) if x["ps"]), None)
    s = dict(n=e["n"], reason=e["reason"], ik=e["ik"], grasp_ever=any(x["grasp"] for x in R), pickup=R[-1]["ps"],
             pinched_ever=any(x["pinched"] for x in R), twopad_ever=any(x["l"] and x["r"] for x in R),
             min_d_mm=float(d.min() * 1e3), tilt_at_min_deg=R[i]["tilt"], ptilt_at_min_deg=R[i]["ptilt"], min_d_step=i + 1,
             gsc_steps=sum(x["gsc"] for x in R), lifted_steps=sum(lifted), held_steps=sum(x["held"] for x in R),
             grasp_steps=sum(x["grasp"] for x in R), pickup_first_step=None if ps_first is None else ps_first + 1,
             lifted_after_first=None if ps_first is None else sum(lifted[ps_first:]),
             steps_after_first=None if ps_first is None else len(R) - ps_first,
             grasp_after_first=None if ps_first is None else sum(x["grasp"] for x in R[ps_first:]),
             lifted_at_end=lifted[-1], grasp_at_end=R[-1]["grasp"], max_lift_mm=max(x["lift"] for x in R) * 1e3)
    summ.append(s)
out = dict(run=a.run, checkpoint=ck, model=str(mp.relative_to(run_dir)), seed=a.seed, deterministic=a.deterministic,
           n=len(eps), obs_rms_identical_before_after=same, episodes=summ)
if a.steps or a.deterministic:
    out["traj"] = [[{k: x[k] for k in ("rel", "ap", "l", "r", "board", "d", "tilt", "lift")} for x in e["rows"]] for e in eps]
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out))
print(a.run, ck, "n", len(eps), "pickup", sum(s["pickup"] for s in summ), "grasp", sum(s["grasp_ever"] for s in summ), "rms_same", same, flush=True)
