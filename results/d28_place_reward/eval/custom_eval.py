#!/usr/bin/env python
"""Untracked, ad-hoc extension of scripts/eval_probe.py for the D28 place-reward
follow-up questions (episode-end state split, truncated-in-S1 geometry, per-step
place_terms reward at the final step, held-within-3cm fraction).

Does not modify scripts/eval_probe.py or scripts/place_eval.py: imports
place_eval's real, unmodified place_episode_summary/place_report for every
number that script would have produced, and adds a second Rec-style wrapper
that also captures info["place_terms"], info["release_height_mm"/"release_xy_error_mm"]
and the real gym `truncated` flag per step (not inferred from a blank
terminated_reason), which eval_probe.py's own JSON output does not retain.

Runs one run/checkpoint combo per process: 300 stochastic episodes (seed 20000)
then 20 deterministic episodes (seed 20000), each frozen (training=False,
norm_reward=False) with its own matching VecNormalize stats for PPO. Writes one
JSON with both raw per-episode rows and the aggregated place_report.

    python custom_eval.py --run _probe_cart_place_ppo_seed0 --checkpoint 500000 --out x.json
"""
import argparse, importlib.util, json, sys, copy
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
import place_eval

import gymnasium as gym, numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--checkpoint", type=int, default=None)
p.add_argument("--seed", type=int, default=20000)
p.add_argument("--stoch-episodes", type=int, default=300)
p.add_argument("--det-episodes", type=int, default=20)
p.add_argument("--runs-dir", type=Path, default=REPO / "runs")
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

run_dir = a.runs_dir / a.run
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
        row = dict(
            st=info["place_state"],
            dxy=float(info["xy_distance_to_goal"]),
            lift=float(t.get("lift", 0.0)),
            held=bool(t.get("held", False)),
            pinched=bool(t.get("pinched", False)),
            ap=float(self.b.data.qpos[self.b._grip_qadr]),
            grasp=bool(info.get("grasp_success", False)),
            succ=bool(info.get("is_success", False)),
            state_reward=float(t.get("state_reward", 0.0)),
            reward=float(r) if np.isscalar(r) else float(np.asarray(r).reshape(-1)[0]),
            release_height_mm=info.get("release_height_mm"),
            release_xy_error_mm=info.get("release_xy_error_mm"),
            pxy=[float(v) for v in self.b._achieved_goal()[:2]],
            gxy=[float(v) for v in self.b._desired_goal[:2]],
            trunc=bool(trunc),
        )
        self.cur.append(row)
        if term or trunc:
            self.episodes.append(dict(
                rows=self.cur,
                reason=info.get("terminated_reason") or "time_limit",
                truncated=bool(trunc),
                terminated=bool(term),
                ik=int(info.get("ik_rejects", 0)),
                n=int(info["elapsed_steps"]),
            ))
            self.cur = []
        return obs, r, term, trunc, info


holder = {}


def mk():
    w = Rec(train.build_env(cfg))
    holder["w"] = w
    return Monitor(w)


def run_rollout(n_episodes: int, deterministic: bool, seed: int):
    holder.clear()
    venv = DummyVecEnv([mk])
    ck = a.checkpoint
    mp = run_dir / ("final_model.zip" if ck is None else f"checkpoints/rc_{ck}_steps.zip")
    rms_before = rms_after = None
    if cfg["algorithm"] == "ppo":
        vn = run_dir / ("final_vecnormalize.pkl" if ck is None else f"checkpoints/rc_vecnormalize_{ck}_steps.pkl")
        venv = VecNormalize.load(str(vn), venv)
        venv.training = False
        venv.norm_reward = False
        snap = lambda: {k: (copy.deepcopy(np.asarray(v.mean)), copy.deepcopy(np.asarray(v.var)), float(v.count))
                        for k, v in (venv.obs_rms.items() if isinstance(venv.obs_rms, dict) else [("obs", venv.obs_rms)])}
        rms_before = snap()
    model = (PPO if cfg["algorithm"] == "ppo" else SAC).load(str(mp), env=venv, device="cpu")
    venv.seed(seed)
    obs = venv.reset()
    w = holder["w"]
    while len(w.episodes) < n_episodes:
        act, _ = model.predict(obs, deterministic=deterministic)
        obs, r, done, infos = venv.step(act)
    if rms_before is not None:
        rms_after = snap()
        same = all(np.array_equal(rms_before[k][0], rms_after[k][0]) and np.array_equal(rms_before[k][1], rms_after[k][1])
                   and rms_before[k][2] == rms_after[k][2] for k in rms_before)
    else:
        same = None
    venv.close()
    return w.episodes[:n_episodes], same


def summarize(eps, deterministic: bool):
    place_summaries = [place_eval.place_episode_summary(e["rows"], e["reason"]) for e in eps]
    same_traj = None
    if deterministic and len(eps) > 1:
        same_traj = all([x["ap"] for x in eps[i]["rows"]] == [x["ap"] for x in eps[0]["rows"]] for i in range(1, len(eps)))
    place_report = place_eval.place_report(place_summaries, deterministic=deterministic,
                                           identical=same_traj if deterministic else None)
    grasp_ever = [any(x["grasp"] for x in e["rows"]) for e in eps]

    # --- truncated_in_s2 using the REAL env truncated flag, not the reason fallback ---
    trunc_s2 = [e for e in eps if e["truncated"] and e["rows"][-1]["st"] == "S2"]

    # --- episode-end place_state split, overall and truncated-only ---
    end_state_all = {}
    end_state_trunc = {}
    for e in eps:
        s = e["rows"][-1]["st"]
        end_state_all[s] = end_state_all.get(s, 0) + 1
        if e["truncated"]:
            end_state_trunc[s] = end_state_trunc.get(s, 0) + 1

    # --- truncated-in-S1 episode geometry ---
    trunc_s1 = [e for e in eps if e["truncated"] and e["rows"][-1]["st"] == "S1"]
    t_s1 = dict(n=len(trunc_s1))
    if trunc_s1:
        final_dxy_mm = [r["rows"][-1]["dxy"] * 1e3 for r in trunc_s1]
        final_lift_mm = [r["rows"][-1]["lift"] * 1e3 for r in trunc_s1]
        min_dxy_mm = [min(x["dxy"] for x in r["rows"]) * 1e3 for r in trunc_s1]
        max_lift_mm = [max(x["lift"] for x in r["rows"]) * 1e3 for r in trunc_s1]
        final_state_reward = [r["rows"][-1]["state_reward"] for r in trunc_s1]
        t_s1.update(
            final_dxy_mm=final_dxy_mm,
            final_lift_mm=final_lift_mm,
            final_pinched=[bool(r["rows"][-1]["pinched"]) for r in trunc_s1],
            final_held=[bool(r["rows"][-1]["held"]) for r in trunc_s1],
            final_gripper_qpos=[float(r["rows"][-1]["ap"]) for r in trunc_s1],
            min_dxy_mm=min_dxy_mm,
            max_lift_mm=max_lift_mm,
            final_state_reward=final_state_reward,
            frac_within_3cm_final=sum(1 for x in final_dxy_mm if x < 30.0) / len(trunc_s1),
        )

    # --- ever within 3cm of target while held ---
    ever_3cm_while_held = sum(1 for e in eps if any(x["held"] and x["dxy"] < 0.03 for x in e["rows"])) / len(eps) if eps else 0.0

    # --- release height / xy error / success step, among successes ---
    successes = [e for e in eps if e["rows"][-1]["succ"]]
    rel_h = [e["rows"][-1]["release_height_mm"] for e in successes if e["rows"][-1]["release_height_mm"] is not None]
    rel_xy = [e["rows"][-1]["release_xy_error_mm"] for e in successes if e["rows"][-1]["release_xy_error_mm"] is not None]
    succ_step = [e["n"] for e in successes]

    reason_counts = {}
    for e in eps:
        reason_counts[e["reason"]] = reason_counts.get(e["reason"], 0) + 1

    return dict(
        n=len(eps),
        place_report=place_report,
        grasp_ever_k=sum(grasp_ever), grasp_ever_n=len(grasp_ever),
        trunc_s2_true_flag=dict(k=len(trunc_s2), n=len(eps),
                                final_gripper_qpos=sorted(round(e["rows"][-1]["ap"], 4) for e in trunc_s2)),
        end_state_all=end_state_all, end_state_truncated=end_state_trunc,
        truncated_in_s1=t_s1,
        ever_within_3cm_while_held=dict(k=sum(1 for e in eps if any(x["held"] and x["dxy"] < 0.03 for x in e["rows"])), n=len(eps),
                                        rate=ever_3cm_while_held),
        release_height_mm=rel_h, release_xy_error_mm=rel_xy, success_step=succ_step,
        reason_counts=reason_counts,
        n_truncated=sum(1 for e in eps if e["truncated"]),
        n_terminated=sum(1 for e in eps if e["terminated"]),
    )


stoch_eps, stoch_rms_same = run_rollout(a.stoch_episodes, deterministic=False, seed=a.seed)
det_eps, det_rms_same = run_rollout(a.det_episodes, deterministic=True, seed=a.seed)

out = dict(
    run=a.run, checkpoint=a.checkpoint, seed=a.seed,
    stochastic=dict(summary=summarize(stoch_eps, deterministic=False), obs_rms_identical_before_after=stoch_rms_same),
    deterministic=dict(summary=summarize(det_eps, deterministic=True), obs_rms_identical_before_after=det_rms_same),
)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(out))
print(a.run, a.checkpoint, "done", "stoch_n", len(stoch_eps), "det_n", len(det_eps), flush=True)
