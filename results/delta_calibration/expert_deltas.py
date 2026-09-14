"""Section 2: per-step change in the scripted expert's COMMANDED arm joint targets
(data.ctrl[:6]) and, as a second reference, in the ACTUAL joint positions
(data.qpos[:6]), over successful expert episodes spanning near and far squares.
One env step = frame_skip 20 x 2 ms = 40 ms. Read-only: env and expert unchanged."""
import json, sys, time
from pathlib import Path
import gymnasium as gym, numpy as np
REPO = Path.home()/"projects/robochess_gymnasium"; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts"))
import scripted_expert as se
from robochess.envs import RoboChessEnv
from robochess.tasks import T1PickAndPlaceTask

class Recorder(gym.Wrapper):
    def reset(self, **kw):
        out = self.env.reset(**kw); d = self.env.unwrapped.data
        self.ctrl = [d.ctrl[:6].copy()]; self.q = [d.qpos[:6].copy()]; return out
    def step(self, a):
        out = self.env.step(a); d = self.env.unwrapped.data
        self.ctrl.append(d.ctrl[:6].copy()); self.q.append(d.qpos[:6].copy()); return out

PAIRS = [(s, t) for s in ("a1", "d1", "h1", "b2", "g2") for t in ("a8", "e8")] + \
        [(s, t) for s in ("a8", "d8", "h8", "b7", "g7") for t in ("h1", "d1")] + \
        [("d4", "e5"), ("e5", "d4"), ("c3", "f6"), ("f6", "c3")]
env = RoboChessEnv(max_episode_steps=350); task = T1PickAndPlaceTask(env); rec = Recorder(task)
expert = se.ScriptedExpert(rec)
J = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
dc, dq, eps = [], [], []
t0 = time.time()
for s, t in PAIRS:
    task.start_square, task.target_square = s, t
    r = expert.run_episode(seed=0)
    eps.append(dict(start=s, target=t, success=r["success"], outcome=r["outcome"], steps=r["steps"]))
    if r["success"]:
        C, Q = np.array(rec.ctrl), np.array(rec.q)
        dc.append(np.abs(np.diff(C, axis=0))); dq.append(np.abs(np.diff(Q, axis=0)))
dc, dq = np.concatenate(dc), np.concatenate(dq)
def table(D):
    return {j: dict(median=float(np.median(D[:, k])), p95=float(np.percentile(D[:, k], 95)),
                    p99=float(np.percentile(D[:, k], 99)), max=float(D[:, k].max()),
                    frac_zero=float(np.mean(D[:, k] < 1e-9))) for k, j in enumerate(J)}
out = dict(episodes=eps, n_success=sum(e["success"] for e in eps), n_steps=int(len(dc)),
           env_step_seconds=0.04, commanded_target_delta_rad=table(dc), actual_joint_delta_rad=table(dq),
           commanded_nonzero_only=table(np.where(dc > 1e-9, dc, np.nan)) if False else None)
# non-zero commanded changes only (waypoint switches), per joint
nz = {}
for k, j in enumerate(J):
    v = dc[:, k][dc[:, k] > 1e-9]
    nz[j] = dict(n=int(v.size), median=float(np.median(v)) if v.size else None,
                 p95=float(np.percentile(v, 95)) if v.size else None, max=float(v.max()) if v.size else None)
out["commanded_nonzero_only"] = nz
Path(REPO/"results/delta_calibration/expert_deltas.json").write_text(json.dumps(out, indent=1))
print(f"{out['n_success']}/{len(eps)} expert episodes succeeded, {out['n_steps']} env steps, {time.time()-t0:.0f}s")
for name, T in (("COMMANDED target delta (rad/step)", out["commanded_target_delta_rad"]), ("ACTUAL joint delta (rad/step)", out["actual_joint_delta_rad"])):
    print(f"\n{name}\n  {'joint':14s} {'median':>9s} {'p95':>9s} {'p99':>9s} {'max':>9s} {'% steps zero':>12s}")
    for j in J: print(f"  {j:14s} {T[j]['median']:9.5f} {T[j]['p95']:9.5f} {T[j]['p99']:9.5f} {T[j]['max']:9.5f} {100*T[j]['frac_zero']:11.1f}%")
print("\nCOMMANDED, non-zero changes only (waypoint switches)")
for j in J: v = nz[j]; print(f"  {j:14s} n={v['n']:4d} median {v['median']:.4f} p95 {v['p95']:.4f} max {v['max']:.4f}")
