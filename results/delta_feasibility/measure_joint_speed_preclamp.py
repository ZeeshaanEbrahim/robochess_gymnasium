import sys, json; sys.path.insert(0, ".")
import mujoco, numpy as np
from robochess.envs import RoboChessEnv
from robochess.envs.robochess_env import N_ARM_ACTUATORS as N
e = RoboChessEnv(action_mode="delta", max_episode_steps=10**6)
QADR = e.model.jnt_qposadr[e.model.actuator_trnid[:N, 0]]
m, d = e.model, e.data
gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(N)]
dt = m.opt.timestep * e.frame_skip
budget = RoboChessEnv().max_episode_steps
out = {}
for j in range(N):
    per_sign = {}
    for sign in (+1, -1):
        e.reset(seed=0)
        a = np.zeros(7, dtype=np.float32); a[j] = sign; a[6] = -1.0
        vs, stall, touching = [], None, None
        for t in range(150):
            e.step(a)
            v = abs(float(d.qvel[QADR[j]])); vs.append(v)
            if t > 5 and stall is None and v < 0.5 * max(vs):
                stall = t
                touching = sorted({f"{gname(int(c.geom1))}|{gname(int(c.geom2))}" for c in d.contact[:d.ncon]})
                break
        free = np.array(vs[3:stall] if stall else vs[3:])
        per_sign[sign] = dict(plateau=float(np.median(free)), peak=float(max(vs)), stall_step=stall, contacts=touching)
    slow = min(per_sign[+1]["plateau"], per_sign[-1]["plateau"])
    out[names[j]] = dict(per_sign=per_sign, limiting_rad_s=slow,
                         travel_per_episode_rad=slow * dt * budget,
                         travel_per_episode_deg=np.degrees(slow * dt * budget))
print(f"macro step {dt*1000:.0f} ms, episode budget {budget} steps = {dt*budget:.1f} s of simulated time\n")
print("| joint | plateau +1 (rad/s) | plateau -1 | limiting | max travel in 200 steps (rad) | (deg) | stall |")
print("|---|---|---|---|---|---|---|")
for k, v in out.items():
    p, q = v["per_sign"][+1], v["per_sign"][-1]
    st = "; ".join(f"{s:+d}@{v['per_sign'][s]['stall_step']}" for s in (+1, -1) if v["per_sign"][s]["stall_step"]) or "none"
    print(f"| {k} | {p['plateau']:.3f} | {q['plateau']:.3f} | {v['limiting_rad_s']:.3f} | {v['travel_per_episode_rad']:.2f} | {v['travel_per_episode_deg']:.0f} | {st} |")
for k, v in out.items():
    for s in (+1, -1):
        if v["per_sign"][s]["contacts"]: print(f"\n{k} {s:+d} stalled touching: {v['per_sign'][s]['contacts']}")
json.dump(out, open(f"{sys.argv[1]}", "w"), indent=1)
