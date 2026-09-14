import importlib.util, json, sys
from pathlib import Path
REPO = Path.home()/"projects/robochess_gymnasium"; sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("train", REPO/"scripts/train.py"); train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
run, ck = Path(sys.argv[1]), int(sys.argv[2])
cfg = json.loads((run/"manifest.json").read_text())["config_resolved"]; train.apply_thread_pinning(1)
import mujoco, numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
env = train.build_env(cfg); b = env.unwrapped; task = env.env; m, d = b.model, b.data
model = (PPO if cfg["algorithm"] == "ppo" else SAC).load(str(run/f"checkpoints/rc_{ck}_steps.zip"), device="cpu")
vn = None
if cfg["algorithm"] == "ppo":
    vn = VecNormalize.load(str(run/f"checkpoints/rc_vecnormalize_{ck}_steps.pkl"), DummyVecEnv([lambda: Monitor(train.build_env(cfg))])); vn.training = False
def act(o):
    o = {k: v[None] for k, v in o.items()}
    if vn is not None: o = vn.normalize_obs(o)
    return model.predict(o, deterministic=True)[0][0]
frc = m.actuator_forcerange[:6]; jr = m.jnt_range[:6]
A, SAT, NEARLIM, ERR = [], [], [], []
for sq in ("a1", "d1", "h1", "a8", "d8", "h8"):
    task.start_square, task.target_square = sq, "d4"
    obs, _ = env.reset(seed=0)
    for t in range(100):
        a = act(obs); obs, r, te, tr, info = env.step(a)
        A.append(np.abs(a[:6]))
        f = d.actuator_force[:6]
        SAT.append(np.abs(f) >= 0.999 * frc[:, 1])
        NEARLIM.append(np.minimum(np.abs(d.qpos[:6] - jr[:, 0]), np.abs(d.qpos[:6] - jr[:, 1])) < np.radians(2))
        ERR.append(np.degrees(np.abs(d.qpos[:6] - d.ctrl[:6])))
        if te or tr: break
A, SAT, NL, ERR = map(np.array, (A, SAT, NEARLIM, ERR))
J = ["pan", "lift", "elbow", "w1", "w2", "w3"]
print(f"{run.name} ck{ck}: {len(A)} policy steps over 6 start squares")
print("  |action| (normalised [-1,1]) median per joint:", {j: round(float(np.median(A[:, k])), 3) for k, j in enumerate(J)})
print("  fraction of steps with |action| > 0.9:        ", {j: round(float(np.mean(A[:, k] > 0.9)), 2) for k, j in enumerate(J)})
print("  |command - actual| deg, median:               ", {j: round(float(np.median(ERR[:, k])), 1) for k, j in enumerate(J)})
print("  fraction of steps actuator force SATURATED:   ", {j: round(float(np.mean(SAT[:, k])), 2) for k, j in enumerate(J)})
print("  fraction of steps joint within 2 deg of limit:", {j: round(float(np.mean(NL[:, k])), 2) for k, j in enumerate(J)})
print("  forcerange:", [tuple(map(float, x)) for x in frc], " ctrlrange lift:", tuple(map(float, m.actuator_ctrlrange[1])))
