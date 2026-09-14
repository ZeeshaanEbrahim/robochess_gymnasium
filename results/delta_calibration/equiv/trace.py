"""Fixed-seed trajectory trace for byte-for-byte action-mode equivalence checks.
Uniform-random actions from a fixed RNG through the real training stack
(scripts/train.py build_env), T1 and T2, shaped reward; records obs, reward,
done flags, data.ctrl and data.qpos every step."""
import importlib.util, sys
from pathlib import Path
import numpy as np
REPO = Path.home()/"projects/robochess_gymnasium"; sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("train", REPO/"scripts/train.py"); train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
out, extra = Path(sys.argv[1]), dict(a.split("=") for a in sys.argv[2:])
arrays = {}
for task in ("t1", "t2"):
    cfg = dict(task=task, reward_type="shaped", max_episode_steps=100, n_obstacles=2 if task == "t2" else 0, **extra)
    env = train.build_env(cfg); base = env.unwrapped
    rng = np.random.default_rng(1234)
    obs, _ = env.reset(seed=7)
    rec = {k: [] for k in ("obs", "reward", "term", "trunc", "ctrl", "qpos")}
    for t in range(600):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 7).astype(np.float32))
        rec["obs"].append(np.concatenate([obs[k].ravel() for k in sorted(obs)])); rec["reward"].append(r)
        rec["term"].append(term); rec["trunc"].append(trunc); rec["ctrl"].append(base.data.ctrl.copy()); rec["qpos"].append(base.data.qpos.copy())
        if term or trunc:
            obs, _ = env.reset(seed=7 + t)
    for k, v in rec.items(): arrays[f"{task}_{k}"] = np.array(v)
np.savez(out, **arrays)
print(f"wrote {out}: " + ", ".join(f"{k}{v.shape}" for k, v in arrays.items() if k.endswith(("obs", "ctrl"))))
