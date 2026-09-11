"""Gate M3 reproducibility check: same config, same seed, twice, then diff.

Records EVERY action the policy emitted and EVERY observation the env
returned, step by step, for two runs of an identical config, and compares
them exactly.

Deliberately captures the tensors at the VecEnv boundary via a wrapper
rather than re-deriving them afterwards: what matters for reproducibility is
the actual byte-level trajectory the learner saw, not a summary of it.
"""
from __future__ import annotations

import argparse, importlib.util, os, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from stable_baselines3.common.vec_env import VecEnvWrapper  # noqa: E402

spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
from robochess.experiment import load_config, resolve_config  # noqa: E402


class TraceRecorder(VecEnvWrapper):
    """Captures (action, obs, reward, done) at the outermost VecEnv boundary."""

    def __init__(self, venv):
        super().__init__(venv)
        self.actions, self.obs, self.rewards, self.dones = [], [], [], []

    def reset(self):
        o = self.venv.reset()
        self.obs.append({k: np.array(v, copy=True) for k, v in o.items()})
        return o

    def step_async(self, actions):
        self.actions.append(np.array(actions, copy=True))
        self.venv.step_async(actions)

    def step_wait(self):
        o, r, d, i = self.venv.step_wait()
        self.obs.append({k: np.array(v, copy=True) for k, v in o.items()})
        self.rewards.append(np.array(r, copy=True))
        self.dones.append(np.array(d, copy=True))
        return o, r, d, i


def run(cfg, steps, tag, outdir: Path):
    from stable_baselines3 import PPO
    venv = TraceRecorder(train.build_vec_env(cfg))
    model = PPO(policy="MultiInputPolicy", env=venv, seed=cfg["seed"],
                device=cfg["device"], verbose=0)
    model.learn(total_timesteps=steps)
    acts = np.concatenate(venv.actions, axis=0)
    obs = {k: np.concatenate([o[k] for o in venv.obs], axis=0) for k in venv.obs[0]}
    rews = np.concatenate(venv.rewards, axis=0)
    np.savez(outdir / f"trace_{tag}.npz", actions=acts, rewards=rews,
             **{f"obs_{k}": v for k, v in obs.items()})
    venv.close()
    return acts, obs, rews


def first_divergence(a, b):
    """Index of the first differing row, or None."""
    n = min(len(a), len(b))
    diff = np.any(a[:n].reshape(n, -1) != b[:n].reshape(n, -1), axis=1)
    idx = np.flatnonzero(diff)
    return int(idx[0]) if idx.size else None


def compare(name, a, b):
    same_shape = a.shape == b.shape
    bit = same_shape and np.array_equal(a, b)
    n = min(len(a), len(b))
    maxabs = float(np.max(np.abs(a[:n].astype(np.float64) - b[:n].astype(np.float64)))) if n else 0.0
    idx = first_divergence(a, b)
    print(f"  {name:22s} shape={str(a.shape):18s} bit-identical={bit}  "
          f"max|diff|={maxabs:.3e}  first-diff-step={idx}")
    return bit, maxabs, idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=4096)
    p.add_argument("--outdir", default=".")
    a = p.parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = resolve_config(load_config(a.config), config_path=Path(a.config))
    print("=== environment provenance (D8: BLAS/threads can move results) ===")
    import torch
    print(f"  torch {torch.__version__} | threads={torch.get_num_threads()} "
          f"interop={torch.get_num_interop_threads()}")
    print(f"  numpy {np.__version__}")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        print(f"  {v}={os.environ.get(v)}")
    print(f"  config={a.config} seed={cfg['seed']} steps={a.steps} device={cfg['device']}")
    print()

    print(f"--- run A ---"); a_act, a_obs, a_rew = run(cfg, a.steps, "A", outdir)
    print(f"--- run B ---"); b_act, b_obs, b_rew = run(cfg, a.steps, "B", outdir)

    print("\n=== diff ===")
    results = [compare("actions", a_act, b_act)]
    for k in sorted(a_obs):
        results.append(compare(f"obs[{k}]", a_obs[k], b_obs[k]))
    results.append(compare("rewards", a_rew, b_rew))

    all_bit = all(r[0] for r in results)
    print()
    if all_bit:
        print("VERDICT: BIT-IDENTICAL across all recorded actions and observations.")
    else:
        worst = max(r[1] for r in results)
        firsts = [r[2] for r in results if r[2] is not None]
        print(f"VERDICT: DIVERGED. first differing step={min(firsts) if firsts else '?'}  "
              f"max|diff| across all arrays={worst:.3e}")
    return 0 if all_bit else 1


if __name__ == "__main__":
    raise SystemExit(main())
