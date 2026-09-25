#!/usr/bin/env python
"""Gate M3 reproducibility check: same config, same seed, twice, then diff.

Records EVERY action the policy emitted and EVERY observation the env
returned, step by step, for two runs of an identical config, and compares
them exactly.

The trace is captured at the VecEnv boundary through a wrapper rather than
re-derived afterwards: what matters for reproducibility is the byte-level
trajectory the learner actually saw, not a summary of it.

Thread pinning is applied through scripts/train.py's `apply_thread_pinning`
before torch is imported, for the reason given in that function's docstring.
Runs at a FIXED thread count are bit-identical; runs at different thread
counts are not, so this script pins from the config exactly as training does.

    python scripts/repro_check.py --config configs/t1_ppo_shaped_state_seed0.yaml \
        --steps 4096 --outdir /tmp/repro

    # two separate interpreter processes, then diff:
    python scripts/repro_check.py --config <cfg> --steps 4096 --outdir D --only A
    python scripts/repro_check.py --config <cfg> --steps 4096 --outdir D --only B
    python scripts/repro_check.py --outdir D --diff-only A B
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from robochess.experiment import load_config, resolve_config  # noqa: E402

_spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)   # safe: train.py imports nothing heavy at module scope


def _recorder_class():
    """Defined lazily so SB3 (and torch) import only after thread pinning."""
    import numpy as np
    from stable_baselines3.common.vec_env import VecEnvWrapper

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

    return TraceRecorder


def run(cfg: dict, steps: int, tag: str, outdir: Path) -> Path:
    """Train once and write the full trace to `outdir/trace_<tag>.npz`."""
    import numpy as np

    venv = _recorder_class()(train.build_vec_env(cfg))
    # The training path's own builder, so the algorithm (PPO / SAC, and SAC's
    # HER replay buffer) is the one the config names. (F13: this used to build
    # PPO unconditionally, so a SAC config was "checked" against a PPO run.)
    model = train.build_model(cfg, venv)
    model.learn(total_timesteps=steps)

    acts = np.concatenate(venv.actions, axis=0)
    obs = {k: np.concatenate([o[k] for o in venv.obs], axis=0) for k in venv.obs[0]}
    rews = np.concatenate(venv.rewards, axis=0)
    path = outdir / f"trace_{tag}.npz"
    np.savez(path, actions=acts, rewards=rews, **{f"obs_{k}": v for k, v in obs.items()})
    venv.close()
    return path


def diff(outdir: Path, tag_a: str, tag_b: str) -> bool:
    """Compare two saved traces array by array. Returns True if bit-identical."""
    import numpy as np

    A = dict(np.load(outdir / f"trace_{tag_a}.npz"))
    B = dict(np.load(outdir / f"trace_{tag_b}.npz"))
    print(f"\n=== diff: {tag_a} vs {tag_b} ===")
    all_bit = True
    for k in sorted(A):
        a, b = A[k], B[k]
        bit = a.shape == b.shape and np.array_equal(a, b)
        n = min(len(a), len(b))
        d = np.abs(a[:n].astype(np.float64) - b[:n].astype(np.float64))
        mx = float(d.max()) if n else 0.0
        rows = np.any(d.reshape(n, -1) != 0, axis=1)
        first = int(np.flatnonzero(rows)[0]) if rows.any() else None
        all_bit &= bit
        print(f"  {k:20s} shape={str(a.shape):16s} bit-identical={str(bit):5s} "
              f"max|diff|={mx:.3e} first-diff-step={first}")
    print(f"\nVERDICT: {'BIT-IDENTICAL' if all_bit else 'DIVERGED'}")
    return all_bit


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--steps", type=int, default=4096)
    p.add_argument("--outdir", default=".")
    p.add_argument("--only", help="run just this tag (for separate-process checks)")
    p.add_argument("--diff-only", nargs=2, metavar=("A", "B"),
                   help="skip training; diff two existing traces")
    a = p.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if a.diff_only:
        return 0 if diff(outdir, *a.diff_only) else 1

    if not a.config:
        p.error("--config is required unless --diff-only is given")
    cfg = resolve_config(load_config(a.config), config_path=Path(a.config))

    # Pin BEFORE importing torch, exactly as training does.
    threads = train.apply_thread_pinning(cfg["num_threads"])

    import numpy as np
    import torch

    print("=== environment provenance (D8: BLAS/threads can move results) ===")
    print(f"  torch {torch.__version__} | numpy {np.__version__}")
    print(f"  num_threads (config) = {cfg['num_threads']}")
    print(f"  torch.get_num_threads() = {threads['torch_num_threads']}  "
          f"interop = {threads['torch_num_interop_threads']}")
    for k, v in threads["env"].items():
        print(f"  {k}={v}")
    print(f"  config={a.config} seed={cfg['seed']} steps={a.steps} device={cfg['device']}")

    if a.only:
        path = run(cfg, a.steps, a.only, outdir)
        print(f"\nrun {a.only} written -> {path}")
        return 0

    print("\n--- run A ---"); run(cfg, a.steps, "A", outdir)
    print("--- run B ---"); run(cfg, a.steps, "B", outdir)
    return 0 if diff(outdir, "A", "B") else 1


if __name__ == "__main__":
    raise SystemExit(main())
