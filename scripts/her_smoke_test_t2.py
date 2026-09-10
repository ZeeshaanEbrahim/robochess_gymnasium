#!/usr/bin/env python
"""HER + SAC integration smoke test for T2 (state-vector observations only).

The T2 counterpart of scripts/her_smoke_test_t1.py, at the same scale and
settings, and for the same purpose: catch integration bugs -- buffer
construction, goal relabelling, observation shape mismatches -- at a small
step count. A short run says nothing about whether a policy can learn T2.

Three things are new here relative to T1's version, and are why this exists
rather than reusing it:

  * the observation is 29-dimensional, not 21 (D12): T2 appends every pool
    slot's (x, y) at the wrapper. HER has never been run against that shape.
  * the wrapper stack is one deeper -- SparsePickAndPlaceReward over
    T2ObstacleTask over T1PickAndPlaceTask over RoboChessEnv.
  * an episode can now end by truncation from D14's solver-instability guard,
    which T1's HER wiring was never exercised against. `--force-guard` lowers
    the guard's thresholds far enough that ordinary motion trips them, so the
    path is tested rather than assumed: at the committed solref = 0.008 no
    natural trigger is expected to occur.

Checks reward/observation for NaN or Inf on every step, and scans the replay
buffer and SAC's losses at the end.

Usage:
    python scripts/her_smoke_test_t2.py --steps 8000 --obstacles 0
    python scripts/her_smoke_test_t2.py --steps 8000 --obstacles 2
    python scripts/her_smoke_test_t2.py --steps 1500 --force-guard
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.her import HerReplayBuffer  # noqa: E402

from robochess.envs import robochess_env as rce  # noqa: E402
from robochess.tasks import (  # noqa: E402
    SparsePickAndPlaceReward,
    T2ObstacleTask,
    make_t2_env,
)


class SanityCallback(BaseCallback):
    """Fails loudly on non-finite values; counts guard truncations."""

    def __init__(self):
        super().__init__()
        self.guard_hits = 0
        self.nonfinite = []
        self.rewards_seen = 0
        self.truncations = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("solver_instability"):
                self.guard_hits += 1
            if info.get("TimeLimit.truncated") or info.get("terminated_reason") == "solver_instability":
                self.truncations += 1

        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float64)
        if rewards.size:
            self.rewards_seen += rewards.size
            if not np.all(np.isfinite(rewards)):
                self.nonfinite.append(("reward", int(self.num_timesteps)))

        obs = self.locals.get("new_obs")
        if isinstance(obs, dict):
            for key, value in obs.items():
                if not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
                    self.nonfinite.append((f"obs[{key}]", int(self.num_timesteps)))
        return True


def make_env(max_episode_steps: int, n_obstacles: int) -> SparsePickAndPlaceReward:
    base = make_t2_env(max_episode_steps=max_episode_steps)
    return SparsePickAndPlaceReward(T2ObstacleTask(base, n_obstacles=n_obstacles))


def scan_buffer(model) -> list[str]:
    """Non-finite values anywhere in the replay buffer's stored arrays."""
    bad = []
    buf = model.replay_buffer
    arrays = {"actions": buf.actions, "rewards": buf.rewards, "dones": buf.dones}
    for key, value in getattr(buf, "observations", {}).items():
        arrays[f"observations[{key}]"] = value
    for key, value in getattr(buf, "next_observations", {}).items():
        arrays[f"next_observations[{key}]"] = value
    for name, arr in arrays.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.size and not np.all(np.isfinite(a)):
            bad.append(name)
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--obstacles", type=int, default=2)
    parser.add_argument("--force-guard", action="store_true",
                        help="lower the solver-instability thresholds so ordinary "
                             "motion trips them, to exercise that ending under HER.")
    args = parser.parse_args()

    if args.force_guard:
        # Ordinary gripper/piece motion sits well under these normally; this
        # makes the guard fire routinely so the truncation path is exercised.
        rce.PIECE_LIN_SPEED_SANITY = 0.02
        rce.PIECE_ANG_SPEED_SANITY = 0.5
        rce.PIECE_KINETIC_ENERGY_SANITY = 1e-6
        print("FORCED GUARD: thresholds lowered to trip on ordinary motion")

    env = make_env(args.max_episode_steps, args.obstacles)
    obs, _ = env.reset(seed=0)
    print(f"observation dim              : {obs['observation'].shape} (T1's is (21,))")
    print(f"obstacles per episode        : {args.obstacles}")

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=256,
        device="cpu",
        verbose=0,
    )
    print(f"policy device                : {model.device}")
    print(f"replay buffer                : {type(model.replay_buffer).__name__}")

    cb = SanityCallback()
    t0 = time.time()
    model.learn(total_timesteps=args.steps, log_interval=1000, callback=cb)
    elapsed = time.time() - t0

    ep = list(model.ep_info_buffer)
    if ep:
        rewards = [e["r"] for e in ep]
        print(f"\ncompleted episodes in buffer : {len(ep)}")
        print(f"mean episode reward          : {sum(rewards)/len(rewards):.3f}")
        print(f"nonzero-reward episodes      : {sum(r > 0 for r in rewards)}/{len(rewards)}")
    else:
        print("\nno completed episodes made it into ep_info_buffer")

    losses = {k: v for k, v in model.logger.name_to_value.items() if "loss" in k}
    print(f"\nsteps with a reward observed : {cb.rewards_seen}")
    print(f"guard truncations (D14)      : {cb.guard_hits}")
    print(f"non-finite obs/reward events : {len(cb.nonfinite)} {cb.nonfinite[:5]}")
    print(f"final losses                 : {losses}")
    print(f"losses finite                : {all(np.isfinite(v) for v in losses.values())}")
    buffer_bad = scan_buffer(model)
    print(f"replay buffer non-finite in  : {buffer_bad or 'nothing'}")
    print(f"buffer entries stored        : {model.replay_buffer.size()}")

    clean = (not cb.nonfinite and not buffer_bad
             and all(np.isfinite(v) for v in losses.values()))
    print(f"\nwall clock                   : {elapsed:.1f} s for {args.steps} steps")
    print(f"STATUS: {'completed without error, no NaNs' if clean else 'PROBLEM -- see above'}")

    env.close()
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
