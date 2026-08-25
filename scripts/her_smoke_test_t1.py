#!/usr/bin/env python
"""HER + SAC integration smoke test for T1 (state-vector observations only).

Catches integration bugs fast -- buffer construction, goal relabelling,
observation shape mismatches -- at a small step count. This is NOT the Gate
M2 convergence check (200k steps); a short run here says nothing about
whether the policy will actually learn to solve T1, only that the
HerReplayBuffer + SAC + T1PickAndPlaceTask + SparsePickAndPlaceReward stack
runs end to end without crashing and that reward is not stuck at a constant.

State-vector only: image-observation HER is out of scope by decision (see
DECISIONS.md D5) -- RoboChessEnv has no image observation mode to begin
with, so there is nothing to accidentally include here.

Usage:
    python scripts/her_smoke_test_t1.py --steps 8000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.her import HerReplayBuffer  # noqa: E402

from robochess.envs import RoboChessEnv  # noqa: E402
from robochess.tasks import SparsePickAndPlaceReward, T1PickAndPlaceTask  # noqa: E402


def make_env(max_episode_steps: int) -> SparsePickAndPlaceReward:
    base = RoboChessEnv(max_episode_steps=max_episode_steps)
    return SparsePickAndPlaceReward(T1PickAndPlaceTask(base))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    args = parser.parse_args()

    env = make_env(args.max_episode_steps)

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=256,
        device="cpu",  # state-vector config -- CPU per the roadmap's device decision
        verbose=1,
    )

    print(f"policy device: {model.device}")
    print(f"replay buffer: {type(model.replay_buffer).__name__}")

    t0 = time.time()
    model.learn(total_timesteps=args.steps, log_interval=4)
    elapsed = time.time() - t0

    ep_rewards = list(model.ep_info_buffer)
    if ep_rewards:
        rewards = [e["r"] for e in ep_rewards]
        print(f"\ncompleted episodes in buffer : {len(ep_rewards)}")
        print(f"mean episode reward          : {sum(rewards)/len(rewards):.3f}")
        print(f"nonzero-reward episodes      : {sum(r > 0 for r in rewards)}/{len(rewards)}")
    else:
        print("\nno completed episodes made it into ep_info_buffer")

    print(f"\nwall clock                   : {elapsed:.1f} s for {args.steps} steps")
    print("STATUS: completed without error")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
