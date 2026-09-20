#!/usr/bin/env python
"""Live-viewer demo of a scripted-expert pick-and-place episode.

Opens MuJoCo's interactive viewer window (GLFW, via WSLg on WSL2 -- the
same backend already confirmed working for manual verification in P1) and
drives ScriptedExpert/T1PickAndPlaceTask/RoboChessEnv through one real
episode in real time, so it can be watched live instead of as a recorded
clip. Must be run from a real interactive terminal with a display -- it
will not work through a headless/offscreen shell.

Usage:
    python scripts/live_demo_pick_and_place.py --start e6 --target e5
    python scripts/live_demo_pick_and_place.py --start a1 --target h8 --speed 0.5   # half speed, easier to follow
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robochess.envs import RoboChessEnv  # noqa: E402
from robochess.tasks import T1PickAndPlaceTask  # noqa: E402
from scripts.scripted_expert import ScriptedExpert  # noqa: E402


class LiveViewerSync(gym.Wrapper):
    """Syncs a live MuJoCo viewer and paces playback to roughly real time
    (scaled by `speed`) after every reset()/step()."""

    def __init__(self, env: gym.Env, viewer, dt: float, speed: float):
        super().__init__(env)
        self._viewer = viewer
        self._sleep = dt / speed if speed > 0 else 0.0

    def _tick(self):
        self._viewer.sync()
        time.sleep(self._sleep)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._tick()
        return result

    def step(self, action):
        result = self.env.step(action)
        self._tick()
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="e6", help="start square, e.g. e6")
    parser.add_argument("--target", default="e5", help="target square, e.g. e5")
    parser.add_argument("--max-episode-steps", type=int, default=350)
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="playback speed multiplier (1.0 = real time, 0.5 = half speed/easier to follow)",
    )
    args = parser.parse_args()

    base_env = RoboChessEnv(max_episode_steps=args.max_episode_steps)
    env = T1PickAndPlaceTask(base_env, start_square=args.start, target_square=args.target)

    viewer = mujoco.viewer.launch_passive(base_env.model, base_env.data)
    dt = base_env.frame_skip * base_env.model.opt.timestep
    env = LiveViewerSync(env, viewer, dt=dt, speed=args.speed)
    expert = ScriptedExpert(env)

    print(f"Live demo: {args.start} -> {args.target}. Watch the viewer window.")
    rec = expert.run_episode(seed=0)
    print(f"outcome: {rec['outcome']}, final distance {rec['final_distance']*1000:.1f} mm, "
          f"steps={rec['steps']}")

    input("Episode finished -- press enter to close the viewer...")
    viewer.close()
    env.close()
    return 0 if rec["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
