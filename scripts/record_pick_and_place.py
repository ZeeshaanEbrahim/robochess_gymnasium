#!/usr/bin/env python
"""Record a scripted-expert pick-and-place episode to an .mp4 file.

Runs the real, unmodified ScriptedExpert/T1PickAndPlaceTask/RoboChessEnv
end to end via run_episode(), capturing one rendered frame per environment
step through a thin wrapper -- meant for showing a supervisor a run of the
current system, not a debugging tool. Uses RoboChessEnv's own rgb_array
renderer, so it works the same way on a headless machine (no live GLFW
window needed).

Usage:
    python scripts/record_pick_and_place.py --start e6 --target e5 --out demo.mp4
    python scripts/record_pick_and_place.py --start a1 --target h8 --out demo.mp4 --fps 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robochess.envs import RoboChessEnv  # noqa: E402
from robochess.tasks import T1PickAndPlaceTask  # noqa: E402
from scripts.scripted_expert import ScriptedExpert  # noqa: E402


class FrameRecorder(gym.Wrapper):
    """Appends a rendered frame to `frames` after every reset()/step()."""

    def __init__(self, env: gym.Env, frames: list):
        super().__init__(env)
        self.frames = frames

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.frames.append(self.env.render())
        return result

    def step(self, action):
        result = self.env.step(action)
        self.frames.append(self.env.render())
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="e6", help="start square, e.g. e6")
    parser.add_argument("--target", default="e5", help="target square, e.g. e5")
    parser.add_argument("--out", default="pick_and_place.mp4", help="output video path")
    parser.add_argument("--fps", type=int, default=25, help="playback frame rate")
    parser.add_argument("--max-episode-steps", type=int, default=350)
    args = parser.parse_args()

    frames: list = []
    env = RoboChessEnv(max_episode_steps=args.max_episode_steps, render_mode="rgb_array")
    env = T1PickAndPlaceTask(env, start_square=args.start, target_square=args.target)
    env = FrameRecorder(env, frames)
    expert = ScriptedExpert(env)

    rec = expert.run_episode(seed=0)
    env.close()

    print(f"outcome: {rec['outcome']}, final distance {rec['final_distance']*1000:.1f} mm, "
          f"{rec['steps']} steps, {len(frames)} frames")

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"wrote {args.out}")
    return 0 if rec["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
