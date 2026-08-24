#!/usr/bin/env python
"""Throughput profiler for RoboChessEnv.

Measures sustained environment steps/sec across observation modes, image
resolutions and worker counts, and projects wall-clock time for a 500k-step
pilot run. The point is to sanity-check P4's training-budget assumptions
before a run schedule is committed to.

This is a THROUGHPUT test, not a task test: actions are uniform random, and no
policy or scripted controller is involved. Nothing here measures learning.

    python scripts/profile_env.py --steps 2000

Backend caveat: image configs are only as fast as the active MuJoCo rendering
backend. Locally that is WSL2 with osmesa (CPU software rasterisation) or GLFW;
the cluster will use EGL on a GPU. The image numbers here are directional for
local debugging and are NOT a promise of cluster throughput. The state-only
numbers do not render at all and travel better.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from robochess.envs import RoboChessEnv  # noqa: E402

# Cameras defined in the scene. "overhead" is the static third-person view in
# assets/scene.xml; "ur_wrist" is the wrist-mounted view, declared in
# assets/ur5e/ur5e.xml and prefixed when the arm is attached.
CAMERAS = ["overhead", "ur_wrist"]

PILOT_BUDGET_STEPS = 500_000  # the pilot-matrix budget the projection is for


class CameraObsWrapper(gym.ObservationWrapper):
    """Adds rendered camera images to the observation dict.

    Deliberately a wrapper rather than a change to RoboChessEnv: image
    observations are a P2 concern, and this profiler should not be the reason
    the env grows a rendering path.
    """

    def __init__(self, env: gym.Env, cameras: list[str], resolution: int):
        super().__init__(env)
        self.cameras = cameras
        self.resolution = resolution
        self._renderer: mujoco.Renderer | None = None

        obs_spaces = dict(env.observation_space.spaces)
        for cam in cameras:
            obs_spaces[f"pixels/{cam}"] = spaces.Box(
                low=0, high=255, shape=(resolution, resolution, 3), dtype=np.uint8
            )
        self.observation_space = spaces.Dict(obs_spaces)

    def _get_renderer(self) -> mujoco.Renderer:
        # Created lazily and inside the worker process: a GL context must not be
        # inherited across a fork.
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.env.unwrapped.model, self.resolution, self.resolution
            )
        return self._renderer

    def observation(self, observation):
        renderer = self._get_renderer()
        data = self.env.unwrapped.data
        for cam in self.cameras:
            renderer.update_scene(data, camera=cam)
            observation[f"pixels/{cam}"] = renderer.render()
        return observation

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        return self.env.close()


def make_env(n_cameras: int, resolution: int):
    """Picklable-by-cloudpickle factory for the vec envs."""

    def _thunk():
        env = RoboChessEnv()
        if n_cameras > 0:
            env = CameraObsWrapper(env, CAMERAS[:n_cameras], resolution)
        return env

    return _thunk


def build_vec_env(n_workers: int, n_cameras: int, resolution: int, start_method: str = "fork"):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = [make_env(n_cameras, resolution) for _ in range(n_workers)]
    if n_workers == 1:
        return DummyVecEnv(fns)
    # "fork" by default, not "spawn". Spawn re-imports SB3 (and therefore torch)
    # in every worker, costing ~460 MB RSS each -- 16 workers OOM-killed this
    # 7 GB machine before producing a number. Fork shares those pages
    # copy-on-write. Renderers are constructed lazily inside the worker, so no
    # OpenGL context is inherited across the fork.
    return SubprocVecEnv(fns, start_method=start_method)


def benchmark(n_workers: int, n_cameras: int, resolution: int, steps: int, warmup: int,
              max_seconds: float, warmup_seconds: float, start_method: str = "fork") -> dict:
    """Time `steps` vec-steps, or `max_seconds` worth of them, whichever comes first.

    The time cap exists because software-rasterised rendering here costs ~215 ms
    per frame; a fixed 2000-step window on the 2-camera configs would run for
    hours. The number of steps actually measured is recorded so a short window
    is visible in the results rather than hidden.
    """
    vec = build_vec_env(n_workers, n_cameras, resolution, start_method)
    try:
        vec.reset()
        rng = np.random.default_rng(0)
        actions = rng.uniform(
            -1.0, 1.0, size=(max(steps, warmup), n_workers, vec.action_space.shape[0])
        ).astype(np.float32)

        # Discard: model load, first render, GL pipeline warm-up.
        t_warm = time.perf_counter()
        for i in range(warmup):
            vec.step(actions[i])
            if time.perf_counter() - t_warm > warmup_seconds:
                break

        t0 = time.perf_counter()
        done_steps = 0
        for i in range(steps):
            vec.step(actions[i])
            done_steps += 1
            if time.perf_counter() - t0 > max_seconds:
                break
        elapsed = time.perf_counter() - t0
    finally:
        vec.close()

    steps = done_steps
    env_steps = steps * n_workers
    sps = env_steps / elapsed
    return {
        "workers": n_workers,
        "cameras": n_cameras,
        "resolution": resolution if n_cameras else 0,
        "vec_steps": steps,
        "env_steps": env_steps,
        "seconds": round(elapsed, 2),
        "steps_per_sec": round(sps, 1),
        "hours_for_500k": round(PILOT_BUDGET_STEPS / sps / 3600.0, 2),
    }


def _kill_process_group(pid: int) -> None:
    """SIGKILL a whole process group, tolerating groups that are already gone."""
    import signal
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_config_isolated(n_cameras, resolution, n_workers, args) -> dict:
    """Run one configuration in a fresh subprocess and return its result dict.

    Each configuration gets its own interpreter for two reasons, both learned
    the hard way on this machine:

      * The workers=1 case uses DummyVecEnv, which builds a renderer *in this
        process*. A later SubprocVecEnv fork would then inherit live OSMesa
        state, and the children deadlock -- the grid hung for 20 minutes with
        every worker at 0% CPU before this was isolated.
      * A configuration that hangs or is OOM-killed is contained: it is
        reported as a failed row and the rest of the grid still runs.
    """
    cmd = [sys.executable, str(Path(__file__).resolve()), "--single",
           "--cameras", str(n_cameras), "--resolution", str(resolution),
           "--workers", str(n_workers), "--steps", str(args.steps),
           "--warmup", str(args.warmup), "--max-seconds", str(args.max_seconds),
           "--warmup-seconds", str(args.warmup_seconds),
           "--start-method", args.start_method]
    timeout = args.max_seconds + args.warmup_seconds + args.config_timeout_slack
    # start_new_session=True puts the child in its own process group, so on a
    # timeout the whole tree (including SubprocVecEnv's forked workers) can be
    # killed by process-GROUP id. Without this, subprocess.run's timeout only
    # kills the direct child -- its forked grandchildren are orphaned and keep
    # running. That happened for real here: a 2-camera/128x128/16-worker config
    # timed out and left 15 renderer processes running unsupervised, together
    # consuming 7.2 GB RSS + 1.8 GB swap until they were found and killed by hand.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc.pid)
        proc.communicate()  # reap, avoid a zombie
        raise
    for line in stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    err = (stderr.strip().splitlines() or ["no output"])[-1]
    raise RuntimeError(f"child exited {proc.returncode}: {err[:160]}")


def obs_label(n_cameras: int, resolution: int) -> str:
    if n_cameras == 0:
        return "state-only"
    return f"state+{n_cameras}cam @{resolution}x{resolution}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000, help="timed vec-steps per worker")
    parser.add_argument("--warmup", type=int, default=50, help="discarded vec-steps")
    parser.add_argument("--max-seconds", type=float, default=60.0,
                        help="cap on the timed window per config (seconds)")
    parser.add_argument("--start-method", default="fork", choices=["fork", "spawn", "forkserver"],
                        help="multiprocessing start method for SubprocVecEnv")
    parser.add_argument("--warmup-seconds", type=float, default=15.0,
                        help="cap on the warm-up phase per config (seconds)")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "profiling" / "p1_throughput.csv")
    parser.add_argument("--config-timeout-slack", type=float, default=180.0,
                        help="extra seconds allowed per config subprocess before it is killed")
    parser.add_argument("--single", action="store_true",
                        help=argparse.SUPPRESS)  # internal: run one config, emit JSON
    parser.add_argument("--cameras", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--resolution", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single:
        row = benchmark(args.workers[0] if isinstance(args.workers, list) else args.workers,
                        args.cameras, args.resolution, args.steps, args.warmup,
                        args.max_seconds, args.warmup_seconds, args.start_method)
        print("RESULT_JSON:" + json.dumps(row))
        return 0

    backend = os.environ.get("MUJOCO_GL", "<unset: mujoco default>")
    print("=" * 78)
    print("RoboChessEnv throughput profile")
    print(f"  MUJOCO_GL       : {backend}")
    print(f"  platform        : {platform.platform()}")
    print(f"  cpu count       : {os.cpu_count()}")
    print(f"  timed window    : up to {args.steps} vec-steps or {args.max_seconds:.0f}s per "
          f"config (after {args.warmup} warm-up steps / {args.warmup_seconds:.0f}s)")
    print("  NOTE: image configs are measured on this local WSL2 backend, NOT the")
    print("        cluster's EGL/GPU backend. Directional for local debugging only.")
    print("=" * 78)

    # (n_cameras, resolution); state-only has no resolution axis.
    obs_configs = [(0, 0)] + [(n, r) for n in (1, 2) for r in args.resolutions]

    rows = []
    for n_cameras, resolution in obs_configs:
        for n_workers in args.workers:
            label = obs_label(n_cameras, resolution)
            print(f"  running {label:<26} workers={n_workers:<3} ...", end="", flush=True)
            try:
                row = run_config_isolated(n_cameras, resolution, n_workers, args)
                row["obs_mode"] = label
                row["backend"] = backend
                rows.append(row)
                print(f" {row['steps_per_sec']:>9.1f} steps/s "
                      f"({row['hours_for_500k']:>6.2f} h for 500k)")
            except subprocess.TimeoutExpired:
                print(" FAILED: timed out (hung)")
                rows.append({
                    "obs_mode": label, "workers": n_workers, "cameras": n_cameras,
                    "resolution": resolution if n_cameras else 0, "vec_steps": 0,
                    "env_steps": 0, "seconds": 0.0, "steps_per_sec": 0.0,
                    "hours_for_500k": 0.0, "backend": backend, "error": "timeout (hung)",
                })
            except Exception as exc:  # a config may simply not fit in RAM
                print(f" FAILED: {type(exc).__name__}: {exc}")
                rows.append({
                    "obs_mode": label, "workers": n_workers, "cameras": n_cameras,
                    "resolution": resolution if n_cameras else 0, "vec_steps": args.steps,
                    "env_steps": 0, "seconds": 0.0, "steps_per_sec": 0.0,
                    "hours_for_500k": 0.0, "backend": backend,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["obs_mode", "cameras", "resolution", "workers", "vec_steps", "env_steps",
              "seconds", "steps_per_sec", "hours_for_500k", "backend", "error"]
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\n" + "=" * 78)
    print(f"{'observation mode':<26} {'workers':>7} {'steps/s':>10} {'500k run':>12} {'measured':>10}")
    print("-" * 78)
    for row in rows:
        proj = "FAILED" if row["steps_per_sec"] == 0 else f"{row['hours_for_500k']:.2f} h"
        print(f"{row['obs_mode']:<26} {row['workers']:>7} {row['steps_per_sec']:>10.1f} {proj:>12}"
              f" {row['env_steps']:>10,}")
    print("=" * 78)
    try:
        shown = args.out.relative_to(REPO_ROOT)
    except ValueError:
        shown = args.out
    print(f"written: {shown}")
    print(f"projection basis: {PILOT_BUDGET_STEPS:,} env steps, sustained rate, no policy "
          "update cost included")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
