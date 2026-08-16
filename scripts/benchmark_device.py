"""Benchmark PPO (MlpPolicy) training throughput on CPU vs CUDA.

Stable-Baselines3 warns that MLP-policy PPO is usually faster on CPU than GPU,
because the per-step tensors are too small to amortise host<->device transfer
overhead. This script measures that claim on this machine rather than assuming
it, so the device choice for the project's training runs is evidence-based.

Both runs use an identical fixed seed and timestep budget. Only the
model.learn() call is timed -- imports, env construction, and model setup are
excluded from the measurement.
"""

import time

import gymnasium as gym
import torch
from stable_baselines3 import PPO

SEED = 42
TOTAL_TIMESTEPS = 20_000
ENV_ID = "Pendulum-v1"


def benchmark(device: str) -> float:
    """Train PPO on `device` and return wall-clock seconds spent in learn()."""
    env = gym.make(ENV_ID)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    model = PPO("MlpPolicy", env, device=device, seed=SEED, verbose=0)

    start = time.perf_counter()
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    elapsed = time.perf_counter() - start

    env.close()
    return elapsed


def main() -> None:
    print(f"torch: {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"env: {ENV_ID} | timesteps: {TOTAL_TIMESTEPS:,} | seed: {SEED}")
    print()

    print("running cpu ...")
    cpu_seconds = benchmark("cpu")
    print(f"  cpu:  {cpu_seconds:.2f} s")

    print("running cuda ...")
    cuda_seconds = benchmark("cuda")
    print(f"  cuda: {cuda_seconds:.2f} s")

    print()
    print("=" * 56)
    print(f"CPU  wall-clock: {cpu_seconds:7.2f} s")
    print(f"CUDA wall-clock: {cuda_seconds:7.2f} s")

    if cpu_seconds < cuda_seconds:
        print(f"CPU was {cuda_seconds / cpu_seconds:.2f}x faster than CUDA.")
    elif cuda_seconds < cpu_seconds:
        print(f"CUDA was {cpu_seconds / cuda_seconds:.2f}x faster than CPU.")
    else:
        print("CPU and CUDA were identical.")
    print("=" * 56)


if __name__ == "__main__":
    main()
