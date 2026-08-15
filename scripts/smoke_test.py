"""Smoke test: verifies mujoco, gymnasium, and stable-baselines3 are
correctly installed and interoperate, before any project-specific code
is written."""

import gymnasium as gym
import mujoco
from stable_baselines3 import PPO

print(f"mujoco version: {mujoco.__version__}")
print(f"gymnasium version: {gym.__version__}")

env = gym.make("Pendulum-v1")
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=500)

obs, info = env.reset()
action, _ = model.predict(obs)
obs, reward, terminated, truncated, info = env.step(action)

print("Smoke test passed: mujoco, gymnasium, and stable-baselines3 "
      "are installed correctly and interoperate.")
env.close()
