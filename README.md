# RoboChessGymnasium

A MuJoCo/Gymnasium robot-arm manipulation benchmark: a UR5e with a Robotiq 2F-85
gripper learning to manipulate chess pieces, trained with Stable-Baselines3
(PPO, SAC, HER).

## Activating the environment

Every command in this project — running scripts, training, tests — must be run
with the project virtual environment active. In each new terminal session, run:

```bash
source ~/venvs/robochess/bin/activate
```

The venv is built on **Python 3.11** and lives outside the repository, so it is
not tracked by git. If `which python` does not point inside
`~/venvs/robochess/bin/`, the venv is not active.

## Setup

The development environment was built by following the project's environment
setup document (`RoboChessGymnasium Dev Environment`, sections 0–12), which is
the authoritative reference for reproducing this machine from scratch. Each step
was verified against that document before proceeding to the next.

Summary of what that produced:

| Component | Version / location |
| --- | --- |
| Platform | WSL2, Ubuntu 22.04 LTS |
| Python | 3.11.15 (`~/venvs/robochess`) |
| mujoco | 3.11.0 |
| gymnasium | 1.3.0 |
| stable-baselines3 | 2.9.0 |
| sb3-contrib | 2.9.0 |
| mink (IK) | 1.2.0 |
| torch | 2.13.0+cu130 (CUDA available) |
| Rendering backend | GLFW (default, via WSLg); `MUJOCO_GL` unset |
| Reference models | `~/reference_assets/mujoco_menagerie` |

Reference robot and gripper models come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), cloned
separately from this repository. Files are **copied** from there into `assets/`
and modified here — never edited in place inside that clone.

For headless/cluster runs (no display), see the commented `MUJOCO_GL` block in
`~/.bashrc`: `egl` for GPU-accelerated headless rendering, `osmesa` as a CPU
fallback.

## Layout

```
assets/              model XML and mesh assets used by the environments
robochess/envs/      Gymnasium environment implementations
robochess/wrappers/  observation/action/reward wrappers
scripts/             entry points and utilities (training, smoke tests)
configs/             experiment and hyperparameter configs
tests/               test suite
```

## Training defaults

State-vector (MLP-policy) configs set `device: cpu` explicitly for both PPO and
SAC; image/CNN-policy configs stay on the default so they resolve to CUDA. See
`configs/README.md` for the rationale and `scripts/benchmark_device.py` for the
measurement behind it.

## Reproducing the dependency set

Pinned versions are in `requirements.txt`. MuJoCo and Gymnasium both make
breaking API changes between minor versions, so experiments should be run
against this frozen set:

```bash
pip install -r requirements.txt
```
