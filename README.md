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

The model XML files under `assets/` are tracked in git, because they carry local
modifications. The mesh files they reference (~34 MB of `.obj`/`.stl`) are **not**
tracked, and are fetched on demand instead. **After cloning, run this before
anything else — no scene will load until you do:**

```bash
scripts/fetch_assets.sh
```

It copies the meshes out of the Menagerie clone into `assets/ur5e/assets/` and
`assets/robotiq_2f85/assets/`, and is safe to re-run. It defaults to
`~/reference_assets/mujoco_menagerie`; override with `$MUJOCO_MENAGERIE_DIR` or
by passing the path as an argument.

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

## Gotchas

Things that look like bugs but are not. Add to this list rather than
rediscovering them.

- **UR5e position actuators droop ~1.5° under gravity.** Settling 5000 steps from
  the `home` keyframe in `assets/scene.xml`, the gripper approach axis drifts from
  the keyframe's exact `(0, 0, -1)` to `(0, -0.0258, -0.9997)`. This is normal
  finite-gain position-actuator behaviour holding against gravity, not a modelling
  error. If `reset`/`step` reports an end-effector orientation slightly off the
  keyframe value, that is why — don't chase it.

- **`failed_phase` and `outcome` mark where a failure was *detected*, not where it
  was *caused*.** `_piece_off_board()` is a plain XY/Z extent test with no reference
  to grip state, so it fires whenever the piece is outside the footprint —
  including while it is securely gripped and airborne, and including many
  macro-steps after whatever actually went wrong. D11 in `DECISIONS.md` has one
  instance labelled `failed_phase: "release"` whose grip was already lost two
  steps *before* the release waypoint began, and another that never sets
  `piece_off_board` at all and surfaces only as `sequence_finished_not_at_goal`.
  Classify failures from the contact/velocity trace, not from these labels.

## Reproducing the dependency set

Pinned versions are in `requirements.txt`. MuJoCo and Gymnasium both make
breaking API changes between minor versions, so experiments should be run
against this frozen set:

```bash
pip install -r requirements.txt
```
