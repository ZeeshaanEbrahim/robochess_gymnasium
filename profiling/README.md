# P1 throughput profiling

Sanity-checks the training-budget assumptions behind P4's pilot run schedule
by measuring sustained `RoboChessEnv` steps/sec across observation mode, image
resolution, and parallel-worker count. This is a throughput benchmark, not a
task benchmark: actions are uniform random (`scripts/scripted_expert.py` is
the task-validity check; this is a different question).

## How to reproduce

```bash
MUJOCO_GL=osmesa python scripts/profile_env.py
```

Full options: `--steps` (timed vec-steps per config, default 2000), `--warmup`
(discarded vec-steps first, default 50), `--max-seconds` / `--warmup-seconds`
(hard per-config time caps, since a slow config would otherwise run for hours
before hitting its step count), `--workers`, `--resolutions`, `--out`.

Each grid cell runs in its own subprocess (see `run_config_isolated` in
`scripts/profile_env.py`). That is not incidental: the `workers=1` case builds
its renderer in-process via `DummyVecEnv`, and if the next config's
`SubprocVecEnv` then forked from that same process, the children would inherit
live OSMesa state and deadlock. Isolating each cell in its own interpreter
also means one hung or OOM-killed config can't take the rest of the grid down
with it -- it's just recorded as a failed row.

## What was measured

| axis | values |
| --- | --- |
| observation mode | state-only, state + 1 camera, state + 2 cameras |
| image resolution | 64x64, 128x128 (camera configs only) |
| parallel workers | 1 (`DummyVecEnv`), 8, 16 (`SubprocVecEnv`) |

15 configurations total. Cameras are `overhead` (static third-person, defined
in `assets/scene.xml`) and `ur_wrist` (wrist-mounted, defined in
`assets/ur5e/ur5e.xml` since it has to hang off the wrist body inside the
attached arm submodel) -- both added to the scene as part of this task, since
P2's image-observation configs need them and there was no reason to add them
twice.

## Results

Full table: [`p1_throughput.csv`](p1_throughput.csv). Summary:

| observation mode | workers | steps/s | 500k-step projection |
| --- | ---: | ---: | ---: |
| state-only | 1 | 428.5 | 0.32 h |
| state-only | 8 | 1149.4 | 0.12 h |
| state-only | 16 | 1420.3 | 0.10 h |
| state + 1 camera @64x64 | 1 | 3.8 | 36.9 h |
| state + 1 camera @64x64 | 8 | 9.6 | 14.5 h |
| state + 1 camera @64x64 | 16 | 10.1 | 13.8 h |
| state + 1 camera @128x128 | 1 | 4.0 | 34.7 h |
| state + 1 camera @128x128 | 8 | 9.8 | 14.2 h |
| state + 1 camera @128x128 | 16 | 9.3 | 14.9 h |
| state + 2 cameras @64x64 | 1 | 2.0 | 68.3 h |
| state + 2 cameras @64x64 | 8 | 5.6 | 24.8 h |
| state + 2 cameras @64x64 | 16 | 4.1 | 34.1 h |
| state + 2 cameras @128x128 | 1 | 2.2 | 62.4 h |
| state + 2 cameras @128x128 | 8 | 5.5 | 25.3 h |
| state + 2 cameras @128x128 | 16 | **unreliable -- see below** | -- |

"500k-step projection" is `500,000 / steps_per_sec`, the pilot-matrix budget
mentioned in the roadmap. It does not include policy-update cost (gradient
steps, optimizer, logging) -- it is env throughput only, a lower bound on
wall-clock, not a training-time estimate.

## What this says, directionally

- **Rendering dominates, resolution barely matters.** State-only is ~100-400x
  faster than any image config. But going from 64x64 to 128x128 changes
  almost nothing (e.g. 1-camera/8-workers: 9.6 vs 9.8 steps/s). Direct
  measurement of `mujoco.Renderer.render()` in isolation confirmed why: the
  scene has 185,489 mesh faces (mostly the UR5e's visual meshes), and at these
  resolutions OSMesa's software rasterizer is spending its time on geometry
  processing, not pixel fill -- so resolution is nearly free and mesh
  complexity is the real cost. This may not hold on a GPU backend, where
  fill-rate scales differently.
- **Parallelism has a resolution-dependent ceiling.** State-only scales well
  through 16 workers (428 to 1420 steps/s). 1-camera configs gain from 1 to 8
  workers but roughly plateau from 8 to 16 (9.6 to 10.1 steps/s, 64x64). For
  2-camera configs, 16 workers is *worse* than 8 (64x64: 4.1 vs 5.6 steps/s;
  128x128: see below) -- consistent with CPU contention among 12 available
  cores plus the memory pressure noted next, not with genuine additional
  throughput.
- **The heaviest config (2 cameras, 128x128, 16 workers) does not fit
  comfortably in 8 GB of RAM.** On the first attempt it ran the machine to
  7.2/7.7 GB RSS plus 1.8 GB of swap and effectively hung; the profiling
  script's subprocess-timeout handling had a real bug where the 15 forked
  worker processes were left running unsupervised after the parent was killed
  (fixed: the child now runs in its own process group and the whole group is
  killed on timeout). A retry after clearing that memory succeeded
  mechanically, but captured only 3 vec-steps in its 60-second window and
  reported 0.7 steps/s -- *slower* than the same config at 8 workers (5.5
  steps/s), which is a sign of measurement artifact from memory pressure, not
  a real sustained rate. **Do not use that cell for planning.** If the
  2-camera/128x128 condition matters for P4, it needs to be measured on a
  machine with real headroom above 8 GB, not extrapolated from this number.

## Backend caveat -- read this before using these numbers for anything but local debugging

`MUJOCO_GL=osmesa` for every image-observation row (recorded per-row in the
CSV's `backend` column). OSMesa is CPU software rasterization. This machine is
WSL2 with no usable GPU passthrough for rendering: `MUJOCO_GL=egl` was tried
first and is *not* faster (4.3-4.7 steps/s either way for 1 camera/1 worker),
and under `SubprocVecEnv` with 8+ workers, EGL contexts actively crashed
(`D3D12: Removing Device`, then broken pipes to the workers) -- WSL2's GPU
passthrough driver does not tolerate multiple concurrent EGL contexts across
forked processes.

**These image-observation numbers are a local WSL2/CPU-rasterization
measurement. They are directional for local debugging (e.g. "is 8 workers
worth it over 1 locally") and are explicitly NOT a promise of throughput on
the cluster's eventual EGL/GPU backend.** A GPU-backed EGL renderer on real
hardware should be substantially faster per frame, and the resolution-
independence seen here may not hold once fill-rate is GPU-bound rather than
geometry-bound. Re-run this script on the cluster before committing to a P4
schedule; don't plan against these hours.

The state-only row has no rendering cost and is backend-independent -- that
number does travel.
