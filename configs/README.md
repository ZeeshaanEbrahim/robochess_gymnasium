# Config conventions

No config files exist yet — they arrive with Phase P2/P3. This file records the
standing defaults to apply when they are written.

## Standing default: `device`

| Observation type | Policy | `device` | Rationale |
| --- | --- | --- | --- |
| State vector | `MlpPolicy` | `"cpu"` (explicit) | CPU measurably faster — see below |
| Image | `CnnPolicy` | omit (resolves to CUDA) | GPU wins on conv workloads |

Applies to **both PPO and SAC**, and to any other MLP-policy algorithm added later.

Paste this line above `device` in every state-vector config:

```yaml
# device: cpu — MLP policies train faster on CPU than GPU here (scripts/benchmark_device.py: 42.07s vs 133.56s, CPU 3.17x faster); leave image/CNN configs on the default so they resolve to CUDA.
```

### Why

`scripts/benchmark_device.py` trains PPO (`MlpPolicy`) on `Pendulum-v1` for
20,000 timesteps on each device with a fixed seed, timing only `model.learn()`:

```
CPU  wall-clock:   42.07 s
CUDA wall-clock:  133.56 s
CPU was 3.17x faster than CUDA.
```

Measured on an RTX 2060 under WSL2 with torch 2.13.0+cu130. MLP policies issue
many tiny forward/backward passes, so per-call host<->device transfer and launch
overhead dominates and the GPU's parallelism goes unused. Stable-Baselines3
raises a `UserWarning` to this effect; the benchmark confirms it on this machine.

This is a measured result on one machine, not a universal rule — re-run the
benchmark before assuming it holds on the cluster, where the CPU/GPU balance and
interconnect differ.
