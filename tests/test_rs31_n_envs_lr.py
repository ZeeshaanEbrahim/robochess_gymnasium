"""RS31: `n_envs` (parallel environments, PPO only) and `ppo_lr_schedule`
("constant" or "linear" learning rate, PPO only).

Expected results, stated before the run:
  * n_envs defaults to 1, must be an int >= 1 (not bool), is PPO only above 1, and
    must divide checkpoint_freq;
  * n_envs 1 builds the same DummyVecEnv with the same Monitor file (logs/monitor);
    n_envs 4 builds a SubprocVecEnv with four Monitor files logs/monitor_{0..3};
  * with n_envs > 1 the env seeds are seed + i, so the per-env pair sequences differ,
    and each equals a single env reset with seed + i;
  * a real n_envs 2 run writes checkpoints at checkpoint_freq multiples of TOTAL steps;
  * ppo_lr_schedule defaults to "constant" (SB3's 3e-4, no learning_rate passed);
    "linear" is 3e-4 * progress_remaining; it is PPO only and unknown values are
    rejected; the manifest records both keys and the built model's hyperparameters.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from robochess.experiment import ConfigError, resolve_config

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
CONFIG = REPO / "configs" / "_probe_cart_rs31C_ppotanh_seed0.yaml"
SAC_CONFIG = REPO / "configs" / "_probe_cart_place29_sac_seed0.yaml"


def _raw(**over):
    return {**yaml.safe_load(CONFIG.read_text()), **over}


def _sac_raw(**over):
    return {**yaml.safe_load(SAC_CONFIG.read_text()), **over}


def _tensorboard_importable() -> bool:
    try:
        import torch.utils.tensorboard  # noqa: F401
    except ImportError:
        return False
    return True


# Same convention as tests/test_rs29_eval_pairs.py.
requires_tensorboard = pytest.mark.skipif(
    not _tensorboard_importable(),
    reason="train.py's tensorboard logger needs tensorboard, which requirements-ci.txt "
           "does not install (requirements.txt pins it)")


# ---------------------------------------------------------------- n_envs: config
def test_n_envs_default_is_one():
    assert resolve_config(_raw())["n_envs"] == 1


@pytest.mark.parametrize("bad", [0, -1, True, "4", 2.0, None])
def test_n_envs_rejects_bad_values(bad):
    with pytest.raises(ConfigError, match="n_envs"):
        resolve_config(_raw(n_envs=bad))


def test_n_envs_above_one_rejected_for_sac():
    with pytest.raises(ConfigError, match="n_envs=2 with algorithm"):
        resolve_config(_sac_raw(n_envs=2))
    assert resolve_config(_sac_raw(n_envs=1))["n_envs"] == 1


def test_checkpoint_freq_must_be_divisible_by_n_envs():
    with pytest.raises(ConfigError, match="not divisible"):
        resolve_config(_raw(n_envs=3))                        # 100000 % 3 != 0
    assert resolve_config(_raw(n_envs=4))["n_envs"] == 4


# ---------------------------------------------------------------- n_envs: vec env
def test_n_envs_one_is_dummy_vec_env_with_the_old_monitor_path(tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv

    from scripts.train import build_vec_env

    venv = build_vec_env(resolve_config(_raw()), monitor_dir=tmp_path)
    try:
        assert type(venv.venv) is DummyVecEnv and venv.num_envs == 1
        assert sorted(p.name for p in tmp_path.iterdir()) == ["monitor.monitor.csv"]
    finally:
        venv.close()


def test_n_envs_four_is_subproc_vec_env_with_four_monitor_files(tmp_path):
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from scripts.train import build_vec_env

    venv = build_vec_env(resolve_config(_raw(n_envs=4)), monitor_dir=tmp_path)
    try:
        assert type(venv.venv) is SubprocVecEnv and venv.num_envs == 4
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            f"monitor_{i}.monitor.csv" for i in range(4)]
    finally:
        venv.close()


def _pairs(info):
    return f"{info['start_square']}:{info['target_square']}"


def test_n_envs_four_pair_sequences_differ_and_follow_seed_plus_i():
    from scripts.train import build_env, build_vec_env

    cfg = resolve_config(_raw(n_envs=4))
    venv = build_vec_env(cfg)
    try:
        venv.seed(7)                     # what PPO(seed=7) does via set_random_seed
        venv.reset()
        seqs = [[_pairs(info)] for info in venv.venv.reset_infos]
        for _ in range(9):
            for i, (_, info) in enumerate(venv.env_method("reset")):
                seqs[i].append(_pairs(info))
    finally:
        venv.close()
    for i, s in enumerate(seqs):
        print(f"env {i} (seed {7 + i}): {' '.join(s)}")
    assert len({tuple(s) for s in seqs}) == 4

    single = build_env(cfg)              # env i is a single env reset with seed 7 + i
    try:
        for i, s in enumerate(seqs):
            ref = [_pairs(single.reset(seed=7 + i)[1])] + [_pairs(single.reset()[1]) for _ in range(9)]
            assert ref == s
    finally:
        single.close()


# ---------------------------------------------------------------- ppo_lr_schedule
def test_lr_schedule_default_is_constant():
    assert resolve_config(_raw())["ppo_lr_schedule"] == "constant"


@pytest.mark.parametrize("bad", ["cosine", "Linear", None, 1])
def test_lr_schedule_rejects_unknown_values(bad):
    with pytest.raises(ConfigError, match="ppo_lr_schedule"):
        resolve_config(_raw(ppo_lr_schedule=bad))


@pytest.mark.parametrize("value", ["linear", "constant"])
def test_lr_schedule_rejected_for_sac(value):
    with pytest.raises(ConfigError, match="ppo_lr_schedule with algorithm"):
        resolve_config(_sac_raw(ppo_lr_schedule=value))


def test_ppo_lr_is_sb3s_default():
    import inspect

    from stable_baselines3 import PPO

    from scripts.train import PPO_LR

    assert PPO_LR == inspect.signature(PPO).parameters["learning_rate"].default


def _built_lr(**over):
    from scripts.train import build_model, build_vec_env

    venv = build_vec_env(resolve_config(_raw(**over)))
    try:
        model = build_model(resolve_config(_raw(**over)), venv)
        return model.learning_rate, [model.lr_schedule(p) for p in (1.0, 0.5, 0.0)]
    finally:
        venv.close()


def test_constant_builds_with_sb3s_default_learning_rate():
    lr, values = _built_lr()
    assert lr == 3e-4 and values == [3e-4, 3e-4, 3e-4]


def test_linear_values_at_progress_one_half_zero():
    from scripts.train import linear_lr

    lr, values = _built_lr(ppo_lr_schedule="linear")
    assert lr is linear_lr
    assert values == [3e-4, 1.5e-4, 0.0]


# ---------------------------------------------------------------- real runs
def _run(args):
    return subprocess.run([PY, *map(str, args)], cwd=REPO, capture_output=True, text=True)


@requires_tensorboard
def test_n_envs_two_checkpoints_land_at_total_step_multiples(tmp_path):
    """One rollout of 2 x 2048 steps, checkpoint_freq 1024 -> files at 1024..4096 total steps."""
    cfg = _raw(n_envs=2, ppo_lr_schedule="linear", total_timesteps=4096, max_episode_steps=40,
               checkpoint_freq=1024, output_root=str(tmp_path), run_name="tiny")
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    r = _run(["scripts/train.py", "--config", cfg_path, "--allow-dirty"])
    assert r.returncode == 0, r.stderr[-2000:]
    run = tmp_path / "tiny"
    names = sorted(p.name for p in (run / "checkpoints").iterdir())
    steps = [1024, 2048, 3072, 4096]
    assert names == sorted([f"rc_{s}_steps.zip" for s in steps]
                           + [f"rc_vecnormalize_{s}_steps.pkl" for s in steps])
    assert sorted(p.name for p in (run / "logs").glob("*.monitor.csv")) == [
        "monitor_0.monitor.csv", "monitor_1.monitor.csv"]

    m = json.loads((run / "manifest.json").read_text())
    assert m["config_resolved"]["n_envs"] == 2 and m["resolved"]["n_envs"] == 2
    assert m["config_resolved"]["ppo_lr_schedule"] == "linear"
    sb3 = m["resolved"]["sb3"]
    assert sb3["n_envs"] == 2 and sb3["n_steps"] == 2048 and sb3["batch_size"] == 64
    assert sb3["ent_coef"] == 0.01
    assert sb3["learning_rate"] == {"schedule": "linear", "at_1.0": 3e-4, "at_0.5": 1.5e-4, "at_0.0": 0.0}


@requires_tensorboard
def test_manifest_records_constant_schedule_and_n_envs_one(tmp_path):
    cfg = _raw(total_timesteps=64, max_episode_steps=20, output_root=str(tmp_path), run_name="tiny1")
    cfg_path = tmp_path / "tiny1.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    r = _run(["scripts/train.py", "--config", cfg_path, "--allow-dirty"])
    assert r.returncode == 0, r.stderr[-2000:]
    run = tmp_path / "tiny1"
    m = json.loads((run / "manifest.json").read_text())
    assert m["config_resolved"]["n_envs"] == 1 and m["resolved"]["n_envs"] == 1
    assert m["config_resolved"]["ppo_lr_schedule"] == "constant"
    assert m["resolved"]["sb3"]["learning_rate"]["at_0.0"] == 3e-4
    assert [p.name for p in (run / "logs").glob("*.monitor.csv")] == ["monitor.monitor.csv"]
