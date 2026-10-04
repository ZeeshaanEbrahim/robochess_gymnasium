"""`sac_obs_normalize`: observation-only VecNormalize for SAC, parity with PPO.

Default OFF. Allowed for SAC without HER only. Goal keys stay raw, settings are
PPO's (norm_obs_keys ["observation"], clip_obs 10, gamma 0.99) with norm_reward
False. What these tests pin:

* the key's default / validation, and that its HER rejection tracks
  train.HER_CELLS (experiment.py cannot import train, so the rule is repeated);
* build_vec_env: flag unset -> SAC env is the same bare DummyVecEnv as before and
  PPO is untouched; flag set -> the VecNormalize described above;
* every loader that used to gate VecNormalize.load on algorithm == "ppo" now asks
  train.uses_vecnormalize, so a normalised SAC is not silently evaluated raw;
* two SUBPROCESS tests (a real SB3 model in the pytest process segfaults after
  many mujoco tests; see test_repro_check.py): the replay buffer stores RAW
  observations and normalises only at sampling, and train.py -> checkpoints ->
  eval_probe round-trips the statistics for a SAC run.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from robochess.experiment import ConfigError, load_config, resolve_config

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("train", ROOT / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

BASE = dict(task="t1", algorithm="sac", reward_type="shaped", observation="state", seed=0,
            total_timesteps=1, device="cpu", max_episode_steps=15)


def cfg(**over):
    return resolve_config(dict(BASE, **over), config_path="t.yaml")


# ------------------------------------------------------------------ config
def test_default_is_false_on_every_algorithm_and_on_the_d29_sac_config():
    assert cfg()["sac_obs_normalize"] is False
    assert cfg(algorithm="ppo")["sac_obs_normalize"] is False
    path = ROOT / "configs" / "_probe_cart_place29_sac_seed0.yaml"
    assert resolve_config(load_config(path), config_path=path)["sac_obs_normalize"] is False


def test_true_is_accepted_for_sac_without_her():
    assert cfg(sac_obs_normalize=True)["sac_obs_normalize"] is True


@pytest.mark.parametrize("bad", ["yes", 1, 0, None, "true"])
def test_must_be_a_bool(bad):
    with pytest.raises(ConfigError, match="sac_obs_normalize"):
        cfg(sac_obs_normalize=bad)


def test_rejected_with_ppo():
    with pytest.raises(ConfigError, match="SAC only"):
        cfg(algorithm="ppo", sac_obs_normalize=True)


def test_rejected_with_her():
    with pytest.raises(ConfigError, match="HER"):
        cfg(reward_type="sparse", sac_obs_normalize=True)


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
@pytest.mark.parametrize("reward_type", ["shaped", "sparse"])
def test_experiment_rule_matches_train_her_cells(algorithm, reward_type):
    """The rule in experiment.py is `algorithm == sac and not the HER cell`."""
    allowed = algorithm == "sac" and (algorithm, reward_type) not in train.HER_CELLS
    try:
        cfg(algorithm=algorithm, reward_type=reward_type, sac_obs_normalize=True)
        accepted = True
    except ConfigError:
        accepted = False
    assert accepted == allowed


# ----------------------------------------------------------- uses_vecnormalize
def test_uses_vecnormalize_truth_table_and_old_manifests():
    assert train.uses_vecnormalize(cfg(algorithm="ppo")) is True
    assert train.uses_vecnormalize(cfg()) is False
    assert train.uses_vecnormalize(cfg(sac_obs_normalize=True)) is True
    # a manifest's config_resolved written before the key existed
    old = {k: v for k, v in cfg().items() if k != "sac_obs_normalize"}
    assert train.uses_vecnormalize(old) is False
    assert train.uses_vecnormalize(dict(old, algorithm="ppo")) is True


# ------------------------------------------------------------- build_vec_env
def test_flag_unset_sac_env_is_the_bare_dummyvecenv():
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    env = train.build_vec_env(cfg())
    try:
        assert type(env) is DummyVecEnv and not isinstance(env, VecNormalize)
    finally:
        env.close()


def test_flag_set_sac_env_is_observation_only_vecnormalize_with_ppos_settings():
    from stable_baselines3.common.vec_env import VecNormalize
    env = train.build_vec_env(cfg(sac_obs_normalize=True))
    try:
        assert isinstance(env, VecNormalize)
        assert env.norm_obs is True and env.norm_reward is False
        assert env.norm_obs_keys == ["observation"] == train.VECNORMALIZE_OBS_KEYS
        assert set(env.obs_rms) == {"observation"}          # goal keys carry no statistics
        assert env.clip_obs == 10.0 and env.gamma == 0.99
        assert env.training is True
        # reward is passed through untouched
        env.reset()
        _, r, _, _ = env.step([env.action_space.sample()])
        assert r[0] == env.get_original_reward()[0]
    finally:
        env.close()


def test_goal_keys_are_not_normalised_observation_is():
    import numpy as np
    env = train.build_vec_env(cfg(sac_obs_normalize=True))
    try:
        env.reset()
        for _ in range(5):
            obs, _, _, _ = env.step([env.action_space.sample()])
        raw = env.get_original_obs()
        assert np.array_equal(obs["achieved_goal"], raw["achieved_goal"])
        assert np.array_equal(obs["desired_goal"], raw["desired_goal"])
        assert not np.array_equal(obs["observation"], raw["observation"])
    finally:
        env.close()


def test_ppo_env_is_unchanged():
    from stable_baselines3.common.vec_env import VecNormalize
    env = train.build_vec_env(cfg(algorithm="ppo"))
    try:
        assert isinstance(env, VecNormalize)
        assert env.norm_obs and env.norm_reward and env.clip_obs == 10.0 and env.clip_reward == 10.0
        assert env.norm_obs_keys == ["observation"] and env.gamma == 0.99
    finally:
        env.close()


def test_build_vec_env_itself_refuses_her_with_the_flag():
    """Bypassing resolve_config (a hand-built dict) still cannot reach HER + VecNormalize."""
    bad = dict(cfg(), reward_type="sparse", sac_obs_normalize=True)
    with pytest.raises(ValueError, match="HER"):
        train.build_vec_env(bad)


# ------------------------------------------------------------------ loaders
@pytest.mark.parametrize("script", ["eval_probe.py", "eval_final.py", "measure_tracking_error.py"])
def test_loaders_ask_the_helper_not_the_algorithm(script):
    src = (ROOT / "scripts" / script).read_text()
    assert "train.uses_vecnormalize(cfg)" in src
    assert 'if cfg["algorithm"] == "ppo":' not in src       # the old PPO-only gate


def test_manifest_flag_comes_from_the_helper():
    src = (ROOT / "scripts" / "train.py").read_text()
    assert '"vecnormalize": uses_vecnormalize(cfg)' in src
    assert '"vecnormalize": cfg["algorithm"] in VECNORMALIZE_ALGORITHMS' not in src


# --------------------------------------------------------- subprocess tests
_BUFFER_DRIVER = textwrap.dedent(r"""
    import importlib.util, json, sys, warnings
    from pathlib import Path
    import numpy as np
    warnings.simplefilter("ignore")
    root = Path(sys.argv[1]); sys.path.insert(0, str(root))
    from robochess.experiment import resolve_config
    spec = importlib.util.spec_from_file_location("train", root / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
    train.apply_thread_pinning(1)
    from stable_baselines3.common.vec_env import VecNormalize
    cfg = resolve_config(dict(task="t1", algorithm="sac", reward_type="shaped", observation="state", seed=0,
                              total_timesteps=1, device="cpu", max_episode_steps=15, sac_obs_normalize=True),
                         config_path="t.yaml")
    venv = train.build_vec_env(cfg)
    assert isinstance(venv, VecNormalize)
    raw = []                                   # what the inner env really returned, step by step
    inner = venv.venv; orig = inner.step_wait
    def spy():
        o, r, d, i = orig(); raw.append(np.array(o["observation"][0], copy=True)); return o, r, d, i
    inner.step_wait = spy
    model = train.build_model(cfg, venv)
    model.learn(total_timesteps=300)
    buf = model.replay_buffer; n = buf.pos
    assert n == 300 and len(raw) == 300, (n, len(raw))
    nxt = buf.next_observations["observation"][:n, 0]; dones = buf.dones[:n, 0]
    keep = np.flatnonzero(dones == 0)          # a done step stores the terminal obs, not the reset obs the spy saw
    stored_is_raw = bool(np.array_equal(nxt[keep], np.stack(raw)[keep]))
    idx = np.array([0, 1, 2, 7, 20])
    sampled = buf._get_samples(idx, env=venv)
    expect = venv.normalize_obs({k: v[idx, 0] for k, v in buf.observations.items()})
    sample_is_normalised = all(np.array_equal(sampled.observations[k].numpy(), expect[k]) for k in expect)
    sample_differs_from_raw = not np.array_equal(sampled.observations["observation"].numpy(),
                                                 buf.observations["observation"][idx, 0])
    goals_raw = all(np.array_equal(sampled.observations[k].numpy(), buf.observations[k][idx, 0])
                    for k in ("achieved_goal", "desired_goal"))
    rewards_raw = bool(np.array_equal(sampled.rewards.numpy().ravel(), buf.rewards[idx, 0]))
    st = venv.obs_rms["observation"]
    print(json.dumps(dict(stored_is_raw=stored_is_raw, n_compared=int(len(keep)),
                          sample_is_normalised=sample_is_normalised, sample_differs_from_raw=sample_differs_from_raw,
                          goals_raw=goals_raw, rewards_raw=rewards_raw, count=float(st.count),
                          has_stats=bool(np.any(st.var > 0)))))
""")


def test_replay_buffer_stores_raw_observations_and_normalises_at_sampling():
    p = subprocess.run([sys.executable, "-c", _BUFFER_DRIVER, str(ROOT)], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    r = json.loads(p.stdout.strip().splitlines()[-1])
    assert r["stored_is_raw"] and r["n_compared"] > 250
    assert r["sample_is_normalised"] and r["sample_differs_from_raw"]
    assert r["goals_raw"] and r["rewards_raw"]
    assert r["has_stats"] and r["count"] > 300


def test_train_py_saves_stats_for_sac_and_eval_probe_loads_them(tmp_path):
    """train.py -> manifest, checkpoint and final pickles -> eval_probe for a SAC run."""
    out_root = tmp_path / "runs"
    conf = tmp_path / "sac_obsnorm_tiny.yaml"
    conf.write_text(textwrap.dedent(f"""
        task: t1
        algorithm: sac
        reward_type: shaped
        observation: state
        seed: 0
        total_timesteps: 300
        device: cpu
        max_episode_steps: 15
        checkpoint_freq: 200
        sac_obs_normalize: true
        output_root: {out_root}
        run_name: tiny
    """))
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(conf), "--allow-dirty"],
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, (p.stdout + p.stderr)[-2000:]
    run = out_root / "tiny"
    m = json.loads((run / "manifest.json").read_text())
    assert m["resolved"]["vecnormalize"] is True and m["config_resolved"]["sac_obs_normalize"] is True
    assert (run / "final_vecnormalize.pkl").is_file()
    assert (run / "checkpoints" / "rc_vecnormalize_200_steps.pkl").is_file()
    for out_name, extra in (("final", []), ("ck", ["--checkpoint", "200"])):
        out = tmp_path / f"{out_name}.json"
        q = subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_probe.py"), "tiny", "--runs-dir", str(out_root),
                            "--episodes", "2", "--out", str(out), *extra],
                           capture_output=True, text=True, timeout=600)
        assert q.returncode == 0, (q.stdout + q.stderr)[-2000:]
        # None would mean eval_probe built a bare env and never loaded any statistics
        assert json.loads(out.read_text())["obs_rms_identical_before_after"] is True
