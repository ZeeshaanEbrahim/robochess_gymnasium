"""RS30: `train_pairs` (a fixed list of (start, target) pairs sampled uniformly per
episode) and `ppo_ent_coef` (PPO entropy coefficient).

Expected results, stated before the run:
  * with train_pairs unset (or None) the 10,000-pair sampling hash equals the D26
    pre-feature baseline, i.e. no extra RNG draw and an unchanged pool;
  * with it set, draws come only from the list, roughly uniformly, reproducibly
    under a seed; a per-episode `options` pair still wins;
  * bad pairs raise in the constructor and in resolve_config, as do combinations with
    start_square/target_square and task t2;
  * ppo_ent_coef defaults to 0.0, must be >= 0 and is PPO only, and reaches the PPO model.
"""

from __future__ import annotations

import collections
import hashlib
from pathlib import Path

import pytest
import yaml

from robochess.envs import RoboChessEnv
from robochess.experiment import ConfigError, resolve_config
from robochess.tasks import T1PickAndPlaceTask

PAIRS = [("e4", "e5"), ("d4", "d5"), ("f4", "f5"), ("e3", "e4")]
PAIR_STRS = ["e4:e5", "d4:d5", "f4:f5", "e3:e4"]
# D26's pre-feature baseline (10,000 draws, seed 123), see tests/test_d26_excluded_squares.py.
BASELINE_PAIRS_SHA256 = "868586369d198f25602f4a1f2851b75ff878636befc2c659bc3969e5ca24f28e"
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "_probe_cart_place29_ppotanh_seed0.yaml"


@pytest.fixture(scope="module")
def t1_base():
    e = RoboChessEnv()
    yield e
    e.close()


def _draw(task, base, n, seed=123):
    base.reset(seed=seed)
    return [task._resolve_squares(None, None) for _ in range(n)]


def _raw(**over):
    raw = yaml.safe_load(CONFIG.read_text())
    raw.pop("start_square"), raw.pop("target_square")
    return {**raw, **over}


# ---------------------------------------------------------------- 1. unset unchanged
@pytest.mark.parametrize("kw", [{}, {"train_pairs": None}])
def test_unset_sampling_is_identical_to_baseline(t1_base, kw):
    task = T1PickAndPlaceTask(t1_base, **kw)
    assert task.train_pairs is None
    pairs = _draw(task, t1_base, 10_000)
    assert hashlib.sha256(repr(pairs).encode()).hexdigest() == BASELINE_PAIRS_SHA256


# ---------------------------------------------------------------- 2. sampling
def test_draws_come_from_the_list_uniformly_and_reproducibly(t1_base):
    task = T1PickAndPlaceTask(t1_base, train_pairs=PAIRS, excluded_squares=["d1", "e1"])
    a = _draw(task, t1_base, 4000)
    assert set(a) == set(PAIRS)
    counts = collections.Counter(a)
    assert all(abs(c - 1000) < 120 for c in counts.values()), counts   # ~7 sigma
    assert a == _draw(task, t1_base, 4000)                              # same seed, same draws


def test_reset_samples_all_pairs_and_records_them(t1_base):
    task = T1PickAndPlaceTask(t1_base, train_pairs=PAIRS)
    seen = set()
    task.reset(seed=0)
    for _ in range(60):
        _, info = task.reset()
        seen.add((info["start_square"], info["target_square"]))
    assert seen == set(PAIRS)


# ---------------------------------------------------------------- 3. options win
def test_options_pair_beats_train_pairs(t1_base):
    task = T1PickAndPlaceTask(t1_base, train_pairs=PAIRS)
    for _ in range(5):
        _, info = task.reset(seed=1, options={"start_square": "a2", "target_square": "g7"})
        assert (info["start_square"], info["target_square"]) == ("a2", "g7")


# ---------------------------------------------------------------- 4. constructor validation
@pytest.mark.parametrize("bad", [
    [("e4", "e4")],                       # start == target
    [("z9", "e5")],                       # not a square
    [("d1", "e5")],                       # excluded
    [("e4", "e5"), ("e4", "e5")],         # duplicate
    [],                                   # empty
    ["e4e5"],                             # not a pair
    [("e4", 5)],                          # not strings
])
def test_constructor_rejects_bad_train_pairs(t1_base, bad):
    with pytest.raises(ValueError):
        T1PickAndPlaceTask(t1_base, train_pairs=bad, excluded_squares=["d1", "e1"])


@pytest.mark.parametrize("kw", [{"start_square": "e4"}, {"target_square": "e5"}])
def test_constructor_rejects_train_pairs_with_fixed_square(t1_base, kw):
    with pytest.raises(ValueError):
        T1PickAndPlaceTask(t1_base, train_pairs=PAIRS, **kw)


# ---------------------------------------------------------------- 5. config validation
def test_config_default_is_none_and_list_is_accepted():
    assert resolve_config(_raw())["train_pairs"] is None
    cfg = resolve_config(_raw(train_pairs=PAIR_STRS, excluded_squares=["d1", "e1"]))
    assert cfg["train_pairs"] == PAIR_STRS


@pytest.mark.parametrize("over", [
    {"train_pairs": "e4:e5"},                                   # not a list
    {"train_pairs": []},
    {"train_pairs": ["e4e5"]},                                  # malformed
    {"train_pairs": ["e4:e5:e6"]},
    {"train_pairs": ["e4:e4"]},                                 # start == target
    {"train_pairs": ["z9:e5"]},                                 # not a square
    {"train_pairs": ["d1:e5"], "excluded_squares": ["d1", "e1"]},
    {"train_pairs": ["e4:e5", "e4:e5"]},                        # duplicate
    {"train_pairs": ["e4:e5"], "start_square": "e4"},
    {"train_pairs": ["e4:e5"], "target_square": "e5"},
    {"train_pairs": ["e4:e5"], "task": "t2"},
])
def test_config_rejects_bad_train_pairs(over):
    with pytest.raises(ConfigError):
        resolve_config(_raw(**over))


def test_build_env_passes_train_pairs_through():
    from scripts.train import build_env

    cfg = resolve_config(_raw(train_pairs=PAIR_STRS))
    env = build_env(cfg)
    try:
        t = env
        while not isinstance(t, T1PickAndPlaceTask):
            t = t.env
        assert t.train_pairs == tuple(PAIRS)
        env.reset(seed=3)
    finally:
        env.close()


def test_build_env_without_train_pairs_is_unchanged():
    from scripts.train import build_env

    env = build_env(resolve_config(_raw()))
    try:
        t = env
        while not isinstance(t, T1PickAndPlaceTask):
            t = t.env
        assert t.train_pairs is None
    finally:
        env.close()


# ---------------------------------------------------------------- ppo_ent_coef
def test_ent_coef_default_is_zero():
    assert resolve_config(_raw())["ppo_ent_coef"] == 0.0


def test_ent_coef_accepts_non_negative_and_normalises_to_float():
    assert resolve_config(_raw(ppo_ent_coef=0.01))["ppo_ent_coef"] == 0.01
    c = resolve_config(_raw(ppo_ent_coef=0))
    assert c["ppo_ent_coef"] == 0.0 and isinstance(c["ppo_ent_coef"], float)


@pytest.mark.parametrize("bad", [-0.01, -1, "0.01", None, True])
def test_ent_coef_rejects_bad_values(bad):
    with pytest.raises(ConfigError):
        resolve_config(_raw(ppo_ent_coef=bad))


def test_ent_coef_rejected_for_sac():
    raw = _raw(algorithm="sac", ppo_ent_coef=0.01)
    raw.pop("ppo_mean_squash", None)
    with pytest.raises(ConfigError, match="ppo_ent_coef with algorithm"):
        resolve_config(raw)


@pytest.mark.parametrize("value, expected", [(None, 0.0), (0.01, 0.01)])
def test_built_ppo_model_has_the_configured_ent_coef(value, expected):
    from scripts.train import build_model, build_vec_env

    raw = _raw() if value is None else _raw(ppo_ent_coef=value)
    cfg = resolve_config(raw)
    venv = build_vec_env(cfg)
    try:
        assert build_model(cfg, venv).ent_coef == expected
    finally:
        venv.close()


def test_ppo_ent_coef_never_reaches_sac():
    """The D29 SAC config omits ppo_ent_coef: its model keeps SB3's learned ('auto') entropy coefficient."""
    from scripts.train import build_model, build_vec_env

    raw = yaml.safe_load((CONFIG.parent / "_probe_cart_place29_sac_seed0.yaml").read_text())
    assert "ppo_ent_coef" not in raw
    cfg = resolve_config(raw)
    assert cfg["algorithm"] == "sac" and cfg["ppo_ent_coef"] == 0.0   # the default is resolved, never used
    venv = build_vec_env(cfg)
    try:
        model = build_model(cfg, venv)
        assert model.ent_coef == "auto"
        assert model.log_ent_coef is not None and model.log_ent_coef.requires_grad
    finally:
        venv.close()
