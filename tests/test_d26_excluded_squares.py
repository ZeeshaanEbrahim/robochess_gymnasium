"""D26: `excluded_squares` -- d1/e1 are not used by the task in any role.

Expected results, stated before the run:
  * default (empty) exclusion leaves sampling bit-identical: the 10,000-pair
    hash equals the one measured on master a42dfa7, before the feature existed;
  * 10,000 T1 samples and 10,000 T2 resets with ["d1", "e1"] never put a start,
    target or obstacle on d1/e1;
  * invalid names, a bare string, duplicates, and an excluded fixed start/target
    (constructor, per-episode option, config) or fixed obstacle list
    (constructor, per-episode option, with or without allow_piece_overlap) raise;
  * with the default empty list the same overrides still work.
"""

from __future__ import annotations

import hashlib

import pytest

from robochess.envs import RoboChessEnv
from robochess.experiment import ConfigError, resolve_config
from robochess.tasks import T1PickAndPlaceTask, T2ObstacleTask, make_t2_env
from robochess.tasks.squares import ALL_SQUARES

EXCL = ["d1", "e1"]
# Measured on master a42dfa7 before any D26 edit (results/d26_excluded_squares/).
BASELINE_PAIRS_SHA256 = "868586369d198f25602f4a1f2851b75ff878636befc2c659bc3969e5ca24f28e"


@pytest.fixture(scope="module")
def t1_base():
    e = RoboChessEnv()
    yield e
    e.close()


@pytest.fixture(scope="module")
def t2_base():
    e = make_t2_env()
    yield e
    e.close()


def _pairs_hash(task, base, n=10_000, seed=123):
    base.reset(seed=seed)
    pairs = [task._resolve_squares(None, None) for _ in range(n)]
    return len(pairs), hashlib.sha256(repr(pairs).encode()).hexdigest()


# ------------------------------------------------------------ default unchanged
def test_default_empty_sampling_is_identical_to_pre_d26(t1_base):
    task = T1PickAndPlaceTask(t1_base)
    assert task.reachable_squares == ALL_SQUARES  # same contents, same order
    assert _pairs_hash(task, t1_base) == (10_000, BASELINE_PAIRS_SHA256)


def test_explicit_empty_equals_omitted(t1_base):
    a = T1PickAndPlaceTask(t1_base)
    b = T1PickAndPlaceTask(t1_base, excluded_squares=[])
    assert a.reachable_squares == b.reachable_squares
    assert _pairs_hash(b, t1_base) == (10_000, BASELINE_PAIRS_SHA256)


def test_pool_is_order_preserving(t1_base):
    task = T1PickAndPlaceTask(t1_base, excluded_squares=EXCL)
    assert task.reachable_squares == tuple(s for s in ALL_SQUARES if s not in EXCL)
    assert len(task.reachable_squares) == 62


# ------------------------------------------------------------- never sampled
def test_t1_10000_samples_never_use_excluded(t1_base):
    task = T1PickAndPlaceTask(t1_base, excluded_squares=EXCL)
    t1_base.reset(seed=7)
    for _ in range(10_000):
        s, t = task._resolve_squares(None, None)
        assert s not in EXCL and t not in EXCL and s != t


def test_t1_half_fixed_sampling_avoids_excluded(t1_base):
    task = T1PickAndPlaceTask(t1_base, excluded_squares=EXCL)
    t1_base.reset(seed=8)
    for _ in range(500):
        assert task._resolve_squares("a1", None)[1] not in EXCL
        assert task._resolve_squares(None, "a1")[0] not in EXCL


def test_t1_10000_real_resets_never_start_or_end_on_excluded(t1_base):
    task = T1PickAndPlaceTask(t1_base, excluded_squares=EXCL)
    for seed in range(10_000):
        _, info = task.reset(seed=seed)
        assert info["start_square"] not in EXCL
        assert info["target_square"] not in EXCL


def test_t2_10000_resets_no_start_target_or_obstacle_on_excluded(t2_base):
    task = T2ObstacleTask(t2_base, n_obstacles=4, excluded_squares=EXCL)
    for seed in range(10_000):
        task.reset(seed=seed)
        used = {task.current_start_square, task.current_target_square,
                *task.current_obstacle_squares}
        assert not used & set(EXCL), (seed, used)


# ------------------------------------------------------------------- rejects
@pytest.mark.parametrize("bad", [["z9"], ["d1", "d1"], "d1", ["D1"], [1], None, ["d1", None]])
def test_invalid_excluded_squares_raise(t1_base, bad):
    with pytest.raises(ValueError):
        T1PickAndPlaceTask(t1_base, excluded_squares=bad)


@pytest.mark.parametrize("kw", [{"start_square": "d1"}, {"target_square": "e1"}])
def test_excluded_fixed_square_in_constructor_raises(t1_base, kw):
    with pytest.raises(ValueError, match="excluded square"):
        T1PickAndPlaceTask(t1_base, excluded_squares=EXCL, **kw)


@pytest.mark.parametrize("key", ["start_square", "target_square"])
def test_excluded_per_episode_override_raises(t1_base, key):
    task = T1PickAndPlaceTask(t1_base, excluded_squares=EXCL)
    with pytest.raises(ValueError, match="excluded square"):
        task.reset(seed=0, options={key: "d1"})


def test_same_override_works_with_default_empty(t1_base):
    task = T1PickAndPlaceTask(t1_base)
    _, info = task.reset(seed=0, options={"start_square": "d1", "target_square": "e1"})
    assert (info["start_square"], info["target_square"]) == ("d1", "e1")


def test_override_outside_the_pool_raises(t1_base):
    task = T1PickAndPlaceTask(t1_base)
    with pytest.raises(ValueError, match="not in reachable_squares"):
        task.reset(seed=0, options={"start_square": "z9"})


# ------------------------------------------------------------------ obstacles
@pytest.mark.parametrize("overlap", [False, True])
def test_fixed_obstacle_list_with_excluded_square_raises_from_constructor(t2_base, overlap):
    with pytest.raises(ValueError, match="excluded square"):
        T2ObstacleTask(t2_base, n_obstacles=2, obstacle_squares=("e1", "a5"),
                       excluded_squares=EXCL, allow_piece_overlap=overlap)


@pytest.mark.parametrize("overlap", [False, True])
def test_fixed_obstacle_list_with_excluded_square_raises_per_episode(t2_base, overlap):
    task = T2ObstacleTask(t2_base, n_obstacles=2, excluded_squares=EXCL,
                          allow_piece_overlap=overlap)
    with pytest.raises(ValueError, match="excluded square"):
        task.reset(seed=0, options={"obstacle_squares": ["e1", "a5"]})


def test_obstacle_on_e1_still_works_with_default_empty(t2_base):
    task = T2ObstacleTask(t2_base, n_obstacles=2, obstacle_squares=("e1", "a5"))
    task.reset(seed=0)
    assert task.current_obstacle_squares == ("e1", "a5")


def test_t2_excluded_fixed_start_raises(t2_base):
    with pytest.raises(ValueError, match="excluded square"):
        T2ObstacleTask(t2_base, start_square="d1", excluded_squares=EXCL)


# --------------------------------------------------------------------- config
_RAW = None


def _raw(**over):
    from pathlib import Path

    import yaml

    global _RAW
    if _RAW is None:
        path = sorted(Path("configs").glob("*.yaml"))[0]
        _RAW = yaml.safe_load(path.read_text())
    return {**_RAW, **over}


def test_config_default_is_empty_list():
    assert resolve_config(_raw())["excluded_squares"] == []


def test_config_accepts_and_normalises_excluded():
    assert resolve_config(_raw(excluded_squares=["d1", "e1"]))["excluded_squares"] == EXCL


@pytest.mark.parametrize("over", [
    {"excluded_squares": ["z9"]},
    {"excluded_squares": "d1"},
    {"excluded_squares": ["d1", "e1"], "start_square": "d1"},
    {"excluded_squares": ["d1", "e1"], "target_square": "e1"},
])
def test_config_rejects_bad_exclusions(over):
    with pytest.raises(ConfigError):
        resolve_config(_raw(**over))


def test_build_env_passes_exclusion_through():
    from scripts.train import build_env

    cfg = resolve_config(_raw(excluded_squares=EXCL, task="t1", n_obstacles=0))
    env = build_env(cfg)
    try:
        assert env.unwrapped is not None
        t = env
        while not isinstance(t, T1PickAndPlaceTask):
            t = t.env
        assert t.excluded_squares == ("d1", "e1")
    finally:
        env.close()
