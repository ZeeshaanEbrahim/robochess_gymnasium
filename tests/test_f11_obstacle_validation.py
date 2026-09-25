"""F11 (Codex, 21 Sep): reset overrides can place two pieces on one square, and
obstacle count validation has gaps.

Measured on the unfixed code (Section 2 of the F11 investigation):

* `reset(options={"obstacle_squares": ...})` is not validated at all. Accepted:
  duplicate squares (two obstacles at one xy); an obstacle on the working
  piece's start square (piece pushed 5.02 mm off the board after one step, the
  obstacle moved 3.98 mm, `obstacles_undisturbed` still True); an obstacle on
  the target square; a count different from `n_obstacles` (extra squares
  silently dropped, `[]` leaves both obstacles parked); a bare string.
* `resolve_config` only checks `n_obstacles` is an int, so -1, 5 and 99 pass.
* The constructor accepts 2.5 (becomes 2) and True (becomes 1), and raises a
  bare TypeError for a string or None.
* NaN is already rejected by both layers (Codex's NaN claim no longer holds);
  it is kept here as a regression guard.

Fix under test: ONE function, `validate_obstacle_config`, called by the
constructor, by reset(), and by resolve_config, so the three cannot drift. The
start/target overlap rule has an opt-out, `allow_piece_overlap`, reachable only
from code (a constructor keyword), never from the config layer.
"""

from __future__ import annotations

import math

import pytest

from robochess.experiment import ConfigError, resolve_config
from robochess.tasks import T2ObstacleTask, make_t2_env
from robochess.tasks import t2_obstacle_pick_and_place as t2mod

CFG = dict(task="t2", algorithm="ppo", reward_type="sparse", observation="state",
           seed=0, total_timesteps=1000)


@pytest.fixture
def base_env():
    e = make_t2_env()
    yield e
    e.close()


def _task(env, n=2, **kw):
    return T2ObstacleTask(env, start_square="b2", target_square="e5", n_obstacles=n, **kw)


# ------------------------------------------------------ (a) reset overrides
@pytest.mark.parametrize(
    "n, squares, message",
    [
        (2, ["g4", "g4"], "distinct"),
        (2, ["b2", "g4"], "start"),
        (2, ["g4", "e5"], "target"),
        (2, ["a4", "b4", "c4"], "entries"),
        (2, ["a4", "b4", "c4", "d4", "f4"], "entries"),
        (2, [], "entries"),
        (1, "g4", "sequence"),
        (2, ["z9", "g4"], "Not a valid square"),
    ],
)
def test_reset_override_is_rejected(base_env, n, squares, message):
    task = _task(base_env, n=n)
    with pytest.raises(ValueError, match=message):
        task.reset(seed=0, options={"obstacle_squares": squares})


def test_a_rejected_reset_leaves_the_task_usable(base_env):
    task = _task(base_env)
    with pytest.raises(ValueError):
        task.reset(seed=0, options={"obstacle_squares": ["b2", "g4"]})
    _, info = task.reset(seed=0, options={"obstacle_squares": ["c3", "g4"]})
    assert info["obstacle_squares"] == ("c3", "g4")


def test_valid_reset_override_is_still_honoured(base_env):
    task = _task(base_env)
    _, info = task.reset(seed=0, options={"obstacle_squares": ("c3", "f6")})
    assert info["obstacle_squares"] == ("c3", "f6")


def test_constructor_rejects_overlap_with_a_fixed_start_or_target(base_env):
    with pytest.raises(ValueError, match="start"):
        _task(base_env, obstacle_squares=("b2", "g4"))
    with pytest.raises(ValueError, match="target"):
        _task(base_env, obstacle_squares=("g4", "e5"))


def test_opt_out_allows_an_obstacle_on_the_target(base_env):
    """D15's measurement-path probe forced an obstacle onto the target square;
    that stays possible, from code, on purpose."""
    task = _task(base_env, allow_piece_overlap=True)
    _, info = task.reset(seed=0, options={"obstacle_squares": ("g4", "e5")})
    assert info["obstacle_squares"] == ("g4", "e5")


def test_opt_out_does_not_relax_the_other_checks(base_env):
    task = _task(base_env, allow_piece_overlap=True)
    with pytest.raises(ValueError, match="distinct"):
        task.reset(seed=0, options={"obstacle_squares": ["g4", "g4"]})


def test_opt_out_is_not_reachable_from_the_config_layer():
    with pytest.raises(ConfigError, match="Unknown config key"):
        resolve_config({**CFG, "allow_piece_overlap": True})


# ---------------------------------------------------------- (b) obstacle count
@pytest.mark.parametrize("bad", [float("nan"), -1, 5, 99, 2.5, True, "2", None])
def test_constructor_rejects_bad_n_obstacles_with_valueerror(base_env, bad):
    with pytest.raises(ValueError):
        T2ObstacleTask(base_env, n_obstacles=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 5, 99, 2.5, True, "2"])
def test_config_layer_rejects_bad_n_obstacles(bad):
    with pytest.raises(ConfigError):
        resolve_config({**CFG, "n_obstacles": bad})


@pytest.mark.parametrize("ok", [0, 1, 2, 3, 4])
def test_config_layer_accepts_every_valid_count(ok):
    assert resolve_config({**CFG, "n_obstacles": ok})["n_obstacles"] == ok


def test_t1_config_still_requires_zero_obstacles():
    with pytest.raises(ConfigError, match="task=t1"):
        resolve_config({**CFG, "task": "t1", "n_obstacles": 1})


# ------------------------------------------- one shared validator, no drift
def test_validator_exists_and_normalises():
    validate = t2mod.validate_obstacle_config
    assert validate(2, ["d4", "e5"]) == (2, ("d4", "e5"))
    assert validate(0, None) == (0, None)
    assert validate(1, ("d4",), start="a1", target="h8") == (1, ("d4",))
    with pytest.raises(ValueError):
        validate(2, ("d4", "a1"), start="a1", target="h8")
    assert validate(2, ("d4", "a1"), start="a1", target="h8", allow_piece_overlap=True)


def test_constructor_reset_and_config_all_call_the_same_validator(base_env, monkeypatch):
    calls = []
    real = t2mod.validate_obstacle_config

    def spy(*args, **kwargs):
        calls.append(kwargs.get("start"))
        return real(*args, **kwargs)

    monkeypatch.setattr(t2mod, "validate_obstacle_config", spy)
    task = _task(base_env, obstacle_squares=("c3", "f6"))
    n_after_ctor = len(calls)
    assert n_after_ctor >= 1
    task.reset(seed=0)
    assert len(calls) > n_after_ctor  # reset() validated too
    n_after_reset = len(calls)
    resolve_config({**CFG, "n_obstacles": 2})
    assert len(calls) > n_after_reset  # and so did the config layer
