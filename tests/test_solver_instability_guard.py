"""D14: regression tests for RoboChessEnv's solver-instability guard.

The guard exists because the contact solver was measured injecting ~563 J into
the tracked piece inside one macro-step while the arm was stationary at a
converged joint target (D14 in DECISIONS.md). That specific trigger, f4->h1,
no longer fires: the same entry's `solref` fix removed it. So these tests do
not wait for physics to misbehave -- they inject the implausible state
directly through MuJoCo's own data structures and assert the guard's contract
on the way out.

That is deliberate. A guard whose only test is "the bug that motivated it"
stops being tested the moment the bug is fixed, which is exactly the state
this file exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from robochess.envs import RoboChessEnv
from robochess.envs.robochess_env import (
    PIECE_ANG_SPEED_SANITY,
    PIECE_KINETIC_ENERGY_SANITY,
    PIECE_LIN_SPEED_SANITY,
)


@pytest.fixture
def env():
    e = RoboChessEnv(max_episode_steps=200)
    yield e
    e.close()


def _set_piece_velocity(env, lin=(0.0, 0.0, 0.0), ang=(0.0, 0.0, 0.0)):
    """Write the tracked piece's spatial velocity straight into MjData."""
    v = env.data.qvel
    v[env._piece_vadr : env._piece_vadr + 3] = lin
    v[env._piece_vadr + 3 : env._piece_vadr + 6] = ang


def _inject_instability(env):
    """Put the piece into an implausible state that SURVIVES the macro-step.

    Injecting spin alone is not enough and the difference is not academic: a
    piece resting on the board sheds angular velocity to contact friction over
    the macro-step's 20 substeps, and how much it sheds depends on the contact
    parameters -- 400 rad/s decays to ~172 rad/s at `solref = 0.004` and stays
    over 200 at `0.008`. A test that injects spin against the board is
    therefore testing the contact model, not the guard, and would silently
    stop exercising the guard the next time those parameters change.

    So the velocity here is directed straight up: the piece leaves contact
    immediately and only gravity acts on it, which removes 0.39 m/s over a
    0.04 s macro-step out of 60 m/s. Nothing about the contact model can
    damp it below the bound.
    """
    _set_piece_velocity(env, lin=(0.0, 0.0, 60.0), ang=(0.0, 0.0, 4000.0))


# ------------------------------------------------------------- the predicate
def test_kinetic_energy_matches_hand_computation(env):
    env.reset(seed=0)
    _set_piece_velocity(env, lin=(3.0, 4.0, 0.0), ang=(0.0, 0.0, 2.0))

    mass = float(env.model.body_mass[env._piece_bid])
    inertia = env.model.body_inertia[env._piece_bid]
    expected = 0.5 * mass * (3.0**2 + 4.0**2) + 0.5 * float(inertia[2]) * 2.0**2

    assert env._piece_kinetic_energy() == pytest.approx(expected, rel=1e-12)


def test_guard_is_quiet_in_normal_operation(env):
    """It has to be silent on ordinary episodes or it is worse than useless:
    a full 315-episode sweep produced zero false positives, and this is the
    unit-level version of that check."""
    _, info = env.reset(seed=0)
    assert info["solver_instability"] is False
    assert env._solver_instability() is False

    for _ in range(10):
        _, _, terminated, truncated, info = env.step(env.home_action())
        assert info["solver_instability"] is False
        assert info["terminated_reason"] != "solver_instability"
        assert truncated is False
        assert info["piece_kinetic_energy_j"] < PIECE_KINETIC_ENERGY_SANITY


@pytest.mark.parametrize(
    "lin, ang, which",
    [
        ((0.0, 0.0, 0.0), (0.0, 0.0, PIECE_ANG_SPEED_SANITY * 1.5), "angular"),
        ((PIECE_LIN_SPEED_SANITY * 1.5, 0.0, 0.0), (0.0, 0.0, 0.0), "linear"),
        # Under both speed bounds, over the energy bound: 9.9 m/s is inside
        # the 10 m/s limit and 199 rad/s inside the 200 rad/s one, but
        # together they carry ~1.04 J. Exercises the energy term on its own.
        # The spin is about x, not z: this piece's transverse inertia
        # (4.27e-6) is 2.6x its polar one (1.65e-6), and about z the same
        # rates fall just under the bound at ~0.99 J.
        ((9.9, 0.0, 0.0), (199.0, 0.0, 0.0), "energy"),
    ],
)
def test_each_threshold_fires_independently(env, lin, ang, which):
    env.reset(seed=0)
    assert env._solver_instability() is False

    _set_piece_velocity(env, lin=lin, ang=ang)
    assert env._solver_instability() is True, f"{which} bound did not fire"

    if which == "energy":
        # Assert it really is the energy term doing the work, not one of the
        # speed bounds sneaking over.
        assert float(np.linalg.norm(lin)) < PIECE_LIN_SPEED_SANITY
        assert float(np.linalg.norm(ang)) < PIECE_ANG_SPEED_SANITY
        assert env._piece_kinetic_energy() > PIECE_KINETIC_ENERGY_SANITY


def test_state_just_inside_the_bounds_does_not_fire(env):
    env.reset(seed=0)
    _set_piece_velocity(
        env,
        lin=(PIECE_LIN_SPEED_SANITY * 0.5, 0.0, 0.0),
        ang=(0.0, 0.0, PIECE_ANG_SPEED_SANITY * 0.5),
    )
    # 5 m/s on a 19.5 g piece is ~0.24 J, still under the 1 J bound.
    assert env._piece_kinetic_energy() < PIECE_KINETIC_ENERGY_SANITY
    assert env._solver_instability() is False


# --------------------------------------------------------------- step contract
def test_guard_truncates_and_labels_the_step(env):
    env.reset(seed=0)
    env.step(env.home_action())

    _inject_instability(env)
    _, _, terminated, truncated, info = env.step(env.home_action())

    assert info["solver_instability"] is True
    assert info["terminated_reason"] == "solver_instability"
    # Truncation, not termination: a solver artefact is not an MDP terminal
    # state, and SB3 treats the two differently.
    assert truncated is True
    assert terminated is False


def test_guard_returns_the_last_known_good_observation(env):
    """The substitution is the point: SB3 bootstraps the value function off
    the final observation when an episode is TRUNCATED, so the corrupted
    state would otherwise reach the value estimate even though the episode
    ended."""
    env.reset(seed=0)
    good_obs, _, _, _, _ = env.step(env.home_action())
    good_obs = {k: v.copy() for k, v in good_obs.items()}

    _inject_instability(env)
    returned, _, _, truncated, info = env.step(env.home_action())

    assert truncated is True and info["solver_instability"] is True
    for key in ("observation", "achieved_goal", "desired_goal"):
        assert np.array_equal(returned[key], good_obs[key]), f"{key} is not the pre-spike value"

    # ...and it is genuinely a substitution, not a no-op: the live piece is
    # metres above where the returned observation says it is. Compared in 3D,
    # since the injected velocity is vertical.
    live = env._achieved_goal()
    assert not np.allclose(live, returned["achieved_goal"], atol=1e-3)
    assert live[2] - returned["achieved_goal"][2] > 1.0


def test_guard_takes_precedence_over_success(env, monkeypatch):
    """`success` must not win over the guard when both would be true.

    The base env's own criterion is forced rather than physically arranged,
    because the two cannot actually co-occur in this env: D9's settle
    requirement resets `_piece_ang_settled_streak` on any step where the piece
    is spinning, and a piece moving fast enough to trip the linear bound
    travels off its goal square within the same macro-step. The precedence in
    `step()` is therefore defensive -- it is asserted here so that it stays
    correct if a future change (a different success criterion, a lower bound)
    makes the combination reachable.
    """
    env.reset(seed=0)
    env.step(env.home_action())
    monkeypatch.setattr(env, "_is_success", lambda: True)

    _inject_instability(env)
    _, _, terminated, truncated, info = env.step(env.home_action())

    assert info["solver_instability"] is True
    assert info["terminated_reason"] == "solver_instability"
    assert info["is_success"] is False
    assert truncated is True
    assert terminated is False


def test_reset_seeds_a_good_observation_before_any_step(env):
    """An instability on the very first step still needs something
    uncorrupted to hand back."""
    obs, _ = env.reset(seed=0)
    _inject_instability(env)
    returned, _, _, truncated, info = env.step(env.home_action())

    assert truncated is True and info["solver_instability"] is True
    assert np.array_equal(returned["achieved_goal"], obs["achieved_goal"])


def test_guard_state_does_not_leak_between_episodes(env):
    """The stored observation is per-episode; a stale one from a previous
    episode would be a different piece position entirely."""
    env.reset(seed=0)
    env.step(env.home_action())
    _inject_instability(env)
    env.step(env.home_action())

    obs, info = env.reset(seed=1)
    assert info["solver_instability"] is False
    _, _, terminated, truncated, info = env.step(env.home_action())
    assert truncated is False
    assert info["solver_instability"] is False
