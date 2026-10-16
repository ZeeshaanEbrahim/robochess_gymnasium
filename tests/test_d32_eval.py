"""D32 evaluation recorder (scripts/d32_eval.py): latch cause per T2 episode, lift shares per placed episode.

The cause classification is tested on synthetic rows; the measurement path (the same task methods the
env's own latch is built from) is tested on a real T2 task with an obstacle moved or tipped by a direct
qpos write. scripts/eval_probe.py's flag-off identity is checked against a real run, not in this file."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from robochess.tasks import T2ObstacleTask, make_t2_env

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("d32_eval", _ROOT / "scripts" / "d32_eval.py")
d32 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d32)

DISP_TOL, TILT_TOL = 5.0, 0.26


def row(disp=0.0, tilt=0.0, offb=False, latch=False):
    return dict(disp_mm=disp, tilt=tilt, offb=offb, latch=latch)


@pytest.mark.parametrize("at_latch, cause", [
    (row(0.5, 0.29, latch=True), "tilt"),            # what the four d4/f4/d5/f5 contact runs did
    (row(8.0, 0.05, latch=True), "displacement"),
    (row(8.0, 0.40, latch=True), "both"),
    (row(1.0, 0.10, offb=True, latch=True), "off_board"),
    (row(1.0, 0.10, latch=True), "unresolved"),
])
def test_cause_is_read_at_the_first_latch_step(at_latch, cause):
    rows = [row(0.0, 0.0), row(0.1, 0.1), at_latch, row(30.0, 1.5, latch=True)]   # later steps must not matter
    s = d32.obstacle_episode_summary(rows, DISP_TOL, TILT_TOL)
    assert s["latch_fired"] and s["latch_step"] == 3 and s["cause"] == cause
    assert s["disp_at_latch_mm"] == at_latch["disp_mm"] and s["tilt_at_latch_rad"] == at_latch["tilt"]
    assert s["peak_disp_mm"] == 30.0 and s["peak_tilt_rad"] == 1.5


def test_no_latch_has_no_cause_but_still_records_the_peaks():
    s = d32.obstacle_episode_summary([row(0.2, 0.01), row(0.4, 0.02)], DISP_TOL, TILT_TOL)
    assert s["latch_fired"] is False and s["cause"] is None and s["latch_step"] is None
    assert s["peak_disp_mm"] == 0.4 and s["peak_tilt_rad"] == 0.02


def ep(reason, lift, cause=None, fired=None):
    e = dict(reason=reason, max_lift_mm=lift)
    fired = (cause is not None) if fired is None else fired
    e["d32"] = dict(obstacle=dict(latch_fired=fired, cause=cause))
    return e


def test_lift_shares_are_among_placed_episodes_only_and_use_strict_thresholds():
    eps = [ep("success", 4.0), ep("success", 10.0), ep("success", 12.0), ep("success", 51.0),
           ep("success_obstacles_disturbed", 80.0, "tilt"), ep("bad_contact", 99.0), ep("time_limit", 0.0)]
    r = d32.d32_report(eps, is_t2=False)
    assert r["n"] == 7 and r["success"] == 4 and r["placement_only"] == 5
    assert r["lift_among_success"]["gt10mm"] == dict(k=2, n=4, share=0.5)       # 10.0 is not above 10
    assert r["lift_among_success"]["gt50mm"] == dict(k=1, n=4, share=0.25)
    assert r["lift_among_placement_only"]["gt50mm"] == dict(k=2, n=5, share=0.4)
    assert r["reasons"] == {"success": 4, "success_obstacles_disturbed": 1, "bad_contact": 1, "time_limit": 1}
    assert "latch" not in r


def test_latch_block_counts_the_cause_among_failures_and_placed_not_success():
    eps = [ep("success", 3.0, None), ep("success_obstacles_disturbed", 5.0, "tilt"),
           ep("success_obstacles_disturbed", 5.0, "displacement"), ep("time_limit", 1.0, "tilt"),
           ep("bad_contact", 1.0, None)]
    r = d32.d32_report(eps, is_t2=True)["latch"]
    assert r["episodes_fired"] == 3 and r["failures"] == 4 and r["failures_with_latch"] == 3
    assert r["cause_among_failures"] == {"tilt": 2, "displacement": 1}
    assert r["placed_not_success"] == 2 and r["cause_among_placed_not_success"] == {"tilt": 1, "displacement": 1}


# ------------------------------------------------ the real measurement path
def _t2():
    env = make_t2_env(max_episode_steps=50)
    task = T2ObstacleTask(env, n_obstacles=1, start_square="e4", target_square="e5", obstacle_squares=("c3",))
    task.reset(seed=0)
    return env, task


def _step(task):
    return task.step(np.zeros(task.action_space.shape, dtype=np.float32))[4]


def test_obstacle_row_before_any_disturbance_is_quiet_and_the_latch_is_off():
    env, t = _t2()
    try:
        r = d32.obstacle_row(t, _step(t))
        assert r["latch"] is False and r["disp_mm"] < DISP_TOL and r["tilt"] < TILT_TOL and not r["offb"]
        assert d32.find_t2(t) is t
    finally:
        env.close()


def test_a_slid_obstacle_is_recorded_as_displacement():
    env, t = _t2()
    try:
        qa = t._obstacle_qadrs[0]
        env.data.qpos[qa:qa + 2] = t._placed_xy[0] + np.array([0.02, 0.0])        # 20 mm, no tilt
        info = _step(t)
        r = d32.obstacle_row(t, info)
        assert r["latch"] is True and r["disp_mm"] > DISP_TOL
        s = d32.obstacle_episode_summary([r], t.displacement_tol * 1e3, t.tilt_tol)
        assert s["cause"] == "displacement"
    finally:
        env.close()


def test_a_tipped_obstacle_is_recorded_as_tilt_when_it_has_not_slid():
    """Evaluated on the tipped state itself (mj_forward, no physics step: a free obstacle tipped by hand
    would fall back during the 20 substeps), through the task's own latch computation."""
    import mujoco

    env, t = _t2()
    try:
        qa = t._obstacle_qadrs[0]
        a = 0.40                                    # rad about x, set on the free joint's quaternion
        env.data.qpos[qa + 3:qa + 7] = [np.cos(a / 2), np.sin(a / 2), 0.0, 0.0]
        mujoco.mj_forward(env.model, env.data)
        info = t._augment_info(env._build_info(terminated_reason=None))
        assert info["obstacles_ever_disturbed"] is True
        r = d32.obstacle_row(t, info)
        assert r["latch"] is True and r["tilt"] == pytest.approx(a, abs=1e-3) and r["disp_mm"] < DISP_TOL
        s = d32.obstacle_episode_summary([r], t.displacement_tol * 1e3, t.tilt_tol)
        assert s["cause"] == "tilt" and s["tilt_at_latch_rad"] > t.tilt_tol
    finally:
        env.close()
