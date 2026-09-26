"""D28 place-mode evaluation: the pure metrics, and eval_probe.py end to end.

The end-to-end tests run eval_probe.py in a SUBPROCESS on a tiny saved model: building an SB3
model inside the pytest process segfaults torch in the full suite (DECISIONS D28).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import place_eval as pe  # noqa: E402


def row(st, held=False, pinched=False, lift=0.0, dxy=0.2, ap=0.0, succ=False, pxy=None, gxy=None):
    r = dict(st=st, held=held, pinched=pinched, lift=lift, dxy=dxy, ap=ap, succ=succ)
    if pxy is not None:
        r.update(pxy=pxy, gxy=gxy)
    return r


def clean_place():
    return ([row("S0")] * 3 + [row("S1", True, True, 0.0, 0.08, 0.65)]
            + [row("S1", True, True, 0.05, 0.02, 0.65)] + [row("S2", ap=0.5), row("S2", ap=0.1, succ=True)])


def test_wilson_matches_the_probe_report_formula():
    lo, hi = pe.wilson(0, 20)
    assert lo == 0.0 or lo < 1e-12
    assert hi == pytest.approx(0.1611, abs=1e-3)
    lo, hi = pe.wilson(10, 20)
    assert (lo, hi) == pytest.approx((0.299, 0.701), abs=2e-3)
    assert pe.wilson(0, 0) == (0.0, 1.0)


def test_stage_flags_positive_and_negative_controls():
    s = pe.place_episode_summary(clean_place(), "success")
    assert "grasp" not in s and "grasp" not in pe.STAGES        # cannot be confused with D18 grasp_success
    assert (s["held"], s["lifted_30mm"], s["carried_within_3cm"], s["released"], s["placed"]) == (True,) * 5
    assert not s["truncated_in_s2"]
    # never held
    s = pe.place_episode_summary([row("S0")] * 5, "time_limit")
    assert not any(s[k] for k in pe.STAGES)
    # held but never lifted 30 mm
    rows = [row("S0"), row("S1", True, True, 0.01, 0.02)]
    s = pe.place_episode_summary(rows, "time_limit")
    assert s["held"] and not s["lifted_30mm"] and not s["carried_within_3cm"]
    # lifted but never within 3 cm of the target
    rows = [row("S1", True, True, 0.05, 0.2)] * 3
    s = pe.place_episode_summary(rows, "time_limit")
    assert s["lifted_30mm"] and not s["carried_within_3cm"]
    # within 3 cm BEFORE the lift does not count as carried
    rows = [row("S1", True, True, 0.0, 0.01), row("S1", True, True, 0.05, 0.2)]
    assert not pe.place_episode_summary(rows, "time_limit")["carried_within_3cm"]


def test_truncated_in_s2_and_its_final_gripper_qpos():
    rows = [row("S1", True, True, 0.05, 0.01)] + [row("S2", ap=0.42)] * 4
    s = pe.place_episode_summary(rows, "time_limit")
    assert s["truncated_in_s2"] and s["final_gripper_qpos"] == 0.42 and s["s2_steps"] == 4
    assert not s["placed"] and s["released"]
    assert not pe.place_episode_summary(rows, "bad_contact")["truncated_in_s2"]     # negative: not a timeout
    rep = pe.place_report([s, pe.place_episode_summary(clean_place(), "success")])
    assert rep["truncated_in_s2"]["k"] == 1 and rep["truncated_in_s2"]["final_gripper_qpos"] == [0.42]


def test_deterministic_twenty_episodes_are_one_observation():
    s = pe.place_episode_summary(clean_place(), "success")
    stoch = pe.place_report([s] * 20)
    det = pe.place_report([s] * 20, deterministic=True, identical=True)
    assert stoch["effective_n"] == 20 and det["effective_n"] == 1
    assert det["rates"]["placed"]["effective_n"] == 1 and det["rates"]["placed"]["k"] == 20
    assert det["rates"]["placed"]["wilson_lo"] == pytest.approx(pe.wilson(1, 1)[0])   # not the n=20 interval
    assert det["rates"]["placed"]["wilson_lo"] < stoch["rates"]["placed"]["wilson_lo"]
    assert pe.place_report([s] * 20, deterministic=True, identical=False)["effective_n"] == 20


# ------------------------------------------------- push-assisted diagnostic
G = (0.2, 0.0)          # target 200 mm from the start (0, 0)


def carried_episode():
    """Grasped at the start, carried the whole way, released on target."""
    rows = [row("S0", pxy=[0.0, 0.0], gxy=G)] * 2
    rows += [row("S1", True, True, 0.05, 0.2 - 0.02 * k, pxy=[0.02 * k, 0.0], gxy=G) for k in range(10)]
    rows += [row("S2", ap=0.5, pxy=[0.2, 0.0], gxy=G), row("S2", ap=0.1, succ=True, pxy=[0.2, 0.0], gxy=G)]
    return rows


def push_then_token_lift_episode():
    """Pushed 190 mm along the board (never held), then one held step with a token lift, then released."""
    rows = [row("S0", pxy=[0.019 * k, 0.0], gxy=G) for k in range(11)]
    rows += [row("S1", True, True, 0.004, 0.01, pxy=[0.19, 0.0], gxy=G)]
    rows += [row("S2", ap=0.5, pxy=[0.19, 0.0], gxy=G), row("S2", ap=0.1, succ=True, pxy=[0.19, 0.0], gxy=G)]
    return rows


def test_held_steps_and_carry_mm():
    c = pe.place_episode_summary(carried_episode(), "success")
    assert c["held_steps"] == 10 and c["carry_mm"] == pytest.approx(180.0)     # first to last held step
    assert c["start_to_target_mm"] == pytest.approx(200.0)
    p = pe.place_episode_summary(push_then_token_lift_episode(), "success")
    assert p["held_steps"] == 1 and p["carry_mm"] == pytest.approx(0.0)
    n = pe.place_episode_summary([row("S0", pxy=[0.0, 0.0], gxy=G)] * 3, "time_limit")
    assert n["held_steps"] == 0 and n["carry_mm"] is None
    legacy = pe.place_episode_summary(clean_place(), "success")                    # rows without positions
    assert legacy["carry_mm"] is None and legacy["start_to_target_mm"] is None


def test_push_assisted_flags_the_token_lift_and_not_the_carry():
    c = pe.place_episode_summary(carried_episode(), "success")
    p = pe.place_episode_summary(push_then_token_lift_episode(), "success")
    only_carried = pe.push_assisted([c])
    assert (only_carried["successful"], only_carried["push_assisted"]) == (1, 0)
    only_pushed = pe.push_assisted([p])
    assert (only_pushed["successful"], only_pushed["push_assisted"], only_pushed["fraction"]) == (1, 1, 1.0)
    both = pe.place_report([c, p, c])["push_assisted"]
    assert (both["successful"], both["push_assisted"]) == (3, 1) and both["fraction"] == pytest.approx(1 / 3)
    assert both["label"] == "push-assisted" and "50%" in both["rule"]


def test_push_assisted_counts_successes_only():
    failed_push = pe.place_episode_summary(push_then_token_lift_episode()[:-1], "time_limit")
    assert not failed_push["placed"]
    pa = pe.push_assisted([failed_push])
    assert pa["successful"] == 0 and pa["push_assisted"] == 0 and pa["fraction"] is None


def test_push_assisted_boundary_is_half_the_distance():
    half = carried_episode()
    for k, r in enumerate(half):                                           # carry exactly 100 mm of 200
        if r["st"] == "S1":
            r["pxy"] = [0.1 * min(1.0, (k - 2) / 9), 0.0]
    e = pe.place_episode_summary(half, "success")
    assert e["carry_mm"] == pytest.approx(100.0)
    assert pe.push_assisted([e])["push_assisted"] == 0                     # not < 50%


# ---------------------------------------------------------------- end to end
def _make_run(tmp_path, name, **flags):
    """A tiny PPO run directory (manifest + final_model + final_vecnormalize), built in a subprocess."""
    code = f'''
import importlib.util, json, sys
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
from robochess.experiment import resolve_config
spec = importlib.util.spec_from_file_location("train", {str(ROOT / "scripts" / "train.py")!r})
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
cfg = resolve_config(dict(task="t1", algorithm="ppo", reward_type="shaped", observation="state", seed=0,
    total_timesteps=1, action_mode="cartesian", max_episode_steps=15, start_square="e4", target_square="e5",
    device="cpu", **{flags!r}), config_path="t.yaml")
run = Path({str(tmp_path / name)!r}); run.mkdir(parents=True)
(run / "manifest.json").write_text(json.dumps(dict(config_resolved=cfg)))
venv = train.build_vec_env(cfg); model = train.build_model(cfg, venv)
model.save(str(run / "final_model.zip")); venv.save(str(run / "final_vecnormalize.pkl"))
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=300)


def _eval(tmp_path, name, *extra):
    out = tmp_path / f"{name}.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_probe.py"), name, "--runs-dir", str(tmp_path),
                    "--episodes", "3", "--out", str(out), *extra], check=True, capture_output=True, timeout=300)
    return json.loads(out.read_text())


def test_eval_probe_reports_place_metrics_for_a_place_run(tmp_path):
    _make_run(tmp_path, "place", pickup_obs=True, place_obs=True, place_mode=True, vecnormalize_clip_reward=1000)
    d = _eval(tmp_path, "place")
    r = d["place_report"]
    assert r["n"] == 3 and r["effective_n"] == 3 and not r["deterministic"]
    assert set(r["rates"]) == set(pe.STAGES)
    assert all(v["n"] == 3 and 0 <= v["wilson_lo"] <= v["wilson_hi"] <= 1 for v in r["rates"].values())
    assert all("place" in e for e in d["episodes"])
    assert all({"held_steps", "carry_mm", "start_to_target_mm"} <= set(e["place"]) for e in d["episodes"])
    assert r["push_assisted"]["label"] == "push-assisted" and r["push_assisted"]["successful"] == 0
    assert r["truncated_in_s2"]["n"] == 3
    det = _eval(tmp_path, "place", "--deterministic")
    assert det["place_report"]["deterministic"] and det["place_report"]["identical_episodes"] is True
    assert det["place_report"]["effective_n"] == 1


def test_eval_probe_flag_off_output_has_no_place_keys(tmp_path):
    _make_run(tmp_path, "pick", pickup_obs=True, pickup_mode=True)
    d = _eval(tmp_path, "pick")
    assert "place_report" not in d
    assert all("place" not in e for e in d["episodes"])
    assert set(d) == {"run", "checkpoint", "model", "seed", "deterministic", "n", "obs_rms_identical_before_after",
                      "episodes"}
    assert d["episodes"][0]["pickup"] in (True, False) and "pinched_ever" in d["episodes"][0]
