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


def row(st, held=False, pinched=False, lift=0.0, dxy=0.2, ap=0.0, succ=False):
    return dict(st=st, held=held, pinched=pinched, lift=lift, dxy=dxy, ap=ap, succ=succ)


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
    assert (s["grasp"], s["lifted_30mm"], s["carried_within_3cm"], s["released"], s["placed"]) == (True,) * 5
    assert not s["truncated_in_s2"]
    # never grasps
    s = pe.place_episode_summary([row("S0")] * 5, "time_limit")
    assert not any(s[k] for k in pe.STAGES)
    # grasp but never lifted 30 mm
    rows = [row("S0"), row("S1", True, True, 0.01, 0.02)]
    s = pe.place_episode_summary(rows, "time_limit")
    assert s["grasp"] and not s["lifted_30mm"] and not s["carried_within_3cm"]
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
