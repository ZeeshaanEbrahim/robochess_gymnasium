"""RS29: scripts/eval_probe.py --pairs (fixed, cycled (start, target) list), and the
place-mode Monitor keywords that carry the per-episode squares into the run's CSV."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def _tensorboard_importable() -> bool:
    # scripts/train.py configures a tensorboard logger; SB3 imports it through torch.
    try:
        import torch.utils.tensorboard  # noqa: F401
    except ImportError:
        return False
    return True


# Same convention as tests/test_d29_place_reward_v2.py's requires_d28_place_ppo_seed0:
# a dependency that is absent from the CI environment (requirements-ci.txt has no
# tensorboard) skips the tests that need it, with the reason stated, instead of failing.
requires_tensorboard = pytest.mark.skipif(
    not _tensorboard_importable(),
    reason="train.py's tensorboard logger needs tensorboard, which requirements-ci.txt "
           "does not install (requirements.txt pins it)")


def _run(args, **kw):
    return subprocess.run([PY, *map(str, args)], cwd=REPO, capture_output=True, text=True, **kw)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """One PPO update on a random-squares place config, in a scratch runs dir."""
    root = tmp_path_factory.mktemp("rs29_runs")
    cfg = yaml.safe_load((REPO / "configs" / "_probe_cart_randsq29_ppotanh_seed0.yaml").read_text())
    cfg.update(total_timesteps=2048, max_episode_steps=40, output_root=str(root), run_name="tiny")
    cfg_path = root / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    r = _run(["scripts/train.py", "--config", cfg_path, "--allow-dirty"])
    assert r.returncode == 0, r.stderr[-2000:]
    return root


def _eval(root, out, *extra, episodes=4):
    return _run(["scripts/eval_probe.py", "tiny", "--runs-dir", root, "--episodes", episodes,
                 "--deterministic", "--seed", 7, "--out", out, *extra])


@requires_tensorboard
def test_pairs_cycle_in_order(tiny_run, tmp_path):
    out = tmp_path / "o.json"
    r = _eval(tiny_run, out, "--pairs", "e4:e5,a1:h8", episodes=4)
    assert r.returncode == 0, r.stderr[-2000:]
    eps = json.loads(out.read_text())["episodes"]
    assert [(e["start_square"], e["target_square"]) for e in eps] == [
        ("e4", "e5"), ("a1", "h8"), ("e4", "e5"), ("a1", "h8")]


@requires_tensorboard
def test_pairs_refuses_episodes_not_a_multiple(tiny_run, tmp_path):
    out = tmp_path / "o.json"
    r = _eval(tiny_run, out, "--pairs", "e4:e5,a1:h8", episodes=3)
    assert r.returncode != 0 and "multiple of the pair count 2" in r.stderr
    assert not out.exists()


@requires_tensorboard
def test_without_pairs_output_identical_to_before(tiny_run, tmp_path):
    parent = subprocess.run(["git", "show", "8cf881b:scripts/eval_probe.py"], cwd=REPO,
                            capture_output=True, text=True)
    if parent.returncode != 0:
        pytest.skip("8cf881b not available (shallow checkout)")
    old = parent.stdout.replace("Path(__file__).resolve().parents[1]", f"Path({str(REPO)!r})", 1)
    old_path = tmp_path / "eval_probe_before.py"
    old_path.write_text(old)
    new_out, old_out = tmp_path / "new.json", tmp_path / "old.json"
    assert _eval(tiny_run, new_out, episodes=3).returncode == 0
    r = _run([old_path, "tiny", "--runs-dir", tiny_run, "--episodes", 3, "--deterministic",
              "--seed", 7, "--out", old_out])
    assert r.returncode == 0, r.stderr[-2000:]
    assert new_out.read_bytes() == old_out.read_bytes()


def test_place_monitor_keywords_gain_the_squares():
    spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    base = train.MONITOR_INFO_KEYWORDS
    assert train.monitor_info_keywords({"place_mode": True}) == base + (
        "place_state", "start_square", "target_square")
    assert train.monitor_info_keywords({}) == base
    assert train.monitor_info_keywords({"pickup_mode": True}) == base + ("pickup_success",)
