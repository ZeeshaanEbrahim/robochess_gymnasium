"""RS32: six D-recipe configs on 48 start-target pairs (seeds 0, 100, ..., 500).

Expected results, stated before the run:
  * all six configs load through resolve_config;
  * train_pairs has 48 unique entries, every square on the board, none d1 or e1 (D26), and
    equals the list in results/rs32_48_pairs/PREREGISTRATION.md, in the same order;
  * seeds are 0, 100, 200, 300, 400, 500 (spaced by 100: SB3 seeds env i with seed + i, so
    no two runs share an env seed at n_envs 4);
  * the six configs differ from each other only in seed and run_name, and from the RS31 D
    config only in train_pairs, total_timesteps (5,000,000), seed, run_name and notes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from robochess.experiment import resolve_config

REPO = Path(__file__).resolve().parents[1]
SEEDS = (0, 100, 200, 300, 400, 500)
PREREG = REPO / "results" / "rs32_48_pairs" / "PREREGISTRATION.md"
RS31_D = REPO / "configs" / "_probe_cart_rs31D_ppotanh_seed0.yaml"


def _path(seed):
    return REPO / "configs" / f"_probe_cart_rs32D_ppotanh_seed{seed}.yaml"


def _raw(seed):
    return yaml.safe_load(_path(seed).read_text())


def _prereg_pairs():
    m = re.search(r"^Training pairs \(48[^)]*\): (.+)$", PREREG.read_text(), re.M)
    assert m, "no 'Training pairs (48 ...)' line in PREREGISTRATION.md"
    return m.group(1).split()


@pytest.mark.parametrize("seed", SEEDS)
def test_config_loads_and_resolves(seed):
    cfg = resolve_config(_raw(seed), config_path=_path(seed))
    assert cfg["seed"] == seed
    assert cfg["run_name"] == f"_probe_cart_rs32D_ppotanh_seed{seed}"
    assert cfg["total_timesteps"] == 5_000_000
    assert (cfg["n_envs"], cfg["ppo_ent_coef"], cfg["ppo_lr_schedule"]) == (4, 0.01, "linear")
    assert len(cfg["train_pairs"]) == 48


def test_seeds_are_spaced_so_env_seeds_never_overlap():
    seeds = [_raw(s)["seed"] for s in SEEDS]
    assert seeds == list(SEEDS)
    env_seeds = {s + i for s in seeds for i in range(4)}
    assert len(env_seeds) == 4 * len(seeds)


def test_train_pairs_are_the_preregistered_48():
    pairs = _raw(0)["train_pairs"]
    files, ranks = "abcdefgh", "12345678"
    assert len(pairs) == 48 and len(set(pairs)) == 48
    for p in pairs:
        s, t = p.split(":")
        assert s != t
        for sq in (s, t):
            assert len(sq) == 2 and sq[0] in files and sq[1] in ranks, p
            assert sq not in ("d1", "e1"), p
    assert pairs == _prereg_pairs()
    for seed in SEEDS:
        assert _raw(seed)["train_pairs"] == pairs


def test_configs_differ_only_in_the_allowed_keys():
    rs31 = yaml.safe_load(RS31_D.read_text())
    base = _raw(0)
    assert {k for k in set(rs31) | set(base) if rs31.get(k) != base.get(k)} == {
        "train_pairs", "total_timesteps", "run_name", "notes"}  # seed is 0 in both
    for seed in SEEDS[1:]:
        other = _raw(seed)
        assert {k for k in set(base) | set(other) if base.get(k) != other.get(k)} == {
            "seed", "run_name"}
