"""Experiment configuration, resolution, and per-run provenance manifests.

One plain YAML file per experiment cell (deliberately NOT Hydra: the cell grid
is small and enumerable, and a flat file that can be read, diffed and archived
without composing anything is worth more here than override syntax).

Two things live here rather than in the training script, because both are
correctness-critical and want to be unit-testable on their own:

  * `resolve_config` -- strict validation plus the standing defaults. Unknown
    keys are an error, not a shrug: a typo'd key in a config that silently
    does nothing is exactly the failure that makes a run grid untrustworthy
    months later.
  * `build_manifest` -- the provenance record. Its contract is that a manifest
    never reports a git SHA that does not describe the code that actually ran.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- schema ---------------------------------------------------------------
# Every key a config may contain. Keys with a `None` default are REQUIRED and
# must be given explicitly -- seed and total_timesteps especially: inheriting a
# default seed is how two "different" runs quietly become the same run.
REQUIRED_KEYS = ("task", "algorithm", "reward_type", "observation", "seed", "total_timesteps")

OPTIONAL_DEFAULTS: dict[str, Any] = {
    "device": None,            # None -> resolved from `observation` (see below)
    "max_episode_steps": 100,
    "n_obstacles": 0,          # t2 only; must stay 0 for t1
    # "absolute" keeps every pilot-matrix config byte-for-byte reproducible;
    # "delta" is RoboChessEnv's per-step joint-target delta mode.
    "action_mode": "absolute",
    "checkpoint_freq": 100_000,
    # Pinned to 1 by default, and this is a REPRODUCIBILITY setting, not a
    # performance one. Measured: runs are bit-identical across processes at a
    # fixed thread count, but 6 threads vs 1 thread diverges from the very
    # first policy forward pass (BLAS reduction order, the D8 class of
    # effect). Left unpinned, torch infers the count from the host's core
    # count, so the same config on a different machine silently produces
    # different numbers. 1 is the only value that is portable by default.
    "num_threads": 1,
    # Fixed start/target squares. None -> the task wrapper's own random
    # sampling over REACHABLE_SQUARES, which is what every run so far used, so
    # a config that omits these is unchanged by their existence.
    "start_square": None,
    "target_square": None,
    # Staged shaped reward (DECISIONS.md D20). Default OFF: with this false the
    # shaped reward is exactly the one every pilot and sanity run used so far.
    "shaped_staged": False,
    "grasp_bonus": 1.0,
    "lift_bonus": 0.5,
    # Pick-up stage observation features (T1 only). Default OFF: with this
    # false the observation vector is the 21-entry one every run so far used,
    # so a config that omits this key is unchanged by its existence.
    "pickup_obs": False,
    # Pick-up training stage (T1 only). Default OFF: with this false the reward
    # stack is exactly the one every run so far used. When true it REPLACES the
    # reward -- no goal-distance term, no travelling failure penalty -- so the
    # remaining pickup_* keys only mean anything with pickup_mode true.
    # Prefixed names throughout: bare `failure_penalty`/`lift_height` would sit
    # next to the shaped reward's own `grasp_bonus`/`lift_bonus` and read as if
    # they applied to it.
    "pickup_mode": False,
    "pickup_lift_height": 0.03,
    "pickup_hold_steps": 5,
    "pickup_failure_penalty": 5.0,
    "pickup_reach_weight": 1.0,
    "pickup_close_weight": 1.0,
    "pickup_lift_weight": 2.0,
    "pickup_success_weight": 1.0,
    "pickup_first_grasp_bonus": 5.0,
    "output_root": "runs",
    "run_name": None,          # None -> derived from the config filename + seed
    "notes": "",
}

ENUMS = {
    "task": ("t1", "t2"),
    "algorithm": ("ppo", "sac"),
    "reward_type": ("shaped", "sparse"),
    "observation": ("state", "image"),
    "action_mode": ("absolute", "delta", "cartesian"),
    "device": ("cpu", "cuda", "auto"),
}

# The standing device decision (configs/README.md), applied here so that it is
# one rule in one place rather than a comment pasted into every config:
#   state-vector / MLP -> "cpu", measured faster on this hardware
#   image / CNN        -> "auto", so it resolves to CUDA where conv work pays
DEVICE_BY_OBSERVATION = {"state": "cpu", "image": "auto"}


class ConfigError(ValueError):
    """Raised for any malformed or internally inconsistent config."""


class DirtyWorkingTreeError(RuntimeError):
    """Raised when tracked files differ from HEAD, so no SHA describes the run."""


# --- config ---------------------------------------------------------------
def load_config(path: str | Path) -> dict[str, Any]:
    """Parse a config file into a raw dict, with no defaults applied."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"No config file at {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    return raw


def resolve_config(raw: dict[str, Any], config_path: str | Path | None = None) -> dict[str, Any]:
    """Validate `raw` and fill in defaults, returning the resolved config.

    Strict in both directions: every required key must be present, and every
    key present must be one this schema knows about.
    """
    known = set(REQUIRED_KEYS) | set(OPTIONAL_DEFAULTS)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(
            f"Unknown config key(s): {unknown}. Known keys: {sorted(known)}"
        )
    missing = sorted(k for k in REQUIRED_KEYS if k not in raw)
    if missing:
        raise ConfigError(f"Missing required config key(s): {missing}")

    cfg: dict[str, Any] = {**OPTIONAL_DEFAULTS, **raw}

    for key, allowed in ENUMS.items():
        value = cfg[key]
        if value is None and key == "device":
            continue
        if value not in allowed:
            raise ConfigError(f"{key}={value!r} is not one of {list(allowed)}")

    for key in ("seed", "total_timesteps", "max_episode_steps", "checkpoint_freq",
                "n_obstacles", "num_threads"):
        value = cfg[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"{key} must be an int, got {value!r}")
    if cfg["seed"] < 0:
        raise ConfigError(f"seed must be >= 0, got {cfg['seed']}")
    for key in ("total_timesteps", "max_episode_steps", "checkpoint_freq", "num_threads"):
        if cfg[key] <= 0:
            raise ConfigError(f"{key} must be > 0, got {cfg[key]}")

    from robochess.tasks.squares import ALL_SQUARES

    for key in ("start_square", "target_square"):
        value = cfg[key]
        if value is not None and value not in ALL_SQUARES:
            raise ConfigError(f"{key}={value!r} is not a board square")
    if (cfg["start_square"] is not None
            and cfg["start_square"] == cfg["target_square"]):
        raise ConfigError(
            f"start_square and target_square are both {cfg['start_square']!r}; they must differ"
        )

    if not isinstance(cfg["shaped_staged"], bool):
        raise ConfigError(f"shaped_staged must be a bool, got {cfg['shaped_staged']!r}")
    if not isinstance(cfg["pickup_obs"], bool):
        raise ConfigError(f"pickup_obs must be a bool, got {cfg['pickup_obs']!r}")
    if not isinstance(cfg["pickup_mode"], bool):
        raise ConfigError(f"pickup_mode must be a bool, got {cfg['pickup_mode']!r}")
    if not isinstance(cfg["pickup_hold_steps"], int) or isinstance(cfg["pickup_hold_steps"], bool):
        raise ConfigError(
            f"pickup_hold_steps must be an int, got {cfg['pickup_hold_steps']!r}")
    if cfg["pickup_hold_steps"] <= 0:
        raise ConfigError(
            f"pickup_hold_steps must be > 0, got {cfg['pickup_hold_steps']}")
    for key in ("pickup_lift_height", "pickup_failure_penalty", "pickup_reach_weight",
                "pickup_close_weight", "pickup_lift_weight", "pickup_success_weight",
                "pickup_first_grasp_bonus"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)):
            raise ConfigError(f"{key} must be a number, got {cfg[key]!r}")
        if cfg[key] < 0:
            raise ConfigError(f"{key} must be >= 0, got {cfg[key]}")
    if cfg["pickup_lift_height"] <= 0:
        raise ConfigError(
            f"pickup_lift_height must be > 0, got {cfg['pickup_lift_height']}")
    if cfg["pickup_mode"] and cfg["task"] != "t1":
        raise ConfigError(
            f"pickup_mode=True with task={cfg['task']!r}: the pick-up stage is T1 only."
        )
    if cfg["pickup_mode"] and cfg["shaped_staged"]:
        raise ConfigError(
            "pickup_mode=True with shaped_staged=True: both replace the reward, "
            "so one of them would be silently ignored. Choose one."
        )
    if cfg["pickup_mode"] and cfg["reward_type"] != "shaped":
        # pickup_mode replaces the reward outright. Rejecting the combination
        # rather than ignoring reward_type keeps a config from declaring
        # "sparse" and silently training on something else.
        raise ConfigError(
            f"pickup_mode=True with reward_type={cfg['reward_type']!r}: the pick-up "
            "reward replaces reward_type entirely; leave it at 'shaped'."
        )
    if cfg["pickup_obs"] and cfg["task"] != "t1":
        raise ConfigError(
            f"pickup_obs=True with task={cfg['task']!r}: the pick-up observation "
            "features are T1 only. T2 and later tasks extend the observation "
            "vector themselves (D12), so the two extensions would collide."
        )
    for key in ("grasp_bonus", "lift_bonus"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)):
            raise ConfigError(f"{key} must be a number, got {cfg[key]!r}")
        if cfg[key] < 0:
            raise ConfigError(f"{key} must be >= 0, got {cfg[key]}")
    if cfg["shaped_staged"] and cfg["reward_type"] != "shaped":
        raise ConfigError(
            f"shaped_staged=True with reward_type={cfg['reward_type']!r}: the staged "
            "terms belong to the shaped reward only."
        )

    # F11: the same validator the T2 constructor and reset() use, so the range
    # rule lives in one place. Count only; obstacle squares are not configurable.
    from robochess.tasks import t2_obstacle_pick_and_place as _t2

    try:
        _t2.validate_obstacle_config(cfg["n_obstacles"], None)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    if cfg["task"] == "t1" and cfg["n_obstacles"] != 0:
        raise ConfigError(
            f"n_obstacles={cfg['n_obstacles']} but task=t1; obstacles only exist on t2"
        )

    # D5: there is no image observation mode in RoboChessEnv. The enum keeps
    # "image" so image cells can be WRITTEN now and the loader rejects them
    # with a precise reason rather than a shape mismatch deep inside SB3.
    if cfg["observation"] == "image":
        raise ConfigError(
            "observation=image is declared in the schema but not implemented: "
            "RoboChessEnv has no image observation mode (DECISIONS.md D5). "
            "Image cells cannot be run until that lands."
        )

    if cfg["device"] is None:
        cfg["device"] = DEVICE_BY_OBSERVATION[cfg["observation"]]

    if cfg["run_name"] is None:
        stem = Path(config_path).stem if config_path is not None else (
            f"{cfg['task']}_{cfg['algorithm']}_{cfg['reward_type']}_{cfg['observation']}"
        )
        cfg["run_name"] = f"{stem}_seed{cfg['seed']}"

    return cfg


# --- provenance -----------------------------------------------------------
def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_provenance(allow_dirty: bool = False) -> dict[str, Any]:
    """Describe the commit the running code corresponds to.

    Raises `DirtyWorkingTreeError` if any TRACKED file differs from HEAD --
    staged or unstaged. In that case HEAD's SHA does not describe what is
    about to run, and recording it would make the manifest a lie.

    Untracked files are reported but are not an error: they are, by
    definition, not part of the code under version control that ran.

    `allow_dirty=True` is an explicit testing escape hatch. It does NOT
    record a clean-looking SHA: the manifest carries `dirty_tracked: true`,
    the list of modified paths, and a hash of the actual diff, so the record
    still says plainly that the commit does not describe the run.
    """
    sha = _git("rev-parse", "HEAD")
    try:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    except subprocess.CalledProcessError:
        branch = None

    # Asking git for plain path lists rather than parsing --porcelain's status
    # columns: the columns are positional and a leading blank column is easy to
    # lose to an incidental strip(), which silently truncates the first path.
    # `diff --name-only HEAD` covers staged and unstaged changes alike.
    dirty = [ln for ln in _git("diff", "--name-only", "HEAD").splitlines() if ln]
    untracked = [
        ln for ln in _git("ls-files", "--others", "--exclude-standard").splitlines() if ln
    ]

    prov: dict[str, Any] = {
        "sha": sha,
        "branch": branch or None,
        "dirty_tracked": bool(dirty),
        "dirty_paths": sorted(dirty),
        "untracked_paths": sorted(untracked),
        "diff_sha256": None,
    }

    if dirty:
        diff = _git("diff", "HEAD")
        prov["diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
        if not allow_dirty:
            raise DirtyWorkingTreeError(
                "Refusing to write a manifest: "
                f"{len(dirty)} tracked file(s) differ from HEAD ({sha[:12]}), so that "
                "SHA does not describe the code about to run.\n  "
                + "\n  ".join(sorted(dirty))
                + "\n\nCommit (or stash) them, or pass --allow-dirty to record the run as "
                  "explicitly non-reproducible."
            )
    return prov


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "torch", "gymnasium", "mujoco", "stable_baselines3", "yaml"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", None)
        except ImportError:
            versions[name] = None
    return versions


def _thread_provenance() -> dict[str, Any]:
    """Thread/BLAS settings. Recorded because D8 showed that BLAS build and
    thread-count differences alone can move results between otherwise-identical
    environments -- so a reproducibility claim is only meaningful alongside them."""
    import os

    info: dict[str, Any] = {
        "env": {
            var: os.environ.get(var)
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        }
    }
    try:
        import torch

        info["torch_num_threads"] = torch.get_num_threads()
        info["torch_num_interop_threads"] = torch.get_num_interop_threads()
    except ImportError:
        pass
    try:
        import numpy as np

        cfg = getattr(np, "__config__", None)
        info["numpy_blas"] = (
            cfg.get_info("blas_opt").get("libraries") if hasattr(cfg, "get_info") else None
        )
    except Exception:
        info["numpy_blas"] = None
    return info


def _repo_relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def build_manifest(
    config_path: str | Path,
    raw_config: dict[str, Any],
    resolved_config: dict[str, Any],
    allow_dirty: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the run manifest. Raises before returning if the tree is dirty."""
    config_path = Path(config_path)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "run_name": resolved_config["run_name"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": resolved_config["seed"],
        # Repo-relative where possible (stable across machines), absolute
        # otherwise -- a config kept outside the repo must not crash the run.
        "config_path": _repo_relative(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "config_raw": raw_config,
        "config_resolved": resolved_config,
        "git": git_provenance(allow_dirty=allow_dirty),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "argv": list(sys.argv),
            "packages": _package_versions(),
            "threads": _thread_provenance(),
        },
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return path
