#!/usr/bin/env python
"""Config-driven training entry point for the RoboChess experiment grid.

    python scripts/train.py --config configs/t1_ppo_shaped_state_seed0.yaml

One YAML config in, one run directory out. The run directory is
`<output_root>/<run_name>/` and is self-describing: its manifest.json records
the git SHA, the full resolved config and the seed, so a result can be traced
back to the code that produced it without consulting anything outside the
directory.

The script refuses to start if tracked files differ from HEAD, because then no
SHA describes what is running. `--allow-dirty` overrides that for local
pipeline testing and marks the manifest accordingly; it should never be used
for a run whose numbers are going to be reported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.utils import get_device  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from robochess.envs import RoboChessEnv  # noqa: E402
from robochess.experiment import (  # noqa: E402
    DirtyWorkingTreeError,
    build_manifest,
    load_config,
    resolve_config,
    write_manifest,
)
from robochess.tasks import (  # noqa: E402
    ShapedPickAndPlaceReward,
    SparsePickAndPlaceReward,
    T1PickAndPlaceTask,
    T2ObstacleTask,
    make_t2_env,
)

REWARD_WRAPPERS = {
    "shaped": ShapedPickAndPlaceReward,
    "sparse": SparsePickAndPlaceReward,
}

# --- VecNormalize policy --------------------------------------------------
# The standing decision is that observation/reward normalisation applies to the
# PPO branch ONLY, never to SAC. PPO is on-policy and its value target moves
# with the reward scale, so running normalisation helps it; SAC's replay buffer
# holds transitions whose normalisation statistics would drift out from under
# them as the running mean updates, making stored transitions inconsistent with
# current ones.
#
# This set is the single source of truth, and `build_vec_env` below branches on
# it with an exhaustive if/elif/else that RAISES on an unrecognised algorithm.
# Deliberately not a config key, not a dict `.get(..., default)`, and not
# anything that depends on key ordering: a silent default is precisely how this
# would one day get applied to SAC without anyone noticing.
VECNORMALIZE_ALGORITHMS = frozenset({"ppo"})

# Only "observation" is normalised, not the goal keys. achieved_goal and
# desired_goal are both piece positions in the same world frame, and what the
# policy needs from them is their DIFFERENCE. VecNormalize keeps an independent
# RunningMeanStd per dict key, so normalising them separately would rescale the
# two sides of that difference by different factors and distort it. See the
# section-2 report -- this is a judgement call, flagged for confirmation.
VECNORMALIZE_OBS_KEYS = ["observation"]


def build_env(cfg: dict) -> gym.Env:
    """Construct the task+reward stack described by `cfg` (no vectorisation)."""
    if cfg["task"] == "t1":
        base = RoboChessEnv(max_episode_steps=cfg["max_episode_steps"])
        task = T1PickAndPlaceTask(base)
    elif cfg["task"] == "t2":
        base = make_t2_env(max_episode_steps=cfg["max_episode_steps"])
        task = T2ObstacleTask(base, n_obstacles=cfg["n_obstacles"])
    else:  # unreachable -- resolve_config validates the enum
        raise ValueError(f"unknown task {cfg['task']!r}")
    return REWARD_WRAPPERS[cfg["reward_type"]](task)


def build_vec_env(cfg: dict) -> DummyVecEnv | VecNormalize:
    """Vectorise the env, applying VecNormalize to the PPO branch only.

    The algorithm branch here is the one visible place this decision is made.
    """
    venv = DummyVecEnv([lambda: build_env(cfg)])

    algorithm = cfg["algorithm"]
    if algorithm == "ppo":
        venv = VecNormalize(
            venv,
            norm_obs=True,
            norm_reward=True,
            norm_obs_keys=VECNORMALIZE_OBS_KEYS,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )
    elif algorithm == "sac":
        # Intentionally left bare. SAC is never wrapped in VecNormalize; see
        # VECNORMALIZE_ALGORITHMS above for why.
        pass
    else:
        # resolve_config validates the enum, so reaching here means a new
        # algorithm was added without deciding its normalisation. Fail rather
        # than inherit whichever behaviour the last branch happened to leave.
        raise ValueError(
            f"algorithm {algorithm!r} has no VecNormalize decision recorded; "
            f"add it to VECNORMALIZE_ALGORITHMS (or to the SAC branch) explicitly."
        )

    applied = isinstance(venv, VecNormalize)
    expected = algorithm in VECNORMALIZE_ALGORITHMS
    # Cheap invariant, kept in the running code rather than only in a test:
    # the branch above and the declared policy must agree, always.
    assert applied == expected, (
        f"VecNormalize applied={applied} but policy says {expected} for {algorithm!r}"
    )
    return venv


def build_model(cfg: dict, env):
    """Construct the algorithm named by `cfg`.

    The observation space is a Dict (observation/achieved_goal/desired_goal),
    so the policy is MultiInputPolicy -- an MLP feature extractor over vector
    inputs. MlpPolicy cannot accept a Dict space; there is no CNN here.
    """
    common = dict(policy="MultiInputPolicy", env=env, seed=cfg["seed"],
                  device=cfg["device"], verbose=1)
    if cfg["algorithm"] == "ppo":
        return PPO(**common)
    if cfg["algorithm"] == "sac":
        return SAC(**common)
    raise ValueError(f"unknown algorithm {cfg['algorithm']!r}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="path to a YAML experiment cell")
    p.add_argument("--allow-dirty", action="store_true",
                   help="proceed with uncommitted tracked changes; marks the manifest "
                        "as non-reproducible. Local pipeline testing only.")
    p.add_argument("--overwrite", action="store_true",
                   help="reuse a run directory that already has a manifest")
    p.add_argument("--dry-run", action="store_true",
                   help="write the manifest and exit without training")
    args = p.parse_args()

    raw = load_config(args.config)
    cfg = resolve_config(raw, config_path=args.config)

    # `device: auto` is resolved eagerly so the manifest records the device the
    # run ACTUALLY used, not the string "auto" -- otherwise the provenance
    # record cannot distinguish a CPU fallback from an intended GPU run.
    resolved_device = str(get_device(cfg["device"]))

    run_dir = (REPO_ROOT / cfg["output_root"] / cfg["run_name"]).resolve()
    if (run_dir / "manifest.json").exists() and not args.overwrite:
        print(f"ERROR: {run_dir} already holds a manifest. Pass --overwrite to reuse it.",
              file=sys.stderr)
        return 1
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        manifest = build_manifest(
            args.config, raw, cfg, allow_dirty=args.allow_dirty,
            extra={"resolved": {
                "device_requested": cfg["device"],
                "device_actual": resolved_device,
                "vecnormalize": cfg["algorithm"] in VECNORMALIZE_ALGORITHMS,
            }},
        )
    except DirtyWorkingTreeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    manifest_path = write_manifest(run_dir, manifest)

    print(f"run dir  : {run_dir}")
    print(f"manifest : {manifest_path}")
    print(f"git sha  : {manifest['git']['sha']}"
          + ("  [DIRTY -- NOT REPRODUCIBLE]" if manifest["git"]["dirty_tracked"] else ""))
    print(f"device   : {cfg['device']} -> {resolved_device}")
    print(f"config   : {cfg}")

    if args.dry_run:
        print("dry run: manifest written, no training performed")
        return 0

    # --- logging -------------------------------------------------------
    # CSV is primary: it is the format the analysis scripts and the writeup
    # read, it survives without a TensorBoard install, and it diffs. The
    # TensorBoard writer is attached alongside as a secondary view for
    # watching a run in progress -- both come from SB3's own logger, so
    # neither is a reimplementation that could drift from the other.
    # "stdout" is kept so the console still shows the usual training tables.
    log_dir = run_dir / "logs"
    logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])
    print(f"logs     : {log_dir} (csv primary, tensorboard secondary)")

    env = build_vec_env(cfg)
    print(f"vecnormalize: {isinstance(env, VecNormalize)} (algorithm={cfg['algorithm']})")
    model = build_model(cfg, env)
    model.set_logger(logger)
    model.learn(total_timesteps=cfg["total_timesteps"])
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
