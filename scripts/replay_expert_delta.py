"""Replay the scripted expert's commanded joint targets through the delta action space.

Diagnostic for the question "is T1 physically solvable in the delta action space within
the production episode budget", asked before spending compute on a re-pilot. It does NOT
tell you whether a policy can LEARN the task -- only whether a near-optimal action
sequence exists inside the action space the policy is given.

Two stages:

1. Reference (`record_reference`): run the scripted expert normally -- absolute-mode env,
   its own raised step budget -- and record, per macro-step, the absolute actuator command
   it issued (6 arm joint targets + gripper ctrl) and the waypoint phase it was in.

2. Replay (`replay`): feed that command sequence into a delta-mode env built by
   scripts/train.py's own build_env(), i.e. the same wrappers and the same production
   max_episode_steps PPO and SAC get. At each step the converter emits

       arm action = clip(expert_target - current commanded target, -cap, +cap) / cap
       gripper    = the delta space's own absolute mapping of the expert's gripper ctrl

   where "current commanded target" is the env's running delta target (read, never
   written) and cap is the env's max_joint_delta. The env is not modified in any way.

Usage:
    python scripts/replay_expert_delta.py --squares a1,a8,... --target e5 --out results/x.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import scripted_expert as se  # noqa: E402
from robochess.envs import RoboChessEnv  # noqa: E402
from robochess.envs.robochess_env import N_ARM_ACTUATORS as N_ARM  # noqa: E402
from robochess.tasks import T1PickAndPlaceTask  # noqa: E402

JOINTS = ("pan", "lift", "elbow", "w1", "w2", "w3")


def _build_env(action_mode: str, max_episode_steps: int, reward_type: str = "shaped"):
    """The training path's own env builder, so the replay runs what SB3 would run."""
    spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    from robochess.experiment import resolve_config

    cfg = resolve_config(dict(task="t1", algorithm="ppo", reward_type=reward_type,
                              observation="state", seed=0, total_timesteps=1,
                              action_mode=action_mode, max_episode_steps=max_episode_steps,
                              n_obstacles=0))
    return train.build_env(cfg)


# --------------------------------------------------------------------- stage 1
def record_reference(start: str, target: str, budget: int = 350) -> dict:
    """The expert's per-step absolute commands, with the waypoint phase of each."""
    env = RoboChessEnv(max_episode_steps=budget)
    task = T1PickAndPlaceTask(env, start_square=start, target_square=target)

    class Capture(gym.Wrapper):
        def reset(self, **kw):
            self.cmds, self.phases = [], []
            return self.env.reset(**kw)

        def step(self, a):
            out = self.env.step(a)
            self.cmds.append(self.unwrapped.data.ctrl.copy())
            self.phases.append(self.phase)
            return out

    cap = Capture(task)
    cap.phase = None
    ex = se.ScriptedExpert(cap)

    original = se.ScriptedExpert._run_waypoint

    def tagged(self, pinch_target, grip, budget_, label, log, q_arm, compensate):
        cap.phase = label
        return original(self, pinch_target, grip, budget_, label, log, q_arm, compensate)

    se.ScriptedExpert._run_waypoint = tagged
    try:
        record = ex.run_episode(seed=0)
    finally:
        se.ScriptedExpert._run_waypoint = original
    env.close()
    return {"start": start, "target": target, "outcome": record["outcome"],
            "success": bool(record["success"]), "steps": int(record["steps"]),
            "commands": np.array(cap.cmds), "phases": list(cap.phases)}


# --------------------------------------------------------------------- stage 2
def to_delta_action(expert_ctrl: np.ndarray, current_target: np.ndarray,
                    cap: float, ctrl_low: np.ndarray, ctrl_high: np.ndarray) -> np.ndarray:
    """One delta-mode action that moves the running target toward the expert's command.

    Arm: the shortfall, clipped to the cap and rescaled to the action range (the env
    multiplies by the cap again, so a full-magnitude action is exactly one cap of travel).
    Gripper: the delta space keeps it absolute, so invert that same linear mapping.
    """
    arm = np.clip(expert_ctrl[:N_ARM] - current_target, -cap, cap) / cap
    span = ctrl_high[N_ARM:] - ctrl_low[N_ARM:]
    grip = 2.0 * (expert_ctrl[N_ARM:] - ctrl_low[N_ARM:]) / span - 1.0
    return np.concatenate([arm, grip]).astype(np.float32)


def replay(reference: dict, max_episode_steps: int = 200, reward_type: str = "shaped") -> dict:
    """Drive the delta-mode training env with `reference`'s command sequence."""
    env = _build_env("delta", max_episode_steps, reward_type)
    base = env.unwrapped
    cmds, phases = reference["commands"], reference["phases"]

    obs, info = env.reset(seed=0, options={"start_square": reference["start"],
                                           "target_square": reference["target"]})
    cap = base.max_joint_delta
    rows, lag = [], []
    grasp_ever = False
    closest = float("inf")
    term = trunc = False
    i = 0
    for i in range(len(cmds)):
        target_now = base._arm_target.copy()          # read-only
        action = to_delta_action(cmds[i], target_now, cap, base._ctrl_low, base._ctrl_high)
        obs, reward, term, trunc, info = env.step(action)
        lag.append(np.abs(cmds[i][:N_ARM] - base._arm_target))
        pinch = base.data.site_xpos[
            __import__("mujoco").mj_name2id(base.model, __import__("mujoco").mjtObj.mjOBJ_SITE,
                                            "ur_gripper_pinch")]
        closest = min(closest, float(np.linalg.norm(pinch - base._achieved_goal())))
        grasp_ever = grasp_ever or bool(info.get("grasp_success"))
        rows.append(dict(step=i, phase=phases[i], reward=float(reward),
                         dist=float(info["distance_to_goal"]),
                         grasp=bool(info.get("grasp_success"))))
        if term or trunc:
            break
    lag = np.array(lag)
    env.close()
    return dict(start=reference["start"], target=reference["target"],
                is_success=bool(info["is_success"]), grasp_success_ever=grasp_ever,
                final_distance_m=float(info["distance_to_goal"]),
                closest_pinch_to_piece_m=closest, steps=i + 1,
                commands_available=len(cmds), terminated=bool(term), truncated=bool(trunc),
                terminated_reason=info.get("terminated_reason"),
                phase_at_end=phases[min(i, len(phases) - 1)],
                phases_reached=sorted(set(phases[:i + 1]), key=phases.index),
                lag_median={j: float(np.median(lag[:, k])) for k, j in enumerate(JOINTS)},
                lag_max={j: float(lag[:, k].max()) for k, j in enumerate(JOINTS)},
                rows=rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--squares", default="a1,a8,h1,h8,d1,d8,b4,g6,c7,f2")
    p.add_argument("--target", default="e5")
    p.add_argument("--budget", type=int, default=200, help="replay budget (production default)")
    p.add_argument("--reference-budget", type=int, default=350)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    results = []
    for sq in args.squares.split(","):
        ref = record_reference(sq, args.target, args.reference_budget)
        res = replay(ref, args.budget)
        res["reference"] = {k: ref[k] for k in ("outcome", "success", "steps")}
        results.append(res)
        print(f"{sq}->{args.target}: success={res['is_success']} grasp={res['grasp_success_ever']} "
              f"dist={res['final_distance_m']:.3f} closest={res['closest_pinch_to_piece_m']:.3f} "
              f"steps={res['steps']}/{args.budget} end_phase={res['phase_at_end']} "
              f"reason={res['terminated_reason']}")
    n = sum(r["is_success"] for r in results)
    print(f"\nsuccess {n}/{len(results)}, grasp {sum(r['grasp_success_ever'] for r in results)}/{len(results)}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
