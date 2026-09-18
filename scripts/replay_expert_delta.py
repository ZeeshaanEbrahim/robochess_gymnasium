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
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import scripted_expert as se  # noqa: E402
from robochess.envs import RoboChessEnv  # noqa: E402
from robochess.envs.robochess_env import N_ARM_ACTUATORS as N_ARM  # noqa: E402
from robochess.tasks import T1PickAndPlaceTask  # noqa: E402

JOINTS = ("pan", "lift", "elbow", "w1", "w2", "w3")


def _pinch_sid(base) -> int:
    return mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_SITE, "ur_gripper_pinch")


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
            self.cmds, self.phases, self.pinch_dist = [], [], []
            return self.env.reset(**kw)

        def step(self, a):
            out = self.env.step(a)
            b = self.unwrapped
            self.cmds.append(b.data.ctrl.copy())
            self.phases.append(self.phase)
            self.pinch_dist.append(float(np.linalg.norm(
                b.data.site_xpos[_pinch_sid(b)] - b._achieved_goal())))
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
    phases = list(cap.phases)
    close0 = phases.index("close") if "close" in phases else None
    return {"start": start, "target": target, "outcome": record["outcome"],
            "success": bool(record["success"]), "steps": int(record["steps"]),
            "commands": np.array(cap.cmds), "phases": phases,
            "close_step": close0,
            # what the expert's own pinch site had achieved when it commanded the close
            "pinch_at_close_m": None if close0 is None else cap.pinch_dist[close0]}


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


def replay(reference: dict, max_episode_steps: int = 200, reward_type: str = "shaped",
           close_gate: bool = False) -> dict:
    """Drive the delta-mode training env with `reference`'s command sequence.

    `close_gate` changes only WHEN the close command is issued, never what the
    commands are: instead of firing the expert's close on its scheduled step, the
    final descend command is held until the pinch site is at least as close to the
    piece as the expert's own pinch was when IT commanded the close, and only then
    does the rest of the sequence run, unchanged. The expert's schedule is a timer
    fixed by its own settle checks in a faster (absolute-mode) run; a closed-loop
    policy has no such timer, so this separates "the action space cannot do it"
    from "the recorded schedule fires too early for it". The wait is capped at the
    episode's remaining steps. The env is untouched either way.
    """
    env = _build_env("delta", max_episode_steps, reward_type)
    base = env.unwrapped
    cmds, phases = reference["commands"], reference["phases"]
    close0 = reference.get("close_step")
    gate_dist = reference.get("pinch_at_close_m")

    obs, info = env.reset(seed=0, options={"start_square": reference["start"],
                                           "target_square": reference["target"]})
    sid = _pinch_sid(base)
    cap = base.max_joint_delta
    rows, lag = [], []
    grasp_ever = False
    closest = float("inf")
    term = trunc = False
    consumed = 0
    hold_min = hold_last = None
    waited = 0
    wait_cap_hit = False
    gate_reached = None
    pinch_at_close = None

    def drive(cmd, phase):
        """One env step toward `cmd`; returns the pinch-to-piece distance after it."""
        nonlocal obs, info, term, trunc, grasp_ever, closest, consumed
        action = to_delta_action(cmd, base._arm_target.copy(), cap, base._ctrl_low, base._ctrl_high)
        obs, reward, term, trunc, info = env.step(action)
        consumed += 1
        lag.append(np.abs(cmd[:N_ARM] - base._arm_target))
        dist = float(np.linalg.norm(base.data.site_xpos[sid] - base._achieved_goal()))
        closest = min(closest, dist)
        grasp_ever = grasp_ever or bool(info.get("grasp_success"))
        rows.append(dict(step=consumed - 1, phase=phase, reward=float(reward),
                         dist=float(info["distance_to_goal"]),
                         pinch_to_piece=dist, grasp=bool(info.get("grasp_success"))))
        return dist

    for i in range(len(cmds)):
        if close_gate and close0 is not None and i == close0 and gate_dist is not None:
            hold = cmds[i - 1]          # the last descend command, held
            gate_reached = False
            hold_dists = []
            while consumed < max_episode_steps:
                dist = drive(hold, "descend_hold")
                waited += 1
                hold_dists.append(dist)
                if dist <= gate_dist:
                    gate_reached = True
                    break
                if term or trunc:
                    break
            wait_cap_hit = not gate_reached
            hold_min = min(hold_dists) if hold_dists else None
            hold_last = hold_dists[-1] if hold_dists else None
            pinch_at_close = rows[-1]["pinch_to_piece"] if rows else None
            if term or trunc:
                break
        dist = drive(cmds[i], phases[i])
        if close0 is not None and i == close0 and pinch_at_close is None:
            pinch_at_close = dist
        if term or trunc:
            break
    lag = np.array(lag)
    env.close()
    return dict(start=reference["start"], target=reference["target"],
                close_gate=bool(close_gate),
                is_success=bool(info["is_success"]), grasp_success_ever=grasp_ever,
                final_distance_m=float(info["distance_to_goal"]),
                closest_pinch_to_piece_m=closest, steps=consumed,
                steps_waited_before_close=waited, wait_cap_hit=bool(wait_cap_hit),
                gate_reached=gate_reached,
                hold_min_pinch_m=(hold_min if close_gate and gate_reached is not None else None),
                hold_final_pinch_m=(hold_last if close_gate and gate_reached is not None else None),
                pinch_at_close_m=pinch_at_close,
                expert_pinch_at_close_m=gate_dist,
                commands_available=len(cmds), terminated=bool(term), truncated=bool(trunc),
                terminated_reason=info.get("terminated_reason"),
                phase_at_end=rows[-1]["phase"] if rows else None,
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
    p.add_argument("--close-gate", action="store_true",
                   help="hold the final descend command until the pinch reaches the distance "
                        "the expert itself had at close, instead of closing on its schedule")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    results = []
    for sq in args.squares.split(","):
        ref = record_reference(sq, args.target, args.reference_budget)
        res = replay(ref, args.budget, close_gate=args.close_gate)
        res["reference"] = {k: ref[k] for k in ("outcome", "success", "steps")}
        results.append(res)
        print(f"{sq}->{args.target}: success={res['is_success']} grasp={res['grasp_success_ever']} "
              f"pinch_at_close={res['pinch_at_close_m']:.4f} waited={res['steps_waited_before_close']} "
              f"cap_hit={res['wait_cap_hit']} steps={res['steps']}/{args.budget} "
              f"dist={res['final_distance_m']:.3f} reason={res['terminated_reason']}")
    n = sum(r["is_success"] for r in results)
    print(f"\nsuccess {n}/{len(results)}, grasp {sum(r['grasp_success_ever'] for r in results)}/{len(results)}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
