"""Replay the scripted expert's timed commands through Cartesian control.

The reference is recorded in the absolute-mode expert environment. Each arm
command is converted by forward kinematics to its commanded pinch position;
the Cartesian training environment then receives a bounded displacement from
its measured pinch position toward that point. Gripper commands and the expert's
step schedule are retained. This tests that particular replay, not whether some
other Cartesian policy could solve the task.

Usage:
    python scripts/replay_expert_cartesian.py --out results/cartesian_replay/replay_200.json
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

from replay_expert_delta import _build_env, record_reference


def commanded_pinch_positions(model: mujoco.MjModel, commands: np.ndarray,
                               home_key: int, pinch_sid: int) -> np.ndarray:
    """Forward-kinematics positions of the reference arm's commanded poses."""
    data = mujoco.MjData(model)
    positions = []
    for ctrl in commands:
        mujoco.mj_resetDataKeyframe(model, data, home_key)
        data.qpos[:6] = ctrl[:6]
        mujoco.mj_forward(model, data)
        positions.append(data.site_xpos[pinch_sid].copy())
    return np.asarray(positions)


def replay(reference: dict, max_episode_steps: int = 200) -> dict:
    env = _build_env("cartesian", max_episode_steps)
    base = env.unwrapped
    targets = commanded_pinch_positions(base.model, reference["commands"],
                                        base._home_key, base._pinch_sid)
    _, info = env.reset(seed=0, options={"start_square": reference["start"],
                                         "target_square": reference["target"]})
    phases = reference["phases"]
    rows = []
    grasp_ever = False
    closest = float("inf")
    term = trunc = False
    terminal_contacts = []
    for i, (target, ctrl) in enumerate(zip(targets, reference["commands"])):
        pinch = base.data.site_xpos[base._pinch_sid].copy()
        displacement = np.clip((target - pinch) / base.cartesian_step, -1.0, 1.0)
        span = base._ctrl_high[6:] - base._ctrl_low[6:]
        gripper = 2.0 * (ctrl[6:] - base._ctrl_low[6:]) / span - 1.0
        action = np.concatenate([displacement, gripper]).astype(np.float32)
        _, reward, term, trunc, info = env.step(action)
        dist = float(np.linalg.norm(base.data.site_xpos[base._pinch_sid]
                                    - base._achieved_goal()))
        closest = min(closest, dist)
        grasp_ever |= bool(info.get("grasp_success"))
        rows.append(dict(step=i, phase=phases[i], reward=float(reward),
                         pinch_to_piece_m=dist,
                         pinch_to_command_m=float(np.linalg.norm(
                             base.data.site_xpos[base._pinch_sid] - target)),
                         grasp=bool(info.get("grasp_success"))))
        if term or trunc:
            terminal_contacts = [
                [mujoco.mj_id2name(base.model, mujoco.mjtObj.mjOBJ_GEOM,
                                   int(c.geom1)),
                 mujoco.mj_id2name(base.model, mujoco.mjtObj.mjOBJ_GEOM,
                                   int(c.geom2))]
                for c in base.data.contact
            ]
            break
    result = dict(start=reference["start"], target=reference["target"],
                  reference={k: reference[k] for k in ("outcome", "success", "steps")},
                  is_success=bool(info["is_success"]), grasp_success_ever=grasp_ever,
                  final_distance_m=float(info["distance_to_goal"]),
                  closest_pinch_to_piece_m=closest, steps=len(rows),
                  commands_available=len(targets), terminated=bool(term),
                  truncated=bool(trunc), terminated_reason=info.get("terminated_reason"),
                  phase_at_end=rows[-1]["phase"] if rows else None,
                  ik_rejects=int(base._ik_rejects),
                  board_contact_force_n=info.get("board_contact_force_n"),
                  terminal_contacts=terminal_contacts, rows=rows)
    env.close()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--squares", default="a1,a8,h1,h8,d1,d8,b4,g6,c7,f2")
    p.add_argument("--target", default="e5")
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--reference-budget", type=int, default=350)
    p.add_argument("--out", type=str)
    args = p.parse_args()
    results = []
    for square in args.squares.split(","):
        reference = record_reference(square, args.target, args.reference_budget)
        result = replay(reference, args.budget)
        results.append(result)
        print(f"{square}->{args.target}: reference={reference['success']} "
              f"replay={result['is_success']} grasp={result['grasp_success_ever']} "
              f"steps={result['steps']}/{args.budget} phase={result['phase_at_end']} "
              f"reason={result['terminated_reason']} rejects={result['ik_rejects']}",
              flush=True)
    print(f"success {sum(r['is_success'] for r in results)}/{len(results)}, "
          f"grasp {sum(r['grasp_success_ever'] for r in results)}/{len(results)}")
    if args.out:
        from pathlib import Path
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=1))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
