import sys; sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import importlib.util, mujoco, numpy as np
spec = importlib.util.spec_from_file_location("rep", "scripts/replay_expert_delta.py")
rep = importlib.util.module_from_spec(spec); spec.loader.exec_module(rep)
from robochess.envs.robochess_env import N_ARM_ACTUATORS as N
for sq in ("a1", "d1", "b4"):
    ref = rep.record_reference(sq, "e5"); cmds, ph = ref["commands"], ref["phases"]
    env = rep._build_env("delta", 200); b = env.unwrapped; task = env.env
    pid = mujoco.mj_name2id(b.model, mujoco.mjtObj.mjOBJ_SITE, "ur_gripper_pinch")
    env.reset(seed=0, options={"start_square": sq, "target_square": "e5"})
    cap_hits = clamp_hits = n = 0
    grip_at, pads = [], []
    for i in range(len(cmds)):
        tgt_before = b._arm_target.copy()
        a = rep.to_delta_action(cmds[i], tgt_before, b.max_joint_delta, b._ctrl_low, b._ctrl_high)
        _, _, te, tr, info = env.step(a)
        if ph[i] in ("descend", "close"):
            n += 1
            moved = np.abs(b._arm_target - tgt_before)
            want = np.abs(np.clip(cmds[i][:N] - tgt_before, -b.max_joint_delta, b.max_joint_delta))
            cap_hits += int(np.any(np.abs(want - b.max_joint_delta) < 1e-9))
            clamp_hits += int(np.any(moved < want - 1e-9))     # clamp cut the step short
        if ph[i] == "close":
            grip_at.append(round(float(b.data.qpos[b._grip_qadr]), 3))
            pads.append(task._piece_contacts() if hasattr(task, "_piece_contacts") else None)
        if te or tr: break
    env.close()
    print(f"{sq}: descend+close steps {n} | steps where a joint wanted a FULL cap step {cap_hits} | "
          f"steps where the windup clamp cut the step short {clamp_hits}")
    print(f"   gripper qpos through close: {grip_at}")
    print(f"   pad/board contacts through close: {pads}")
