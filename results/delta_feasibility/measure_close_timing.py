import sys; sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import importlib.util, mujoco, numpy as np
spec = importlib.util.spec_from_file_location("rep", "scripts/replay_expert_delta.py")
rep = importlib.util.module_from_spec(spec); spec.loader.exec_module(rep)
from robochess.envs.robochess_env import N_ARM_ACTUATORS as N
print("| start | ok | close cmd step | phase steps descend/close | pinch-piece at close start: expert | replay | replay arm lag (rad) | replay catches within (steps) | pinch-piece at close END replay |")
print("|---|---|---|---|---|---|---|---|---|")
for sq in ("a1", "h1", "d1", "f2", "a8", "b4", "g6"):
    ref = rep.record_reference(sq, "e5")
    cmds, ph = ref["commands"], ref["phases"]
    close0 = ph.index("close"); close1 = len(ph) - 1 - ph[::-1].index("close")
    env = rep._build_env("delta", 200); b = env.unwrapped
    pid = mujoco.mj_name2id(b.model, mujoco.mjtObj.mjOBJ_SITE, "ur_gripper_pinch")
    env.reset(seed=0, options={"start_square": sq, "target_square": "e5"})
    d_at_close = lag_at_close = None; catch = None; d_close_end = None
    for i in range(len(cmds)):
        a = rep.to_delta_action(cmds[i], b._arm_target.copy(), b.max_joint_delta, b._ctrl_low, b._ctrl_high)
        _, _, te, tr, info = env.step(a)
        dist = float(np.linalg.norm(b.data.site_xpos[pid] - b._achieved_goal()))
        lag = np.abs(cmds[i][:N] - b._arm_target)
        if i == close0: d_at_close, lag_at_close = dist, lag.max()
        if i >= close0 and catch is None and lag.max() < 0.01: catch = i - close0
        if i == close1: d_close_end = dist
        if te or tr: break
    env.close()
    # expert's own distance at its close step
    eenv = rep._build_env("absolute", 350); eb = eenv.unwrapped
    epid = mujoco.mj_name2id(eb.model, mujoco.mjtObj.mjOBJ_SITE, "ur_gripper_pinch")
    eenv.reset(seed=0, options={"start_square": sq, "target_square": "e5"})
    ed = None
    span = eb._ctrl_high - eb._ctrl_low
    for i in range(len(cmds)):
        act = (2.0 * (cmds[i] - eb._ctrl_low) / span - 1.0).astype(np.float32)
        _, _, te, tr, _ = eenv.step(act)
        if i == close0: ed = float(np.linalg.norm(eb.data.site_xpos[epid] - eb._achieved_goal()))
        if te or tr: break
    eenv.close()
    ok = sq in ("a8", "b4", "g6", "d8", "h8", "c7")
    print(f"| {sq} | {'Y' if ok else 'N'} | {close0} | {close0-ph.index('descend')}/{close1-close0+1} | {ed:.4f} | {d_at_close:.4f} | {lag_at_close:.3f} | {catch} | {d_close_end:.4f} |")
