"""Verify grasp_success's board-contact term (D18 gap).
(a) Scripted-expert carries: new flag vs the pads-only definition, step by step.
(b) A toppled piece laid on its side on e4, then the expert's own IK/drive closes the gripper
    across it at a sweep of pinch heights (base or stem region, 0-14 mm above the region's
    centre). At +11/+14 mm the pads press the piece into the board: both pads and the board
    touch it every step -- the pads-only definition fires, the new one must not. Lower pinch
    heights squeeze the piece up out of the fingers; those transient steps are reported too."""
import json, sys
from pathlib import Path
import gymnasium as gym, mujoco, numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import scripted_expert as se
from robochess.envs import RoboChessEnv
from robochess.tasks import T1PickAndPlaceTask

def terms(task):
    b = task.unwrapped
    lifted = float(b._achieved_goal()[2]) > b._board_surface_z + b.rest_tolerance
    closed = float(b.data.qpos[b._grip_qadr]) >= b.gripper_open_threshold
    l, r, board = task._piece_contacts()
    return dict(lifted=lifted, closed=closed, left=l, right=r, board=board,
                pads_only=bool(lifted and closed and l and r), h_mm=round(1000 * (float(b._achieved_goal()[2]) - b._board_surface_z), 1))

class Rec(gym.Wrapper):
    def reset(self, **kw):
        self.rows = []; return self.env.reset(**kw)
    def step(self, a):
        out = self.env.step(a); t = terms(self.env); t["new"] = bool(out[4]["grasp_success"]); self.rows.append(t); return out

out = {"carry": {}}
env = RoboChessEnv(max_episode_steps=350); task = T1PickAndPlaceTask(env); rec = Rec(task); ex = se.ScriptedExpert(rec)
for s, t in (("d1", "a8"), ("a8", "h1"), ("h8", "h1"), ("e4", "e5")):
    task.start_square, task.target_square = s, t
    r = ex.run_episode(seed=0); R = rec.rows
    o = dict(outcome=r["outcome"], steps=len(R), pads_only_steps=sum(x["pads_only"] for x in R), new_steps=sum(x["new"] for x in R),
             disagree=sum(x["pads_only"] != x["new"] for x in R), board_contact_while_both_pads=sum(x["board"] and x["left"] and x["right"] for x in R),
             board_contact_while_lifted=sum(x["board"] and x["lifted"] for x in R))
    out["carry"][f"{s}->{t}"] = o; print(f"carry {s}->{t}: {o}")
env.close()

# (b) toppled piece pinched on the board, pinch-height sweep
BASE_R, BASE_HALF = 0.015, 0.009
def run(offset_mm, region):
    env = RoboChessEnv(max_episode_steps=400); task = T1PickAndPlaceTask(env, start_square="e4", target_square="e6"); ex = se.ScriptedExpert(task)
    b = env; m, d = b.model, b.data
    task.reset(seed=0)
    lp, rp = d.xpos[task._left_pad_bid].copy(), d.xpos[task._right_pad_bid].copy()
    ca = lp - rp; ca[2] = 0; ca /= np.linalg.norm(ca)
    ax = np.array([-ca[1], ca[0], 0.0])
    v = np.cross([0, 0, 1], ax); quat = np.concatenate([[np.cos(np.pi / 4)], np.sin(np.pi / 4) * v / np.linalg.norm(v)])
    qa = b._piece_qadr; xy = d.qpos[qa:qa + 2].copy()
    d.qpos[qa:qa + 3] = np.array([xy[0], xy[1], b._board_surface_z + BASE_R]) - ax * BASE_HALF
    d.qpos[qa + 3:qa + 7] = quat; d.qvel[b._piece_vadr:b._piece_vadr + 6] = 0; mujoco.mj_forward(m, d)
    for _ in range(10): task.step(b.home_action())
    R = d.xmat[b._piece_bid].reshape(3, 3)
    along = BASE_HALF if region == "base" else 0.026           # base centre or stem centre along the piece axis
    c = d.xpos[b._piece_bid] + R[:, 2] * along
    z = float(c[2]) + offset_mm / 1000
    rows = []
    for label, zz, grip, budget in (("approach", 0.882, 0.0, 25), ("descend", z, 0.0, 40), ("close", z, 255.0, 45), ("hold", z, 255.0, 30)):
        q = ex.solve_ik(np.array([c[0], c[1], zz]))
        for obs, _, te, tr, info in ex._drive_to(q, grip, budget, label=label):
            l, r, board = task._piece_contacts()
            lifted = float(b._achieved_goal()[2]) > b._board_surface_z + b.rest_tolerance
            closed = float(d.qpos[b._grip_qadr]) >= b.gripper_open_threshold
            rows.append((label, lifted, closed, l, r, board, bool(info["grasp_success"]), 1000 * (float(b._achieved_goal()[2]) - b._board_surface_z)))
            if te or tr: break
    tilt = float(np.degrees(np.arccos(np.clip(d.xmat[b._piece_bid].reshape(3, 3)[2, 2], -1, 1))))
    after_close = [x for x in rows if x[0] in ("close", "hold")]
    hold = [x for x in rows if x[0] == "hold"]
    both = [x for x in after_close if x[3] and x[4]]
    rec = dict(final_tilt_deg=round(tilt, 1), final_h_mm=round(hold[-1][7], 1), final_grip=round(float(d.qpos[b._grip_qadr]), 3),
               close_hold_steps=len(after_close), both_pads=len(both), both_pads_and_board=sum(x[5] for x in both),
               pads_only_def=sum(x[1] and x[2] for x in both), new_flag=sum(x[6] for x in rows),
               h_mm_while_new_flag=[round(x[7], 1) for x in rows if x[6]])
    print(f"{region:4s} +{offset_mm:2d}mm | final tilt {tilt:5.1f} deg, h {hold[-1][7]:5.1f} mm, grip {float(d.qpos[b._grip_qadr]):.3f} | "
          f"close+hold steps {len(after_close)}: both pads {len(both)}, both pads & board {sum(x[5] for x in both)}, "
          f"lifted&closed&both pads (old D18 def) {sum(x[1] and x[2] for x in both)}, new flag {sum(x[6] for x in rows)} | "
          f"h range while flag {[round(x[7],1) for x in rows if x[6]][:6]} | hold last: L{hold[-1][3]:d} R{hold[-1][4]:d} B{hold[-1][5]:d} new{hold[-1][6]:d}")
    env.close()
    return rec
out["toppled_sweep"] = {f"{region}+{off}mm": run(off, region) for region in ("base", "stem") for off in (0, 4, 8, 11, 14)}
(Path(__file__).parent / "board_term_verification.json").write_text(json.dumps(out, indent=1, default=bool))
