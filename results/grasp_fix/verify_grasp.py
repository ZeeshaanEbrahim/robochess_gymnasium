"""Section 4 verification: old (lifted AND closed) vs new (lifted AND closed AND both pads
touching the piece) grasp_success, step by step, on scripted-expert episodes."""
import json, sys
from pathlib import Path
import gymnasium as gym, numpy as np
REPO = Path.home()/"projects/robochess_gymnasium"; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts"))
import scripted_expert as se
from robochess.envs import RoboChessEnv
from robochess.tasks import T1PickAndPlaceTask

class Rec(gym.Wrapper):
    def reset(self, **kw):
        self.rows = []; return self.env.reset(**kw)
    def step(self, a):
        out = self.env.step(a); info = out[4]; b = self.env.unwrapped; t = self.env
        lifted = float(b._achieved_goal()[2]) > b._board_surface_z + b.rest_tolerance
        closed = float(b.data.qpos[b._grip_qadr]) >= b.gripper_open_threshold
        l, r = t._pad_piece_contact()
        self.rows.append(dict(step=info["elapsed_steps"], old=bool(lifted and closed), new=bool(info["grasp_success"]),
                              lifted=lifted, closed=closed, left_pad=l, right_pad=r,
                              piece_h_mm=round(1000*(float(b._achieved_goal()[2]) - b._board_surface_z), 1)))
        return out

env = RoboChessEnv(max_episode_steps=350); task = T1PickAndPlaceTask(env); rec = Rec(task); ex = se.ScriptedExpert(rec)
out = {}
for s, t in (("d1", "a8"), ("a8", "h1"), ("h8", "h1")):
    task.start_square, task.target_square = s, t
    r = ex.run_episode(seed=0); R = rec.rows
    first = lambda k: next((x["step"] for x in R if x[k]), None)
    mism = [x for x in R if x["old"] != x["new"]]
    out[f"{s}->{t}"] = dict(outcome=r["outcome"], success=r["success"], steps=len(R),
                            old_true_steps=sum(x["old"] for x in R), new_true_steps=sum(x["new"] for x in R),
                            first_old=first("old"), first_new=first("new"), last_new=max((x["step"] for x in R if x["new"]), default=None),
                            old_only_steps=[(x["step"], x["piece_h_mm"], x["left_pad"], x["right_pad"]) for x in mism if x["old"]][:8])
    o = out[f"{s}->{t}"]
    print(f"{s}->{t}: outcome={o['outcome']} steps={o['steps']} | old grasp steps {o['old_true_steps']} (first {o['first_old']}) | new grasp steps {o['new_true_steps']} (first {o['first_new']}, last {o['last_new']})")
    if o["old_only_steps"]: print(f"    steps old=True but new=False (step, piece h mm, left pad, right pad): {o['old_only_steps']}")
Path(REPO/"results/grasp_fix/expert_episodes.json").write_text(json.dumps(out, indent=1))
