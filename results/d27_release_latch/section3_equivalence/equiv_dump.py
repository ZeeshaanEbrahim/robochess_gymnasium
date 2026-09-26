"""Per-step dump for old-vs-new equivalence: contact flags, D21 reward, pinched, held.
Run under either tree (PYTHONPATH selects the code). Output: JSON lines, floats via repr."""
import sys, json
sys.path.insert(0, "/root/projects/robochess_gymnasium/tests")
import numpy as np, robochess
import d27_scenarios as sc
import scripted_expert as se
from robochess.envs import RoboChessEnv
from robochess.tasks import T1PickAndPlaceTask, PickupReward

out_path = sys.argv[1]
rows = []
fresh_mismatch = [0, 0]  # [steps, cached != fresh scan]  (new tree only)

def record(tag, env_top, task, out, k):
    obs, r, term, trunc, info = out
    t = info["pickup_terms"]
    flags = task._piece_contacts()
    if hasattr(task.unwrapped, "_piece_contact_flags"):
        fresh_mismatch[0] += 1
        fresh_mismatch[1] += int(tuple(flags) != tuple(task.unwrapped._piece_contact_flags()))
    rows.append(dict(tag=tag, k=k, reward=repr(float(r)), flags=[bool(x) for x in flags],
                     pinched=bool(t["pinched"]), held=bool(t["held"]),
                     first_grasp_bonus=repr(float(t["first_grasp_bonus"])),
                     terms={kk: repr(float(v)) if isinstance(v, (float, np.floating)) else v for kk, v in sorted(t.items())}))

# --- joint (absolute) expert replay, pickup_mode on ---------------------------------
for start, target in (("e4", "e5"), ("c1", "f6")):
    base = RoboChessEnv(max_episode_steps=350)
    task = T1PickAndPlaceTask(base, start_square=start, target_square=target)
    top = PickupReward(task)
    k = [0]
    orig = top.step
    def stepper(a, orig=orig, top=top, task=task, k=k, tag=f"joint {start}->{target}"):
        o = orig(a); record(tag, top, task, o, k[0]); k[0] += 1; return o
    top.step = stepper
    se.ScriptedExpert(top).run_episode(seed=0)
    base.close()

# --- Cartesian replays, pickup_mode on: a clean carry and a push (pinched, not held) -
def factory(task): return PickupReward(task)
for tag, start, target, kind in (("cart c1->f6 place", "c1", "f6", "place"), ("cart e4->e5 push", "e4", "e5", "push")):
    dr = sc.Driver(start, target, env_factory=factory)
    task = dr.env.env
    k = [0]
    dr.on_step = lambda o, dr=dr, task=task, k=k, tag=tag: (record(tag, dr.env, task, o, k[0]), k.__setitem__(0, k[0] + 1))
    if kind == "place":
        dr.grasp_and_lift(); dr.place_and_release(height_m=0.0015)
    else:
        sc.push_then_open(dr)

json.dump(dict(code=robochess.__file__, fresh_scan_check=fresh_mismatch, rows=rows), open(out_path, "w"), indent=0, sort_keys=True)
first = {}
for r in rows:
    if r["held"] and r["tag"] not in first: first[r["tag"]] = r["k"]
print(robochess.__file__, "| steps:", len(rows), "| first-grasp (first held) step per replay:", first, "| fresh-scan check [steps, mismatches]:", fresh_mismatch)
