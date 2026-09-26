"""Run the repo's own fullboard validation (joint or T2) unchanged, and additionally capture
the terminal info per episode (release_height_mm etc.). Tree chosen by PYTHONPATH/cwd."""
import sys, json, time
from pathlib import Path
which, out = sys.argv[1], sys.argv[2]
n_obs = int(sys.argv[3]) if len(sys.argv) > 3 else 0
root = Path.cwd()
sys.path.insert(0, str(root / "scripts")); sys.path.insert(0, str(root))
import robochess
targets = ["e5", "a1", "h8", "a8", "h1"]
last = {}
import robochess.tasks as tasks
orig_step = tasks.T1PickAndPlaceTask.step
def step(self, a):
    o = orig_step(self, a); last["info"] = o[4]; return o
tasks.T1PickAndPlaceTask.step = step
if which == "joint":
    import fullboard_validation as m
    import scripted_expert as se
else:
    import fullboard_validation_t2 as m
    import scripted_expert as se
infos = []
orig_run = se.ScriptedExpert.run_episode
def run_episode(self, *a, **k):
    rec = orig_run(self, *a, **k)
    i = last.get("info", {})
    infos.append({kk: i.get(kk) for kk in ("release_height_mm", "release_xy_error_mm", "ever_held", "release_ok", "is_success", "terminated_reason")})
    return rec
se.ScriptedExpert.run_episode = run_episode
t0 = time.time()
if which == "joint":
    results, el = m.run(targets, 350, "new", progress_every=0)
else:
    results, el = m.run(targets, 350, n_obs, progress_every=0)
assert len(results) == len(infos)
for r, i in zip(results, infos): r.update(i)
json.dump(dict(code=robochess.__file__, results=results), open(out, "w"), indent=1)
print(robochess.__file__, which, n_obs, "successes", sum(r["success"] for r in results), "/", len(results), "elapsed %.0fs" % el)
