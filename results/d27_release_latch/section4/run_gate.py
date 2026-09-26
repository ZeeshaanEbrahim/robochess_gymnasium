import sys, json
from pathlib import Path
root = Path.cwd(); sys.path.insert(0, str(root / "scripts")); sys.path.insert(0, str(root))
import robochess, robochess.tasks as tasks
last = {}
orig = tasks.T1PickAndPlaceTask.step
def step(self, a):
    o = orig(self, a); last["info"] = o[4]; return o
tasks.T1PickAndPlaceTask.step = step
import gate_expert_cartesian as g, scripted_expert as se
from robochess.envs import RoboChessEnv
expert = se.ScriptedExpert(RoboChessEnv())
out = []
for pair in g.DEFAULT_PAIRS.split(","):
    s, t = pair.split(":")
    r = g.run_pair(s, t, expert)
    i = last["info"]
    out.append(dict(pair=pair, harness_clean_place=r["clean_place"], harness_success=r["success"], failure=r["failure"],
                    env_is_success=bool(i["is_success"]), terminated_reason=i["terminated_reason"], steps=r["steps"],
                    release_height_mm=i.get("release_height_mm"), release_xy_error_mm=i.get("release_xy_error_mm"),
                    ever_held=i.get("ever_held"), release_ok=i.get("release_ok"),
                    base_above_rest_at_release_start_mm=(r["release_start"] or {}).get("base_above_rest_mm")))
json.dump(dict(code=robochess.__file__, pairs=out), open(sys.argv[1], "w"), indent=1)
print(robochess.__file__, "env is_success", sum(o["env_is_success"] for o in out), "/", len(out))
