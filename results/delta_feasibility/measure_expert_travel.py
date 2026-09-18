"""Section 2: total absolute joint travel the scripted expert needs, vs the per-episode
ceiling measured by measure_joint_speed.py. Usage: python <this> joint_speed.json out.json"""
import json, sys; sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import gymnasium as gym, numpy as np
import scripted_expert as se
from robochess.envs import RoboChessEnv
from robochess.tasks import T1PickAndPlaceTask
from robochess.envs.robochess_env import N_ARM_ACTUATORS as N
J = ["pan", "lift", "elbow", "w1", "w2", "w3"]
SQUARES = ["a1", "a8", "h1", "h8", "d1", "d8", "b4", "g6", "c7", "f2"]
TARGET = "e5"
ceil = json.load(open(sys.argv[1]))
CEIL = np.array([ceil[k]["travel_per_episode_rad"] for k in
                 ("ur_shoulder_pan", "ur_shoulder_lift", "ur_elbow", "ur_wrist_1", "ur_wrist_2", "ur_wrist_3")])

class Travel(gym.Wrapper):
    def reset(self, **kw):
        out = self.env.reset(**kw)
        self.prev = self.unwrapped.data.qpos[self.unwrapped._arm_qadr].copy()
        self.total = np.zeros(N); self.steps = 0
        return out
    def step(self, a):
        out = self.env.step(a)
        q = self.unwrapped.data.qpos[self.unwrapped._arm_qadr].copy()
        self.total += np.abs(q - self.prev); self.prev = q; self.steps += 1
        return out

env = RoboChessEnv(max_episode_steps=350); task = T1PickAndPlaceTask(env, target_square=TARGET); rec = Travel(task)
ex = se.ScriptedExpert(rec)
marks = {}
orig = se.ScriptedExpert._run_waypoint
def patched(self, pinch_target, grip, budget, label, log, q_arm, compensate):
    r = orig(self, pinch_target, grip, budget, label, log, q_arm, compensate)
    marks[label] = (rec.total.copy(), rec.steps)
    return r
se.ScriptedExpert._run_waypoint = patched
rows = {}
for sq in SQUARES:
    task.start_square = sq; marks.clear()
    r = ex.run_episode(seed=0)
    g_t, g_s = marks.get("close", (np.full(N, np.nan), None))
    p_t, p_s = marks.get("place", (np.full(N, np.nan), None))
    rows[sq] = dict(outcome=r["outcome"], success=r["success"], steps=r["steps"],
                    to_grasp=g_t.tolist(), steps_to_grasp=g_s, to_place=p_t.tolist(), steps_to_place=p_s,
                    full=rec.total.tolist())
    print(f"{sq}->{TARGET} {r['outcome']:<12} steps {r['steps']:3d} | to grasp (step {g_s}): " +
          " ".join(f"{j}{v:5.2f}" for j, v in zip(J, g_t)) + f" | to place (step {p_s}): " +
          " ".join(f"{j}{v:5.2f}" for j, v in zip(J, p_t)))
print("\nceiling (rad per 200-step episode):", " ".join(f"{j}{c:5.2f}" for j, c in zip(J, CEIL)))
print("\n| start | phase | " + " | ".join(J) + " | worst joint (fraction of ceiling) |")
print("|---|---|" + "---|" * (N + 1))
for sq, r in rows.items():
    for ph, key in (("to grasp", "to_grasp"), ("to place", "to_place")):
        v = np.array(r[key]); frac = v / CEIL
        worst = int(np.nanargmax(frac))
        print(f"| {sq} | {ph} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {J[worst]} {frac[worst]:.2f}x |")
over = {sq: {J[i]: float(np.array(r['to_place'])[i] / CEIL[i]) for i in range(N) if np.array(r['to_place'])[i] > CEIL[i]} for sq, r in rows.items()}
print("\nover budget (to place):", json.dumps(over))
json.dump(dict(rows=rows, ceiling=CEIL.tolist(), squares=SQUARES, target=TARGET), open(sys.argv[2], "w"), indent=1)
