"""D28 Section 4 acceptance (pre-registered). Usage: python acceptance.py OUT.json

4a  expert path: the Cartesian gate harness (scripts/gate_expert_cartesian.py, unmodified)
    run with the env built through train.build_env with place_mode on; per-step reward
    and info are logged by a wrapper.  Recurring reward (state_reward) at nine checkpoints,
    must be non-decreasing in order.
4b  cheats vs the clean place on the same pair, T = 300: discounted (0.99) return is the
    pass test, undiscounted is reported.
4c  smoothness of the recurring reward on the 4a replays.
"""
import importlib.util, json, sys, warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "tests")]
import gymnasium as gym
import numpy as np
warnings.simplefilter("ignore")
import d27_scenarios as sc
import scripted_expert as se
import gate_expert_cartesian as gate
from robochess.envs import RoboChessEnv
from robochess.experiment import resolve_config
from robochess.tasks import PlaceReward, PlaceObservation, PickupObservation, T1PickAndPlaceTask

spec = importlib.util.spec_from_file_location("train", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)

T = 300
GAMMA = 0.99
PAIRS = [("e4", "e5"), ("a1", "h8"), ("c1", "f6"), ("b7", "g2")]
ORDER = ["home", "hover", "close", "first_held", "lifted_30mm", "mid_carry",
         "over_target_at_height", "set_down_pinched", "placed"]


def place_cfg(budget=T):
    return resolve_config(dict(task="t1", algorithm="ppo", reward_type="shaped", observation="state",
                               seed=0, total_timesteps=1, action_mode="cartesian",
                               max_episode_steps=budget, n_obstacles=0, pickup_obs=True,
                               place_obs=True, place_mode=True, vecnormalize_clip_reward=1000))


class Log(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env); self.log = []
    def step(self, a):
        out = self.env.step(a)
        self.log.append((float(out[1]), out[4]))
        return out


# ------------------------------------------------------------------------ 4a
def run_gate(start, target):
    holder = {}
    def build(mode, budget):
        holder["env"] = Log(train.build_env(place_cfg(budget)))
        return holder["env"]
    gate._build_env = build
    expert = se.ScriptedExpert(RoboChessEnv())
    res = gate.run_pair(start, target, expert, budget=T)
    return res, holder["env"].log


def checkpoints(res, log):
    rows = res["rows"]
    assert len(rows) == len(log), (len(rows), len(log))
    terms = [i["place_terms"] for _, i in log]
    ph = [r["phase"] for r in rows]
    last = lambda p: max(k for k, x in enumerate(ph) if x == p)
    first = lambda p: min(k for k, x in enumerate(ph) if x == p)
    st = [t["state"] for t in terms]
    k_first_held = next(k for k, t in enumerate(terms) if t["held"])
    k_close = max(k for k in range(k_first_held) if st[k] == "S0")     # last step at grasp pose before the first hold
    k_lift30 = next(k for k, t in enumerate(terms) if t["lift"] >= 0.03)
    tr = [k for k, x in enumerate(ph) if x == "traverse"]
    k_mid = tr[len(tr) // 2]
    k_placed = next(k for k, s in enumerate(st) if s == "S2")
    ks = dict(home=0, hover=last("approach"), close=k_close, first_held=k_first_held,
              lifted_30mm=k_lift30, mid_carry=k_mid, over_target_at_height=last("traverse"),
              set_down_pinched=last("place"), placed=k_placed)
    out = {}
    for name in ORDER:
        k = ks[name]; t = terms[k]
        out[name] = dict(step=k + 1, phase=ph[k], state=t["state"], recurring=t["state_reward"],
                         d_xy_mm=round(t["d_xy"] * 1e3, 2), lift_mm=round(t["lift"] * 1e3, 2),
                         reward_incl_bonuses=log[k][0])
    vals = [out[n]["recurring"] for n in ORDER]
    s2 = [t["state_reward"] for t in terms if t["state"] == "S2"]
    out["_s2_steps"] = len(s2)
    out["_s2_rewards"] = [round(v, 4) for v in s2]
    out["_s2_non_decreasing_and_ge_8"] = bool(all(b >= a - 1e-12 for a, b in zip(s2, s2[1:])) and min(s2) >= 8.0)
    out["_placed_between_8_and_9"] = bool(8.0 <= out["placed"]["recurring"] <= 9.0)
    out["_non_decreasing"] = all(b >= a for a, b in zip(vals, vals[1:]))
    out["_first_violation"] = next(((ORDER[i], vals[i], ORDER[i + 1], vals[i + 1])
                                    for i in range(len(vals) - 1) if vals[i + 1] < vals[i]), None)
    return out


def smoothness(log, res):
    terms = [i["place_terms"] for _, i in log]
    rec = np.array([t["state_reward"] for t in terms])
    d = np.diff(rec)
    ad = np.abs(d)
    ph = [r["phase"] for r in res["rows"]]
    top = np.argsort(-ad)[:5]
    return dict(
        n=int(len(d)), median_abs=float(np.median(ad)), p95=float(np.percentile(ad, 95)),
        p99=float(np.percentile(ad, 99)), max=float(ad.max()),
        top5=[dict(step=int(k + 2), phase=ph[k + 1], change=round(float(d[k]), 4),
                   from_state=terms[k]["state"], to_state=terms[k + 1]["state"]) for k in top],
        n_state_switch_steps=int(sum(terms[k]["state"] != terms[k + 1]["state"] for k in range(len(d)))),
        max_abs_within_state=float(max((ad[k] for k in range(len(d)) if terms[k]["state"] == terms[k + 1]["state"]), default=0)),
        one_time_events=[dict(step=t["t"], first_grasp=t["first_grasp_bonus"], lift=t["lift_bonus"],
                              placed=t["placed_bonus"], lump=round(t["success_lump"], 4))
                         for t in terms if t["first_grasp_bonus"] or t["lift_bonus"] or t["placed_bonus"] or t["success_lump"]])


# ------------------------------------------------------------------------ 4b
class Rec:
    def __init__(self, start, target):
        self.rewards, self.infos = [], []
        self.dr = sc.Driver(start, target, budget=T, env_factory=lambda task: PlaceReward(task))
        self.dr.on_step = self._on
        # "half": commanded aperture ~0.40 rad -- pads off the piece (not pinched) but above the
        # 0.2 open threshold, so the env never counts the gripper as released.
        self.dr._grip_action["half"] = 0.0
    def _on(self, out):
        self.rewards.append(float(out[1])); self.infos.append(out[4])
    @property
    def done(self): return self.dr.trace.terminated
    def to_end(self, grip):
        while not self.done:
            self.dr.step(np.zeros(3), grip)
    def summary(self):
        r = np.array(self.rewards)
        disc = float(np.sum(r * GAMMA ** np.arange(len(r))))
        last = self.infos[-1]
        terms = [i["place_terms"] for i in self.infos]
        return dict(rewards=[round(float(x), 6) for x in r],
                    state_rewards=[round(t["state_reward"], 6) for t in terms],
                    state_seq="".join(t["state"][1] for t in terms),
                    steps=len(r), undiscounted=float(r.sum()), discounted=disc,
                    success=bool(last["is_success"]), end_reason=last.get("terminated_reason"),
                    final_state=last["place_state"],
                    bonuses=dict(first_grasp=sum(t["first_grasp_bonus"] > 0 for t in terms),
                                 lift=sum(t["lift_bonus"] > 0 for t in terms),
                                 placed=sum(t["placed_bonus"] > 0 for t in terms),
                                 lump=sum(t["success_lump"] > 0 for t in terms),
                                 penalty=sum(t["failure_penalty"] > 0 for t in terms)))
    def state(self): return self.infos[-1]["place_state"] if self.infos else "S0"


def until_state(rec, state, grip, steps):
    for _ in range(steps):
        if rec.done or rec.state() == state:
            break
        rec.dr.step(np.zeros(3), grip)


def clean(rec):
    rec.dr.grasp_and_lift(); rec.dr.place_and_release(height_m=0.0015)

def push_then_open(rec):
    sc.push_then_open(rec.dr); rec.to_end("open")

def drop(h):
    def f(rec):
        rec.dr.grasp_and_lift(); rec.dr.carry_to(rec.dr.goal_xy, h); rec.dr.hold("open", 300); rec.to_end("open")
    return f

def hold_pinched(rec):
    rec.dr.grasp_and_lift(); rec.dr.carry_to(rec.dr.goal_xy, 0.0015); rec.to_end("closed")

def hover_lifted(rec):
    rec.dr.grasp_and_lift(); rec.dr.goto([*rec.dr.goal_xy, se.CLEAR_Z], "closed"); rec.to_end("closed")

def regrasp_release(rec):
    d = rec.dr
    d.grasp_and_lift(); d.carry_to(d.goal_xy, 0.0015)
    until_state(rec, "S2", "open", 60)              # released, on target
    if not rec.done:
        d.hold("closed", 4)                          # re-grasp
        d.hold("open", 300)                          # release again, until the end

def release_then_topple(rec):
    d = rec.dr
    d.grasp_and_lift(); d.carry_to(d.goal_xy, 0.0015)
    until_state(rec, "S2", "open", 60)              # released, on target
    for _ in range(8):                               # sweep sideways through the piece at grasp height
        if rec.done: break
        d.step([0.012, 0.0, 0.0], "open")
    rec.to_end("open")

def drop_off_target(rec):
    d = rec.dr
    d.grasp_and_lift(); d.carry_to(d.goal_xy + np.array([0.078 * 2, 0.0]), 0.0015)
    d.hold("open", 300); rec.to_end("open")

def reclose_on_air(rec):
    d = rec.dr
    d.grasp_and_lift(); d.carry_to(d.goal_xy, 0.0015)
    until_state(rec, "S2", "open", 60)              # released, on target
    if not rec.done:
        d.step([0.0, 0.0, 0.03], "open")            # clear the piece (a 20 mm/step step is the cap)
        for _ in range(3):
            if rec.done: break
            d.step([0.0, 0.0, 0.03], "open")
        d.hold("closed", 300)                        # close on air, stay
    rec.to_end("closed")

def partial_open_hold(action):
    """Release to a partial aperture and stay: action 0.0 parks at ~0.40 rad, -0.5 at ~0.203 rad
    (just above the 0.2 released threshold, where S2 pays almost its ceiling)."""
    def f(rec):
        d = rec.dr
        d._grip_action["half"] = action
        d.grasp_and_lift(); d.carry_to(d.goal_xy, 0.0015)
        rec.to_end("half")
    return f

def partial_open_topple(rec):
    d = rec.dr
    d.grasp_and_lift(); d.carry_to(d.goal_xy, 0.0015)
    until_state(rec, "S2", "half", 60)
    for _ in range(8):
        if rec.done: break
        d.step([0.012, 0.0, 0.0], "half")
    rec.to_end("half")

CASES = dict(clean=clean, push_then_open=push_then_open, drop_30mm=drop(0.030), drop_80mm=drop(0.080),
             hold_pinched_on_target=hold_pinched, hover_lifted_over_target=hover_lifted,
             release_regrasp_release=regrasp_release, release_then_topple_open_gripper=release_then_topple,
             drop_off_target=drop_off_target, release_lift_reclose_literal=reclose_on_air,
             release_partial_open_hold_0p40=partial_open_hold(0.0),
             release_partial_open_hold_0p203=partial_open_hold(-0.5), release_partial_open_topple=partial_open_topple)


def main(out_path):
    result = dict(T=T, gamma=GAMMA, acceptance_4a={}, smoothness={}, cheats={})
    for s, t in PAIRS:
        res, log = run_gate(s, t)
        pair = f"{s}->{t}"
        info = dict(gate_success=res["success"], clean_place=res["clean_place"], failure=res["failure"],
                    steps=res["steps"])
        try:
            cp = checkpoints(res, log)
        except Exception as e:                       # report, do not hide
            cp = dict(error=repr(e))
        result["acceptance_4a"][pair] = dict(harness=info, checkpoints=cp)
        try:
            result["smoothness"][pair] = smoothness(log, res)
        except Exception as e:
            result["smoothness"][pair] = dict(error=repr(e))
        print(pair, info, "non_decreasing:", cp.get("_non_decreasing"), flush=True)
    for s, t in PAIRS:
        pair = f"{s}->{t}"; result["cheats"][pair] = {}
        for name, fn in CASES.items():
            rec = Rec(s, t)
            try:
                fn(rec)
                sm = rec.summary()
            except Exception as e:
                sm = dict(error=repr(e))
            result["cheats"][pair][name] = sm
            rec.dr.base.close()
            print(pair, name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in sm.items() if k in ("steps", "success", "discounted", "undiscounted")}, flush=True)
    Path(out_path).write_text(json.dumps(result, indent=1))

if __name__ == "__main__":
    main(sys.argv[1])
