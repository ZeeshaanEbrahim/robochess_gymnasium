"""D29 Section 4 acceptance (pre-registered). Usage: python acceptance.py OUT.json

Adapted from results/d28_place_reward/acceptance.py with the D29 gate and
carry-potential flags on throughout (place_posture_gate, place_carry_potential,
default tilt/k). Additions over the D28 version:

4a  each logged step also reports the D28-equivalent state_reward (recomputed
    from the logged lift_frac/closeness/near -- the posture gate is the only
    thing that can move state_reward off the D28 value, and only in S1), the
    max |diff| restricted to S1 steps with tilt <= 10 deg, the max per-step
    "gate jump" (the gate's own contribution to a step-to-step state_reward
    change), and shaping per checkpoint plus the episode's shaping total.
4b  the 12 D28 cheats, run through the D29-flagged reward; plus shaping_farm
    (oscillate toward/away from the target while held, to timeout) and
    push_then_grasp_release_on_target (report only, not judged); plus
    toppled_hold, e4->e5 ONLY: replays the exact 150 actions recorded from a
    real D28 SAC seed0 episode (find_topple_episode.py, topple_actions.json)
    that ends pinched at tilt 90.61 deg at t=150, then holds (zero
    displacement, closed gripper) to timeout. No SB3 model is built in this
    process; the recorded actions are pure numbers.
4c  reported from the toppled_hold replay: S1 reward at every pinched step
    with tilt >= 45 deg (must all be <= 2.2).
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

CARRY_WEIGHT, LOWER_WEIGHT = 2.0, 4.0     # D28's defaults; every existing place config uses them (verified D29 step 2)
GATE_LO, GATE_HI, POT_K = 10.0, 45.0, 50.0

PLACE_KWARGS = dict(posture_gate=True, gate_tilt_lo_deg=GATE_LO, gate_tilt_hi_deg=GATE_HI,
                    carry_potential=True, carry_potential_k=POT_K)


def place_cfg(budget=T):
    return resolve_config(dict(task="t1", algorithm="ppo", reward_type="shaped", observation="state",
                               seed=0, total_timesteps=1, action_mode="cartesian",
                               max_episode_steps=budget, n_obstacles=0, pickup_obs=True,
                               place_obs=True, place_mode=True, vecnormalize_clip_reward=1000,
                               place_posture_gate=True, place_carry_potential=True))


def d28_equivalent_state_reward(t: dict) -> float:
    """What D28 (flag-off) would have paid this step. Only S1 can differ from
    the logged D29 state_reward -- S0/S2/S3 are untouched by the posture gate."""
    if t["state"] != "S1":
        return t["state_reward"]
    return (2.0 + 2.0 * t["lift_frac"] + CARRY_WEIGHT * t["closeness"]
            + LOWER_WEIGHT * t["near"] * (1.0 - t["lift_frac"]))


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
    k_close = max(k for k in range(k_first_held) if st[k] == "S0")
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
                         reward_incl_bonuses=log[k][0], shaping=round(t["shaping"], 6),
                         posture_gate=round(t["posture_gate"], 4))
    vals = [out[n]["recurring"] for n in ORDER]
    s2 = [t["state_reward"] for t in terms if t["state"] == "S2"]
    out["_s2_steps"] = len(s2)
    out["_s2_rewards"] = [round(v, 4) for v in s2]
    out["_s2_non_decreasing_and_ge_8"] = bool(all(b >= a - 1e-12 for a, b in zip(s2, s2[1:])) and min(s2) >= 8.0)
    out["_placed_between_8_and_9"] = bool(8.0 <= out["placed"]["recurring"] <= 9.0)
    out["_non_decreasing"] = all(b >= a for a, b in zip(vals, vals[1:]))
    out["_first_violation"] = next(((ORDER[i], vals[i], ORDER[i + 1], vals[i + 1])
                                    for i in range(len(vals) - 1) if vals[i + 1] < vals[i]), None)

    # D29: state_reward vs D28-equivalent, restricted to S1 steps with tilt <= 10 deg.
    d28_seq = [d28_equivalent_state_reward(t) for t in terms]
    d29_seq = [t["state_reward"] for t in terms]
    s1_le10_diffs = [d29_seq[k] - d28_seq[k] for k in range(len(terms))
                     if terms[k]["state"] == "S1" and terms[k]["piece_tilt_deg"] <= 10.0]
    out["_n_s1_tilt_le_10_steps"] = len(s1_le10_diffs)
    out["_max_abs_diff_vs_d28_tilt_le_10"] = float(max((abs(d) for d in s1_le10_diffs), default=0.0))

    # gate-attributable per-step jump: (D29 step change) - (D28-equivalent step change).
    gate_jumps = [(d29_seq[k] - d29_seq[k - 1]) - (d28_seq[k] - d28_seq[k - 1]) for k in range(1, len(terms))]
    out["_max_abs_gate_jump"] = float(max((abs(g) for g in gate_jumps), default=0.0))

    out["_shaping_total"] = float(sum(t["shaping"] for t in terms))
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
        self.dr = sc.Driver(start, target, budget=T,
                            env_factory=lambda task: PlaceReward(task, **PLACE_KWARGS))
        self.dr.on_step = self._on
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
                    shaping=[round(t["shaping"], 6) for t in terms],
                    state_seq="".join(t["state"][1] for t in terms),
                    steps=len(r), undiscounted=float(r.sum()), discounted=disc,
                    shaping_total=float(sum(t["shaping"] for t in terms)),
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
    until_state(rec, "S2", "open", 60)
    if not rec.done:
        d.hold("closed", 4)
        d.hold("open", 300)

def release_then_topple(rec):
    d = rec.dr
    d.grasp_and_lift(); d.carry_to(d.goal_xy, 0.0015)
    until_state(rec, "S2", "open", 60)
    for _ in range(8):
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
    until_state(rec, "S2", "open", 60)
    if not rec.done:
        d.step([0.0, 0.0, 0.03], "open")
        for _ in range(3):
            if rec.done: break
            d.step([0.0, 0.0, 0.03], "open")
        d.hold("closed", 300)
    rec.to_end("closed")

def partial_open_hold(action):
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

# ---- D29 additions ----
def shaping_farm(rec):
    """While held, oscillate toward and away from the target to timeout (never releases)."""
    d = rec.dr
    d.grasp_and_lift()
    if d.trace.terminated:
        return
    to_goal = d.goal_xy - d.piece_xy
    n = np.linalg.norm(to_goal)
    u = to_goal / n if n > 1e-9 else np.array([1.0, 0.0])
    swing = 0.01  # 10 mm per half-cycle, 5 steps each way
    sign = 1.0
    while not d.trace.terminated:
        for _ in range(5):
            if d.trace.terminated: break
            d.step(sign * (swing / 5.0) * np.array([u[0], u[1], 0.0]), "closed")
        sign *= -1.0

def push_then_grasp_release_on_target(rec):
    """Push most of the way with a closed, empty gripper, then actually grasp and
    place cleanly (report only -- not a cheat vs the clean place, a push-assisted
    but genuine placement)."""
    d = rec.dr
    px, py = d.piece_xy; gx, gy = d.goal_xy
    u = np.array([gx - px, gy - py]) / np.hypot(gx - px, gy - py)
    behind = np.array([px, py]) - u * 0.06
    d.goto([behind[0], behind[1], se.CLEAR_Z], "closed")
    d.goto([behind[0], behind[1], se.GRASP_Z], "closed")
    for _ in range(300):
        piece = d.base._achieved_goal()[:2].astype(float)
        err = np.array([gx, gy]) - piece
        if np.hypot(*err) <= 0.02 or d.trace.terminated:
            break
        d.step(np.array([u[0], u[1], 0.0]) * 3.0e-3, "closed")
    d.hold("closed", 5)
    if not d.trace.terminated:
        piece_xy = d.base._achieved_goal()[:2].astype(float)
        d.goto([piece_xy[0], piece_xy[1], se.CLEAR_Z], "open")
        d.grasp_and_lift(xy=piece_xy)
        d.place_and_release(height_m=0.0015)


CASES = dict(clean=clean, push_then_open=push_then_open, drop_30mm=drop(0.030), drop_80mm=drop(0.080),
             hold_pinched_on_target=hold_pinched, hover_lifted_over_target=hover_lifted,
             release_regrasp_release=regrasp_release, release_then_topple_open_gripper=release_then_topple,
             drop_off_target=drop_off_target, release_lift_reclose_literal=reclose_on_air,
             release_partial_open_hold_0p40=partial_open_hold(0.0),
             release_partial_open_hold_0p203=partial_open_hold(-0.5), release_partial_open_topple=partial_open_topple,
             shaping_farm=shaping_farm, push_then_grasp_release_on_target=push_then_grasp_release_on_target)

REPORT_ONLY = {"push_then_grasp_release_on_target"}


# ---- toppled_hold: e4->e5 only, replays a real recorded SAC episode ----
def toppled_hold_e4e5():
    found = json.loads((Path(__file__).with_name("topple_actions.json")).read_text())
    assert found["found"], "topple_actions.json has no recorded episode"
    actions150 = np.asarray(found["actions"], dtype=np.float32)
    assert actions150.shape[0] == 150

    base = RoboChessEnv(action_mode="cartesian", max_episode_steps=T)
    task = T1PickAndPlaceTask(base, start_square="e4", target_square="e5")
    task = PickupObservation(task); task = PlaceObservation(task)
    env = PlaceReward(task, **PLACE_KWARGS)
    env.reset(seed=20000)

    rewards, terms, infos = [], [], []
    for a in actions150:
        obs, r, term, trunc, info = env.step(a)
        rewards.append(float(r)); terms.append(info["place_terms"]); infos.append(info)
        if term or trunc:
            break

    pose150 = dict(piece_xyz=base._achieved_goal().astype(float).tolist(),
                  piece_quat=base.data.qpos[base._piece_qadr + 3:base._piece_qadr + 7].tolist())
    pose_check = dict(
        xyz_max_abs_diff=float(max(abs(a - b) for a, b in zip(found["pose150"]["piece_xyz"], pose150["piece_xyz"]))),
        original_tilt=found["t150"]["piece_tilt_deg"], replay_tilt=terms[-1]["piece_tilt_deg"])

    hold_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    terminated = truncated = False
    while not (terminated or truncated):
        obs, r, terminated, truncated, info = env.step(hold_action)
        rewards.append(float(r)); terms.append(info["place_terms"]); infos.append(info)
    base.close()

    r = np.array(rewards)
    disc = float(np.sum(r * GAMMA ** np.arange(len(r))))
    last = infos[-1]
    pinched_terms = [t for t in terms if t["pinched"]]
    tilt_ge_45 = [t for t in pinched_terms if t["piece_tilt_deg"] >= 45.0]
    s1_over_2p2 = [t for t in tilt_ge_45 if t["state"] == "S1" and t["state_reward"] > 2.2]
    per_pinched_step = [
        dict(t=t["t"], state=t["state"], tilt=round(t["piece_tilt_deg"], 3),
             posture_gate=round(t["posture_gate"], 4),
             d28_s1=round(d28_equivalent_state_reward(t), 4) if t["state"] == "S1" else None,
             d29_s1_paid=round(t["state_reward"], 4) if t["state"] == "S1" else None,
             gate_lt_1=bool(t["posture_gate"] < 1.0))
        for t in pinched_terms
    ]
    summary = dict(rewards=[round(float(x), 6) for x in r],
                   state_rewards=[round(t["state_reward"], 6) for t in terms],
                   shaping=[round(t["shaping"], 6) for t in terms],
                   state_seq="".join(t["state"][1] for t in terms),
                   steps=len(r), undiscounted=float(r.sum()), discounted=disc,
                   shaping_total=float(sum(t["shaping"] for t in terms)),
                   success=bool(last["is_success"]), end_reason=last.get("terminated_reason"),
                   final_state=last["place_state"],
                   bonuses=dict(first_grasp=sum(t["first_grasp_bonus"] > 0 for t in terms),
                                lift=sum(t["lift_bonus"] > 0 for t in terms),
                                placed=sum(t["placed_bonus"] > 0 for t in terms),
                                lump=sum(t["success_lump"] > 0 for t in terms),
                                penalty=sum(t["failure_penalty"] > 0 for t in terms)))
    summary["_source"] = "replays 150 actions recorded from a real D28 SAC seed0 episode (seed 20000, episode 0), then holds to timeout"
    summary["_pose_check_at_t150"] = pose_check
    summary["_n_pinched_steps"] = len(pinched_terms)
    summary["_n_gate_lt_1"] = sum(1 for t in pinched_terms if t["posture_gate"] < 1.0)
    summary["_n_tilt_ge_45"] = len(tilt_ge_45)
    summary["_n_s1_over_2p2_at_tilt_ge_45"] = len(s1_over_2p2)
    summary["_per_pinched_step"] = per_pinched_step
    return summary


def main(out_path):
    result = dict(T=T, gamma=GAMMA, acceptance_4a={}, smoothness={}, cheats={})
    for s, t in PAIRS:
        res, log = run_gate(s, t)
        pair = f"{s}->{t}"
        info = dict(gate_success=res["success"], clean_place=res["clean_place"], failure=res["failure"],
                    steps=res["steps"])
        try:
            cp = checkpoints(res, log)
        except Exception as e:
            cp = dict(error=repr(e))
        result["acceptance_4a"][pair] = dict(harness=info, checkpoints=cp)
        try:
            result["smoothness"][pair] = smoothness(log, res)
        except Exception as e:
            result["smoothness"][pair] = dict(error=repr(e))
        print(pair, info, "non_decreasing:", cp.get("_non_decreasing"),
              "max_diff_vs_d28_tilt_le_10:", cp.get("_max_abs_diff_vs_d28_tilt_le_10"),
              "max_gate_jump:", cp.get("_max_abs_gate_jump"), flush=True)
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
    result["cheats"]["e4->e5"]["toppled_hold"] = toppled_hold_e4e5()
    print("e4->e5 toppled_hold",
          {k: result["cheats"]["e4->e5"]["toppled_hold"][k] for k in
           ("_n_pinched_steps", "_n_gate_lt_1", "_n_tilt_ge_45", "_n_s1_over_2p2_at_tilt_ge_45")}, flush=True)
    Path(out_path).write_text(json.dumps(result, indent=1))

if __name__ == "__main__":
    main(sys.argv[1])
