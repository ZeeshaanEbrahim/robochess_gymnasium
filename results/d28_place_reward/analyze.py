"""Tables for the D28 Section 4 acceptance from acceptance.json (pre-registered checks 7-9)."""
import json, sys
from pathlib import Path
import numpy as np
G = 0.99; T = 300
r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("acceptance.json")))
ok = True
def G_from(x, k): x = np.array(x[k:]); return float(np.sum(x * G ** np.arange(len(x))))
def boot(s, k):
    """G from step k, plus gamma^(T-k) * V_tail for a case truncated at T (V_tail = last reward / (1-gamma));
    terminated cases (success, bad_contact, piece_off_board) get no tail."""
    g = G_from(s["rewards"], k)
    if s["end_reason"] is None and not s["success"] and s["steps"] >= T:
        g += G ** (T - k) * s["rewards"][-1] / (1 - G)
    return g
print("== 4a")
for p, v in r["acceptance_4a"].items():
    cp = v["checkpoints"]
    row = " ".join(f"{n}={cp[n]['recurring']:.3f}" for n in ["home","hover","close","first_held","lifted_30mm","mid_carry","over_target_at_height","set_down_pinched","placed"])
    flag = cp["_non_decreasing"] and cp["_placed_between_8_and_9"] and cp["_s2_non_decreasing_and_ge_8"]
    ok &= bool(flag)
    print(p, "PASS" if flag else "FAIL", "| non_decreasing", cp["_non_decreasing"], "placed in [8,9]", cp["_placed_between_8_and_9"], "S2 nondecr>=8", cp["_s2_non_decreasing_and_ge_8"], "| gate steps", v["harness"]["steps"])
    print("   ", row)
print("== 4b   (Δ = cheat - clean; discounted gamma=0.99)")
park = []
for p, d in r["cheats"].items():
    c = d["clean"]; kr = c["state_seq"].index("2")
    print(p, f"clean: success step {c['steps']}, first release/S2 step {kr+1}; disc0 {c['discounted']:.2f}, from-release {G_from(c['rewards'], kr):.2f}")
    for n, s in d.items():
        if n == "clean": continue
        d0 = s["discounted"] - c["discounted"]; dr = G_from(s["rewards"], kr) - G_from(c["rewards"], kr)
        b0 = boot(s, 0) - boot(c, 0); br = boot(s, kr) - boot(c, kr)
        lower = d0 < 0 and dr < 0 and b0 < 0 and br < 0
        # the two literal-action cheats cannot run (success fires first): identical to clean, reported not judged
        literal = n in ("release_then_topple_open_gripper", "release_lift_reclose_literal")
        if not literal: ok &= bool(lower)
        tag = "n/a (cheat never ran: success first)" if literal else ("ok" if lower else "FAIL")
        print(f"   {n:34s} succ={s['success']!s:5s} end={s['end_reason']}  step0 {d0:+9.2f}  from-release {dr:+9.2f}  boot-step0 {b0:+9.2f}  boot-release {br:+9.2f}  undisc {s['undiscounted']-c['undiscounted']:+9.1f}  {tag}")
        if n.startswith("release_partial_open_hold"): park.append((p, n, dr, br))
print("== parking margin from the release step (must be >= 100)")
for p, n, dr, br in park:
    f = dr <= -100 and br <= -100; ok &= f
    print(f"   {p} {n:34s} plain {dr:+8.2f}  bootstrapped {br:+8.2f}  {'PASS' if f else 'FAIL'}")
print("ALL PRE-REGISTERED CHECKS PASS" if ok else "AT LEAST ONE CHECK FAILED")
