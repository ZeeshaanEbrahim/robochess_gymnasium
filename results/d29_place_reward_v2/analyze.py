"""Tables for the D29 Section 4 acceptance from acceptance.json (pre-registered checks)."""
import json, sys
from pathlib import Path
import numpy as np
G = 0.99; T = 300
r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("acceptance.json")))
ok = True
def G_from(x, k): x = np.array(x[k:]); return float(np.sum(x * G ** np.arange(len(x))))
def boot(s, k):
    g = G_from(s["rewards"], k)
    if s["end_reason"] is None and not s["success"] and s["steps"] >= T:
        g += G ** (T - k) * s["rewards"][-1] / (1 - G)
    return g

print("== 4a")
for p, v in r["acceptance_4a"].items():
    cp = v["checkpoints"]
    row = " ".join(f"{n}={cp[n]['recurring']:.3f}" for n in ["home","hover","close","first_held","lifted_30mm","mid_carry","over_target_at_height","set_down_pinched","placed"])
    shap_row = " ".join(f"{n}={cp[n]['shaping']:+.4f}" for n in ["home","hover","close","first_held","lifted_30mm","mid_carry","over_target_at_height","set_down_pinched","placed"])
    flag = cp["_non_decreasing"] and cp["_placed_between_8_and_9"] and cp["_s2_non_decreasing_and_ge_8"]
    gate_ok = cp["_max_abs_gate_jump"] <= 1.0
    ok &= bool(flag) and bool(gate_ok)
    print(p, "PASS" if flag else "FAIL", "| non_decreasing", cp["_non_decreasing"], "placed in [8,9]", cp["_placed_between_8_and_9"],
          "S2 nondecr>=8", cp["_s2_non_decreasing_and_ge_8"], "| gate steps", v["harness"]["steps"])
    print("   recurring:", row)
    print("   shaping  :", shap_row, "| shaping_total", f"{cp['_shaping_total']:+.4f}")
    print("   D29 vs D28 (S1, tilt<=10):", cp["_n_s1_tilt_le_10_steps"], "steps, max|diff|",
          f"{cp['_max_abs_diff_vs_d28_tilt_le_10']:.3e}", "| max gate-caused per-step jump",
          f"{cp['_max_abs_gate_jump']:.4f}", "(must be <= 1.0)", "PASS" if gate_ok else "FAIL")

print("== 4b   (Δ = cheat - clean; discounted gamma=0.99)")
park = []
for p, d in r["cheats"].items():
    c = d["clean"]; kr = c["state_seq"].index("2")
    print(p, f"clean: success step {c['steps']}, first release/S2 step {kr+1}; disc0 {c['discounted']:.2f}, from-release {G_from(c['rewards'], kr):.2f}, shaping_total {c['shaping_total']:+.3f}")
    for n, s in d.items():
        if n == "clean": continue
        if "error" in s:
            print(f"   {n:40s} ERROR {s['error']}"); ok = False; continue
        d0 = s["discounted"] - c["discounted"]; dr = G_from(s["rewards"], kr) - G_from(c["rewards"], kr)
        b0 = boot(s, 0) - boot(c, 0); br = boot(s, kr) - boot(c, kr)
        lower = d0 < 0 and dr < 0 and b0 < 0 and br < 0
        literal = n in ("release_then_topple_open_gripper", "release_lift_reclose_literal")
        report_only = n in ("push_then_grasp_release_on_target",)
        if not literal and not report_only:
            ok &= bool(lower)
        tag = ("n/a (cheat never ran: success first)" if literal else
               "report only (not judged)" if report_only else
               ("ok" if lower else "FAIL"))
        print(f"   {n:40s} succ={s['success']!s:5s} end={s['end_reason']}  step0 {d0:+9.2f}  from-release {dr:+9.2f}  "
              f"boot-step0 {b0:+9.2f}  boot-release {br:+9.2f}  undisc {s['undiscounted']-c['undiscounted']:+9.1f}  "
              f"shaping_total {s['shaping_total']:+7.3f}  {tag}")
        if n.startswith("release_partial_open_hold"): park.append((p, n, dr, br))

print("== parking margin from the release step (must be >= 100)")
for p, n, dr, br in park:
    f = dr <= -100 and br <= -100; ok &= f
    print(f"   {p} {n:34s} plain {dr:+8.2f}  bootstrapped {br:+8.2f}  {'PASS' if f else 'FAIL'}")

print("== 4c / toppled_hold (e4->e5, real physics: replays a recorded D28 SAC seed0 episode, then holds to timeout)")
th = r["cheats"]["e4->e5"]["toppled_hold"]
print("   source:", th["_source"])
pc = th["_pose_check_at_t150"]
print(f"   pose match at t=150: xyz_max_abs_diff {pc['xyz_max_abs_diff']:.3e} m, tilt original {pc['original_tilt']:.4f} vs replay {pc['replay_tilt']:.4f}")
pose_ok = pc["xyz_max_abs_diff"] < 1e-9 and abs(pc["original_tilt"] - pc["replay_tilt"]) < 1e-9
print("   pose match:", "PASS" if pose_ok else "FAIL (replay diverged from the original SAC trajectory)")
ok &= bool(pose_ok)
print(f"   {th['_n_pinched_steps']} pinched steps, {th['_n_gate_lt_1']} with gate<1, {th['_n_tilt_ge_45']} with tilt>=45 deg")
c4c = th["_n_s1_over_2p2_at_tilt_ge_45"] == 0
ok &= bool(c4c)
print(f"   4c: S1 reward <= 2.2 whenever tilt >= 45 deg: {'PASS' if c4c else 'FAIL'} "
      f"({th['_n_s1_over_2p2_at_tilt_ge_45']} violations out of {th['_n_tilt_ge_45']} such steps)")
print("   per pinched step (t, state, tilt, gate, D28 S1, D29 S1 paid):")
rows = th["_per_pinched_step"]
step_idx = {148, 149, 150, 151, 152, 153, 154, 155, 156, 200, 250, 296, 297, 298, 299, 300}
for row in rows:
    if row["t"] <= 65 or row["t"] in step_idx:
        print(f"     t={row['t']:>3} {row['state']} tilt={row['tilt']:>7.3f} gate={row['posture_gate']:.4f} "
              f"d28_s1={row['d28_s1']} d29_s1_paid={row['d29_s1_paid']} gate<1={row['gate_lt_1']}")
print(f"   ({len(rows)} pinched-step rows total; showing t<=65 and the hold-transition/timeout windows)")

print("ALL PRE-REGISTERED CHECKS PASS" if ok else "AT LEAST ONE CHECK FAILED")
