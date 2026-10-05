#!/usr/bin/env python
"""RS29 Section 3 diagnostics (3a-3d, k; h and i are in rs29_monitor_analysis.py).
Inputs: final_seed*.json (pre-registered eval output) and trace_final_seed*.json.gz (rs29_trace.py rerun).
Pre-registered items are 3a-3d; k is EXPLORATORY. Reads results/ only."""
import csv, gzip, json, math, statistics as st, sys, collections
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
import numpy as np
from robochess.tasks.squares import square_to_local_xy
E = Path(__file__).parent
PAIRS = "e4:e5,c1:c2,d2:e2,a6:a5,e8:d7,d4:e5,f3:c6,h5:d3,g5:b3,c1:f6,h1:h8,h8:h1,a4:h4,b7:g2,b2:g7,f1:c8,a1:h8,h8:a1,h1:a8,a8:h1".split(",")
BANDS = dict(short=PAIRS[0:6], mid=PAIRS[6:10], long=PAIRS[10:16], diagonal=PAIRS[16:20])
def dist(p): s, t = p.split(":"); return float(np.linalg.norm(square_to_local_xy(s) - square_to_local_xy(t)))
D = {p: dist(p) for p in PAIRS}
def wilson(k, n, z=1.959964):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d, (c + h) / d
def pct(k, n): lo, hi = wilson(k, n); return f"{k}/{n} ({100*k/n:.1f}%, [{100*lo:.1f},{100*hi:.1f}])"
cols = None; S = {}
for i in range(6):
    ev = json.load(open(E / f"final_seed{i}.json")); tr = json.load(gzip.open(E / f"trace_final_seed{i}.json.gz", "rt"))
    cols = {c: j for j, c in enumerate(tr["cols"])}
    assert len(ev["episodes"]) == len(tr["episodes"]) == 300
    # the trace rerun must reproduce the eval exactly (same episodes, reasons, lengths, held steps)
    for e, t in zip(ev["episodes"], tr["episodes"]):
        assert e["reason"] == t["reason"] and e["n"] == len(t["rows"]), (i, e["reason"], t["reason"], e["n"], len(t["rows"]))
        assert e["place"]["held_steps"] == sum(r[cols["held"]] for r in t["rows"])
        assert f'{e["start_square"]}:{e["target_square"]}' == t["pair"]
    S[i] = (ev, tr)
print("CHECK: trace rerun reproduces final_seed*.json exactly (reason, length, held steps, pair) for all 6 x 300 episodes; obs_rms identical:",
      [S[i][1]["obs_rms_identical"] for i in range(6)])
C = lambda r, k: r[cols[k]]

print("\n=== 3a. placement per pair (placed/15) per seed; held-episodes/15 in brackets ===")
print("pair      d_mm  " + "  ".join(f"s{i}" for i in range(6)))
for p in PAIRS:
    cells = []
    for i in range(6):
        eps = [(e, t) for e, t in zip(*[S[i][0]["episodes"], S[i][1]["episodes"]]) if t["pair"] == p]
        cells.append(f'{sum(t["success"] for _, t in eps)}[{sum(e["place"]["held"] for e, _ in eps)}]')
    print(f"{p:8s} {1000*D[p]:6.1f} " + " ".join(f"{c:>6s}" for c in cells))
def grp(i, pairs):
    eps = [(e, t) for e, t in zip(S[i][0]["episodes"], S[i][1]["episodes"]) if t["pair"] in pairs]
    return len(eps), sum(t["success"] for _, t in eps), sum(e["place"]["held"] for e, _ in eps)
print("\nper band: placed (Wilson95) | held episodes   [n = 15 x pairs]")
for b, ps in BANDS.items():
    print(f"{b:9s}(n={15*len(ps)} per seed): " + " ; ".join(f"s{i} " + pct(grp(i, ps)[1], grp(i, ps)[0]) + f" held {grp(i, ps)[2]}" for i in range(6)))
lt = [p for p in PAIRS if D[p] < 0.4]; ge = [p for p in PAIRS if D[p] >= 0.4]
print(f"\nstart-to-target < 0.4 m ({len(lt)} pairs: {','.join(lt)}) vs >= 0.4 m ({len(ge)} pairs: {','.join(ge)})")
for nm, ps in (("< 0.4 m", lt), (">= 0.4 m", ge)):
    print(f"{nm:9s}: " + " ; ".join(f"s{i} placed " + pct(grp(i, ps)[1], grp(i, ps)[0]) + f" held {grp(i, ps)[2]}/{grp(i, ps)[0]}" for i in range(6)))

print("\n=== 3b. funnel per seed (n=300; eval place_report + eval_probe grasp_ever) ===")
for i in range(6):
    ev = S[i][0]; r = ev["place_report"]["rates"]; g = sum(e["grasp_ever"] for e in ev["episodes"])
    print(f"s{i}: grasp_ever(D18) {pct(g,300)} | held(D27) {pct(r['held']['k'],300)} | lifted_30mm {pct(r['lifted_30mm']['k'],300)} | carried_within_3cm {pct(r['carried_within_3cm']['k'],300)} | released {pct(r['released']['k'],300)} | placed {pct(r['placed']['k'],300)}")

print("\n=== 3c. failure classification (all six seeds fail the rule); measured numbers ===")
LAB = {}
for i in range(6):
    ev, tr = S[i]; rows_by_ep = [t["rows"] for t in tr["episodes"]]
    held_eps = [k for k, rs in enumerate(rows_by_ep) if any(C(r, "held") for r in rs)]
    ratio = []; maxlift = []; mind = []; hs = []; first = []
    for k in held_eps:
        rs = rows_by_ep[k]; h = [r for r in rs if C(r, "held")]
        hs.append(len(h)); maxlift.append(1e3 * max(C(r, "lift") for r in h)); mind.append(1e3 * min(C(r, "d_xy") for r in h))
        first.append(next(j for j, r in enumerate(rs) if C(r, "held")) + 1)
        d0 = D[tr["episodes"][k]["pair"]]; ratio.append(ev["episodes"][k]["place"]["carry_mm"] / (1e3 * d0))
    any_lift = sum(1 for k in held_eps if max(C(r, "lift") for r in rows_by_ep[k] if C(r, "held")) >= 0.03)
    carried3 = ev["place_report"]["rates"]["carried_within_3cm"]["k"]; released = ev["place_report"]["rates"]["released"]["k"]
    if len(held_eps) < 5: lab = "never grasps (held in <5/300 episodes)"
    elif carried3 == 0 and (not ratio or float(np.median(ratio)) < 0.5): lab = "grasps but never carries (held-phase displacement median < 50% of start-to-target; never within 3 cm of target while lifted)"
    else: lab = "other"
    LAB[i] = lab
    f = lambda v, q: f"{np.percentile(v, q):.1f}" if v else "n/a"
    print(f"s{i}: {lab}\n    held episodes {len(held_eps)}/300; held steps/ep median {f(hs,50)} (p10 {f(hs,10)}, p90 {f(hs,90)}); first-held step median {f(first,50)};"
          f" max lift while held, median {f(maxlift,50)} mm (p90 {f(maxlift,90)}, max {max(maxlift):.1f}); episodes with lift>=30mm while held {any_lift};"
          f" closest d_xy to target while held, median {f(mind,50)} mm; held-phase displacement / start-to-target, median {f([100*x for x in ratio],50)}%; released {released}")

print("\n=== 3d. truncated-in-S2 tripwire and shaping total per episode ===")
for i in range(6):
    ev, tr = S[i]; k2 = ev["place_report"]["truncated_in_s2"]["k"]
    tot = [sum(C(r, "shaping") for r in t["rows"]) for t in tr["episodes"]]
    idle = [0.5 * D[t["pair"]] * len(t["rows"]) for t in tr["episodes"]]  # (1-gamma)*k*d per step, d = start-to-target
    dd = [tt - ii for tt, ii in zip(tot, idle)]
    print(f"s{i}: truncated_in_S2 {k2}/300 (tripwire >= 10% -> {'TRIPPED' if k2 >= 30 else 'not tripped'}) | shaping total/episode: median {np.median(tot):.2f}, mean {np.mean(tot):.2f}, "
          f"min {min(tot):.2f}, max {max(tot):.2f} | idle baseline (0.5*d_pair*steps) median {np.median(idle):.2f} | total - idle: median {np.median(dd):+.2f}")

print("\n=== k. [EXPLORATORY] held-but-not-lifted episodes (lift < 30 mm throughout), per seed; source: trace rerun (all 300 episodes, per-step) ===")
out = []
for i in range(6):
    ev, tr = S[i]; sel = []
    for k, t in enumerate(tr["episodes"]):
        rs = t["rows"]; h = [j for j, r in enumerate(rs) if C(r, "held")]
        if not h: continue
        if max(C(rs[j], "lift") for j in h) >= 0.03: continue
        # last hold run end
        j_end = h[-1]
        if j_end == len(rs) - 1: how = {"success": "success", "time_limit": "time limit (still held)", "piece_off_board": "piece_off_board (while held)", "bad_contact": "bad_contact (while held)"}.get(t["reason"], t["reason"])
        else:
            nxt = rs[j_end + 1]; cmd = C(nxt, "grip_cmd")
            how = ("released into S2" if nxt[0] == "S2" else ("gripper opening (cmd<0.5) -> not held" if cmd < 0.5 else "slipped (cmd>=0.5, not held)"))
            if t["reason"] != "time_limit": how += f"; episode then {t['reason']}"
        hr = [rs[j] for j in h]
        out.append(dict(seed=i, ep=k, pair=t["pair"], first_held_step=h[0] + 1, held_steps=len(h), max_lift_mm=1e3 * max(C(r, "lift") for r in hr),
                        grip_cmd_med=float(np.median([C(r, "grip_cmd") for r in hr])), grip_cmd_min=min(C(r, "grip_cmd") for r in hr),
                        grip_qpos_med=float(np.median([C(r, "grip_qpos") for r in hr])), tilt_med=float(np.median([C(r, "tilt") for r in hr])),
                        tilt_max=max(C(r, "tilt") for r in hr), ended=how, reason=t["reason"], n=len(rs)))
with open(E / "k_held_not_lifted_episodes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
for i in range(6):
    o = [x for x in out if x["seed"] == i]
    nheld = S[i][0]["place_report"]["rates"]["held"]["k"]
    print(f"s{i}: held {nheld}/300; held-but-not-lifted {len(o)}")
    if not o: continue
    q = lambda key, qq: float(np.percentile([x[key] for x in o], qq))
    print(f"    first-held step: median {q('first_held_step',50):.0f} (p10 {q('first_held_step',10):.0f}, p90 {q('first_held_step',90):.0f}, min {min(x['first_held_step'] for x in o)})")
    print(f"    steps held: median {q('held_steps',50):.0f} (p10 {q('held_steps',10):.0f}, p90 {q('held_steps',90):.0f}, max {max(x['held_steps'] for x in o)})")
    print(f"    max lift while held (mm): median {q('max_lift_mm',50):.1f} (p90 {q('max_lift_mm',90):.1f}, max {max(x['max_lift_mm'] for x in o):.1f})")
    print(f"    gripper command while held (0 open .. 1 closed, per-episode median): median {q('grip_cmd_med',50):.3f} (p10 {q('grip_cmd_med',10):.3f}); lowest single-step {min(x['grip_cmd_min'] for x in o):.3f}")
    print(f"    gripper aperture qpos while held (per-episode median): median {q('grip_qpos_med',50):.4f} (p10 {q('grip_qpos_med',10):.4f}, p90 {q('grip_qpos_med',90):.4f})")
    print(f"    piece tilt while held (deg): per-episode median {q('tilt_med',50):.1f}; max over episodes {max(x['tilt_max'] for x in o):.1f}; episodes with tilt>45 at some held step {sum(x['tilt_max']>45 for x in o)}")
    print(f"    how the (last) hold ended: {dict(collections.Counter(x['ended'] for x in o))}")
print("\npiece_off_board episodes (all 300 episodes per seed): ever held? / state and held flag at the final step")
for i in range(6):
    tr = S[i][1]; c = collections.Counter(); ever = 0; n = 0
    for t in tr["episodes"]:
        if t["reason"] != "piece_off_board": continue
        n += 1; rs = t["rows"]; ev_ = any(C(r, "held") for r in rs); ever += ev_
        last = rs[-1]; ph = ("approach (never held)" if not C(last, "ever_held") else "while held" if C(last, "held") else
                             {"S1": "while pinched with board contact (S1, not held)", "S2": "after release (S2)", "S3": "after hold lost (S3)"}.get(last[0], last[0]))
        c[ph] += 1
    print(f"s{i}: piece_off_board {n}; ever D27-held in episode before leaving: {ever}/{n}; phase at the final step: {dict(c)}")
