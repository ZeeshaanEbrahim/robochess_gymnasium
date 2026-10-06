#!/usr/bin/env python
"""RS31 Section 2: pooled and per-pair placement with Wilson 95% (z=1.96, no continuity correction), reasons,
stage counts, rms check, and the mechanical rule table. Reads eval/*.json, *_rms.json, EVAL_MANIFEST.json only."""
import collections, json, math, re
from pathlib import Path
EV = Path(__file__).resolve().parent
PREREG = EV.parent / "PREREGISTRATION.md"
line = next(l for l in PREREG.read_text().splitlines() if l.startswith("Training pairs"))
PAIRS = re.findall(r"\b[a-h][1-8]:[a-h][1-8]\b", line.split("):", 1)[-1] if "):" in line else line.split(":", 1)[1])
assert len(PAIRS) == 24
NAMES = {"C": ["rs31C_seed0", "rs31C_seed1"], "D": ["rs31D_seed0", "rs31D_seed1"]}

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

man = json.loads((EV / "EVAL_MANIFEST.json").read_text())
res = {}
for arm, names in NAMES.items():
    for name in names:
        d = json.load(open(EV / f"{name}_final.json")); rms = json.load(open(EV / f"{name}_final_rms.json"))
        assert d["n"] == 1200 and d["seed"] == 20000 and not d["deterministic"] and d["model"] == "final_model.zip" and d["checkpoint"] is None
        k, n, rs = collections.Counter(), collections.Counter(), collections.Counter()
        for e in d["episodes"]:
            p = f'{e["start_square"]}:{e["target_square"]}'; n[p] += 1; k[p] += e["place"]["placed"]; rs[e["reason"]] += 1
        assert [n[p] for p in PAIRS] == [50] * 24 and sum(n.values()) == 1200
        # episode i must use pair i % 24 (eval_probe cycles in the given order)
        assert [f'{e["start_square"]}:{e["target_square"]}' for e in d["episodes"]] == [PAIRS[i % 24] for i in range(1200)]
        kk = [k[p] for p in PAIRS]; tot = sum(kk); pr = d["place_report"]["rates"]
        assert tot == pr["placed"]["k"] == rs.get("success", 0), (name, tot, pr["placed"]["k"], rs.get("success", 0))
        ev = [e for e in man["evals"] if e["label"] == name][0]
        res[name] = dict(arm=arm, k=kk, tot=tot, reasons=dict(rs), grasp_ever=sum(e["grasp_ever"] for e in d["episodes"]),
                         stages={s: pr[s]["k"] for s in ("held", "released", "placed")},
                         rms_probe=d["obs_rms_identical_before_after"], rms_wrap=rms["identical_before_after"], pkl_same=rms["pkl_unchanged"],
                         rms_keys=rms["keys"], exit=ev["exit"], start=ev["start"], end=ev["end"], cmd=ev["cmd"])
        res[name]["fail_pooled"] = tot < 600
        res[name]["fail_pairs"] = [(p, x) for p, x in zip(PAIRS, kk) if x < 10]
        res[name]["learns"] = not res[name]["fail_pooled"] and not res[name]["fail_pairs"]

print("EVAL command (one per model; wrapper runs scripts/eval_probe.py unmodified, same argv):")
for name, r in res.items(): print(f"  {name}: {r['cmd']}")
print("\nExit status / times (manifest):")
for name, r in res.items(): print(f"  {name}: exit={r['exit']} start={r['start']} end={r['end']}")
print("\nVecNormalize frozen check (before = pkl read before eval; after = loaded venv after rollout; keys: " + ", ".join(res["rs31C_seed0"]["rms_keys"]) + ")")
for name, r in res.items():
    print(f"  {name}: identical before/after (obs_rms + ret_rms, mean/var/count) = {'yes' if r['rms_wrap'] else 'NO'}; eval_probe own obs_rms flag = {r['rms_probe']}; pkl sha256 unchanged = {r['pkl_same']}")
print("\n1. pooled placements k/1200 (Wilson 95%, %)")
for name, r in res.items():
    lo, hi = wilson(r["tot"], 1200); print(f"  {name}: {r['tot']}/1200 = {100*r['tot']/1200:.2f}%  [{lo:.2f}, {hi:.2f}]")
print("\n2. per pair k/50 [Wilson 95%, %], preregistration order")
print(f'  {"pair":<7}' + "".join(f"{n:>24}" for n in res))
for i, p in enumerate(PAIRS):
    print(f"  {p:<7}" + "".join(f"{r['k'][i]:>7}/50 [{wilson(r['k'][i],50)[0]:5.1f},{wilson(r['k'][i],50)[1]:5.1f}]" for r in res.values()))
print("\n3. pairs below 10/50")
for name, r in res.items(): print(f"  {name}: {len(r['fail_pairs'])} pairs: " + (", ".join(f"{p} {x}/50" for p, x in r["fail_pairs"]) or "none"))
print("\n4. terminated_reason counts (success endings vs placed checked equal by assertion)")
for name, r in res.items(): print(f"  {name}: {dict(sorted(r['reasons'].items()))}")
print("\n5. funnel counts /1200 (grasp_ever = lift over 10 mm (undercounts); held = both pads on piece, no piece-board contact; released = S2 reached; placed = env is_success)")
for name, r in res.items():
    s = r["stages"]; print(f"  {name}: grasp_ever (lift over 10 mm, undercounts) {r['grasp_ever']}/1200  held {s['held']}/1200  released {s['released']}/1200  placed {s['placed']}/1200")
print("\n6. MECHANICAL RULE TABLE (PREREGISTRATION.md rules 1 to 5)")
print(f'  {"model":<13}{"pooled":>10}{"pooled>=600":>13}{"pairs<10/50":>13}{"RULE1 learns":>14}  failing conditions   rule5 pooled>=960')
for name, r in res.items():
    fc = []
    if r["fail_pooled"]: fc.append(f"pooled {r['tot']}<600")
    if r["fail_pairs"]: fc.append(f"{len(r['fail_pairs'])} pair(s) <10/50: " + ", ".join(p for p, _ in r["fail_pairs"]))
    print(f'  {name:<13}{r["tot"]:>6}/1200{("yes" if not r["fail_pooled"] else "no"):>13}{len(r["fail_pairs"]):>13}{("yes" if r["learns"] else "no"):>14}  {"; ".join(fc) or "none":<20}  {"yes" if r["tot"] >= 960 else "no"}')
L = {a: sum(res[n]["learns"] for n in NAMES[a]) for a in NAMES}
print(f"  Rule 2: seeds that learn: C {L['C']} of 2, D {L['D']} of 2")
print(f"  Rule 3: D learns on {L['D']} seeds, C on {L['C']} -> " + ("D replaces C" if L["D"] > L["C"] else "C kept (D does not learn on strictly more seeds)"))
print("  Rule 4: " + ("24 pairs at 3M steps is not achieved (no seed in either arm learns)" if L["C"] + L["D"] == 0 else "not triggered (at least one seed learns)"))
print("  Rule 5 (reported, not gated): " + ", ".join(f"{n} {'yes' if r['tot'] >= 960 else 'no'}" for n, r in res.items()))
