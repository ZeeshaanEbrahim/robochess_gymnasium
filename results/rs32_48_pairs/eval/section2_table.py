#!/usr/bin/env python
"""RS32 Section 2 (mechanical only): pooled and per-pair placement with Wilson 95% (z=1.96, no continuity correction),
terminated_reason counts, funnel counts, rms check and the rule table. Reads eval/*.json, *_rms.json, EVAL_MANIFEST.json only."""
import collections, json
from rs32_lib import *
N_EP = 2400
man = json.loads((EV / "EVAL_MANIFEST.json").read_text())
res = {}
for name in NAMES:
    d = json.load(open(EV / f"{name}_final.json")); rms = json.load(open(EV / f"{name}_final_rms.json"))
    assert d["n"] == N_EP and d["seed"] == 20000 and not d["deterministic"] and d["model"] == "final_model.zip" and d["checkpoint"] is None
    k, n, rs = collections.Counter(), collections.Counter(), collections.Counter()
    for e in d["episodes"]:
        p = f'{e["start_square"]}:{e["target_square"]}'; n[p] += 1; k[p] += e["place"]["placed"]; rs[e["reason"]] += 1
    assert [n[p] for p in PAIRS] == [50] * 48 and sum(n.values()) == N_EP
    assert [f'{e["start_square"]}:{e["target_square"]}' for e in d["episodes"]] == [PAIRS[i % 48] for i in range(N_EP)]
    kk = [k[p] for p in PAIRS]; tot = sum(kk); pr = d["place_report"]["rates"]
    assert tot == pr["placed"]["k"] == rs.get("success", 0), (name, tot, pr["placed"]["k"], rs.get("success", 0))
    ev = [e for e in man["evals"] if e["label"] == name][0]
    res[name] = dict(k=kk, tot=tot, reasons=dict(rs), lift10=sum(e["grasp_ever"] for e in d["episodes"]),
                     stages={s: pr[s]["k"] for s in ("held", "released", "placed")},
                     rms_probe=d["obs_rms_identical_before_after"], rms_wrap=rms["identical_before_after"], pkl_same=rms["pkl_unchanged"],
                     rms_keys=rms["keys"], exit=ev["exit"], start=ev["start"], end=ev["end"], cmd=ev["cmd"],
                     steps=man["final_num_timesteps"][RUN(name)])
    r = res[name]
    r["fail_pooled"] = tot < 1200
    r["fail_pairs"] = [(p, x) for p, x in zip(PAIRS, kk) if x < 10]
    r["learns"] = not r["fail_pooled"] and not r["fail_pairs"]
    r["target"] = tot >= 1920
    r["json_sha256_ok"] = ev.get("output_json_sha256") is not None

print("EVAL command (one per model; wrapper runs scripts/eval_probe.py unmodified, same argv):")
for name, r in res.items(): print(f"  {name}: {r['cmd']}")
print("\nFinal model step counts (num_timesteps in final_model.zip; expected 5,005,312):")
for name, r in res.items(): print(f"  {name}: {r['steps']:,}  {'as expected' if r['steps'] == 5005312 else 'NOT as expected'}")
print("\nExit status / times (manifest):")
for name, r in res.items(): print(f"  {name}: exit={r['exit']} start={r['start']} end={r['end']}")
print("\nVecNormalize frozen check (before = pkl read before eval; after = loaded venv after rollout; keys: " + ", ".join(res[NAMES[0]]["rms_keys"]) + ")")
for name, r in res.items():
    print(f"  {name}: identical before/after (obs_rms + ret_rms, mean/var/count) = {'yes' if r['rms_wrap'] else 'NO'}; eval_probe own obs_rms flag = {r['rms_probe']}; pkl sha256 unchanged = {r['pkl_same']}")
print(f"\n1. pooled placements k/{N_EP} (Wilson 95%, %)")
for name, r in res.items(): print(f"  {name}: {fmt(r['tot'], N_EP)}")
print("\n2. per pair k/50 [Wilson 95%, %], preregistration order")
print(f'  {"pair":<7}' + "".join(f"{n:>24}" for n in res))
for i, p in enumerate(PAIRS):
    print(f"  {p:<7}" + "".join(f"{r['k'][i]:>7}/50 [{wilson(r['k'][i],50)[0]:5.1f},{wilson(r['k'][i],50)[1]:5.1f}]" for r in res.values()))
print("\n3. pairs below 10/50")
for name, r in res.items(): print(f"  {name}: {len(r['fail_pairs'])} pairs: " + (", ".join(f"{p} {x}/50" for p, x in r["fail_pairs"]) or "none"))
print("\n4. terminated_reason counts (success endings vs placed checked equal by assertion)")
for name, r in res.items(): print(f"  {name}: {dict(sorted(r['reasons'].items()))}")
print(f"\n5. funnel counts /{N_EP} (lift over 10 mm = eval_probe grasp_ever (undercounts); held = both pads on piece, no piece-board contact; released = S2 reached; placed = env is_success)")
for name, r in res.items():
    s = r["stages"]; print(f"  {name}: lift over 10 mm (undercounts) {r['lift10']}/{N_EP}  held {s['held']}/{N_EP}  released {s['released']}/{N_EP}  placed {s['placed']}/{N_EP}")
print("\n6. MECHANICAL RULE TABLE (PREREGISTRATION.md rules 1 to 4)")
print(f'  {"model":<13}{"pooled":>11}{"pooled>=1200":>14}{"pairs<10/50":>13}{"RULE1 learns":>14}{"target>=1920":>14}  failing conditions')
for name, r in res.items():
    fc = []
    if r["fail_pooled"]: fc.append(f"pooled {r['tot']}<1200")
    if r["fail_pairs"]: fc.append(f"{len(r['fail_pairs'])} pair(s) <10/50: " + ", ".join(p for p, _ in r["fail_pairs"]))
    print(f'  {name:<13}{r["tot"]:>6}/{N_EP}{("yes" if not r["fail_pooled"] else "no"):>14}{len(r["fail_pairs"]):>13}{("yes" if r["learns"] else "no"):>14}{("yes" if r["target"] else "no"):>14}  {"; ".join(fc) or "none"}')
L = sum(r["learns"] for r in res.values())
print(f"  Rule 2: seeds that learn: {L} of 6 ({', '.join(n for n, r in res.items() if r['learns']) or 'none'})")
print(f"  Rule 3: claim 'PPO with the D recipe learns 48 mixed pairs' holds if >= 4 of 6 seeds learn: {L} of 6 -> " + ("HOLDS" if L >= 4 else "NOT ACHIEVED"))
print("  Rule 4 (target tier, reported): pooled >= 1920/2400 per seed: " + ", ".join(f"{n} {'yes' if r['target'] else 'no'}" for n, r in res.items()))
