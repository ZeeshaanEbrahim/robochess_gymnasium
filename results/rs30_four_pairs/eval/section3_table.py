#!/usr/bin/env python
"""RS30 Section 3: per-pair placement, Wilson CIs, decision rule 1 and 2, reasons. Reads eval/*.json only."""
import collections, json, math
from pathlib import Path
EV = Path(__file__).resolve().parent
PAIRS = ["e4:e5", "d4:d5", "f4:f5", "e3:e4"]
D29 = [92.3, 0, 96.0, 99.3, 97.3]          # D29 squash seeds 0-4, percent e4:e5

def wilson(k, n, z=1.96):                   # centre = (p + z^2/2n)/(1+z^2/n); half = z*sqrt(p(1-p)/n + z^2/4n^2)/(1+z^2/n)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

res = {}
for arm in "AB":
    for i in range(4):
        name = f"rs30{arm}_seed{i}"; d = json.load(open(EV / f"{name}_final.json"))
        assert d["n"] == 300 and d["seed"] == 20000 and not d["deterministic"] and d["obs_rms_identical_before_after"] is True
        k, n, rs = collections.Counter(), collections.Counter(), collections.Counter()
        for e in d["episodes"]:
            p = f'{e["start_square"]}:{e["target_square"]}'; n[p] += 1; k[p] += e["place"]["placed"]; rs[e["reason"]] += 1
        assert [n[p] for p in PAIRS] == [75] * 4 and sum(n.values()) == 300
        res[name] = dict(k=[k[p] for p in PAIRS], tot=sum(k.values()), reasons=dict(rs), placed_rate=d["place_report"]["rates"]["placed"]["k"])
        assert res[name]["tot"] == res[name]["placed_rate"]
        res[name]["learns"] = all(x >= 15 for x in res[name]["k"])

print("a/b. placements k/75 per training pair (Wilson 95%, z=1.96, no continuity correction, in %)")
print(f'{"run":<13}' + "".join(f"{p:>22}" for p in PAIRS) + f'{"total/300":>12}  learns(>=15 on each)')
for name, r in res.items():
    cells = "".join(f'{k:>6}/75 [{wilson(k,75)[0]:5.1f},{wilson(k,75)[1]:5.1f}]' for k in r["k"])
    print(f"{name:<13}{cells}{r['tot']:>8}/300  {r['learns']}")
print("\nc. seed-learns flag and arm verdict")
for arm in "AB":
    L = [res[f"rs30{arm}_seed{i}"]["learns"] for i in range(4)]
    print(f"  seed flags arm {arm}: {L}")
    print(f"  arm {arm} verdict: {sum(L)}/4 seeds learn -> {'PASS' if sum(L) >= 3 else 'FAIL'} (needs >= 3)")
print("\nd. e4:e5 placed per seed (k/75, %) vs D29 squash seeds 0-4 (%)", D29)
for arm in "AB":
    print(f"  arm {arm}: " + ", ".join(f"s{i}: {res[f'rs30{arm}_seed{i}']['k'][0]}/75={100*res[f'rs30{arm}_seed{i}']['k'][0]/75:.1f}%" for i in range(4)))
print("\ne. terminated_reason histogram; success endings vs placed")
for name, r in res.items():
    s = r["reasons"].get("success", 0)
    print(f"  {name}: {dict(sorted(r['reasons'].items()))}  success={s} placed={r['tot']} {'MATCH' if s == r['tot'] else 'MISMATCH'}")
