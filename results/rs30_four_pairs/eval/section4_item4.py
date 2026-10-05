#!/usr/bin/env python
"""RS30 Section 4 item 4: zero-shot held-out placement. Reads eval/zs_*.json only."""
import collections, json, math
from pathlib import Path
EV = Path(__file__).resolve().parent
NAMES = [f"rs30{a}_seed{i}" for a in "AB" for i in range(4)]
BANDS = dict(short="e4:e5 c1:c2 d2:e2 a6:a5 e8:d7 d4:e5".split(), mid="f3:c6 h5:d3 g5:b3 c1:f6".split(),
             long="h1:h8 h8:h1 a4:h4 b7:g2 b2:g7 f1:c8".split(), diagonal="a1:h8 h8:a1 h1:a8 a8:h1".split())
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)
def by_pair(f, n_exp):
    d = json.load(open(EV / f)); assert d["n"] == n_exp and d["seed"] == 20000 and d["obs_rms_identical_before_after"] is True
    k, n = collections.Counter(), collections.Counter()
    for e in d["episodes"]:
        p = f'{e["start_square"]}:{e["target_square"]}'; n[p] += 1; k[p] += e["place"]["placed"]
    assert sum(k.values()) == d["place_report"]["rates"]["placed"]["k"]
    return k, n
fmt = lambda k, n: f"{k}/{n} [{wilson(k,n)[0]:.1f},{wilson(k,n)[1]:.1f}]"
print("(a) near held-out pairs d5:d6, c4:c5, 75 episodes each (Wilson 95%, z=1.96, in %)")
for nm in NAMES:
    k, n = by_pair(f"zs_near_{nm}.json", 150)
    print(f"  {nm}: d5:d6 {fmt(k['d5:d6'], n['d5:d6'])}   c4:c5 {fmt(k['c4:c5'], n['c4:c5'])}   both {fmt(sum(k.values()), 150)}")
print("\n(b) the 20 RS29 pairs (list from results/rs29_random_squares/PREREGISTRATION.md), 15 episodes per pair, 300 total")
for nm in NAMES:
    k, n = by_pair(f"zs_rs29pairs_{nm}.json", 300)
    assert len(n) == 20 and set(n.values()) == {15}
    nz = {p: k[p] for p in n if k[p]}
    print(f"  {nm}: total {fmt(sum(k.values()), 300)}; pairs with >=1 placement {len(nz)}/20 {nz}")
    print("      bands: " + "; ".join(f"{b} {fmt(sum(k[p] for p in ps), 15*len(ps))}" for b, ps in BANDS.items()))
