#!/usr/bin/env python
"""[EXPLORATORY] zero-shot: D29 squash final models (trained on e4->e5 only, 300-step budget) on the 20 RS29 pairs.
Placement/pairs from zeroshot_d29_seed*.json (scripts/eval_probe.py --pairs); ever-held and steps held (D27 `held`,
same definition as Section 3k) from trace_zeroshot_d29_seed*.json.gz (rs29_trace.py), checked against the eval output."""
import gzip, json, math, sys
from pathlib import Path
import numpy as np
E = Path(__file__).parent
PAIRS = "e4:e5,c1:c2,d2:e2,a6:a5,e8:d7,d4:e5,f3:c6,h5:d3,g5:b3,c1:f6,h1:h8,h8:h1,a4:h4,b7:g2,b2:g7,f1:c8,a1:h8,h8:a1,h1:a8,a8:h1".split(",")
BANDS = dict(short=PAIRS[0:6], mid=PAIRS[6:10], long=PAIRS[10:16], diagonal=PAIRS[16:20])
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d, (c + h) / d
def pct(k, n): lo, hi = wilson(k, n); return f"{k}/{n} {100*k/n:.1f}% [{lo*100:.1f},{hi*100:.1f}]"
for i in range(5):
    ev = json.load(open(E / f"zeroshot_d29_seed{i}.json")); tr = json.load(gzip.open(E / f"trace_zeroshot_d29_seed{i}.json.gz", "rt"))
    cols = {c: j for j, c in enumerate(tr["cols"])}
    for e, t in zip(ev["episodes"], tr["episodes"]):
        assert e["reason"] == t["reason"] and e["n"] == len(t["rows"]) and e["place"]["held_steps"] == sum(r[cols["held"]] for r in t["rows"])
    pair = [f'{e["start_square"]}:{e["target_square"]}' for e in ev["episodes"]]
    assert pair == [PAIRS[j % 20] for j in range(300)]
    succ = [e["reason"] == "success" for e in ev["episodes"]]
    hs = [e["place"]["held_steps"] for e in ev["episodes"]]; held = [h > 0 for h in hs]
    r = ev["place_report"]["rates"]
    print(f"seed{i} (budget 300 steps; trace rerun reproduces eval exactly; obs_rms identical {ev['obs_rms_identical_before_after']})")
    print(f"  placed {pct(sum(succ), 300)} | pairs with >=1 placement {len({p for p, s in zip(pair, succ) if s})}/20 | ever held (D27) {pct(sum(held), 300)} | "
          f"median steps held (held episodes) {np.median([h for h in hs if h]) if any(held) else float('nan'):.1f} | funnel held {r['held']['k']} lifted30 {r['lifted_30mm']['k']} carried3cm {r['carried_within_3cm']['k']} released {r['released']['k']} placed {r['placed']['k']}")
    print("  per band placed: " + " ; ".join(f"{b} " + pct(sum(s for s, p in zip(succ, pair) if p in ps), 15 * len(ps)) for b, ps in BANDS.items()))
    print("  per band ever-held: " + " ; ".join(f"{b} {sum(h for h, p in zip(held, pair) if p in ps)}/{15*len(ps)}" for b, ps in BANDS.items()))
    print("  placed per pair: " + " ".join(f"{p}={sum(s for s, q in zip(succ, pair) if q == p)}" for p in PAIRS))
    print("  terminations: " + str({k: sum(e['reason'] == k for e in ev['episodes']) for k in sorted({e['reason'] for e in ev['episodes']})}))
