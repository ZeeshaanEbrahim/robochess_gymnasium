#!/usr/bin/env python
"""Rule 3 helper (mechanical): which pre-registered subset the pairs under 10/50 belong to, per seed. Reads eval/*_final.json only."""
import collections, json
from rs32_lib import *
print("pairs under 10/50 split by subset (30 neighbour = first 30 pairs in PREREGISTRATION.md order, 18 longer = last 18)")
for name in NAMES:
    d = json.load(open(EV / f"{name}_final.json")); k = collections.Counter()
    for e in d["episodes"]: k[f'{e["start_square"]}:{e["target_square"]}'] += e["place"]["placed"]
    fn = [p for p in NEIGH if k[p] < 10]; fl = [p for p in LONG if k[p] < 10]
    print(f"  {name}: neighbour {len(fn)}/30 {fn}   long {len(fl)}/18 {fl}")
