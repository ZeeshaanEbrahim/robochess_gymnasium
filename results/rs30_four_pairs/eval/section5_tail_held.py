#!/usr/bin/env python
"""RS30 Section 5 item 1: Monitor ever_held over the last 20k and last 100k training steps. Reads runs/ read-only."""
import csv
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
EVAL_HELD = {"A0": 145, "A1": 209, "A3": 175, "B1": 240}
C10 = {"A0": 76.6, "A1": 68.6, "A3": 80.4, "B1": 79.6}     # 10th-chunk values reported in Section 4 (items 2)
print("window = episodes whose end step (cumulative sum of Monitor `l`) is within the last N steps of the Monitor total")
for tag in ("A0", "A1", "A3", "B1"):
    p = REPO / "runs" / f"_probe_cart_rs30{tag[0]}_ppotanh_seed{tag[1]}" / "logs" / "monitor.monitor.csv"
    rows = list(csv.DictReader(l for l in open(p) if not l.startswith("#")))
    L = np.array([int(r["l"]) for r in rows]); H = np.array([r["ever_held"] == "True" for r in rows]); cum = np.cumsum(L)
    out = []
    for N in (20000, 100000):
        m = cum > cum[-1] - N
        out.append(f"last {N//1000}k: {H[m].sum()}/{m.sum()} = {100*H[m].mean():.1f}%")
    print(f"{tag}: Monitor steps {cum[-1]}; " + "; ".join(out) + f"; chunk10 {C10[tag]}%; eval held {EVAL_HELD[tag]}/300 = {100*EVAL_HELD[tag]/300:.1f}%")
