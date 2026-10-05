#!/usr/bin/env python
"""RS29 exploratory h and i, and pre-registered 3e, from runs/<run>/logs/monitor.monitor.csv (read-only).
h, i: EXPLORATORY (not pre-registered). 3e: training-log placement by band, last 10% of episodes per seed."""
import math, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from robochess.tasks.squares import ALL_SQUARES, square_to_local_xy
PAIRS = "e4:e5,c1:c2,d2:e2,a6:a5,e8:d7,d4:e5,f3:c6,h5:d3,g5:b3,c1:f6,h1:h8,h8:h1,a4:h4,b7:g2,b2:g7,f1:c8,a1:h8,h8:a1,h1:a8,a8:h1".split(",")
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d, (c + h) / d
def pct(k, n):
    if n == 0: return "0/0"
    lo, hi = wilson(k, n); return f"{k}/{n} ({100*k/n:.1f}%, [{100*lo:.1f},{100*hi:.1f}])"
def chunks(df): return [df.iloc[ix] for ix in np.array_split(np.arange(len(df)), 10)]
dfs = {}
for i in range(6):
    df = pd.read_csv(REPO / f"runs/_probe_cart_randsq29_ppotanh_seed{i}/logs/monitor.monitor.csv", skiprows=1)
    df["reason"] = df.terminated_reason.fillna("time_limit"); df["ever_held"] = df.ever_held.astype(str).str.lower().isin(["true", "1", "1.0"])
    df["d_mm"] = [1000 * np.linalg.norm(square_to_local_xy(s) - square_to_local_xy(t)) for s, t in zip(df.start_square, df.target_square)]
    dfs[i] = df
print("[EXPLORATORY] h. ten equal chunks of episodes in time order, per seed")
for i, df in dfs.items():
    print(f"seed{i}: {len(df)} episodes; chunk sizes {[len(c) for c in chunks(df)]}")
    print("  chunk  eps_end_step  ever_held   S0    S1    S2    S3   piece_off_board  bad_contact  success")
    for j, c in enumerate(chunks(df)):
        n = len(c); sc = c.place_state.value_counts()
        print(f"  {j+1:>4}  {int(c.l.sum()):>10}   {100*c.ever_held.mean():6.1f}%  " + " ".join(f"{100*sc.get(s,0)/n:4.1f}%" for s in ("S0", "S1", "S2", "S3"))
              + f"   {100*(c.reason=='piece_off_board').mean():8.1f}%        {100*(c.reason=='bad_contact').mean():6.1f}%     {int((c.reason=='success').sum())}")
edge = lambda sq: sq[1] in "18" or sq[0] in "ah"
n_edge = sum(edge(s) for s in ALL_SQUARES if s not in ("d1", "e1")); print(f"\nreference: {n_edge}/62 allowed squares are edge squares (rank 1/8 or file a/h) = {100*n_edge/62:.1f}%")
print("\n[EXPLORATORY] i. piece_off_board rate by edge vs non-edge square, last 50% of episodes per seed")
for role in ("start_square", "target_square"):
    print(f"-- {role}")
    for i, df in dfs.items():
        h = df.iloc[len(df) // 2:]; e = h[role].map(edge); off = h.reason == "piece_off_board"
        print(f"seed{i} (last {len(h)} eps): edge {pct(int((off & e).sum()), int(e.sum()))} | non-edge {pct(int((off & ~e).sum()), int((~e).sum()))} | all {pct(int(off.sum()), len(h))}")
print("\n-- piece_off_board by (start edge?, target edge?) , last 50%")
for i, df in dfs.items():
    h = df.iloc[len(df) // 2:]; es = h.start_square.map(edge); et = h.target_square.map(edge); off = h.reason == "piece_off_board"
    print(f"seed{i}: " + " | ".join(f"start {'edge' if a else 'in  '} target {'edge' if b else 'in  '}: {int((off&(es==a)&(et==b)).sum())}/{int(((es==a)&(et==b)).sum())}" for a in (True, False) for b in (True, False)))

print("\n3e. training-log placement by band, last 10% of episodes per seed (success terminations; Monitor start_square/target_square/place_state)")
def band(d): return "short" if d <= 110.5 else "mid" if d < 500 else "long" if d < 600 else "diagonal"
for i, df in dfs.items():
    t = df.iloc[-(len(df) // 10):].copy(); t["band"] = t.d_mm.map(band)
    print(f"seed{i}: last {len(t)} eps (steps {int(t.l.sum())}); success {int((t.reason=='success').sum())}; end place_state {t.place_state.value_counts().to_dict()}")
    for b in ("short", "mid", "long", "diagonal"):
        g = t[t.band == b]; print(f"    {b:8s}: eps {len(g):4d} success {int((g.reason=='success').sum())}  S2-ending {int((g.place_state=='S2').sum())}  ever_held {pct(int(g.ever_held.sum()), len(g))}")
    print(f"    d<0.4m: success {int(((t.reason=='success')&(t.d_mm<400)).sum())}/{int((t.d_mm<400).sum())}; >=0.4m: {int(((t.reason=='success')&(t.d_mm>=400)).sum())}/{int((t.d_mm>=400).sum())}")
print("\nwhole-run totals: success terminations per seed:", {i: int((df.reason == 'success').sum()) for i, df in dfs.items()})
