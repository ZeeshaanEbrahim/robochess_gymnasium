#!/usr/bin/env python
"""RS30 Section 4 items 1-3 (no env rollouts). Reads runs/ (read-only) and eval/*.json."""
import csv, io, json, zipfile
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; EV = Path(__file__).resolve().parent
RUNS = [(f"rs30{a}_seed{i}", f"_probe_cart_rs30{a}_ppotanh_seed{i}") for a in "AB" for i in range(4)]
CKS = [100000, 300000, 500000, 1000000, 2000000]
PAIRS = ["e4:e5", "d4:d5", "f4:f5", "e3:e4"]

print("ITEM 1. policy std = exp(log_std), dims (dx, dy, dz, gripper), from the checkpoint zips' policy.pth")
def std_of(zp):
    with zipfile.ZipFile(zp) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    return np.exp(sd["log_std"].numpy())
for name, run in RUNS:
    first = "none"
    print(name)
    for ck in CKS:
        s = std_of(REPO / "runs" / run / "checkpoints" / f"rc_{ck}_steps.zip")
        ok = bool((s < 0.1).all()); first = ck if (first == "none" and ok) else first
        print(f"  {ck:>8}: " + " ".join(f"{v:.4f}" for v in s) + f"   all<0.1: {ok}")
    print(f"  first checkpoint (of 100k,300k,500k,1M,2M) with all 4 dims < 0.1: {first}")

print("\nITEM 2. Monitor ever_held fraction per 10% chunk of training STEPS (column `ever_held` of monitor.monitor.csv)")
print("  ever_held = env latch: True once any step had both pads on the piece and no piece-board contact (held);")
print("  episode takes its chunk from the cumulative step count (sum of column `l`) at its end; chunk edges at total/10 steps")
for name, run in RUNS:
    rows = list(csv.DictReader(open(REPO / "runs" / run / "logs" / "monitor.monitor.csv").readlines()[1:] and
                               [l for l in open(REPO / "runs" / run / "logs" / "monitor.monitor.csv") if not l.startswith("#")]))
    L = np.array([int(r["l"]) for r in rows]); H = np.array([r["ever_held"] == "True" for r in rows])
    cum = np.cumsum(L); tot = cum[-1]; ch = np.minimum((cum - 1) * 10 // tot, 9)
    print(f"{name}: episodes {len(rows)}, steps {tot}")
    print("   held %: " + " ".join(f"{100*H[ch==c].mean():5.1f}" for c in range(10)) + "   | n eps: " + " ".join(str(int((ch==c).sum())) for c in range(10)))

print("\nITEM 3. per model, per pair: grasp_ever / place.held / placed, out of 75 (from the 8 RS30 eval JSONs)")
print("  grasp_ever = any step with info['grasp_success'] (_grasp_success: piece origin > 10 mm above board, gripper closed,")
print("               BOTH pads on piece, piece not touching board)")
print("  place.held = any step with pinched and not board (both pads on piece, no piece-board contact; no lift or closure needed)")
for name, _ in RUNS:
    d = json.load(open(EV / f"{name}_final.json")); out = {}
    for p in PAIRS:
        E = [e for e in d["episodes"] if f'{e["start_square"]}:{e["target_square"]}' == p]
        out[p] = (sum(e["grasp_ever"] for e in E), sum(e["place"]["held"] for e in E), sum(e["place"]["placed"] for e in E), len(E))
    print(f"{name}: " + " | ".join(f"{p} grasp {g}/{n} held {h}/{n} placed {k}/{n}" for p, (g, h, k, n) in out.items()))
    E = d["episodes"]
    print(f"    total grasp {sum(e['grasp_ever'] for e in E)}/300 held {sum(e['place']['held'] for e in E)}/300 "
          f"released(S2 reached) {sum(e['place']['released'] for e in E)}/300 placed {sum(e['place']['placed'] for e in E)}/300 "
          f"max_lift_mm over episodes {max(e['max_lift_mm'] for e in E):.1f}")
for name in ("rs30A_seed0", "rs30A_seed3"):
    E = json.load(open(EV / f"{name}_final.json"))["episodes"]
    hs = np.array([e["place"]["held_steps"] for e in E]); gs = np.array([e["grasp_steps"] for e in E])
    print(f"{name}: episodes with held_steps>0: {(hs>0).sum()}, held_steps mean over those {hs[hs>0].mean() if (hs>0).any() else 0:.1f}, "
          f"max {hs.max()}; episodes with grasp_steps>0: {(gs>0).sum()}; min_d_mm min/median {min(e['min_d_mm'] for e in E):.1f}/"
          f"{np.median([e['min_d_mm'] for e in E]):.1f}")
