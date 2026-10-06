#!/usr/bin/env python
"""RS31 Section 3 item 3 (follow-up): policy std = exp(log_std) for C seed 0 and C seed 1 at every one of the 30 rc checkpoints
and the final model (dims dx, dy, dz, gripper) from policy.pth inside the zips. Reads runs/ read-only. No interpretation."""
import io, zipfile
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]
def std_of(zp):
    with zipfile.ZipFile(zp) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    return np.exp(sd["log_std"].numpy())
for name in ("rs31C_seed0", "rs31C_seed1"):
    run = REPO / "runs" / f"_probe_cart_{name.split('_')[0]}_ppotanh_{name.split('_')[1]}"
    rows = [(ck, std_of(run / "checkpoints" / f"rc_{ck}_steps.zip")) for ck in range(100000, 3000001, 100000)]
    assert len(rows) == 30
    rows.append(("final", std_of(run / "final_model.zip")))
    print(f"{name}: exp(log_std) at the 30 rc checkpoints and final_model.zip")
    print(f'  {"step":>8} {"dx":>9} {"dy":>9} {"dz":>9} {"gripper":>10} {"mean":>9}')
    for ck, s in rows: print(f"  {str(ck):>8} " + " ".join(f"{v:9.4f}" if i < 3 else f"{v:10.4f}" for i, v in enumerate(s)) + f" {s.mean():9.4f}")
    g = [s[3] for ck, s in rows[:30]]
    dec = [(rows[i][0], rows[i + 1][0], g[i], g[i + 1]) for i in range(29) if g[i + 1] < g[i]]
    print(f"  gripper std over the 30 checkpoints: first {g[0]:.4f}, max {max(g):.4f} at {rows[int(np.argmax(g))][0]}, last {g[-1]:.4f}; "
          f"strictly increasing at every step: {not dec}; decreases at {len(dec)} step(s)" + "".join(f"\n    {a}->{b}: {x:.4f} -> {y:.4f}" for a, b, x, y in dec))
    gf = rows[30][1][3]; print(f"  final_model.zip gripper std {gf:.4f} (3,000,000 checkpoint {g[-1]:.4f}); including final: monotone non-decreasing {all(g[i] <= g[i+1] for i in range(29)) and g[-1] <= gf}\n")
