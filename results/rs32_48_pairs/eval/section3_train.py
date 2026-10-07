#!/usr/bin/env python
"""RS32 Step 3 items b, c, d (no env rollouts). Reads runs/ read-only.
b: policy std = exp(log_std) from policy.pth inside rc_{1M,2M,3M,4M,5M}_steps.zip and final_model.zip (5,005,312 steps).
c: Monitor CSV (column `ever_held`). Chunk of an episode = min((cum_l - 1) * 10 // total_l, 9), cum_l the cumulative `l` at the
   episode's end, per file; D is computed per Monitor file (4 files, own totals) and the counts pooled.
d: training placement (Monitor terminated_reason == "success") per 100k TRAINING steps: bin = (cum_l - 1) // 25000 per file
   (n_envs 4, each file holds ~1/4 of the steps; same convention as RS31), counts pooled over the four files."""
import io, zipfile
import numpy as np, torch
from rs32_lib import *
CKS = [1000000, 2000000, 3000000, 4000000, 5000000]

print("ITEM b. policy std = exp(log_std) per action dim (dx, dy, dz, gripper) and their mean")
for name in NAMES:
    run = REPO / "runs" / RUN(name); print(name)
    srcs = [(f"{ck:>8} (checkpoint)", run / "checkpoints" / f"rc_{ck}_steps.zip") for ck in CKS] + [("final_model (5,005,312)", run / "final_model.zip")]
    for lab, zp in srcs:
        with zipfile.ZipFile(zp) as z:
            sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
        s = np.exp(sd["log_std"].numpy())
        print(f"  {lab:<24}: " + " ".join(f"{v:.4f}" for v in s) + f"   mean {s.mean():.4f}")

mon = {name: monitor(RUN(name)) for name in NAMES}
for name in NAMES: assert len(mon[name]) == 4

print("\nITEM c. Monitor ever_held per 10% chunk of training (%; pooled over the four files)")
for name in NAMES:
    K = np.zeros(10, int); N = np.zeros(10, int); tots = []
    for m in mon[name]:
        cum = np.cumsum(m["l"]); tot = int(cum[-1]); tots.append(tot); ch = np.minimum((cum - 1) * 10 // tot, 9)
        for c in range(10): K[c] += m["held"][ch == c].sum(); N[c] += (ch == c).sum()
    print(f"{name}: files {[m['file'] for m in mon[name]]} steps per file {tots} (sum {sum(tots):,}) episodes {sum(len(m['l']) for m in mon[name])}")
    print("   ever_held %: " + " ".join(f"{100*K[c]/N[c]:5.1f}" for c in range(10)) + "   | n eps: " + " ".join(str(int(N[c])) for c in range(10)))

print("\nITEM d. rolling training placement per 100k training steps; k/n eps = % (bin j = training steps (j-1)*100k+1 .. j*100k; 25k steps of each of the 4 files)")
roll = {}
for name in NAMES:
    K = {}; N = {}
    for m in mon[name]:
        cum = np.cumsum(m["l"]); b = (cum - 1) // 25000
        for j in np.unique(b): K[j] = K.get(j, 0) + int(m["placed"][b == j].sum()); N[j] = N.get(j, 0) + int((b == j).sum())
    roll[name] = (K, N)
nb = max(max(K) for K, N in roll.values()) + 1
print(f'{"bin(100k)":>10}' + "".join(f"{n:>17}" for n in NAMES))
for j in range(nb):
    cells = ""
    for name in NAMES:
        K, N = roll[name]; cells += f"{(format(100*K[j]/N[j],'.1f')+'% '+str(K[j])+'/'+str(N[j])) if j in N else '-':>17}"
    print(f"{((j+1)*100000 if j + 1 <= 50 else '5.0M+'):>10}" + cells + ("   (partial bin: the last ~5k training steps after 5,000,000; Monitor files hold 5,004,9xx-5,005,0xx steps, the rest of the 5,005,312 is unfinished episodes)" if j + 1 > 50 else ""))
