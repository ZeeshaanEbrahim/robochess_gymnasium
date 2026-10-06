#!/usr/bin/env python
"""RS31 Section 3 items 4 and 5 and addition B (no env rollouts). Reads runs/ read-only and eval/*.json.
item 4: policy std = exp(log_std) from the rc_{500000,1000000,2000000,3000000}_steps.zip policy.pth (NOT the final model).
item 5: Monitor CSV (column `ever_held`; placement = column `terminated_reason` == "success"; episode length = column `l`).
  Chunk of an episode = min((cum_l - 1) * 10 // total_l, 9) with cum_l the cumulative `l` at the episode's end, per file.
  D: computed per Monitor file (4 files, own totals) and the counts pooled.
  Rolling placement per 100k TRAINING steps: bin = (cum_l - 1) // (100000 / n_envs) per file, counts pooled. For D (n_envs 4) each
  file holds ~1/4 of the 3M steps, so 100k training steps = 25k file steps (interpretation, disclosed; RS30 had a single file).
B: eval vs training cross-check (Wilson 95%, z=1.96, no continuity correction; flag |gap| > 10 percentage points; no explanation).
"""
import csv, io, json, math, zipfile
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; EV = Path(__file__).resolve().parent
RUNS = [(f"rs31{a}_seed{i}", f"_probe_cart_rs31{a}_ppotanh_seed{i}", 1 if a == "C" else 4) for a in "CD" for i in range(2)]
CKS = [500000, 1000000, 2000000, 3000000]
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h))
overlap = lambda a, b: "intervals OVERLAP" if (a[0] <= b[1] and b[0] <= a[1]) else "intervals do NOT overlap"
fmt = lambda k, n: f"{k}/{n} = {100*k/n:5.1f}% [{wilson(k,n)[0]:5.1f},{wilson(k,n)[1]:5.1f}]"

print("ITEM 4. policy std = exp(log_std) per action dim (dx, dy, dz, gripper) and their mean; checkpoint zips policy.pth (3,000,000 checkpoint, not final_model)")
for name, run, ne in RUNS:
    print(name)
    for ck in CKS:
        with zipfile.ZipFile(REPO / "runs" / run / "checkpoints" / f"rc_{ck}_steps.zip") as z:
            sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
        s = np.exp(sd["log_std"].numpy())
        print(f"  {ck:>8}: " + " ".join(f"{v:.4f}" for v in s) + f"   mean {s.mean():.4f}")

def monitor(run):
    d = REPO / "runs" / run / "logs"; fs = sorted(d.glob("monitor*.csv")); out = []
    for f in fs:
        rows = list(csv.DictReader(l for l in open(f) if not l.startswith("#")))
        out.append(dict(file=f.name, l=np.array([int(r["l"]) for r in rows]), held=np.array([r["ever_held"] == "True" for r in rows]),
                        placed=np.array([r["terminated_reason"] == "success" for r in rows])))
    return out

mon = {name: monitor(run) for name, run, ne in RUNS}
print("\nITEM 5a. Monitor ever_held per 10% chunk of training (k/n = %; pooled over files for D)")
chunk_held = {}
for name, run, ne in RUNS:
    K = np.zeros(10, int); N = np.zeros(10, int); tots = []
    for m in mon[name]:
        cum = np.cumsum(m["l"]); tot = int(cum[-1]); tots.append(tot); ch = np.minimum((cum - 1) * 10 // tot, 9)
        for c in range(10): K[c] += m["held"][ch == c].sum(); N[c] += (ch == c).sum()
    chunk_held[name] = (int(K[9]), int(N[9]))
    print(f"{name}: files {[m['file'] for m in mon[name]]} steps per file {tots} episodes {sum(len(m['l']) for m in mon[name])}")
    print("   ever_held %: " + " ".join(f"{100*K[c]/N[c]:5.1f}" for c in range(10)) + "   | n eps: " + " ".join(str(int(N[c])) for c in range(10)))

print("\nITEM 5b. rolling training placement (Monitor terminated_reason == success) per 100k training steps; k/n eps, % (bin = steps (j-1)*100k+1 .. j*100k)")
roll = {}
for name, run, ne in RUNS:
    width = 100000 // ne; K = {}; N = {}
    for m in mon[name]:
        cum = np.cumsum(m["l"]); b = (cum - 1) // width
        for j in np.unique(b): K[j] = K.get(j, 0) + int(m["placed"][b == j].sum()); N[j] = N.get(j, 0) + int((b == j).sum())
    roll[name] = (K, N)
nb = max(max(K) for K, N in roll.values()) + 1
print(f'{"bin(100k)":>10}' + "".join(f"{n:>22}" for n, _, _ in RUNS))
for j in range(nb):
    cells = ""
    for name, _, _ in RUNS:
        K, N = roll[name]; cells += f"{(str(K[j])+'/'+str(N[j])+' = '+format(100*K[j]/N[j],'.1f')+'%') if j in N else '-':>22}"
    print(f"{(j+1)*100000:>10}" + cells + ("   (partial bin: last steps only)" if j == nb - 1 and nb > 30 else ""))

print("\nB. eval vs training cross-check (Wilson 95%); gap = eval - training, percentage points; FLAG if |gap| > 10")
print("(1) eval held rate (place.held over the 1200 final-model episodes) vs ever_held in the LAST 10% chunk of the Monitor files")
print("(2) eval pooled placement k/1200 vs training placement over the last 100k training steps (Monitor; D: last 25k steps of each of the 4 files, pooled)")
for name, run, ne in RUNS:
    d = json.load(open(EV / f"{name}_final.json")); E = d["episodes"]
    eh = sum(e["place"]["held"] for e in E); ep = sum(e["place"]["placed"] for e in E)
    th, tn = chunk_held[name]
    pk = pn = 0
    for m in mon[name]:
        cum = np.cumsum(m["l"]); w = cum > cum[-1] - 100000 // ne
        pk += int(m["placed"][w].sum()); pn += int(w.sum())
    g1 = 100 * eh / 1200 - 100 * th / tn; g2 = 100 * ep / 1200 - 100 * pk / pn
    print(f"{name}")
    print(f"  (1) held: eval {fmt(eh,1200)}   training last-10% chunk {fmt(th,tn)}   gap {g1:+.1f} pts {'FLAG' if abs(g1) > 10 else 'ok'}; {overlap(wilson(eh,1200), wilson(th,tn))} (eval [{wilson(eh,1200)[0]:.3f},{wilson(eh,1200)[1]:.3f}] vs training [{wilson(th,tn)[0]:.3f},{wilson(th,tn)[1]:.3f}])")
    print(f"  (2) placed: eval {fmt(ep,1200)}   training last 100k steps {fmt(pk,pn)}   gap {g2:+.1f} pts {'FLAG' if abs(g2) > 10 else 'ok'}; {overlap(wilson(ep,1200), wilson(pk,pn))} (eval [{wilson(ep,1200)[0]:.3f},{wilson(ep,1200)[1]:.3f}] vs training [{wilson(pk,pn)[0]:.3f},{wilson(pk,pn)[1]:.3f}])")
