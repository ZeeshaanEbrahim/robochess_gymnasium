#!/usr/bin/env python
"""POST HOC / EXPLORATORY. Tables from the training artifacts only (Monitor CSVs, checkpoint zips). No rollouts.
Training step of an episode = 4 x (cumulative `l` in its Monitor file) (n_envs 4, four files of ~1.25M steps each; an approximation of
training progress, the same convention as the 100k bins in ../section3_train_output.txt)."""
import io, zipfile
import torch
from diag_common import *
tag = lambda n: n.replace("rs32D_", "")
print("1. POLICY STD = exp(log_std) per action dim (dx, dy, dz, gripper) and mean, from policy.pth in rc_{1,2,3,4,5}M_steps.zip and final_model.zip")
for n in NAMES:
    print(n)
    for lab, zp in [(f"{ck//10**6}M ckpt", REPO/"runs"/RUN(n)/"checkpoints"/f"rc_{ck}_steps.zip") for ck in (1000000, 2000000, 3000000, 4000000, 5000000)] + [("final 5,005,312", REPO/"runs"/RUN(n)/"final_model.zip")]:
        with zipfile.ZipFile(zp) as z: sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
        s = np.exp(sd["log_std"].numpy()); print(f"   {lab:<16}" + " ".join(f"{v:.4f}" for v in s) + f"   mean {s.mean():.4f}")
mon = {n: monitor(RUN(n)) for n in NAMES}
print("\n2. EVER_HELD per 10% chunk of training (%; pooled over the 4 Monitor files; chunk = min((cum_l-1)*10//total_l, 9) per file)")
print(f'{"seed":<12}' + "".join(f"{i*10:>3}-{i*10+10:<3}" for i in range(10)))
for n in NAMES:
    K = np.zeros(10, int); N = np.zeros(10, int)
    for m in mon[n]:
        ch = chunk_counts(m)
        for c in range(10): K[c] += m["held"][ch == c].sum(); N[c] += (ch == c).sum()
    print(f"{tag(n):<12}" + "".join(f"{100*K[c]/N[c]:>7.1f}" for c in range(10)))
def rows_of(n):
    """per-episode (training step, pair, placed) over the four files"""
    out = []
    for f, m in zip(sorted((REPO/"runs"/RUN(n)/"logs").glob("monitor_*.monitor.csv")), mon[n]):
        rows = list(csv.DictReader(l for l in open(f) if not l.startswith("#")))
        step = np.cumsum(m["l"]) * 4
        out += [(int(step[i]), f'{r["start_square"]}:{r["target_square"]}', bool(m["placed"][i])) for i, r in enumerate(rows)]
    return out
R_ = {n: rows_of(n) for n in NAMES}
print("\n3. TRAINING PLACEMENT in the LAST 500k training steps (steps 4.5M to 5.0M; Monitor terminated_reason == success), pooled and by pair category; k/n and %")
print(f'{"category":<16}' + "".join(f"{tag(n):>18}" for n in NAMES))
def cell(n, sel, lo=4_500_000, hi=5_010_000):
    k = t = 0
    for s, p, ok in R_[n]:
        if lo < s <= hi and sel(p): k += ok; t += 1
    return k, t
for lab, sel in [("all 48 pairs", lambda p: True)] + [(c, (lambda p, c=c: pair_cat(p) == c)) for c in CATS] + [("neighbour 30", lambda p: p in NEIGH), ("long 18", lambda p: p in LONG)]:
    print(f"{lab:<16}" + "".join(f"{cell(n, sel)[0]:>8}/{cell(n, sel)[1]:<5}{100*cell(n, sel)[0]/cell(n, sel)[1]:4.0f}%" for n in NAMES))
print("\n   the same for 1M-step windows (all seeds), by category: placement % (k/n omitted)")
W = [(0, 1e6), (1e6, 2e6), (2e6, 3e6), (3e6, 4e6), (4e6, 4.5e6), (4.5e6, 5.01e6)]
print(f'{"seed":<10}{"category":<16}' + "".join(f"{a/1e6:.1f}-{b/1e6:.1f}M".rjust(11) for a, b in W[:-1]) + "4.5-5.0M".rjust(11))
for n in NAMES:
    for c in CATS:
        cells = []
        for a, b in W:
            k, t = cell(n, lambda p, c=c: pair_cat(p) == c, a, b); cells.append(f"{100*k/t:5.1f}" if t else "   - ")
        print(f"{tag(n):<10}{c:<16}" + "".join(x.rjust(11) for x in cells))
print("\n4. TRAINING PLACEMENT PER PAIR in the last 500k steps (k/n), order of PREREGISTRATION.md")
print(f'{"pair":<8}' + "".join(f"{tag(n):>12}" for n in NAMES))
for p in PAIRS: print(f"{p:<8}" + "".join(f"{cell(n, lambda q: q == p)[0]:>6}/{cell(n, lambda q: q == p)[1]:<5}" for n in NAMES))
