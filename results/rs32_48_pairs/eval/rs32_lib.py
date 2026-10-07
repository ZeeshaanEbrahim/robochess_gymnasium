"""Shared helpers for the RS32 evaluation scripts (pair lists, Wilson interval, formatting). No rollouts."""
import csv, json, math, re
import numpy as np
from pathlib import Path
EV = Path(__file__).resolve().parent
REPO = EV.parents[2]
PREREG = EV.parent / "PREREGISTRATION.md"
SEEDS = [0, 100, 200, 300, 400, 500]
NAMES = [f"rs32D_seed{s}" for s in SEEDS]
RUN = lambda name: f"_probe_cart_{name.replace('_seed', '_ppotanh_seed')}"

def prereg_pairs():
    line = next(l for l in PREREG.read_text().splitlines() if l.startswith("Training pairs"))
    p = re.findall(r"\b[a-h][1-8]:[a-h][1-8]\b", line.split("in this order):", 1)[1])
    assert len(p) == 48 and len(set(p)) == 48
    return p
PAIRS = prereg_pairs()

def cheb(p):
    s, t = p.split(":"); return max(abs(ord(s[0]) - ord(t[0])), abs(int(s[1]) - int(t[1])))
NEIGH = PAIRS[:30]; LONG = PAIRS[30:]
assert all(cheb(p) == 1 for p in NEIGH) and all(cheb(p) in (2, 3) for p in LONG) and len(LONG) == 18
# the 24 RS31 pairs, parsed from the RS31 pre-registration on branch rs31 (read with git show)
import subprocess
_l = next(l for l in subprocess.run(["git", "show", "rs31:results/rs31_24_pairs/PREREGISTRATION.md"], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines() if l.startswith("Training pairs"))
RS31_PAIRS = re.findall(r"\b[a-h][1-8]:[a-h][1-8]\b", _l.split("):", 1)[1])
assert len(RS31_PAIRS) == 24 and all(p in PAIRS for p in RS31_PAIRS)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h))
fmt = lambda k, n: f"{k}/{n} = {100*k/n:.1f}% [{wilson(k,n)[0]:.1f},{wilson(k,n)[1]:.1f}]"
overlap = lambda a, b: a[0] <= b[1] and b[0] <= a[1]

def monitor(run):
    fs = sorted((REPO / "runs" / run / "logs").glob("monitor_*.monitor.csv")); out = []
    for f in fs:
        rows = list(csv.DictReader(l for l in open(f) if not l.startswith("#")))
        out.append(dict(file=f.name, l=np.array([int(r["l"]) for r in rows]), held=np.array([r["ever_held"] == "True" for r in rows]),
                        placed=np.array([r["terminated_reason"] == "success" for r in rows])))
    return out

def chunk_counts(m, n_chunks=10):
    """per-file 10% chunk index of every episode (same definition as RS31)."""
    cum = np.cumsum(m["l"]); tot = int(cum[-1]); return np.minimum((cum - 1) * n_chunks // tot, n_chunks - 1)
