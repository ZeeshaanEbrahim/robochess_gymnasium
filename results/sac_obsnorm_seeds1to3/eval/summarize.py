"""Per-seed summary of the eval_probe JSONs for the SAC obs-norm cell. Pure arithmetic over saved JSON; no simulator.
Wilson 95% interval with z = 1.96 and no continuity correction. Verdict: placed >= 60 of 300 (prereg e0d8981)."""
import collections, json, math, sys
from pathlib import Path
GATE = 60
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
here = Path(__file__).resolve().parent
files = {0: Path(sys.argv[1]) if len(sys.argv) > 1 else None, 1: here / "seed1_final_300.json", 2: here / "seed2_final_300.json", 3: here / "seed3_final_300.json"}
passes = 0; n_seeds = 0
for s, f in files.items():
    if f is None: continue
    d = json.load(open(f)); E = d["episodes"]; n = len(E); k = sum(e["place"]["placed"] for e in E); lo, hi = wilson(k, n)
    v = "PASS" if k >= GATE else "FAIL"; passes += v == "PASS"; n_seeds += 1
    r = d["place_report"]["rates"]
    print(f"seed{s}: placed {k}/{n}  Wilson95 [{lo*100:.2f}%, {hi*100:.2f}%]  verdict {v} (gate {GATE}/300)")
    print(f"   run {d['run']} model {d['model']} eval seed {d['seed']} deterministic {d['deterministic']} obs_rms_identical_before_after {d['obs_rms_identical_before_after']}")
    print(f"   terminated_reason {dict(collections.Counter(e['reason'] for e in E))}")
    print(f"   grasp_ever {sum(e['grasp_ever'] for e in E)}/{n} | held {r['held']['k']}/{n} | lifted_30mm {r['lifted_30mm']['k']}/{n} | carried_within_3cm {r['carried_within_3cm']['k']}/{n} | reached S2 (released) {r['released']['k']}/{n}")
print(f"CELL: {passes} of {n_seeds} seeds pass the per-seed gate of {GATE}/300 (N = {n_seeds}; the prereg defines no cell-level threshold)")
