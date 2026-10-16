"""500k vs 1M placement and training-log losses for the SAC obs-norm cell (prereg e0d8981 fail branch). Reads saved JSON and
progress.csv only; no simulator. Wilson 95% with z = 1.96, no continuity correction. Seed 0 rows read the local gitignored
runs/_eval_sac_obsnorm_seed0/{final,ck500k}_300.json (the files behind results/sac_obsnorm_seed0/RESULT.md)."""
import collections, csv, json, math
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[3]            # repo root
here = Path(__file__).resolve().parent; one = here.parent / "eval"
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
def load(f):
    d = json.load(open(f)); E = d["episodes"]; k = sum(e["place"]["placed"] for e in E)
    return d, k, len(E), dict(collections.Counter(e["reason"] for e in E))
src = {0: (R / "runs/_eval_sac_obsnorm_seed0/ck500k_300.json", R / "runs/_eval_sac_obsnorm_seed0/final_300.json")}
for s in (1, 2, 3): src[s] = (here / f"seed{s}_ck500k_300.json", one / f"seed{s}_final_300.json")
print("PLACEMENT, 500k checkpoint vs final model (1M); 300 stochastic episodes, eval seed 20000, each with its own frozen VecNormalize pickle")
print(f"{'seed':>4} | {'500k placed':>11} {'Wilson95':>17} {'reasons':>22} {'rms_same':>8} | {'1M placed':>9} {'Wilson95':>17} {'reasons':>22}")
for s, (f5, f1) in src.items():
    d5, k5, n5, r5 = load(f5); d1, k1, n1, r1 = load(f1); a5 = wilson(k5, n5); a1 = wilson(k1, n1)
    print(f"{s:>4} | {k5:>5}/{n5:<5} [{a5[0]*100:5.2f}%,{a5[1]*100:5.2f}%] {str(r5):>22} {str(d5['obs_rms_identical_before_after']):>8} | {k1:>4}/{n1:<4} [{a1[0]*100:5.2f}%,{a1[1]*100:5.2f}%] {str(r1):>22}")
print("\nother 500k eval fields (descriptive): ")
for s, (f5, _) in src.items():
    d5 = json.load(open(f5)); r = d5["place_report"]["rates"]; E = d5["episodes"]
    print(f"  seed{s}: grasp_ever {sum(e['grasp_ever'] for e in E)}/300 | held {r['held']['k']} | lifted_30mm {r['lifted_30mm']['k']} | carried_within_3cm {r['carried_within_3cm']['k']} | reached S2 {r['released']['k']}")
print("\nTRAINING-LOG LOSSES: mean over a +-5k-step window of logged rows (1M window one-sided; ends ~999k-1,000k); -actor_loss is a rough Q proxy")
keys = ("train/critic_loss", "train/actor_loss", "train/ent_coef", "train/ent_coef_loss")
print(f"{'seed':>4} {'step':>8} {'rows':>4} {'critic_loss':>12} {'actor_loss':>11} {'-actor_loss':>12} {'ent_coef':>9} {'ent_coef_loss':>14}")
for s in (0, 1, 2, 3):
    rows = list(csv.DictReader(open(R / f"runs/_probe_cart_place30_sac_obsnorm_seed{s}/logs/progress.csv"))); ts = np.array([float(r["time/total_timesteps"]) for r in rows])
    for S in (100_000, 500_000, 1_000_000):
        m = (ts >= S - 5000) & (ts <= S + 5000); v = [np.mean([float(r[k]) for r, mm in zip(rows, m) if mm]) for k in keys]
        print(f"{s:>4} {S:>8} {int(m.sum()):>4} {v[0]:>12.2f} {v[1]:>11.2f} {-v[1]:>12.2f} {v[2]:>9.4f} {v[3]:>14.4f}")
