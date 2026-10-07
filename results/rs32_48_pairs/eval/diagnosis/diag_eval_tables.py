#!/usr/bin/env python
"""POST HOC / EXPLORATORY. Tables computed from the committed eval JSONs only (no dumps, no rollouts): the 2400 Step-2 episodes per seed,
zs_*.json and det48_*.json. Reads ../*.json."""
from diag_common import *
import scipy.stats  # noqa (not used for p-values; no tests are run)
E = {n: eval_eps(n) for n in NAMES}
K = {n: collections.Counter() for n in NAMES}
for n in NAMES:
    for e in E[n]: K[n][pk(e)] += e["place"]["placed"]
tag = lambda n: n.replace("rs32D_", "")

print("A. PLACEMENT BY PAIR CATEGORY (full 2400 episodes per seed, 50 per pair). back = target rank below start rank. k/n and %")
print(f'{"category":<16}{"pairs":>6}{"n/seed":>8}' + "".join(f"{tag(n):>16}" for n in NAMES))
for c in CATS:
    ps = [p for p in PAIRS if pair_cat(p) == c]
    print(f"{c:<16}{len(ps):>6}{50*len(ps):>8}" + "".join(f"{sum(K[n][p] for p in ps):>8}/{50*len(ps):<4}{100*sum(K[n][p] for p in ps)/(50*len(ps)):4.0f}%" for n in NAMES))
print("\nB. GAP DECOMPOSITION: mean pooled placement of seeds 100,300,400 minus mean of seeds 0,200,500, split by category (pts of the 48-pair pooled rate)")
print("   (seeds 100, 300 and 400 are grouped here only by their pooled result; see section 2 of DIAGNOSIS.md: seed 400 differs from seeds 100 and 300)")
good = ["rs32D_seed100", "rs32D_seed300", "rs32D_seed400"]; tot = 0; rows = []
for c in CATS:
    ps = [p for p in PAIRS if pair_cat(p) == c]; r = {n: 100*sum(K[n][p] for p in ps)/(50*len(ps)) for n in NAMES}
    fa = np.mean([r[n] for n in FAIL_SEEDS]); ga = np.mean([r[n] for n in good]); gap = (ga - fa)*len(ps)/48; tot += gap; rows.append((c, len(ps), fa, ga, gap))
for c, npairs, fa, ga, gap in rows: print(f"   {c:<16} pairs {npairs:>2}  failing-seed mean {fa:5.1f}%  good-seed mean {ga:5.1f}%  contribution {gap:5.1f} pts ({100*gap/tot:4.1f}% of {tot:.1f})")

print("\nC. NEIGHBOUR (30) PAIRS BELOW 10/50 AND BELOW 25/50, per seed (also the 18 long pairs and all 48)")
print(f'{"seed":<14}{"neigh <10/50":>14}{"neigh <25/50":>14}{"long <10/50":>13}{"long <25/50":>13}{"all48 <10":>11}{"all48 <25":>11}')
for n in NAMES:
    f = lambda ps, th: sum(K[n][p] < th for p in ps)
    print(f"{tag(n):<14}{f(NEIGH,10):>14}{f(NEIGH,25):>14}{f(LONG,10):>13}{f(LONG,25):>13}{f(PAIRS,10):>11}{f(PAIRS,25):>11}")
print("   neighbour pairs below 25/50, listed:")
for n in NAMES: print(f"   {tag(n):<12}" + (", ".join(f"{p} {K[n][p]}" for p in NEIGH if K[n][p] < 25) or "none"))

print("\nD. BACKWARD PAIRS, placed counts from the FULL 2400-episode eval (50 episodes per pair)")
for c in ("neigh_back", "long_back"):
    ps = [p for p in PAIRS if pair_cat(p) == c]; print(f"   {c} pairs: {ps}")
    for n in NAMES: print(f"   {tag(n):<12} {sum(K[n][p] for p in ps)}/{50*len(ps)} = {100*sum(K[n][p] for p in ps)/(50*len(ps)):.1f}%   per pair " + " ".join(f"{p} {K[n][p]}" for p in ps))

print("\nE. LIFT AMONG PLACED EPISODES (full 2400; max_lift_mm = eval_probe's per-episode maximum of the piece lift, mm)")
print("   The question: do placements come from sliding the piece along the board (low lift) or from lifting and carrying?")
print(f'{"seed":<12}{"placed":>8}{"placed with max_lift>10mm":>27}{">20mm":>8}{">30mm":>8}{"median max_lift (placed)":>26}{"p90":>7}{"carry_mm<50% of dist (push-assisted, place_report)":>52}')
for n in NAMES:
    pl = [e for e in E[n] if e["place"]["placed"]]; ml = np.array([e["max_lift_mm"] for e in pl])
    d = json.load(open(EV / f"{n}_final.json"))["place_report"]["push_assisted"]
    print(f"{tag(n):<12}{len(pl):>8}{(ml>10).sum():>14} = {100*(ml>10).mean():5.1f}%{100*(ml>20).mean():7.1f}%{100*(ml>30).mean():7.1f}%{np.median(ml):>26.1f}{np.percentile(ml,90):>7.1f}{d['push_assisted']:>14}/{d['successful']} = {100*d['fraction']:.1f}%")
print("   lift_30mm-style flags are not used (known unreliable); only max_lift_mm thresholds above.")
print("\nF. OUTCOME BY WHETHER THE PIECE EVER LIFTED > 10 mm (association only; lift can be a consequence of what happens in the episode)")
print("   all 48 pairs:")
for n in NAMES:
    ml = np.array([e["max_lift_mm"] for e in E[n]]); pl = np.array([e["place"]["placed"] for e in E[n]]); hi = ml > 10
    print(f"   {tag(n):<12} episodes with max_lift>10mm {hi.sum():>4} ({100*hi.mean():4.1f}%)  placed|lift>10 {100*pl[hi].mean():5.1f}%  placed|lift<=10 {100*pl[~hi].mean():5.1f}%")
print("   neighbour forward/lateral pairs only (28 pairs, 1400 episodes):")
for n in NAMES:
    S = [e for e in E[n] if pair_cat(pk(e)) == "neigh_fwd/lat"]; ml = np.array([e["max_lift_mm"] for e in S]); pl = np.array([e["place"]["placed"] for e in S]); hi = ml > 10
    print(f"   {tag(n):<12} episodes with max_lift>10mm {hi.sum():>4} ({100*hi.mean():4.1f}%)  placed|lift>10 {100*pl[hi].mean():5.1f}%  placed|lift<=10 {100*pl[~hi].mean():5.1f}%")

print("\nG. DETERMINISTIC vs STOCHASTIC (UNITS DIFFER). deterministic: 96 episodes = 2 per pair; the two episodes of every pair were identical in all six seeds,")
print("   so its denominator is 48 PAIRS (a pair is placed or not). stochastic: 2400 EPISODES (50 per pair). Wilson 95% of each is over its own unit.")
for n in NAMES:
    d = json.load(open(EV / f"det48_{n}.json"))["episodes"]; kp = collections.Counter()
    for e in d: kp[pk(e)] += e["place"]["placed"]
    assert all(v in (0, 2) for v in kp.values()) and len(kp) == 48
    pairs_placed = sum(v == 2 for v in kp.values()); ks = sum(K[n].values())
    mean_pair = 100*np.mean([K[n][p]/50 for p in PAIRS])
    print(f"   {tag(n):<12} deterministic: {pairs_placed}/48 pairs placed = {100*pairs_placed/48:.1f}% {[round(x,1) for x in wilson(pairs_placed,48)]}"
          f"   stochastic: {ks}/2400 episodes = {100*ks/2400:.1f}% {[round(x,1) for x in wilson(ks,2400)]} (= mean over the 48 pairs of per-pair placement {mean_pair:.1f}%)")

print("\nH. ZERO-SHOT OVERLAP CHECK (from the pairs actually present in the committed zs_*.json files, against the 48 trained pairs)")
for n in NAMES:
    for f, lab in (("zs_near", "zero-shot (a) b/g pairs"), ("zs_rs29pairs", "zero-shot (b) RS29 pairs")):
        d = json.load(open(EV / f"{f}_{n}.json"))["episodes"]; ps = sorted({pk(e) for e in d}); ov = sorted(set(ps) & set(PAIRS))
        print(f"   {tag(n):<12} {lab}: {len(ps)} distinct pairs, {len(d)} episodes; overlap with the 48 trained pairs: {ov or 'none'}")
