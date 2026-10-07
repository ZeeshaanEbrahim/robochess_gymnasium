#!/usr/bin/env python
"""RS32 Step 3 items a, e, f, g. Reads eval/*.json, *_rms.json, EVAL_MANIFEST*.json and the Monitor CSVs only (no rollouts).
Wilson 95%, z=1.96, no continuity correction. Reported, not gated; no interpretation."""
import collections, json
from rs32_lib import *
import run_section3 as S
man = json.load(open(EV / "EVAL_MANIFEST.json")); man3 = json.load(open(EV / "EVAL_MANIFEST_section3.json"))
print("Section-3 manifest exits:", {e["label"]: e["exit"] for e in man3["evals"]})

def load(f, n_exp, pairs, det):
    d = json.load(open(EV / f)); rms = json.load(open(EV / (f[:-5] + "_rms.json")))
    assert d["n"] == n_exp and d["seed"] == 20000 and d["deterministic"] is det and d["obs_rms_identical_before_after"] is True and rms["identical_before_after"] is True
    assert [f'{e["start_square"]}:{e["target_square"]}' for e in d["episodes"]] == [pairs[i % len(pairs)] for i in range(n_exp)]
    k, n = collections.Counter(), collections.Counter()
    for e in d["episodes"]: p = f'{e["start_square"]}:{e["target_square"]}'; n[p] += 1; k[p] += e["place"]["placed"]
    assert sum(k.values()) == d["place_report"]["rates"]["placed"]["k"]
    E = d["episodes"]
    fun = dict(lift10=sum(e["grasp_ever"] for e in E), held=sum(e["place"]["held"] for e in E), released=sum(e["place"]["released"] for e in E), placed=sum(k.values()))
    return d, k, n, fun, collections.Counter(e["reason"] for e in E)
funnel = lambda f, n: f"lift over 10 mm (undercounts) {f['lift10']}/{n}  held {f['held']}/{n}  released {f['released']}/{n}  placed {f['placed']}/{n}"

main = {}
for name in NAMES:
    d = json.load(open(EV / f"{name}_final.json")); k = collections.Counter()
    for e in d["episodes"]: k[f'{e["start_square"]}:{e["target_square"]}'] += e["place"]["placed"]
    main[name] = (d, k)

print("\nITEM a. success on three subsets (stochastic, final model, the 2400 Step-2 episodes; 50 per pair)")
for name in NAMES:
    k = main[name][1]; tot = sum(k.values())
    print(f"{name}: all 48 {fmt(tot, 2400)}")
    for lab, ps in (("30 neighbour pairs", NEIGH), ("18 long pairs (2 or 3 squares)", LONG), ("24 RS31 pairs (all neighbour, subset of the 30)", RS31_PAIRS)):
        print(f"   {lab:<48}{fmt(sum(k[p] for p in ps), 50*len(ps))}")
    nb = [p for p in NEIGH if p not in RS31_PAIRS]
    print(f"   {'6 neighbour pairs not in RS31 (extra)':<48}{fmt(sum(k[p] for p in nb), 50*len(nb))}")

print("\nITEM e(a). zero-shot: 12 pairs on files b and g, 300 episodes, 25 per pair, stochastic, seed 20000")
print("   overlap with the 48 trained pairs: " + (str(sorted(set(S.NEAR12) & set(PAIRS))) or "none") + " (asserted empty; the 12 pairs: " + ",".join(S.NEAR12) + ")")
for name in NAMES:
    d, k, n, fun, rs = load(f"zs_near_{name}.json", 300, S.NEAR12, False)
    print(f"{name}: pooled {fmt(sum(k.values()), 300)}   rms identical before/after: yes")
    print("   funnel: " + funnel(fun, 300) + f"   reasons {dict(sorted(rs.items()))}")
    print("   per pair k/25: " + "  ".join(f"{p} {k[p]}" for p in S.NEAR12))

TRAIN = set(PAIRS); UNTR = [p for p in S.RS29 if p not in TRAIN]; assert len(UNTR) == 19 and [p for p in S.RS29 if p in TRAIN] == ["e4:e5"]
BANDS = dict(short="e4:e5 c1:c2 d2:e2 a6:a5 e8:d7 d4:e5".split(), mid="f3:c6 h5:d3 g5:b3 c1:f6".split(),
             long="h1:h8 h8:h1 a4:h4 b7:g2 b2:g7 f1:c8".split(), diagonal="a1:h8 h8:a1 h1:a8 a8:h1".split())
print("\nITEM e(b). zero-shot: the 20 RS29 pairs, 300 episodes, 15 per pair, stochastic, seed 20000")
print("   " + ",".join(S.RS29))
print("   overlap with the 48 trained pairs: ['e4:e5'] (asserted; e4:e5 is a training pair) -> excluded from the headline; 19 untrained pairs, n = 285")
lab = lambda p: p + (" (TRAINING pair)" if p in TRAIN else "")
for name in NAMES:
    d, k, n, fun, rs = load(f"zs_rs29pairs_{name}.json", 300, S.RS29, False)
    assert len(n) == 20 and set(n.values()) == {15}
    k19 = sum(k[p] for p in UNTR); nz = {lab(p): k[p] for p in S.RS29 if k[p]}
    print(f"{name}: *** 19 UNTRAINED pairs, n = 285: {fmt(k19, 285)} ***   (all 20 incl. e4:e5: {fmt(sum(k.values()), 300)}; e4:e5 alone {k['e4:e5']}/15)")
    print(f"   pairs with >=1 placement: {len(nz)}/20 {nz}   rms identical before/after: yes")
    print("   funnel (all 20 pairs; includes e4:e5): " + funnel(fun, 300) + f"   reasons {dict(sorted(rs.items()))}")
    print("   bands, 19 untrained: " + "; ".join(f"{b} {fmt(sum(k[p] for p in ps if p not in TRAIN), 15*len([p for p in ps if p not in TRAIN]))}" for b, ps in BANDS.items()))
    print("   per pair k/15: " + " ".join(f"{lab(p)} {k[p]}" for p in S.RS29))

print("\nITEM f. deterministic vs stochastic on the 48 pairs: 96 deterministic episodes = 2 per pair, seed 20000 (reporting choice carried over from RS31)")
for name in NAMES:
    d, k, n, fun, rs = load(f"det48_{name}.json", 96, PAIRS, True)
    assert all(v == 2 for v in n.values()) and len(n) == 48
    ks = sum(main[name][1].values()); cnt = collections.Counter(k[p] for p in PAIRS); ones = [p for p in PAIRS if k[p] == 1]
    E = d["episodes"]; T = d["traj"]; same_traj = same_rec = 0
    for i in range(48):
        assert E[i]["start_square"] == E[i + 48]["start_square"] and E[i]["target_square"] == E[i + 48]["target_square"]
        same_traj += T[i] == T[i + 48]; same_rec += E[i] == E[i + 48]
    print(f"{name}: deterministic {fmt(sum(k.values()), 96)} [96-episode interval assumes independent episodes; NOT valid if the two episodes of a pair are identical]   stochastic (Step 2, 2400 ep) {fmt(ks, 2400)}")
    print(f"   per-pair placed count (0, 1 or 2): pairs with 0: {cnt[0]}, with 1: {cnt[1]}, with 2: {cnt[2]}")
    print(f"   two episodes of a pair: identical per-step trajectory records in {same_traj}/48 pairs; identical whole episode records in {same_rec}/48 pairs")
    if not ones and same_traj == 48:
        kp = sum(k[p] == 2 for p in PAIRS)
        print(f"   all 48 pairs give identical results across their 2 episodes -> effective n = 48 pairs: {fmt(kp, 48)}  (pairs placed / 48 pairs)")
    else:
        print(f"   NOT all pairs identical; pairs with exactly 1 placed: {ones}; effective n not reduced to 48 by this check")
    print("   deterministic funnel: " + funnel(fun, 96) + f"   reasons {dict(sorted(rs.items()))}")
    print("   per pair placed (0/1/2): " + " ".join(f"{p} {k[p]}" for p in PAIRS))

print("\nITEM g. eval vs training cross-check (Wilson 95%, z=1.96, no continuity correction); gap = eval - training, percentage points; overlap computed from the printed intervals")
print("(1) PRIMARY: eval pooled placement (Step 2, k/2400) vs training placement in the LAST 10% chunk of training (Monitor terminated_reason == success, chunk 9, pooled over the 4 files)")
print("(2) supplementary (RS31 convention): eval placement vs the last 100k training steps (last 25k steps of each file)")
print("(3) supplementary (RS31 convention): eval held (place.held /2400) vs ever_held in the last 10% chunk")
for name in NAMES:
    d = main[name][0]; E = d["episodes"]; ep = sum(e["place"]["placed"] for e in E); eh = sum(e["place"]["held"] for e in E)
    mon = monitor(RUN(name)); k1 = n1 = h1 = k2 = n2 = 0
    for m in mon:
        ch = chunk_counts(m); k1 += int(m["placed"][ch == 9].sum()); n1 += int((ch == 9).sum()); h1 += int(m["held"][ch == 9].sum())
        cum = np.cumsum(m["l"]); w = cum > cum[-1] - 25000; k2 += int(m["placed"][w].sum()); n2 += int(w.sum())
    ev = wilson(ep, 2400)
    print(name)
    for tag, (tk, tn), ek in (("(1) placed, last 10% chunk", (k1, n1), ep), ("(2) placed, last 100k steps", (k2, n2), ep), ("(3) held,   last 10% chunk", (h1, n1), eh)):
        e_ = wilson(ek, 2400); t_ = wilson(tk, tn); gap = 100 * ek / 2400 - 100 * tk / tn
        print(f"  {tag}: eval {fmt(ek,2400)}   training {fmt(tk,tn)}   gap {gap:+.1f} pts; Wilson intervals {'OVERLAP' if overlap(e_, t_) else 'do NOT overlap'} (eval [{e_[0]:.2f},{e_[1]:.2f}] vs training [{t_[0]:.2f},{t_[1]:.2f}])")
