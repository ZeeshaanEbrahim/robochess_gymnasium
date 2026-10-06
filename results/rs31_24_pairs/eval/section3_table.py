#!/usr/bin/env python
"""RS31 Section 3 items 1-3 and addition C. Reads eval/*.json, *_rms.json and EVAL_MANIFEST_section3.json only.
Funnel columns: grasp_ever (labelled "lift over 10 mm (undercounts)"), held, released, placed. Wilson 95%, z=1.96, no continuity correction."""
import collections, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_section3 as S
EV = Path(__file__).resolve().parent
NAMES = [f"rs31{a}_seed{i}" for a in "CD" for i in range(2)]
BANDS = dict(short="e4:e5 c1:c2 d2:e2 a6:a5 e8:d7 d4:e5".split(), mid="f3:c6 h5:d3 g5:b3 c1:f6".split(),
             long="h1:h8 h8:h1 a4:h4 b7:g2 b2:g7 f1:c8".split(), diagonal="a1:h8 h8:a1 h1:a8 a8:h1".split())
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h))
fmt = lambda k, n: f"{k}/{n} = {100*k/n:.1f}% [{wilson(k,n)[0]:.1f},{wilson(k,n)[1]:.1f}]"
man = json.load(open(EV / "EVAL_MANIFEST_section3.json"))
def load(f, n_exp, pairs, det):
    d = json.load(open(EV / f)); rms = json.load(open(EV / (f[:-5] + "_rms.json")))
    assert d["n"] == n_exp and d["seed"] == 20000 and d["deterministic"] is det and d["obs_rms_identical_before_after"] is True and rms["identical_before_after"] is True
    assert [f'{e["start_square"]}:{e["target_square"]}' for e in d["episodes"]] == [pairs[i % len(pairs)] for i in range(n_exp)]
    k, n = collections.Counter(), collections.Counter()
    for e in d["episodes"]: p = f'{e["start_square"]}:{e["target_square"]}'; n[p] += 1; k[p] += e["place"]["placed"]
    assert sum(k.values()) == d["place_report"]["rates"]["placed"]["k"]
    E = d["episodes"]
    fun = dict(grasp=sum(e["grasp_ever"] for e in E), held=sum(e["place"]["held"] for e in E), released=sum(e["place"]["released"] for e in E), placed=sum(k.values()))
    return d, k, n, fun, collections.Counter(e["reason"] for e in E)
def funnel(f, n): return f"lift over 10 mm (undercounts) {f['grasp']}/{n}  held {f['held']}/{n}  released {f['released']}/{n}  placed {f['placed']}/{n}"

print("Manifest exits:", {e["label"]: e["exit"] for e in man["evals"]})
print("Commands:"); [print(f"  {e['label']}: {e['cmd']}") for e in man["evals"]]
print("\nITEM 1. zero-shot (a): 12 pairs on files b and g, 300 episodes, 25 per pair, stochastic, seed 20000 (pair order as in PREREGISTRATION.md)")
for nm in NAMES:
    d, k, n, fun, rs = load(f"zs_near_{nm}.json", 300, S.NEAR12, False)
    print(f"{nm}: pooled {fmt(sum(k.values()), 300)}   rms identical before/after: yes")
    print("   funnel: " + funnel(fun, 300) + f"   reasons {dict(sorted(rs.items()))}")
    print("   " + "  ".join(f"{p} {k[p]}/25 [{wilson(k[p],25)[0]:.0f},{wilson(k[p],25)[1]:.0f}]" for p in S.NEAR12))
print("\nITEM 2. zero-shot (b): the 20 RS29 pairs (list from results/rs29_random_squares/PREREGISTRATION.md, asserted equal to the RS30 tooling's list):")
print("   " + ",".join(S.RS29))
TRAIN24 = set(S.PAIRS24); TRAINED_IN_RS29 = [p for p in S.RS29 if p in TRAIN24]
assert TRAINED_IN_RS29 == ["e4:e5"], TRAINED_IN_RS29
UNTRAINED19 = [p for p in S.RS29 if p not in TRAIN24]; assert len(UNTRAINED19) == 19
print("   e4:e5 is one of the 24 RS31 TRAINING pairs (the only one in this list: intersection asserted) -> labelled 'training pair' below; 19 untrained pairs remain")
print("   300 episodes, 15 per pair, stochastic, seed 20000")
lab = lambda p: p + (" (training pair)" if p in TRAIN24 else "")
BANDS19 = {b: [p for p in ps if p in UNTRAINED19] for b, ps in BANDS.items()}
for nm in NAMES:
    d, k, n, fun, rs = load(f"zs_rs29pairs_{nm}.json", 300, S.RS29, False)
    assert len(n) == 20 and set(n.values()) == {15}
    nz = {lab(p): k[p] for p in S.RS29 if k[p]}
    k19 = sum(k[p] for p in UNTRAINED19)
    print(f"{nm}: ALL 20 pairs pooled {fmt(sum(k.values()), 300)}; pairs with >=1 placement {len(nz)}/20 {nz}   rms identical before/after: yes")
    print(f"   *** 19 UNTRAINED pairs (e4:e5 excluded), n = 285: {fmt(k19, 285)} ***")
    print("   funnel (all 20 pairs): " + funnel(fun, 300) + f"   reasons {dict(sorted(rs.items()))}")
    print("   bands, 19 untrained pairs: " + "; ".join(f"{b} {fmt(sum(k[p] for p in ps), 15*len(ps))}" for b, ps in BANDS19.items()))
    print("   bands, all 20 pairs: " + "; ".join(f"{b} {fmt(sum(k[p] for p in ps), 15*len(ps))}" for b, ps in BANDS.items()))
    print("   per pair k/15: " + " ".join(f"{lab(p)} {k[p]}" for p in S.RS29))
print("\nITEM 3. deterministic vs stochastic on the 24 pairs. REPORTING CHOICE (Zeeshaan's, not in PREREGISTRATION.md): 48 deterministic episodes = 2 per pair, seed 20000")
for nm in NAMES:
    d, k, n, fun, rs = load(f"det24_{nm}.json", 48, S.PAIRS24, True)
    s_ = json.load(open(EV / f"{nm}_final.json")); ks = sum(e["place"]["placed"] for e in s_["episodes"])
    assert all(v == 2 for v in n.values()) and len(n) == 24
    cnt = collections.Counter(k[p] for p in S.PAIRS24)          # how many pairs have 0, 1 or 2 placed episodes
    ones = [p for p in S.PAIRS24 if k[p] == 1]
    E = d["episodes"]; T = d["traj"]; same_traj = same_rec = 0
    for i in range(24):                                          # episode i and i+24 are the two episodes of pair i (cycled in order)
        assert E[i]["start_square"] == E[i + 24]["start_square"] and E[i]["target_square"] == E[i + 24]["target_square"]
        same_traj += T[i] == T[i + 24]; same_rec += E[i] == E[i + 24]
    print(f"{nm}: deterministic {fmt(sum(k.values()), 48)} [48-episode interval assumes independent episodes, NOT valid when the two episodes of a pair are identical]   stochastic (Section 2, 1200 ep) {fmt(ks, 1200)}")
    print(f"   per-pair placed count (0, 1 or 2): pairs with 0: {cnt[0]}, with 1: {cnt[1]}, with 2: {cnt[2]}")
    if not ones:
        kp = sum(k[p] == 2 for p in S.PAIRS24)
        print(f"   every pair is 0 or 2 -> effective-n = 24 row: {fmt(kp, 24)}  (pairs with 2 placed / 24 pairs)")
    else:
        print(f"   pairs with exactly 1 placed: {ones}")
    print(f"   the two episodes of a pair: identical per-step trajectory records (rel, ap, l, r, board, d, tilt, lift each step) in {same_traj}/24 pairs; identical whole episode records (all summary fields) in {same_rec}/24 pairs")
    print("   deterministic funnel: " + funnel(fun, 48) + f"   reasons {dict(sorted(rs.items()))}")
    print("   per pair placed (0/1/2): " + " ".join(f"{p} {k[p]}" for p in S.PAIRS24))
print("\nADDITION C. wrapper equivalence")
print("  The 298/300 positive control (D29 squash seed 3, e4:e5) is the RS30 control: results/rs30_four_pairs/eval/control_d29_seed3_e4e5.json on branch rs30.")
print("  Its command (rs30 EVAL_MANIFEST.json): /root/venvs/robochess/bin/python scripts/eval_probe.py _probe_cart_place29_ppotanh_seed3 --episodes 300 --pairs e4:e5 --seed 20000 --out results/rs30_four_pairs/eval/control_d29_seed3_e4e5.json")
print("  = PLAIN eval_probe.py, not rms_wrapper.py (rms_wrapper.py did not exist then). No control was run in RS31. Therefore: wrapper vs plain compared directly below.")
a = json.load(open(EV / "rs31D_seed0_final.json")); b = json.load(open(EV / "equiv_plain_rs31D_seed0.json"))
print(f"  model rs31D_seed0, 1200 episodes, wrapper JSON (Section 2) vs plain eval_probe.py JSON (equiv_plain_rs31D_seed0.json)")
print(f"  top-level keys equal: {set(a)==set(b)}; per-key equal: " + str({k: a[k] == b[k] for k in a}))
print(f"  episodes compared: {len(a['episodes'])} vs {len(b['episodes'])}; identical per-episode records: {sum(x == y for x, y in zip(a['episodes'], b['episodes']))}/{len(a['episodes'])}")
print(f"  whole JSON identical: {a == b}")
