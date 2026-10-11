#!/usr/bin/env python
"""POST HOC / EXPLORATORY addendum (2026-10-11). Read-only: committed eval JSONs, Monitor CSVs, configs, manifests, PREREGISTRATION.md.
 X1 piece height in the scene and what `lift` measures; X2 lift among PLACED episodes per seed (median, p90, ...); X3 pair e4:e5 only;
 X4 long pairs under 25/50 per seed; X5 was training placement still rising at the end; X6 pair-order check against PREREGISTRATION.md."""
import re, subprocess
import yaml
import mujoco
from diag_common import *
tag = lambda n: n.replace("rs32D_", "")
E = {n: eval_eps(n) for n in NAMES}

print("X1. PIECE GEOMETRY (measured from assets/scene.xml with MuJoCo: world z extent of piece_a's collision geoms above the piece origin, at the home keyframe)")
m = mujoco.MjModel.from_xml_path(str(REPO / "assets/scene.xml")); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "piece_a"); z0 = d.xpos[bid][2]; lo = []; hi = []
for g in range(m.ngeom):
    if m.geom_bodyid[g] == bid and m.geom_contype[g] > 0:
        R = d.geom_xmat[g].reshape(3, 3); ab = m.geom_aabb[g]
        cor = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * ab[3:] + ab[:3]; z = ((R @ cor.T).T + d.geom_xpos[g])[:, 2] - z0
        lo.append(z.min()); hi.append(z.max()); print(f"   {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g):<14} z from {1e3*z.min():5.1f} to {1e3*z.max():5.1f} mm above the piece origin")
PIECE_H = 1e3 * (max(hi) - min(lo)); print(f"   piece height {PIECE_H:.1f} mm (origin at the base); base diameter 30 mm; board square pitch 78 mm")
bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
CFG = lambda n: REPO / "configs" / ("_probe_cart_" + n.replace("_seed", "_ppotanh_seed") + ".yaml")
nobs = sorted({yaml.safe_load(open(CFG(n)))["n_obstacles"] for n in NAMES})
print(f"   bodies named piece*: {[b for b in bodies if b.startswith('piece')]} (the scene holds ONE piece); n_obstacles in the six configs: {nobs}")
print("   `lift` (eval_probe max_lift_mm) = piece ORIGIN z minus its resting z at the home keyframe (pickup_geometry in t1_pick_and_place.py). It is the rise of the piece base.")
print(f"   A piece of the same height standing on a neighbouring square would have its top {PIECE_H:.0f} mm above the board, so a carried piece clears it only if its BASE is lifted by more than {PIECE_H:.0f} mm")
print("   (more with margin). This task has no neighbouring piece (one piece, n_obstacles 0); the comparison is for scale only.")

def lifts(n, pairs=None, placed=True):
    return np.array([e["max_lift_mm"] for e in E[n] if (not placed or e["place"]["placed"]) and (pairs is None or pk(e) in pairs)])
print("\nX2. MAXIMUM PIECE LIFT (mm) PER EPISODE, PLACED EPISODES ONLY, full 2400-episode eval")
print(f'{"seed":<12}{"placed":>7}{"median":>8}{"p90":>7}{"p99":>7}{"max":>7}{">10mm":>14}{">30mm":>8}{f">{PIECE_H:.0f}mm":>8}')
for n in NAMES:
    x = lifts(n); print(f"{tag(n):<12}{len(x):>7}{np.median(x):>8.1f}{np.percentile(x,90):>7.1f}{np.percentile(x,99):>7.1f}{x.max():>7.1f}{(x>10).sum():>7} ={100*(x>10).mean():4.1f}%{(x>30).sum():>8}{(x>PIECE_H).sum():>8}")
print("   same, FAILED episodes (for scale):")
for n in NAMES:
    x = np.array([e["max_lift_mm"] for e in E[n] if not e["place"]["placed"]]); print(f"   {tag(n):<12}n={len(x):<5} median {np.median(x):5.1f}  p90 {np.percentile(x,90):5.1f}  max {x.max():5.1f}  >{PIECE_H:.0f}mm: {(x>PIECE_H).sum()}")

print("\nX3. PAIR e4:e5 ONLY (50 episodes per seed in the full eval): placed episodes whose max piece lift exceeded 10 mm")
print(f'{"seed":<12}{"placed":>8}{"lift>10mm":>11}{"share":>8}{"median max lift (placed)":>26}{"p90":>7}{"max":>7}')
for n in NAMES:
    x = lifts(n, {"e4:e5"}); k = len(x)
    print(f"{tag(n):<12}{k:>5}/50{(x>10).sum():>11}{(f'{100*(x>10).mean():.0f}%' if k else 'n/a'):>8}{np.median(x):>26.1f}{np.percentile(x,90):>7.1f}{x.max():>7.1f}")

K = {n: collections.Counter() for n in NAMES}
for n in NAMES:
    for e in E[n]: K[n][pk(e)] += e["place"]["placed"]
print("\nX4. LONG PAIRS (18) BELOW 25/50 AND BELOW 10/50, per seed (placed k/50 in the full eval)")
print(f'{"seed":<12}{"long <10/50":>13}{"long <25/50":>13}   long pairs below 25/50 (k/50)')
for n in NAMES:
    print(f"{tag(n):<12}{sum(K[n][p]<10 for p in LONG):>13}{sum(K[n][p]<25 for p in LONG):>13}   " + ", ".join(f"{p} {K[n][p]}" for p in LONG if K[n][p] < 25))

print("\nX5. WAS TRAINING PLACEMENT STILL RISING AT THE END OF TRAINING?  (Monitor terminated_reason == success, pooled over the 4 files)")
print("    100k bins as in ../section3_train_output.txt (25k steps of each file; bin j = training steps (j-1)*100k+1..j*100k); the partial bin after 5.0M is excluded.")
print("    slope = weighted least-squares slope of the bin placement rates against bin index, in percentage points per 100k steps, with a nominal binomial SE")
print("    (episodes of one evolving policy are not independent, so the SE is only nominal). 'rising' = slope - 1.96 SE > 0; 'falling' = slope + 1.96 SE < 0; else 'no detectable change'.")
print("    diff = placement in 4.5-5.0M minus 4.0-4.5M (episode counts pooled), points, with an approximate 95% interval (unpooled normal).")
def bins(n):
    K_ = {}; N_ = {}
    for m_ in monitor(RUN(n)):
        cum = np.cumsum(m_["l"]); b = (cum - 1) // 25000
        for j in np.unique(b):
            if j < 50: K_[j] = K_.get(j, 0) + int(m_["placed"][b == j].sum()); N_[j] = N_.get(j, 0) + int((b == j).sum())
    return K_, N_
def wls(K_, N_, js):
    x = np.array(js, float); nn = np.array([N_[j] for j in js], float); y = np.array([K_[j]/N_[j] for j in js])
    xw = (nn*x).sum()/nn.sum(); yw = (nn*y).sum()/nn.sum(); sxx = (nn*(x-xw)**2).sum(); b = (nn*(x-xw)*(y-yw)).sum()/sxx
    var = ((nn**2)*((x-xw)**2)*(y*(1-y)/nn)).sum()/sxx**2; return 100*b, 100*np.sqrt(var)
print(f'{"seed":<12}{"last10 bins (4.1-5.0M) %":>74}')
rows = []
for n in NAMES:
    K_, N_ = bins(n); last = list(range(40, 50))
    pct = " ".join(f"{100*K_[j]/N_[j]:5.1f}" for j in last)
    b10, s10 = wls(K_, N_, last); b5, s5 = wls(K_, N_, last[5:])
    k1 = sum(K_[j] for j in range(40, 45)); n1 = sum(N_[j] for j in range(40, 45)); k2 = sum(K_[j] for j in range(45, 50)); n2 = sum(N_[j] for j in range(45, 50))
    p1, p2 = k1/n1, k2/n2; se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2); dif = 100*(p2-p1); lo_, hi_ = dif - 196*se, dif + 196*se
    verdict = lambda b, s: "rising" if b - 1.96*s > 0 else ("falling" if b + 1.96*s < 0 else "no detectable change")
    rows.append((n, pct, b10, s10, b5, s5, dif, lo_, hi_, p1, p2, verdict(b10, s10), verdict(b5, s5), "rising" if lo_ > 0 else ("falling" if hi_ < 0 else "no detectable change")))
for n, pct, *_ in rows: print(f"{tag(n):<12}{pct:>74}")
print(f'\n{"seed":<12}{"4.0-4.5M":>9}{"4.5-5.0M":>10}{"diff pts [95%]":>22}{"slope last10 (pts/100k)":>26}{"slope last5":>22}   verdicts (diff / last10 slope / last5 slope)')
for n, pct, b10, s10, b5, s5, dif, lo_, hi_, p1, p2, v10, v5, vd in rows:
    print(f"{tag(n):<12}{100*p1:>8.1f}%{100*p2:>9.1f}%{dif:>+9.1f} [{lo_:+.1f},{hi_:+.1f}]{b10:>+14.2f} +/- {1.96*s10:4.2f}{b5:>+12.2f} +/- {1.96*s5:4.2f}   {vd} / {v10} / {v5}")
print("    Learning rate at 4.0M, 4.5M, 5.0M steps (linear decay 3e-4 to 0 over 5,000,000 steps): " + ", ".join(f"{3e-4*(1-t/5e6):.1e}" for t in (4e6, 4.5e6, 5e6)))

print("\nX6. PAIR ORDER CHECK against PREREGISTRATION.md (every file that carries the 48 pairs; order and content)")
line = next(l for l in PREREG.read_text().splitlines() if l.startswith("Training pairs"))
ref = [t for t in line.split("in this order):", 1)[1].split() if re.fullmatch(r"[a-h][1-8]:[a-h][1-8]", t)]   # independent parse: whitespace tokens
assert len(ref) == 48 and len(set(ref)) == 48
chk = lambda lab, got: print(f"   {'MATCH   ' if list(got) == ref else 'MISMATCH'} {lab}" + ("" if list(got) == ref else f"   (len {len(list(got))}; set equal: {set(got) == set(ref)})"))
chk("PREREGISTRATION.md 'Training pairs' line, parsed here as whitespace tokens vs rs32_lib.PAIRS (regex parse)", PAIRS)
man = json.load(open(EV / "EVAL_MANIFEST.json")); chk("../EVAL_MANIFEST.json  pairs", man["pairs"])
for e in man["evals"]: chk(f"../EVAL_MANIFEST.json  --pairs in the command of {e['label']}", e["cmd"].split("--pairs ")[1].split(" ")[0].split(","))
m3 = json.load(open(EV / "EVAL_MANIFEST_section3.json")); chk("../EVAL_MANIFEST_section3.json  pairs48", m3["pairs48"])
for e in m3["evals"]:
    if e["label"].startswith("det48"): chk(f"../EVAL_MANIFEST_section3.json  --pairs in the command of {e['label']}", e["cmd"].split("--pairs ")[1].split(" ")[0].split(","))
for n in NAMES:
    ok = all(pk(e) == ref[i % 48] for i, e in enumerate(E[n])); print(f"   {'MATCH   ' if ok else 'MISMATCH'} ../{n}_final.json  episode i is pair ref[i mod 48] for all {len(E[n])} episodes")
    dd = json.load(open(EV / f"det48_{n}.json"))["episodes"]; ok = all(pk(e) == ref[i % 48] for i, e in enumerate(dd)); print(f"   {'MATCH   ' if ok else 'MISMATCH'} ../det48_{n}.json  episode i is pair ref[i mod 48] for all {len(dd)} episodes")
for n in NAMES:
    cfg = yaml.safe_load(open(CFG(n))); chk("configs/" + CFG(n).name + "  train_pairs", cfg["train_pairs"])
    mf = json.load(open(REPO / "runs" / RUN(n) / "manifest.json"))["config_resolved"]; chk(f"runs/{RUN(n)}/manifest.json  config_resolved.train_pairs (what the run used)", mf["train_pairs"])
rows_ = [l for l in (REPO / "results/rs32_48_pairs/expert_gate/TABLE.md").read_text().splitlines() if re.match(r"\| [a-h][1-8]:[a-h][1-8] \|", l)]
gate = [l.split("|")[1].strip() for l in rows_]; chk("results/rs32_48_pairs/expert_gate/TABLE.md  first 48 rows", gate[:48]); print(f"        (that table has {len(gate)} rows: 48 pairs + {len(gate)-48} reserves; reserves not in the 48: {sorted(set(gate[48:]) - set(ref)) == sorted(gate[48:])})")
seen = collections.Counter()
for n in NAMES:
    for f in sorted((REPO/"runs"/RUN(n)/"logs").glob("monitor_*.monitor.csv")):
        for r in csv.DictReader(l for l in open(f) if not l.startswith("#")): seen[f'{r["start_square"]}:{r["target_square"]}'] += 1
print(f"   {'MATCH   ' if set(seen) == set(ref) else 'MISMATCH'} the (start, target) pairs that occur in the six runs' Monitor CSVs are exactly the 48 pairs ({len(seen)} distinct; episodes per pair min {min(seen.values())} max {max(seen.values())}, all six runs pooled)")
print("   Note: training draws one pair uniformly at random per episode, so the ORDER of train_pairs does not affect training; order matters for the eval (episode i uses pair i mod 48) and for the tables.")
