#!/usr/bin/env python
"""Aggregate the JSONs written by scripts/eval_probe.py into the probe report.

Prints, per run and model: D18 `grasp_success` ever-in-episode, D21
`pickup_success`, any two-pad contact (each with a 95% Wilson interval), closest
pinch-to-g distance (g = piece + 0.03725 m in z), tool/piece tilt at closest
approach, `terminated_reason` counts, `ik_rejects`; the deterministic-identity
check; the D21 lifted-steps behaviour; and ppo_seed2's first five trajectories.
Expects <dir>/<run>_{ck100000,ck200000,ck500000,final,det}.json for
run in ppo_seed0, ppo_seed1, ppo_seed2, sac_seed0 (default dir:
results/cart_pickup_probes).

    python scripts/eval_probe_agg.py [dir]
"""
import json, math, collections, statistics as st, os, sys
from pathlib import Path
import numpy as np
E=sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "results" / "cart_pickup_probes")
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
def f(k,n): lo,hi=wil(k,n); return f"{k}/{n}={100*k/n:.1f}% [{100*lo:.1f},{100*hi:.1f}]"
runs=["ppo_seed0","ppo_seed1","ppo_seed2","sac_seed0"]
print("SECTION 2: stochastic eval, seed 20000")
for a in runs:
    for tag in ("ck100000","ck200000","ck500000","final"):
        fn=f"{E}/{a}_{tag}.json"
        if not os.path.exists(fn): print(a,tag,"MISSING"); continue
        d=json.load(open(fn)); ep=d["episodes"]; n=len(ep)
        md=np.array([e["min_d_mm"] for e in ep]); tl=[e["tilt_at_min_deg"] for e in ep]
        print(f"{a:10s} {tag:9s} n={n} rms_identical={d['obs_rms_identical_before_after']}")
        print(f"   grasp_ever {f(sum(e['grasp_ever'] for e in ep),n)} | pickup {f(sum(e['pickup'] for e in ep),n)} | two-pad(any) {f(sum(e['twopad_ever'] for e in ep),n)} | pinched(any) {f(sum(e['pinched_ever'] for e in ep),n)}")
        print(f"   closest pinch-to-g mm: median {np.median(md):.2f} p10 {np.percentile(md,10):.2f} | median tool tilt at closest {np.median(tl):.2f} deg, piece tilt {np.median([e['ptilt_at_min_deg'] for e in ep]):.2f} deg")
        print(f"   reasons {dict(collections.Counter(e['reason'] for e in ep))} | ik_rejects total {sum(e['ik'] for e in ep)} | gsc steps total {sum(e['gsc_steps'] for e in ep)} in {sum(e['gsc_steps']>0 for e in ep)} eps")
print("\nDETERMINISTIC 20 eps at final")
for a in runs:
    d=json.load(open(f"{E}/{a}_det.json")); tr=d["traj"]
    same=all(tr[i]==tr[0] for i in range(1,len(tr)))
    print(a,"identical trajectories across 20 episodes:",same,"| pickup",sum(e['pickup'] for e in d['episodes']),"/20 grasp",sum(e['grasp_ever'] for e in d['episodes']),"| distinct n-steps",sorted(set(e['n'] for e in d['episodes'])),"| rms_identical",d['obs_rms_identical_before_after'])
print("\n(c) D21 behaviour at final_model (stochastic 300)")
for a in runs:
    ep=json.load(open(f"{E}/{a}_final.json"))["episodes"]; s=[e for e in ep if e["pickup"]]
    print(f"{a}: pickup eps {len(s)}/{len(ep)}")
    if s:
        med=lambda k,x=s: float(np.median([e[k] for e in x]))
        print(f"   all-episode median lifted(>=30mm) steps {np.median([e['lifted_steps'] for e in ep]):.1f}; among pickup eps: median lifted steps {med('lifted_steps'):.1f}, median first pickup step {med('pickup_first_step'):.1f}, median steps after first {med('steps_after_first'):.1f}, median lifted steps after first {med('lifted_after_first'):.1f}")
        fr=[e['lifted_after_first']/e['steps_after_first'] for e in s]
        print(f"   fraction of post-success steps lifted: median {np.median(fr):.3f} min {min(fr):.3f}; eps lifted at final step {sum(e['lifted_at_end'] for e in s)}/{len(s)}, grasp_success at final step {sum(e['grasp_at_end'] for e in s)}/{len(s)}; eps with every post-success step lifted {sum(e['lifted_after_first']==e['steps_after_first'] for e in s)}/{len(s)}; ended by: {dict(collections.Counter(e['reason'] for e in s))}")
print("\n(a) ppo_seed2 final: first 5 stochastic episodes; rel = pinch - g (mm x,y,z), aperture, pads L/R/board")
tr=json.load(open(f"{E}/ppo_seed2_final.json"))["traj"]; eps=json.load(open(f"{E}/ppo_seed2_final.json"))["episodes"]
for i in range(5):
    print(f" ep{i}: n={eps[i]['n']} reason={eps[i]['reason']} grasp_ever={eps[i]['grasp_ever']} min_d={eps[i]['min_d_mm']:.1f}mm at step {eps[i]['min_d_step']}")
    for k in (25,50,100,200,300):
        if k<=len(tr[i]):
            x=tr[i][k-1]; print(f"    step {k:3d}: rel {x['rel']} |d|={x['d']*1e3:.1f}mm ap={x['ap']:.3f} pads L={int(x['l'])} R={int(x['r'])} board={int(x['board'])} tilt={x['tilt']:.1f}")
        else: print(f"    step {k:3d}: (episode ended at {len(tr[i])})")

# D28 place-mode runs: printed only when the directory holds eval_probe.py outputs that carry a
# `place_report`, so the pick-up report above is unchanged for the existing result directories.
_place = sorted(p for p in Path(E).glob("*.json")
                if p.name.endswith((".json",)) and "place_report" in json.loads(p.read_text()))
if _place:
    print("\nPLACE MODE: grasp -> lifted 30 mm -> carried within 3 cm -> released -> placed (95% Wilson)")
    for pth in _place:
        d = json.loads(pth.read_text()); r = d["place_report"]
        lab = "deterministic, EFFECTIVE n = %d of %d episodes" % (r["effective_n"], r["n"]) if r["deterministic"] else f"stochastic n={r['n']}"
        print(f"{d['run']} {pth.stem}: {lab}")
        for st_, v in r["rates"].items():
            print(f"   {st_:20s} {v['k']}/{v['n']}  Wilson[{100*v['wilson_lo']:.1f},{100*v['wilson_hi']:.1f}] at n={v['effective_n']}")
        t = r["truncated_in_s2"]
        print(f"   truncated in S2 {t['k']}/{t['n']}  final gripper qpos {t['final_gripper_qpos']}")
