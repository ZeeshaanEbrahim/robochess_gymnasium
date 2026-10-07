#!/usr/bin/env python
"""POST HOC / EXPLORATORY. Tables from the diagnostic rollouts of diag_rollout.py (first 480 episodes = 10 per pair of each seed's committed
eval stream; step arrays in DUMPS/ro_seedN_steps.json, not committed). usage: diag_rollout_tables.py <dumps dir>"""
from diag_common import *
DUMPS = Path(sys.argv[1])
tag = lambda n: n.replace("rs32D_", "")
RO = {n: json.load(open(DUMPS / f"ro_seed{n.split('seed')[1]}_steps.json")) for n in NAMES}
EV_ = {n: eval_eps(n)[:480] for n in NAMES}
loc = lambda sq: np.array([(Fl(sq)-3.5)*0.078, (Rk(sq)-4.5)*0.078])
ecat = lambda e: pair_cat(f'{e["start"]}:{e["target"]}')
TOPPLE_DEG = 45

print("R1. REPRODUCTION of the committed eval by the diagnostic rollout (episode-for-episode: end reason, length, placed, start and target square)")
for n in NAMES:
    a, b = RO[n], EV_[n]
    same = sum(x["reason"] == y["reason"] and x["n"] == y["n"] and x["succ"] == y["place"]["placed"] and x["start"] == y["start_square"] and x["target"] == y["target_square"] for x, y in zip(a, b))
    print(f"   {tag(n):<10} {same}/{len(b)} episodes identical" + ("" if same == len(b) else "   <-- NOT identical"))

def info(e):
    st = np.array(e["st"]); pt = np.array(e["ptilt"]); return dict(final=int(st[-1]), maxt=float(pt.max()), n=len(st))
def mode(e):
    i = info(e)
    if e["succ"]: return "placed"
    if i["maxt"] > TOPPLE_DEG: return "toppled"
    if e["reason"] == "time_limit" and i["final"] == 1: return "upright, still pinching at time limit"
    if e["reason"] == "time_limit" and i["final"] == 3: return "upright, lost (S3) at time limit"
    return "other"
MODES = ["placed", "toppled", "upright, still pinching at time limit", "upright, lost (S3) at time limit", "other"]
print(f"\nR2. EXCLUSIVE EPISODE OUTCOMES, 480 episodes per seed (every row sums to 480; asserted). Evaluated in this order:")
print(f"   placed = env is_success at the last step; toppled = piece tilt > {TOPPLE_DEG} deg at ANY step of the episode (includes every piece_off_board end that tilted);")
print("   upright, pinching/lost = not toppled, ended by time limit in place state S1 (both pads on the piece) / S3 (held once, now lost, not in S2);")
print("   other = everything else: not toppled and (time limit in S2 [released, but not success] or in S0 [never held], or ended by bad_contact / piece_off_board without tilt > 45).")
print(f'{"outcome":<44}' + "".join(f"{tag(n):>9}" for n in NAMES))
for m in MODES: print(f"{m:<44}" + "".join(f"{sum(mode(e) == m for e in RO[n]):>9}" for n in NAMES))
print(f'{"SUM":<44}' + "".join(f"{sum(mode(e) in MODES for e in RO[n]):>9}" for n in NAMES))
for n in NAMES: assert sum(mode(e) in MODES for e in RO[n]) == 480
print("   composition of 'other' (reason / final place state S0..S3):")
for n in NAMES:
    c = collections.Counter(f'{e["reason"]}/S{info(e)["final"]}' for e in RO[n] if mode(e) == "other"); print(f"   {tag(n):<10} {dict(sorted(c.items()))}")
print("   UPRIGHT-AND-STUCK = 'still pinching' + 'lost at time limit':")
for n in NAMES:
    a = sum(mode(e) == MODES[2] for e in RO[n]); b = sum(mode(e) == MODES[3] for e in RO[n]); print(f"   {tag(n):<10} {a} + {b} = {a+b}    toppled {sum(mode(e)=='toppled' for e in RO[n])}")
print("\nR2b. the same outcomes by pair category (episodes per category per seed: neigh_fwd/lat 280, neigh_back 20, long_fwd/lat 130, long_back 50)")
for c in CATS:
    print(f"-- {c}")
    for m in MODES: print(f"   {m:<41}" + "".join(f"{sum(mode(e) == m for e in RO[n] if ecat(e) == c):>9}" for n in NAMES))

print("\nR3. BACKWARD NEIGHBOUR PAIRS e5:e4 and c6:c5, n = 20 episodes per seed (10 per pair) from this 480-episode re-run; progress = piece displacement along start->target / pair distance at the last step (<0 = ended on the far side of its start square)")
for n in NAMES:
    S = [e for e in RO[n] if ecat(e) == "neigh_back"]; assert len(S) == 20
    prog = []
    for e in S:
        g = np.array(e["gxy"]); s0 = g - (loc(e["target"]) - loc(e["start"])); v = g - s0; L = np.linalg.norm(v); prog.append(float((np.array(e["pxy"][-1]) - s0) @ (v/L)) / L)
    prog = np.array(prog); fail = [(e, p) for e, p in zip(S, prog) if not e["succ"]]
    to = [(e, p) for e, p in fail if e["reason"] == "time_limit"]
    print(f"   {tag(n):<10} n=20: placed {sum(e['succ'] for e in S)}, time-limit ends {len(to)} (final state S1: {sum(info(e)['final']==1 for e,_ in to)}, S3: {sum(info(e)['final']==3 for e,_ in to)}), toppled {sum(mode(e)=='toppled' for e in S)}; "
          f"non-placed with progress<0: {sum(p<0 for _, p in fail)}/{len(fail)}; median progress of non-placed {np.median([p for _, p in fail]) if fail else float('nan'):.2f}")

print("\nR4. WHERE THE PIECE ENDS IN THE UPRIGHT-AND-STUCK EPISODES (piece-target xy distance at the last step, mm; groups with n>=5)")
for m in MODES[2:4]:
    for n in NAMES:
        x = np.array([e["dxy"][-1] for e in RO[n] if mode(e) == m])
        if len(x) >= 5: print(f"   {m:<40}{tag(n):>9} n={len(x):<3} p25 {np.percentile(x,25):5.0f} p50 {np.percentile(x,50):5.0f} p75 {np.percentile(x,75):5.0f}  share<30mm {np.mean(x<30):.2f}")
print("   number of separate grasp phases (runs of place state S1) per episode:")
for n in NAMES:
    ph = lambda e: int(((np.diff(np.array(e["st"]) == 1, prepend=0)) == 1).sum())
    a = np.mean([ph(e) for e in RO[n] if mode(e) == "placed"]); st = [ph(e) for e in RO[n] if mode(e) in MODES[2:4]]
    print(f"   {tag(n):<10} placed {a:.1f} per episode; upright-and-stuck {np.mean(st) if st else float('nan'):.1f} per episode (n={len(st)})")

print(f"\nR5. WHEN DOES THE PIECE TOPPLE (first step with tilt > {TOPPLE_DEG} deg): place state at that step, and time since the last pinched step")
for n in NAMES:
    ev = []
    for e in RO[n]:
        t = np.array(e["ptilt"]); idx = np.nonzero(t > TOPPLE_DEG)[0]
        if not len(idx): continue
        i = int(idx[0]); st = e["st"]; last = max([k for k in range(i) if st[k] == 1], default=None)
        ev.append(dict(st=st[i], since=None if last is None else i - last, lift=e["lift"][i], d=e["d"][i], ap=e["ap"][i], tool=e["tilt"][i], i=i, succ=e["succ"], long=f'{e["start"]}:{e["target"]}' in LONG))
    c = collections.Counter(x["st"] for x in ev); ns = [x for x in ev if x["st"] != 1]
    sin = [x["since"] for x in ns if x["since"] is not None]
    print(f"   {tag(n):<10} toppled episodes {len(ev)}/480 (long pairs {sum(x['long'] for x in ev)}), placed after toppling {sum(x['succ'] for x in ev)}; state at first topple step: S0 {c[0]}, S1 {c[1]}, S3 {c[3]}; "
          f"not pinched at topple: median steps since last pinch {np.median(sin) if sin else float('nan'):.0f}, pinch-site-to-piece distance median {np.median([x['d'] for x in ns]):.0f} mm, tool tilt median {np.median([x['tool'] for x in ns]):.1f} deg")

print("\nR6. HOW PLACED EPISODES MOVE THE PIECE (placed episodes only; step level, 480-episode re-run). Along-target displacement D_t = (piece_t - piece_{t-1}) . u, u = unit vector start->target; shares of the SUM of D_t over steps")
print(f'{"seed":<10}{"placed":>7}{"max_lift>10mm":>15}{"share of D at steps with lift<=10mm":>38}{"share of D at pinched steps":>29}{"pinched steps with lift>10mm":>30}{"share of D at pinched steps with lift<=10mm":>45}')
for n in NAMES:
    S = [e for e in RO[n] if e["succ"]]; Dl = Dp = Dpl = Dtot = 0.0; pst = psl = 0; mx = 0
    for e in S:
        g = np.array(e["gxy"]); s0 = g - (loc(e["target"]) - loc(e["start"])); v = g - s0; u = v/np.linalg.norm(v)
        P = np.vstack([s0, np.array(e["pxy"])]); D = np.diff(P, axis=0) @ u * 1e3; lift = np.array(e["lift"]); pin = np.array(e["st"]) == 1
        Dtot += D.sum(); Dl += D[lift <= 10].sum(); Dp += D[pin].sum(); Dpl += D[pin & (lift <= 10)].sum(); pst += pin.sum(); psl += (pin & (lift > 10)).sum(); mx += lift.max() > 10
    print(f"{tag(n):<10}{len(S):>7}{mx:>8} ={100*mx/len(S):5.1f}%{100*Dl/Dtot:>37.1f}%{100*Dp/Dtot:>28.1f}%{100*psl/pst:>29.1f}%{100*Dpl/Dtot:>44.1f}%")
print("   (lift = eval_probe's per-step piece lift in mm. A step with lift <= 10 mm can still have the piece pressed by the pads; this table does not tell dragging from pushing.)")
