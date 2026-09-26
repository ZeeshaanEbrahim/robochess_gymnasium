"""Reproduction gate: compare a run of the recovered scripts on ff7ea92 with the sweep's own records.
usage: python compare_gate.py <out dir of the ff7ea92 run> -> prints one line per case, writes recovered_gate_ff7ea92.json"""
import json, os, sys
from pathlib import Path
H = Path(__file__).resolve().parent
ev = json.load(open(H / "sweep_evidence.json")); out = Path(sys.argv[1]); rows = []
keys = ("steps", "success", "final_tilt_deg", "final_target_distance_mm", "reward_shaped", "reward_pickup")
for f in sorted(os.listdir(out)):
    r = json.load(open(out / f))["result"]; e = ev.get(f, {}).get("result")
    same = None if e is None else all(
        (abs(float(r[k]) - float(e[k])) < 1e-9) if isinstance(r[k], (int, float)) and not isinstance(r[k], bool) else r[k] == e[k] for k in keys)
    rows.append(dict(file=f, **{k: r[k] for k in keys}, identical_to_sweep_record=same))
    print(f, {k: r[k] for k in keys}, "identical to sweep record:", same)
json.dump(rows, open(H / "recovered_gate_ff7ea92.json", "w"), indent=1)
