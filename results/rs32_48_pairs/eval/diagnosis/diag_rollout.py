#!/usr/bin/env python
"""POST HOC / EXPLORATORY diagnostic rollout. Runs scripts/eval_probe.py UNMODIFIED (runpy, same argv as the Step 2 eval, but
--episodes 480 = the first 10 episodes of every pair of the committed eval stream) and dumps per-step arrays from its Rec wrapper.

    diag_rollout.py <run> --episodes 480 --pairs <48 pairs> --seed 20000 --out <dir>/ro_seedN.json   (eval_probe's own arguments)

Writes <out stem>_steps.json (large; not committed). All diagnostic tables are computed from it by diag_rollout_tables.py."""
import json, runpy, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
args = sys.argv[1:]; out = Path(args[args.index("--out") + 1]); N = int(args[args.index("--episodes") + 1])
sys.argv = [str(REPO / "scripts" / "eval_probe.py")] + args
g = runpy.run_path(sys.argv[0], run_name="__main__")
w = g["w"]; ext = []
for e in w.episodes[:N]:
    R = e["rows"]
    ext.append(dict(start=e["start"], target=e["target"], reason=e["reason"], n=len(R), gxy=R[-1]["gxy"], succ=bool(R[-1]["succ"]),
        st=[int(r["st"][1]) for r in R], pxy=[[round(v, 5) for v in r["pxy"]] for r in R], ptilt=[round(r["ptilt"], 1) for r in R],
        lift=[round(r["lift"] * 1e3, 1) for r in R], dxy=[round(r["dxy"] * 1e3, 1) for r in R], ph=[round(r["ph"] * 1e3, 1) for r in R],
        d=[round(r["d"] * 1e3, 1) for r in R], l=[int(r["l"]) for r in R], r=[int(r["r"]) for r in R],
        ap=[round(r["ap"], 3) for r in R], tilt=[round(r["tilt"], 1) for r in R]))
out.with_name(out.stem + "_steps.json").write_text(json.dumps(ext))
print("steps written", len(ext), flush=True)
