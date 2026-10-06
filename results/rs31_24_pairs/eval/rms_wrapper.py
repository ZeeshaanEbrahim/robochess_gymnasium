#!/usr/bin/env python
"""Runs scripts/eval_probe.py UNMODIFIED (runpy, same argv) and records VecNormalize obs_rms / ret_rms
(mean, var, count) before and after, to <out>_rms.json.

    rms_wrapper.py <run> --episodes N --pairs ... --seed S --out X.json      (eval_probe.py's own arguments)

before = the statistics unpickled from the run's final_vecnormalize.pkl (file read, before eval_probe.py starts);
after  = the `venv` object that eval_probe.py loaded and rolled out (read from its globals once it finishes).
Also records sha256 of the pkl before and after. Writes only <out>_rms.json (and whatever eval_probe.py writes).
"""
import hashlib, json, pickle, runpy, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
args = sys.argv[1:]
run = args[0]; out = Path(args[args.index("--out") + 1])
pkl = REPO / "runs" / run / "final_vecnormalize.pkl"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def stats(vn):
    d = {}
    for name, rms in (vn.obs_rms.items() if isinstance(vn.obs_rms, dict) else [("obs", vn.obs_rms)]):
        d[f"obs_rms/{name}"] = rms
    d["ret_rms"] = vn.ret_rms
    return {k: dict(mean=np.asarray(v.mean).tolist(), var=np.asarray(v.var).tolist(), count=float(v.count)) for k, v in d.items()}

sha_before = sha(pkl)
with open(pkl, "rb") as f:
    before = stats(pickle.load(f))
sys.argv = [str(REPO / "scripts" / "eval_probe.py")] + args
g = runpy.run_path(sys.argv[0], run_name="__main__")
after = stats(g["venv"])
same = before.keys() == after.keys() and all(before[k] == after[k] for k in before)
rec = dict(run=run, pkl=str(pkl.relative_to(REPO)), sha256_before=sha_before, sha256_after=sha(pkl), pkl_unchanged=sha_before == sha(pkl),
           identical_before_after=same, keys=sorted(before), before=before, after=after)
out.with_name(out.stem + "_rms.json").write_text(json.dumps(rec))
print("rms_wrapper", run, "rms_identical_before_after", same, "pkl_unchanged", rec["pkl_unchanged"], "keys", rec["keys"], flush=True)
