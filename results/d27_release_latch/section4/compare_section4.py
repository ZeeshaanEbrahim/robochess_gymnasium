"""Compare the old (ff7ea92) and new runs written by run_val.py / run_gate.py; prints the Section 4 tables."""
import json, statistics as st
from collections import Counter
from pathlib import Path
H = Path(__file__).resolve().parent
L = lambda n: json.load(open(H / n))["results"]
for name in ("joint", "t2o0", "t2o2"):
    o, n = L(f"val_{name}_old.json"), L(f"val_{name}_new.json")
    assert len(o) == len(n) == 315 and all((a["start"], a["target"]) == (b["start"], b["target"]) for a, b in zip(o, n))
    key = lambda r: (r["success"], r["outcome"], r["failed_phase"], r["steps"])
    diffs = [(a["start"], a["target"], key(a), key(b)) for a, b in zip(o, n) if key(a) != key(b)]
    fo = sorted((a["start"], a["target"], a["outcome"], a["failed_phase"]) for a in o if not a["success"])
    fn = sorted((b["start"], b["target"], b["outcome"], b["failed_phase"]) for b in n if not b["success"])
    succ = [b for b in n if b["success"]]; h = [b["release_height_mm"] for b in succ]
    print(f"{name}: old {sum(a['success'] for a in o)}/315 new {sum(b['success'] for b in n)}/315; episodes differing "
          f"(success, outcome, phase, steps): {len(diffs)}; failure sets equal: {fo == fn} ({len(fn)}); release height over "
          f"{len(succ)} successes mm min {min(h):.2f} median {st.median(h):.2f} max {max(h):.2f}; "
          f"ever_held/release_ok all True: {all(b['ever_held'] and b['release_ok'] for b in succ)}")
    if name == "joint": print("  failures:", Counter((r[2], r[3]) for r in fn))
go = json.load(open(H / "gate_old.json"))["pairs"]; gn = json.load(open(H / "gate_new.json"))["pairs"]
for a, b in zip(go, gn):
    print(b["pair"], "harness clean", a["harness_clean_place"], b["harness_clean_place"], "env is_success", a["env_is_success"], b["env_is_success"],
          "release_height_mm %.2f" % b["release_height_mm"], "steps", a["steps"], b["steps"])
