#!/usr/bin/env python
"""RS32 Step 3 evaluation runner (reported, not gated). Same method as run_evals.py (scripts/eval_probe.py unmodified via
rms_wrapper.py), at most 3 processes at once, no launch while MemAvailable < 1.5 GB. Manifest: EVAL_MANIFEST_section3.json.

Jobs per model (6 models): zs_near (item e(a): 12 b/g pairs, 300 ep = 25 per pair), zs_rs29pairs (item e(b): the 20 RS29 pairs,
300 ep = 15 per pair, of which 19 are untrained; e4:e5 is a training pair and is excluded in the table, as in RS31),
det48 (item f: --deterministic, 48 pairs x 2 = 96 ep).
Start only after run_evals.py models has finished (the 3-process limit covers both).
"""
import json, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_evals as R

MAN3 = R.EVAL / "EVAL_MANIFEST_section3.json"
PAIRS48 = R.prereg_pairs()
NEAR12 = [f"{f}{r}:{f}{r+1}" for f in "bg" for r in range(2, 8)]
RS29 = ("e4:e5,c1:c2,d2:e2,a6:a5,e8:d7,d4:e5,f3:c6,h5:d3,g5:b3,c1:f6,h1:h8,h8:h1,a4:h4,b7:g2,b2:g7,f1:c8,"
        "a1:h8,h8:a1,h1:a8,a8:h1").split(",")
def check_lists():
    txt = (R.REPO / "results/rs29_random_squares/PREREGISTRATION.md").read_text()
    band = [l for l in txt.splitlines() if re.match(r"\s+(short|mid|long|diagonal) \(", l)]
    p = [x for l in band for x in re.findall(r"\b[a-h][1-8]:[a-h][1-8]\b", l)]
    assert p == RS29 and len(p) == 20, (p, RS29)
    pre = next(l for l in R.PREREG.read_text().splitlines() if l.startswith("Reported, not gated"))
    assert " ".join(NEAR12) in pre, (NEAR12, pre)
    assert not set(NEAR12) & set(PAIRS48) and set(RS29) & set(PAIRS48) == {"e4:e5"}
check_lists()

def jobs():
    j = []
    for r in R.RUNS:
        s = R.short(r)
        j.append(dict(label=f"zs_near_{s}", run=r, episodes=300, pairs=NEAR12, det=False, out=R.EVAL / f"zs_near_{s}.json"))
        j.append(dict(label=f"zs_rs29pairs_{s}", run=r, episodes=300, pairs=RS29, det=False, out=R.EVAL / f"zs_rs29pairs_{s}.json"))
    for r in R.RUNS:
        s = R.short(r)
        j.append(dict(label=f"det48_{s}", run=r, episodes=96, pairs=PAIRS48, det=True, out=R.EVAL / f"det48_{s}.json"))
    return j

def cmd(j):
    a = [j["run"], "--episodes", str(j["episodes"]), "--pairs", ",".join(j["pairs"]), "--seed", str(R.SEED)]
    if j["det"]: a.append("--deterministic")
    a += ["--out", str(j["out"].relative_to(R.REPO))]
    return [R.PY, str(Path(R.__file__).with_name("rms_wrapper.py").relative_to(R.REPO))] + a

def main():
    assert not MAN3.exists(), "section3 manifest exists"
    files = {f"runs/{r}/{f}": R.sha256(R.REPO / "runs" / r / f) for r in R.RUNS for f in ("final_model.zip", "final_vecnormalize.pkl")}
    files["scripts/eval_probe.py"] = R.sha256(R.REPO / "scripts/eval_probe.py")
    m = dict(start=R.now(), git_start=R.git_state(), sha256=files, seed=R.SEED, near12=NEAR12, rs29_20=RS29, pairs48=PAIRS48,
             free_m_at_start=R.sh("free", "-m"), evals=[], end=None)
    save = lambda: MAN3.write_text(json.dumps(m, indent=1) + "\n")
    save()
    pending, running = jobs(), []
    while pending or running:
        while pending and len(running) < R.MAX_PROCS and R.mem_avail_kb() >= R.MIN_AVAIL_KB:
            j = pending.pop(0); c = cmd(j); log = j["out"].with_suffix(".log")
            rec = dict(label=j["label"], run=j["run"], cmd=" ".join(c), episodes=j["episodes"], seed=R.SEED, deterministic=j["det"],
                       start=R.now(), end=None, exit=None, log=str(log.relative_to(R.REPO)),
                       model_sha256=files[f"runs/{j['run']}/final_model.zip"], vecnormalize_sha256=files[f"runs/{j['run']}/final_vecnormalize.pkl"])
            m["evals"].append(rec); save()
            running.append((rec, j["out"], subprocess.Popen(c, cwd=R.REPO, stdout=open(log, "w"), stderr=subprocess.STDOUT)))
        time.sleep(10)
        for item in list(running):
            rec, out, p = item
            if p.poll() is not None:
                rec["exit"] = p.returncode; rec["end"] = R.now()
                if out.exists(): rec["output_json_sha256"] = R.sha256(out)
                rms = out.with_name(out.stem + "_rms.json")
                if rms.exists(): rec["rms_json_sha256"] = R.sha256(rms)
                running.remove(item); save()
    m["end"] = R.now(); m["git_end"] = R.git_state(); m["free_m_at_end"] = R.sh("free", "-m"); save()

if __name__ == "__main__":
    main()
