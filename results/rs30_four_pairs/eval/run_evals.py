#!/usr/bin/env python
"""RS30 Section 3 runner. Runs scripts/eval_probe.py UNMODIFIED, records EVAL_MANIFEST.json.

    run_evals.py start      write the manifest header (HEAD, dirty/untracked, sha256s, free -m)
    run_evals.py control    positive control: D29 squash seed 3, e4:e5, 300 episodes
    run_evals.py models     the 8 RS30 final models, batches of 3 (max 3 processes at once)
    run_evals.py zeroshot   Section 4 item 4: held-out d5:d6,c4:c5 (150 ep) and the 20 RS29 pairs (300 ep), 8 models
    run_evals.py end        close the manifest (end timestamp, HEAD, dirty/untracked again)
"""
import hashlib, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "results" / "rs30_four_pairs" / "eval"
MAN = EVAL / "EVAL_MANIFEST.json"
PY = str(Path.home() / "venvs" / "robochess" / "bin" / "python")
PAIRS = "e4:e5,d4:d5,f4:f5,e3:e4"
CONTROL = "_probe_cart_place29_ppotanh_seed3"
RUNS = [f"_probe_cart_rs30{arm}_ppotanh_seed{i}" for arm in "AB" for i in range(4)]
MAX_PROCS = 3

now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
def sh(*cmd):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True).stdout
def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git_state():
    por = [l for l in sh("git", "status", "--porcelain").splitlines() if l]
    return dict(head=sh("git", "rev-parse", "HEAD").strip(), branch=sh("git", "branch", "--show-current").strip(),
                tracked_dirty_count=sum(not l.startswith("??") for l in por),
                tracked_dirty_paths=[l for l in por if not l.startswith("??")],
                untracked_paths=[l[3:] for l in por if l.startswith("??")])
def load(): return json.loads(MAN.read_text())
def save(m): MAN.write_text(json.dumps(m, indent=1) + "\n")
def free_m(): return sh("free", "-m")

def start():
    assert not MAN.exists(), "manifest exists; refusing to overwrite"
    files = {"scripts/eval_probe.py": sha256(REPO / "scripts/eval_probe.py"),
             "results/rs30_four_pairs/PREREGISTRATION.md": sha256(REPO / "results/rs30_four_pairs/PREREGISTRATION.md")}
    for r in [CONTROL] + RUNS:
        for f in ("final_model.zip", "final_vecnormalize.pkl"):
            files[f"runs/{r}/{f}"] = sha256(REPO / "runs" / r / f)
    save(dict(start_utc=now(), git_start=git_state(), sha256=files, free_m_at_start=free_m(), evals=[], end_utc=None))

def launch(m, label, run, out, pairs, episodes):
    cmd = [PY, "scripts/eval_probe.py", run, "--episodes", str(episodes), "--pairs", pairs, "--seed", "20000",
           "--out", str(out.relative_to(REPO))]
    log = out.with_suffix(".log")
    rec = dict(label=label, cmd=" ".join(cmd), start_utc=now(), end_utc=None, exit=None, log=str(log.relative_to(REPO)))
    m["evals"].append(rec); save(m)
    return rec, subprocess.Popen(cmd, cwd=REPO, stdout=open(log, "w"), stderr=subprocess.STDOUT)

def run_batches(jobs):
    m = load()
    for b in range(0, len(jobs), MAX_PROCS):
        batch = jobs[b:b + MAX_PROCS]
        m.setdefault("free_m_before_batch", []).append(dict(batch=[j[0] for j in batch], utc=now(), free_m=free_m()))
        procs = [launch(m, *j) for j in batch]
        for rec, p in procs:
            p.wait(); rec["exit"] = p.returncode; rec["end_utc"] = now()
        save(m)
        if any(rec["exit"] != 0 for rec, _ in procs): sys.exit("an eval exited non-zero; stopping")

def control():
    run_batches([("control_d29_seed3_e4e5", CONTROL, EVAL / "control_d29_seed3_e4e5.json", "e4:e5", 300)])

def models():
    run_batches([(r.replace("_probe_cart_", "").replace("_ppotanh", ""), r,
                  EVAL / (r.replace("_probe_cart_", "").replace("_ppotanh", "") + "_final.json"), PAIRS, 300) for r in RUNS])

# the 20 RS29 pairs, in the order of results/rs29_random_squares/PREREGISTRATION.md (and of its final_seed*.json)
RS29_PAIRS = ("e4:e5,c1:c2,d2:e2,a6:a5,e8:d7,d4:e5,f3:c6,h5:d3,g5:b3,c1:f6,h1:h8,h8:h1,a4:h4,b7:g2,b2:g7,f1:c8,"
              "a1:h8,h8:a1,h1:a8,a8:h1")
NEAR_PAIRS = "d5:d6,c4:c5"

def zeroshot():
    jobs = []
    for r in RUNS:
        short = r.replace("_probe_cart_", "").replace("_ppotanh", "")
        jobs.append((f"zs_near_{short}", r, EVAL / f"zs_near_{short}.json", NEAR_PAIRS, 150))
        jobs.append((f"zs_rs29pairs_{short}", r, EVAL / f"zs_rs29pairs_{short}.json", RS29_PAIRS, 300))
    run_batches(jobs)

def end():
    m = load(); m["end_utc"] = now(); m["git_end"] = git_state(); m["free_m_at_end"] = free_m(); save(m)

if __name__ == "__main__":
    {"start": start, "control": control, "models": models, "zeroshot": zeroshot, "end": end}[sys.argv[1]]()
