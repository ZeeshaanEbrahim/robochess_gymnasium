#!/usr/bin/env python
"""RS32 main evaluation runner (adapted from results/rs31_24_pairs/eval/run_evals.py).
Runs scripts/eval_probe.py UNMODIFIED via rms_wrapper.py, at most 3 processes at once, records EVAL_MANIFEST.json.

    run_evals.py start    write the manifest header (HEAD, dirty/untracked, sha256s, step counts, free -m)
    run_evals.py models   the 6 RS32 final models (pool of 3 workers; no launch while MemAvailable < 1.5 GB)
    run_evals.py end      close the manifest
Pair list and order are parsed from results/rs32_48_pairs/PREREGISTRATION.md (line starting "Training pairs").
"""
import hashlib, json, re, subprocess, sys, time, zipfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "results" / "rs32_48_pairs" / "eval"
PREREG = REPO / "results" / "rs32_48_pairs" / "PREREGISTRATION.md"
MAN = EVAL / "EVAL_MANIFEST.json"
PY = str(Path.home() / "venvs" / "robochess" / "bin" / "python")
SEEDS = [0, 100, 200, 300, 400, 500]
RUNS = [f"_probe_cart_rs32D_ppotanh_seed{s}" for s in SEEDS]
MAX_PROCS = 3
MIN_AVAIL_KB = 1.5 * 1024 * 1024
EPISODES, SEED = 2400, 20000

now = lambda: datetime.now().astimezone().isoformat(timespec="seconds")
def sh(*cmd): return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True).stdout
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def short(r): return r.replace("_probe_cart_", "").replace("_ppotanh", "")
def mem_avail_kb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"): return int(l.split()[1])
def n_timesteps(run):
    with zipfile.ZipFile(REPO / "runs" / run / "final_model.zip") as z:
        return json.loads(z.read("data"))["num_timesteps"]
def prereg_pairs():
    line = next(l for l in PREREG.read_text().splitlines() if l.startswith("Training pairs"))
    pairs = re.findall(r"\b[a-h][1-8]:[a-h][1-8]\b", line.split("in this order):", 1)[1])
    assert len(pairs) == 48 and len(set(pairs)) == 48, pairs
    return pairs
def git_state():
    por = [l for l in sh("git", "status", "--porcelain").splitlines() if l]
    return dict(head=sh("git", "rev-parse", "HEAD").strip(), branch=sh("git", "branch", "--show-current").strip(),
                tracked_dirty_count=sum(not l.startswith("??") for l in por),
                tracked_dirty_paths=[l for l in por if not l.startswith("??")], untracked_paths=[l[3:] for l in por if l.startswith("??")])
def load(): return json.loads(MAN.read_text())
def save(m): MAN.write_text(json.dumps(m, indent=1) + "\n")

def start():
    assert not MAN.exists(), "manifest exists; refusing to overwrite"
    files = {"scripts/eval_probe.py": sha256(REPO / "scripts/eval_probe.py"),
             "results/rs32_48_pairs/PREREGISTRATION.md": sha256(PREREG)}
    for r in RUNS:
        for f in ("final_model.zip", "final_vecnormalize.pkl"):
            files[f"runs/{r}/{f}"] = sha256(REPO / "runs" / r / f)
    save(dict(start=now(), git_start=git_state(), pairs=prereg_pairs(), episodes=EPISODES, seed=SEED, sha256=files,
              final_num_timesteps={r: n_timesteps(r) for r in RUNS},
              free_m_at_start=sh("free", "-m"), evals=[], end=None))

def cmd_for(run, out):
    return [PY, str(Path(__file__).with_name("rms_wrapper.py").relative_to(REPO)), run, "--episodes", str(EPISODES),
            "--pairs", ",".join(prereg_pairs()), "--seed", str(SEED), "--out", str(out.relative_to(REPO))]

def models():
    m = load(); pending = list(RUNS); running = []
    while pending or running:
        while pending and len(running) < MAX_PROCS and mem_avail_kb() >= MIN_AVAIL_KB:
            r = pending.pop(0); out = EVAL / f"{short(r)}_final.json"; log = out.with_suffix(".log"); cmd = cmd_for(r, out)
            rec = dict(label=short(r), run=r, cmd=" ".join(cmd),
                       equivalent_eval_probe_cmd=" ".join([PY, "scripts/eval_probe.py"] + cmd[3:]),
                       seed=SEED, episodes=EPISODES, deterministic=False, start=now(), end=None, exit=None, log=str(log.relative_to(REPO)),
                       model_path=f"runs/{r}/final_model.zip", vecnormalize_path=f"runs/{r}/final_vecnormalize.pkl",
                       model_sha256=m["sha256"][f"runs/{r}/final_model.zip"], vecnormalize_sha256=m["sha256"][f"runs/{r}/final_vecnormalize.pkl"])
            m["evals"].append(rec); m.setdefault("free_m_before_launch", []).append(dict(label=short(r), at=now(), free_m=sh("free", "-m")))
            save(m)
            running.append((rec, out, subprocess.Popen(cmd, cwd=REPO, stdout=open(log, "w"), stderr=subprocess.STDOUT)))
        if pending and len(running) < MAX_PROCS and mem_avail_kb() < MIN_AVAIL_KB:
            m.setdefault("mem_holds", []).append(dict(at=now(), mem_available_kb=mem_avail_kb())); save(m)
        time.sleep(10)
        for item in list(running):
            rec, out, p = item
            if p.poll() is not None:
                rec["exit"] = p.returncode; rec["end"] = now()
                if out.exists(): rec["output_json_sha256"] = sha256(out)
                rms = out.with_name(out.stem + "_rms.json")
                if rms.exists(): rec["rms_json_sha256"] = sha256(rms)
                running.remove(item); save(m)

def end():
    m = load(); m["end"] = now(); m["git_end"] = git_state(); m["free_m_at_end"] = sh("free", "-m"); save(m)

if __name__ == "__main__":
    {"start": start, "models": models, "end": end}[sys.argv[1]]()
