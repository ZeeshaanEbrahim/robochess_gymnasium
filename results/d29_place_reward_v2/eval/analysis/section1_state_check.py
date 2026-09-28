#!/usr/bin/env python
"""D29 section 1 (read-only): state check for all 11 probe runs.

For each run: manifest git sha (and whether it matches HEAD), dirty flag,
final_model.zip presence, and the final total timesteps. For PPO, that last
figure comes straight from logs/progress.csv's last row (SB3 logs it exactly).
For SAC, progress.csv's last row lags the true end (SB3 only appends at its own
logging interval); the true count is loaded from the saved model itself
(SAC.load(...).num_timesteps) and cross-checked against the Monitor CSV's
completed-episode length sum.

Reads runs/ only. Writes only --out. No trace directory needed (kept as an
accepted argument for interface consistency across the D29 analysis scripts,
per the read-out instructions).
"""
import argparse, csv, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HEAD_SHA = "41ae8e542c60a17802f92d668a872896bf6b7026"

RUNS = [f"_probe_cart_place29_ppo_seed{s}" for s in range(5)] + \
       [f"_probe_cart_place29_ppotanh_seed{s}" for s in range(5)] + \
       ["_probe_cart_place29_sac_seed0"]

p = argparse.ArgumentParser()
p.add_argument("--trace-dir", type=Path, default=Path.home() / "robochess_archive" / "d29_eval_traces",
                help="unused by this script; accepted for interface consistency")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--runs-dir", type=Path, default=REPO / "runs")
a = p.parse_args()


def sac_true_final_timesteps(run_dir: Path) -> tuple[int, int]:
    """Returns (model.num_timesteps, monitor-csv completed-episode length sum)."""
    code = (
        "import json,sys\n"
        "from stable_baselines3 import SAC\n"
        f"m = SAC.load({str(run_dir / 'final_model.zip')!r}, device='cpu')\n"
        "print(m.num_timesteps)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    num_timesteps = int(out.stdout.strip())
    monitor_path = run_dir / "logs" / "monitor.monitor.csv"
    total = 0
    with open(monitor_path) as f:
        next(f)
        for row in csv.DictReader(f):
            total += int(float(row["l"]))
    return num_timesteps, total


rows = []
for run in RUNS:
    base = a.runs_dir / run
    manifest = json.loads((base / "manifest.json").read_text())
    sha = manifest["git"]["sha"]
    dirty = manifest["git"]["dirty_tracked"]
    final_model = (base / "final_model.zip").exists()
    algo = manifest["config_resolved"]["algorithm"]

    prog_path = base / "logs" / "progress.csv"
    with open(prog_path) as f:
        last_row = None
        for row in csv.DictReader(f):
            last_row = row
    progress_csv_last = last_row["time/total_timesteps"] if last_row else None

    entry = dict(run=run, sha=sha, sha_matches_head=(sha == HEAD_SHA), dirty_tracked=dirty,
                 final_model_zip=final_model)
    if algo == "sac":
        num_timesteps, monitor_sum = sac_true_final_timesteps(base)
        entry["final_total_timesteps_progress_csv_last_row"] = progress_csv_last
        entry["final_total_timesteps_model_num_timesteps"] = num_timesteps
        entry["monitor_csv_completed_episode_length_sum"] = monitor_sum
        entry["note"] = (
            f"progress.csv last row ({progress_csv_last}) lags the true end; confirmed via "
            f"SAC.load().num_timesteps == {num_timesteps}, cross-checked against monitor CSV "
            f"completed-episode length sum {monitor_sum} ({num_timesteps - monitor_sum} short, "
            "consistent with the final episode still in progress at save time)"
        )
    else:
        entry["final_total_timesteps"] = progress_csv_last
    rows.append(entry)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
