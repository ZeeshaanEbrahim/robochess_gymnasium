#!/usr/bin/env bash
# D32 evaluation driver: one run at a time, final model + frozen final_vecnormalize.pkl, 200 stochastic episodes,
# eval seed 20000, --d32 recorder. free -h is printed (and MemAvailable >= 3.5 GiB required) before each run.
REPO=/root/projects/robochess_t2d29; OUTD=$REPO/results/d32_t1_t2/eval
cd $REPO; source ~/venvs/robochess/bin/activate
MIN_KB=$((35*1024*1024/10))
ev() {  # runs_dir run_name
  local rd="$1" name="$2"
  while :; do a=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo); [ "$a" -ge "$MIN_KB" ] && break; echo "$(date '+%F %T') MemAvailable ${a} kB < 3.5 GiB; waiting 60 s"; sleep 60; done
  echo "=== $(date '+%F %T') before $name:"; free -h
  nice -n 19 python -W error::UserWarning scripts/eval_probe.py "$name" --runs-dir "$rd" --episodes 200 --seed 20000 --d32 --out "$OUTD/$name.json" > "$OUTD/$name.log" 2>&1
  rc=$?
  echo "=== $(date '+%F %T') $name eval exit=$rc : $(tail -1 $OUTD/$name.log)"
  return $rc
}
for s in 0 100 200 300 400; do ev /root/projects/robochess_recipe/runs d32_t1_e4e5_seed$s || exit 1; done
for s in 0 100 200 300 400; do ev /root/projects/robochess_t2d29_run/runs d32_t2_d4f4_seed$s || exit 1; done
echo "ALL EVALS DONE $(date '+%F %T')"
