#!/usr/bin/env bash
# D33 evaluation chain: usage run_evals_d33.sh SEED [SEED ...]  (run two chains at most at once)
# final model + frozen final_vecnormalize.pkl, 200 stochastic episodes, eval seed 20000, --d32; free -h printed before each run.
REPO=/root/projects/robochess_d33_eval; OUTD=$REPO/results/d33_t1/eval
cd $REPO; source ~/venvs/robochess/bin/activate; export PYTHONDONTWRITEBYTECODE=1
for s in "$@"; do
  name=d33_t1_e4e5_seed$s
  while :; do a=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo); [ "$a" -ge 3670016 ] && break; echo "$(date '+%F %T') MemAvailable ${a} kB < 3.5 GiB; waiting 60 s"; sleep 60; done
  echo "=== $(date '+%F %T') before $name:"; free -h
  nice -n 19 python -W error::UserWarning scripts/eval_probe.py $name --runs-dir /root/projects/robochess_d33/runs --episodes 200 --seed 20000 --d32 --out $OUTD/$name.json > $OUTD/$name.log 2>&1
  rc=$?; echo "=== $(date '+%F %T') $name eval exit=$rc : $(tail -1 $OUTD/$name.log)"
  [ $rc -eq 0 ] || exit 1
done
echo "CHAIN DONE ($*) $(date '+%F %T')"
