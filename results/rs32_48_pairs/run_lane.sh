#!/usr/bin/env bash
# RS32 lane runner: trains the given seeds one after another, one D run at a time.
#   usage: bash results/rs32_48_pairs/run_lane.sh 0 200 400
# Per seed S (config configs/_probe_cart_rs32D_ppotanh_seed${S}.yaml):
#   * runs/<run_name>_train.log has an EXIT=0 line      -> skip (already finished)
#   * run dir or log exists without EXIT=0              -> STOP the lane, print the path;
#                                                          nothing is deleted or overwritten
#   * otherwise                                         -> launch scripts/train.py exactly as
#                                                          the RS31 launch did, append EXIT=$?
# A non-zero EXIT also stops the lane (the next seed is not started). After the last seed,
# or on a stop, the script sleeps so the tmux session stays open for reading the output.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
source ~/venvs/robochess/bin/activate
export CUDA_VISIBLE_DEVICES=''

[ "$#" -ge 1 ] || { echo "usage: $0 SEED [SEED ...]"; sleep 86400; exit 2; }

for S in "$@"; do
  NAME="_probe_cart_rs32D_ppotanh_seed${S}"
  CFG="configs/${NAME}.yaml"
  LOG="runs/${NAME}_train.log"
  DIR="runs/${NAME}"
  [ -f "$CFG" ] || { echo "STOP: missing config $CFG"; break; }
  if [ -f "$LOG" ] && grep -q '^EXIT=0$' "$LOG"; then
    echo "skip seed ${S}: ${LOG} has EXIT=0"
    continue
  fi
  if [ -e "$DIR" ] || [ -e "$LOG" ]; then
    echo "STOP: seed ${S} has a partial or failed run without EXIT=0:"
    [ -e "$DIR" ] && echo "  $REPO/$DIR"
    [ -e "$LOG" ] && echo "  $REPO/$LOG"
    echo "Move both aside (e.g. prefix _died_YYYYMMDD_), then re-run this lane."
    break
  fi
  mkdir -p runs
  echo "=== $(date '+%F %T') launching seed ${S} (${NAME})"
  python -W error::UserWarning scripts/train.py --config "$CFG" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  echo "EXIT=${RC}" >> "$LOG"
  echo "=== $(date '+%F %T') seed ${S} finished EXIT=${RC}"
  if [ "$RC" -ne 0 ]; then
    echo "STOP: seed ${S} exited ${RC}; lane halted, remaining seeds not started."
    break
  fi
done
echo "lane done or stopped; session left open"
sleep 86400
