#!/usr/bin/env bash
# E8 + post-hoc IoU-filter analysis + test-stage predictions, one background job (idempotent).
#
#   scripts/run_E8_stage.sh --dry-run
#   scripts/run_E8_stage.sh
#
# Order: (1) E8 CV via run_E2.sh (train, collapse check, predict, evaluate; needs the E8 tile cache),
# (2) iou_filter_analysis.py predict for E2c, then E8 (only if every E8 fold passed), (3) select (CPU),
# (4) 5-fold-ensemble test predictions for E2a, E2b, E2c and E8 (scripts/predict_test_medsam.py) - NOT
# scored. Steps already done are skipped by the scripts themselves.
# Log: logs/stage/job.log; PID logs/stage/job.pid
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO/scripts/env.sh"
JOBDIR="$MSBENCH_LOGS/stage"; PIDFILE="$JOBDIR/job.pid"; mkdir -p "$JOBDIR" "$MSBENCH_LOGS/E8" "$MSBENCH_LOGS/iou"
RUNS="$(python -c 'from msbench.config import paths; print(paths()["medsam_runs"])')"
e8_ok() { for k in 0 1 2 3 4; do [ -f "$RUNS/E8/fold_$k/checkpoint_final.pt" ] || return 1; done; }

if [ "${1:-}" = "--worker" ]; then
    set +e
    echo "=== $(date -Is) stage start (pid $$), commit $(git -C "$REPO" rev-parse --short HEAD)"
    "$REPO/scripts/run_E2.sh" --worker --exps E8
    for e in E2c E8; do
        if [ "$e" = E8 ] && ! e8_ok; then echo "=== $(date -Is) E8 not complete (collapse/failure): skipping its IoU analysis"; continue; fi
        echo "=== $(date -Is) IoU filter $e: inner models + levels"
        python "$REPO/scripts/iou_filter_analysis.py" predict --exp "$e" >> "$MSBENCH_LOGS/iou/$e.log" 2>&1
        rc=$?; echo "=== $(date -Is) IoU filter $e predict: exit $rc"
        if [ $rc -eq 0 ]; then
            python "$REPO/scripts/iou_filter_analysis.py" select --exp "$e" >> "$MSBENCH_LOGS/iou/$e.log" 2>&1
            echo "=== $(date -Is) IoU filter $e select: exit $?"
        fi
    done
    for e in E2a E2b E2c E8; do
        if [ "$e" = E8 ] && ! e8_ok; then echo "=== $(date -Is) E8 not complete: no test predictions"; continue; fi
        echo "=== $(date -Is) test predictions $e"
        python "$REPO/scripts/predict_test_medsam.py" --exp "$e" >> "$MSBENCH_LOGS/stage/test_$e.log" 2>&1
        echo "=== $(date -Is) test predictions $e: exit $?"
    done
    echo "=== $(date -Is) stage done"
    exit 0
fi

python "$REPO/scripts/cache_e8_tiles.py" --dry-run | head -1 | grep -q ", 0 to do" \
    || { echo "E8 tile cache incomplete" >&2; exit 1; }
if [ -f "$PIDFILE" ] && grep -q run_E8_stage "/proc/$(cat "$PIDFILE")/cmdline" 2>/dev/null; then
    echo "stage job already running (pid $(cat "$PIDFILE"))" >&2; exit 1; fi
"$REPO/scripts/run_E2.sh" --exps E8 --dry-run | sed 's/^/  /'
[ "${1:-}" = "--dry-run" ] && { echo "[dry-run] then IoU analysis (E2c, E8) and test predictions (E2a E2b E2c E8)"; exit 0; }
python "$REPO/scripts/gpu_check.py" || { echo "GPU looks busy; not launching" >&2; exit 1; }
cd "$REPO"
setsid nohup "$0" --worker >> "$JOBDIR/job.log" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "launched, pid $(cat "$PIDFILE"), log $JOBDIR/job.log"
