#!/usr/bin/env bash
# E7 (exploratory, after the results freeze): MS-SAM-LESS + HarDNet baseline, one background job (idempotent).
#
#   scripts/run_E7.sh --dry-run
#   scripts/run_E7.sh
#
# Per fold 0-4: MS-SAMLess phase 1 (SAM-guided prompt learner), phase 2 (SAM-free segmentor), HarDNet
# baseline (scripts/e7_train.py; finished runs are skipped). Then predictions (CV per fold model, test sets
# as 5-fold ensembles; scripts/e7_predict.py), CV scoring with scripts/evaluate.py (results/E7*/fold<k>_cases.csv)
# and the report (scripts/e7_summary.py -> results/E7/summary.md).
# Logs: logs/E7/{job.log, <run>.log}; PID logs/E7/job.pid
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO/scripts/env.sh"
LOG="$MSBENCH_LOGS/E7"; PIDFILE="$LOG/job.pid"; mkdir -p "$LOG"
RUNS="$(python -c 'from msbench.config import paths; print(paths()["e7_runs"])')"
val_cases() { python -c "import json; print(' '.join(json.load(open('$REPO/experiments/E1/splits_final.json'))[$1]['val']))"; }

if [ "${1:-}" = "--worker" ]; then
    set +e
    echo "=== $(date -Is) E7 start (pid $$), commit $(git -C "$REPO" rev-parse --short HEAD)"
    fail=0
    for k in 0 1 2 3 4; do
        for spec in "samless --phase 1:samless_p1" "samless --phase 2:samless_p2" "hardnet:hardnet"; do
            a="${spec%%:*}"; n="${spec##*:}_fold$k"
            echo "=== $(date -Is) $n"
            python "$REPO/scripts/gpu_check.py" | sed 's/^/    /'
            SECONDS=0
            python "$REPO/scripts/e7_train.py" --model $a --fold "$k" >> "$LOG/$n.log" 2>&1
            rc=$?; echo "=== $(date -Is) $n: exit $rc after $((SECONDS / 60)) min"
            [ $rc -eq 0 ] || fail=1
        done
    done
    if [ $fail -eq 1 ]; then echo "=== $(date -Is) a training run failed; stopping before prediction"; exit 1; fi
    for e in E7 E7_hardnet; do
        python "$REPO/scripts/e7_predict.py" --exp "$e" --cv --test >> "$LOG/predict_$e.log" 2>&1
        echo "=== $(date -Is) predictions $e: exit $?"
        for k in 0 1 2 3 4; do
            python "$REPO/scripts/evaluate.py" --exp "$e" --fold "$k" --pred-dir "$RUNS/$e/fold_$k/validation" \
                --cases $(val_cases "$k") >> "$LOG/evaluate_$e.log" 2>&1
            echo "=== $(date -Is) $e fold $k evaluation: exit $?"
        done
    done
    python "$REPO/scripts/e7_summary.py" >> "$LOG/summary.log" 2>&1
    echo "=== $(date -Is) summary: exit $?"
    echo "=== $(date -Is) E7 done"
    exit 0
fi

if [ -f "$PIDFILE" ] && grep -q run_E7 "/proc/$(cat "$PIDFILE")/cmdline" 2>/dev/null; then
    echo "E7 job already running (pid $(cat "$PIDFILE"))" >&2; exit 1; fi
for k in 0 1 2 3 4; do for n in samless_p1 samless_p2 hardnet; do
    e=$([ $n = samless_p1 ] && echo 200 || ([ $n = samless_p2 ] && echo 100 || echo 300))
    f=$(ls "$RUNS"/results/*_${n}_fold$k/ckpt/net_last_$(printf %05d $((e - 1))).pth 2>/dev/null | tail -1 || true)
    echo "  ${n}_fold$k: ${f:+done}${f:-to train}"
done; done
[ "${1:-}" = "--dry-run" ] && { echo "[dry-run] then predictions, CV scoring and results/E7/summary.md"; exit 0; }
python "$REPO/scripts/gpu_check.py" || { echo "GPU looks busy; not launching" >&2; exit 1; }
cd "$REPO"
setsid nohup "$0" --worker >> "$LOG/job.log" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "launched, pid $(cat "$PIDFILE"), log $LOG/job.log"
