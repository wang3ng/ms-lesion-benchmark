#!/usr/bin/env bash
# E1: nnU-Net v2 3d_fullres on MSSEG-2016 training, patient-level folds.
#
#   scripts/run_E1.sh --folds "0 1"     # one background job: fold 0, then fold 1
#   scripts/run_E1.sh                   # all five folds, one after another
#   scripts/run_E1.sh --dry-run         # show the plan, launch nothing
#   scripts/run_E1.sh --force           # launch even if the GPU looks busy
#
# One background job runs the requested folds in order. For each fold:
#   1. skip training if validation/summary.json exists (fold finished);
#      otherwise `nnUNetv2_train ... --c` (fresh start, resume, or validation only);
#   2. if the fold finished, run scripts/evaluate.py --exp E1 --fold k
#      -> results/E1/fold<k>_cases.csv.
# A failed fold is logged and the job moves on to the next fold. Re-running this
# script is safe: finished folds are skipped, unfinished ones resume.
#
# Logs:  logs/E1/job.log (job events), logs/E1/fold<k>.log (nnU-Net + evaluation)
# PID:   logs/E1/job.pid (process group leader; stop with: kill -- -$(cat logs/E1/job.pid))
# Uses tmux (session msbench-E1-f<folds>) if installed, otherwise setsid + nohup.
# Epochs come from configs/experiment.yaml (models.nnunet.epochs), same for all folds.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO/scripts/env.sh"

FOLDS="0 1 2 3 4"
DRY=0
FORCE=0
WORKER=0
while [ $# -gt 0 ]; do
    case "$1" in
        --folds) FOLDS="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --force) FORCE=1; shift ;;
        --worker) WORKER=1; shift ;;
        -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

EXP=E1
TRAINER="$MSBENCH_NNUNET_TRAINER"
EPOCHS="$(python -c 'from msbench.config import load_config; print(load_config()["models"]["nnunet"]["epochs"])')"
PRE="$nnUNet_preprocessed/$MSBENCH_NNUNET_DATASET"
OUT="$nnUNet_results/$MSBENCH_NNUNET_DATASET/${TRAINER}__nnUNetPlans__${MSBENCH_NNUNET_CONFIG}"
LOGDIR="$MSBENCH_LOGS/$EXP"
PIDFILE="$LOGDIR/job.pid"
mkdir -p "$LOGDIR"

finished() { [ -f "$OUT/fold_$1/validation/summary.json" ]; }
evaluated() { [ -f "$MSBENCH_RESULTS/$EXP/fold$1_cases.csv" ] && [ "$MSBENCH_RESULTS/$EXP/fold$1_cases.csv" -nt "$OUT/fold_$1/validation/summary.json" ]; }
job_alive() { [ -f "$PIDFILE" ] && grep -q "run_E1" "/proc/$(cat "$PIDFILE")/cmdline" 2>/dev/null; }

# ------------------------------------------------------------------ worker
if [ "$WORKER" -eq 1 ]; then
    set +e
    echo "=== $(date -Is) job start (pid $$): folds [$FOLDS], trainer $TRAINER, $EPOCHS epochs, commit $(git -C "$REPO" rev-parse --short HEAD)"
    for k in $FOLDS; do
        log="$LOGDIR/fold$k.log"
        if finished "$k"; then
            echo "=== $(date -Is) fold $k already finished, no training"
        else
            echo "=== $(date -Is) fold $k training (log $log)"
            python "$REPO/scripts/gpu_check.py" | sed 's/^/    /'
            "$REPO/scripts/nnunet_fold.sh" "$k" "$TRAINER" "$log" --npz
            rc=$?
            echo "=== $(date -Is) fold $k training exit $rc"
        fi
        if finished "$k"; then
            if evaluated "$k"; then
                echo "=== $(date -Is) fold $k already evaluated"
            else
                echo "=== $(date -Is) fold $k evaluation"
                python "$REPO/scripts/evaluate.py" --exp "$EXP" --fold "$k" >> "$log" 2>&1
                rc=$?
                echo "=== $(date -Is) fold $k evaluation exit $rc$( [ $rc -eq 0 ] && echo " -> $MSBENCH_RESULTS/$EXP/fold${k}_cases.csv")"
            fi
        else
            echo "=== $(date -Is) fold $k NOT finished; skipping evaluation (see $log)"
        fi
    done
    echo "=== $(date -Is) job done"
    exit 0
fi

# ------------------------------------------------------------------ launcher
[ -f "$PRE/nnUNetPlans.json" ] || { echo "missing $PRE/nnUNetPlans.json; run plan_and_preprocess first" >&2; exit 1; }
[ -f "$PRE/splits_final.json" ] || { echo "missing $PRE/splits_final.json; run: python -m msbench.nnunet_export --splits" >&2; exit 1; }
cmp -s "$PRE/splits_final.json" "$REPO/experiments/E1/splits_final.json" \
    || { echo "splits_final.json differs from experiments/E1/splits_final.json; refusing" >&2; exit 1; }
python - "$TRAINER" <<'EOF'
import os, sys, nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
if recursive_find_python_class(os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                               sys.argv[1], "nnunetv2.training.nnUNetTrainer") is None:
    sys.exit(f"nnU-Net has no trainer class {sys.argv[1]}")
EOF

if job_alive; then
    echo "an E1 job is already running (pid $(cat "$PIDFILE")); see $LOGDIR/job.log" >&2
    exit 1
fi

echo "E1: $MSBENCH_NNUNET_DATASET $MSBENCH_NNUNET_CONFIG trainer=$TRAINER epochs=$EPOCHS folds=[$FOLDS] (sequential)"
echo "results: $OUT"
work=0
for k in $FOLDS; do
    if finished "$k"; then
        if evaluated "$k"; then echo "  fold $k: finished, evaluated"; else echo "  fold $k: finished, will evaluate"; work=1; fi
    elif [ -f "$OUT/fold_$k/checkpoint_final.pth" ]; then echo "  fold $k: validation only, then evaluate"; work=1
    elif [ -f "$OUT/fold_$k/checkpoint_latest.pth" ]; then echo "  fold $k: resume, then evaluate"; work=1
    else echo "  fold $k: train from scratch, then evaluate"; work=1
    fi
done
[ "$work" -eq 1 ] || { echo "nothing to do"; exit 0; }

if command -v tmux >/dev/null; then LAUNCHER=tmux; else LAUNCHER=nohup; fi
SESSION="msbench-$EXP-f$(echo "$FOLDS" | tr -d ' ')"
if [ "$DRY" -eq 1 ]; then
    echo "[dry-run] $LAUNCHER ($SESSION): scripts/run_E1.sh --worker --folds \"$FOLDS\"  >> $LOGDIR/job.log"
    exit 0
fi

if [ "$FORCE" -eq 0 ]; then
    python "$REPO/scripts/gpu_check.py" || { echo "GPU looks busy; not launching (use --force to override)" >&2; exit 1; }
fi

touch "$LOGDIR/job.log"
log_lines_before=$(wc -l < "$LOGDIR/job.log")
if [ "$LAUNCHER" = tmux ]; then
    tmux new-session -d -s "$SESSION" -c "$REPO" "'$0' --worker --folds '$FOLDS' >> '$LOGDIR/job.log' 2>&1"
    tmux list-panes -t "=$SESSION" -F '#{pane_pid}' > "$PIDFILE"
else
    cd "$REPO"
    setsid nohup "$0" --worker --folds "$FOLDS" >> "$LOGDIR/job.log" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
fi
sleep 3
# Alive, or already finished quickly after logging its start (e.g. evaluation only).
job_alive || tail -n +"$((log_lines_before + 1))" "$LOGDIR/job.log" | grep -q "job start" \
    || { echo "job did not start; see $LOGDIR/job.log" >&2; exit 1; }
echo "launched $SESSION ($LAUNCHER), pid $(cat "$PIDFILE"), log $LOGDIR/job.log"
