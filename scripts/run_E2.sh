#!/usr/bin/env bash
# E2 MedSAM conditions: fine-tune + predict + evaluate, one background job.
#
#   scripts/run_E2.sh --exps "E2a E2c"              # E2a folds 0-4, then E2c folds 0-4
#   scripts/run_E2.sh --exps "E2c E2b"              # E2b starts only if every E2c fold passed
#   scripts/run_E2.sh --exps E2a --folds "0 1"
#   scripts/run_E2.sh --dry-run                     # show the plan, launch nothing
#   scripts/run_E2.sh --force                       # launch even if the GPU looks busy
#
# For each (exp, fold), in order: scripts/train_medsam.py (train or resume, then predict the
# fold's 3 validation patients), then scripts/evaluate.py -> results/<exp>/fold<k>_cases.csv.
# A failed fold is logged and the job moves on to the next fold of the same experiment.
# Gate: a later experiment starts only if every fold of the earlier ones passed the collapse
# check (models.medsam.finetune.collapse_check) and was predicted. A fold whose collapse check
# failed (collapse_check.json, train_medsam.py exit 3) is never resumed; delete its run dir to
# retry. Re-running is safe: finished folds are skipped, unfinished ones resume.
# Logs: logs/<exp>/fold<k>.log, logs/E2/job.log; PID logs/E2/job.pid
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO/scripts/env.sh"

EXPS="E2a E2c"; FOLDS="0 1 2 3 4"; DRY=0; FORCE=0; WORKER=0
while [ $# -gt 0 ]; do
    case "$1" in
        --exps) EXPS="$2"; shift 2 ;;
        --folds) FOLDS="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --force) FORCE=1; shift ;;
        --worker) WORKER=1; shift ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

RUNS="$(python -c 'from msbench.config import paths; print(paths()["medsam_runs"])')"
JOBDIR="$MSBENCH_LOGS/E2"; PIDFILE="$JOBDIR/job.pid"; mkdir -p "$JOBDIR"
val_cases() { python -c "import json; print(' '.join(json.load(open('$REPO/experiments/E1/splits_final.json'))[$1]['val']))"; }
predicted() { local d="$RUNS/$1/fold_$2/validation"; for c in $(val_cases "$2"); do [ -f "$d/$c.nii.gz" ] || return 1; done; }
evaluated() { local f="$MSBENCH_RESULTS/$1/fold$2_cases.csv"; [ -f "$f" ] && [ "$f" -nt "$RUNS/$1/fold_$2/validation" ]; }
collapsed() { local f="$RUNS/$1/fold_$2/collapse_check.json"
    [ -f "$f" ] && python -c "import json,sys; sys.exit(1 if json.load(open('$f'))['passed'] else 0)"; }
job_alive() { [ -f "$PIDFILE" ] && grep -q "run_E2" "/proc/$(cat "$PIDFILE")/cmdline" 2>/dev/null; }

if [ "$WORKER" -eq 1 ]; then
    set +e
    echo "=== $(date -Is) job start (pid $$): exps [$EXPS] folds [$FOLDS], commit $(git -C "$REPO" rev-parse --short HEAD)"
    for e in $EXPS; do
        mkdir -p "$MSBENCH_LOGS/$e"
        bad=""
        for k in $FOLDS; do
            log="$MSBENCH_LOGS/$e/fold$k.log"
            if collapsed "$e" "$k"; then
                echo "=== $(date -Is) $e fold $k: COLLAPSE CHECK FAILED (see $RUNS/$e/fold_$k/collapse_check.json)"
                bad="$bad $k"; continue
            fi
            if ! predicted "$e" "$k"; then
                echo "=== $(date -Is) $e fold $k: train + predict (log $log)"
                python "$REPO/scripts/gpu_check.py" | sed 's/^/    /'
                SECONDS=0
                python "$REPO/scripts/train_medsam.py" --exp "$e" --fold "$k" >> "$log" 2>&1
                rc=$?
                echo "=== $(date -Is) $e fold $k: exit $rc after $((SECONDS / 60)) min"
                if collapsed "$e" "$k"; then
                    echo "=== $(date -Is) $e fold $k: COLLAPSE CHECK FAILED; fold stopped, not predicted"
                    grep -A3 "collapse check at step" "$log" | tail -3 | sed 's/^/    /'
                    bad="$bad $k"; continue
                fi
            fi
            if predicted "$e" "$k"; then
                if evaluated "$e" "$k"; then
                    echo "=== $(date -Is) $e fold $k: already evaluated"
                else
                    python "$REPO/scripts/evaluate.py" --exp "$e" --fold "$k" \
                        --pred-dir "$RUNS/$e/fold_$k/validation" --cases $(val_cases "$k") >> "$log" 2>&1
                    rc=$?
                    echo "=== $(date -Is) $e fold $k: evaluation exit $rc"
                fi
            else
                echo "=== $(date -Is) $e fold $k: NOT predicted; skipping evaluation (see $log)"
                bad="$bad $k"
            fi
        done
        if [ -n "$bad" ]; then
            echo "=== $(date -Is) $e: folds [${bad# }] collapsed or failed; not starting later experiments"
            break
        fi
    done
    echo "=== $(date -Is) job done"
    exit 0
fi

[ -f "$REPO/experiments/E1/splits_final.json" ] || { echo "missing experiments/E1/splits_final.json" >&2; exit 1; }
python "$REPO/scripts/cache_medsam_embeddings.py" --dry-run | tail -1 | grep -q "^74 already complete" \
    || { echo "MedSAM embedding cache incomplete; run scripts/cache_medsam_embeddings.py" >&2; exit 1; }
if job_alive; then echo "an E2 job is already running (pid $(cat "$PIDFILE"))" >&2; exit 1; fi
echo "E2: exps [$EXPS] folds [$FOLDS] -> $RUNS"
work=0
for e in $EXPS; do for k in $FOLDS; do
    if collapsed "$e" "$k"; then s="COLLAPSE CHECK FAILED (delete run dir to retry)"
    elif predicted "$e" "$k"; then
        if evaluated "$e" "$k"; then s="done"; else s="evaluate"; work=1; fi
    elif [ -f "$RUNS/$e/fold_$k/checkpoint_final.pt" ]; then s="predict + evaluate"; work=1
    elif [ -f "$RUNS/$e/fold_$k/checkpoint_latest.pt" ]; then s="resume + predict + evaluate"; work=1
    else s="train + predict + evaluate"; work=1; fi
    echo "  $e fold $k: $s"
done; done
[ "$work" -eq 1 ] || { echo "nothing to do"; exit 0; }
[ "$DRY" -eq 1 ] && { echo "[dry-run] nohup $0 --worker --exps \"$EXPS\" --folds \"$FOLDS\" >> $JOBDIR/job.log"; exit 0; }
if [ "$FORCE" -eq 0 ]; then
    python "$REPO/scripts/gpu_check.py" || { echo "GPU looks busy; not launching (use --force)" >&2; exit 1; }
fi
cd "$REPO"
setsid nohup "$0" --worker --exps "$EXPS" --folds "$FOLDS" >> "$JOBDIR/job.log" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
sleep 3
job_alive || tail -3 "$JOBDIR/job.log" | grep -q "job start" || { echo "job did not start; see $JOBDIR/job.log" >&2; exit 1; }
echo "launched (nohup), pid $(cat "$PIDFILE"), log $JOBDIR/job.log"
