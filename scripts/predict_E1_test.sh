#!/usr/bin/env bash
# E1 5-fold ensemble inference on the evaluation sets. Generates predictions only; scores nothing.
#
#   scripts/predict_E1_test.sh             # background job: MSSEG test (incl. GE) then ISBI 2015
#   scripts/predict_E1_test.sh --dry-run   # show the plan, launch nothing
#   scripts/predict_E1_test.sh --force     # launch even if the GPU looks busy
#
# Inputs:  $nnUNet_raw/<dataset>/imagesTs_{msseg2016,isbi2015}  (python -m msbench.nnunet_export --test)
# Model:   all five folds of E1, checkpoint_final.pth, nnU-Net default test-time mirroring.
# Output:  results/E1/test_predictions/<set>/<case_id>.nii.gz  (+ .npz/.pkl probabilities)
#          results/E1/test_predictions/manifest.csv
# Logs:    logs/E1/predict_test.log, PID logs/E1/predict.pid
# Re-running is safe: nnU-Net's --continue_prediction skips cases that already have an output.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO/scripts/env.sh"

DRY=0; FORCE=0; WORKER=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --force) FORCE=1; shift ;;
        --worker) WORKER=1; shift ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

EXP=E1
SETS="msseg2016 isbi2015"
TRAINER="$MSBENCH_NNUNET_TRAINER"
MODEL="$nnUNet_results/$MSBENCH_NNUNET_DATASET/${TRAINER}__nnUNetPlans__${MSBENCH_NNUNET_CONFIG}"
RAW="$nnUNet_raw/$MSBENCH_NNUNET_DATASET"
OUT="$MSBENCH_RESULTS/$EXP/test_predictions"
LOGDIR="$MSBENCH_LOGS/$EXP"
LOG="$LOGDIR/predict_test.log"
PIDFILE="$LOGDIR/predict.pid"
mkdir -p "$LOGDIR"

n_inputs() { find "$RAW/imagesTs_$1" -name '*_0000.nii.gz' | wc -l; }
n_outputs() { [ -d "$OUT/$1" ] && find "$OUT/$1" -maxdepth 1 -name '*.nii.gz' | wc -l || echo 0; }
job_alive() { [ -f "$PIDFILE" ] && grep -q "predict_E1_test" "/proc/$(cat "$PIDFILE")/cmdline" 2>/dev/null; }

# ------------------------------------------------------------------ worker
if [ "$WORKER" -eq 1 ]; then
    set +e
    echo "=== $(date -Is) predict start (pid $$): $MODEL folds 0-4 checkpoint_final, commit $(git -C "$REPO" rev-parse --short HEAD)"
    status=0
    for s in $SETS; do
        echo "=== $(date -Is) $s: $(n_inputs "$s") cases -> $OUT/$s"
        python "$REPO/scripts/gpu_check.py" | sed 's/^/    /'
        SECONDS=0
        nnUNetv2_predict -i "$RAW/imagesTs_$s" -o "$OUT/$s" -d "$MSBENCH_NNUNET_ID" -c "$MSBENCH_NNUNET_CONFIG" \
            -tr "$TRAINER" -f 0 1 2 3 4 -chk checkpoint_final.pth --save_probabilities --continue_prediction
        rc=$?
        echo "=== $(date -Is) $s: exit $rc after $((SECONDS / 60)) min, $(n_outputs "$s") / $(n_inputs "$s") predictions"
        [ "$rc" -eq 0 ] && [ "$(n_outputs "$s")" -eq "$(n_inputs "$s")" ] || status=1
    done
    python - "$OUT" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
from msbench.config import load_config
from msbench.nnunet_export import TEST_SETS, test_cases
out = Path(sys.argv[1]); cfg = load_config(); rows = []
for name in TEST_SETS:
    for r in test_cases(cfg, name).itertuples():
        pred = out / name / f"{r.case_id}.nii.gz"
        rows.append({"case_id": r.case_id, "dataset": r.dataset, "release_split": r.release_split,
                     "scanner": r.scanner, "subject_id": r.subject_id,
                     "prediction": str(pred.relative_to(out)), "exists": pred.exists()})
m = pd.DataFrame(rows); m.to_csv(out / "manifest.csv", index=False)
print(f"manifest: {int(m.exists.sum())} / {len(m)} predictions present -> {out / 'manifest.csv'}")
EOF
    echo "=== $(date -Is) predict done (status $status)"
    exit $status
fi

# ------------------------------------------------------------------ launcher
for k in 0 1 2 3 4; do
    [ -f "$MODEL/fold_$k/checkpoint_final.pth" ] || { echo "missing $MODEL/fold_$k/checkpoint_final.pth" >&2; exit 1; }
done
python -m msbench.nnunet_export --test
if job_alive; then echo "a prediction job is already running (pid $(cat "$PIDFILE")); see $LOG" >&2; exit 1; fi
echo "E1 ensemble inference: folds 0-4 of $MODEL"
work=0
for s in $SETS; do
    echo "  $s: $(n_outputs "$s") / $(n_inputs "$s") predictions exist"
    [ "$(n_outputs "$s")" -lt "$(n_inputs "$s")" ] && work=1
done
[ "$work" -eq 1 ] || { echo "nothing to do"; exit 0; }
if [ "$DRY" -eq 1 ]; then echo "[dry-run] nohup $0 --worker >> $LOG"; exit 0; fi
if [ "$FORCE" -eq 0 ]; then
    python "$REPO/scripts/gpu_check.py" || { echo "GPU looks busy; not launching (use --force to override)" >&2; exit 1; }
fi
cd "$REPO"
setsid nohup "$0" --worker >> "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
sleep 3
job_alive || { echo "job did not start; see $LOG" >&2; exit 1; }
echo "launched (nohup), pid $(cat "$PIDFILE"), log $LOG"
