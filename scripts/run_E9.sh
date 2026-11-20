#!/usr/bin/env bash
# E9 (exploratory, after the results freeze): data scaling, one background job (idempotent).
#
#   scripts/run_E9.sh --dry-run
#   scripts/run_E9.sh
#
# Steps, in order (each logs start / end / exit code to logs/E9/job.log; the job continues past a failed step,
# and a step whose inputs are missing fails fast with a message):
#   1 counts             training-set counts -> results/E9/training_set.json
#   2 nnunet_prep        export Dataset502 (MSSEG train + MSLesSeg non-GE) + automatic planning/preprocessing
#   3 nnunet_train       nnUNetv2_train 502 3d_fullres all (1000 epochs, default trainer; resumes)
#   4 medsam_cache       MedSAM embeddings of the 84 MSLesSeg scans
#   5 e2c_train          MedSAM E2c, 82,500 steps, collapse check
#   6 hardnet_train      HarDNet Segmentor, 300 epochs x8 on 99 scans
#   7-9 predict_*        test-set predictions of the three models (MSSEG test incl. GE, ISBI)
#   10 summary           scoring + results/E9/summary.md
# Logs: logs/E9/{job.log, <step>.log}; PID logs/E9/job.pid
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO/scripts/env.sh"
LOG="$MSBENCH_LOGS/E9"; PIDFILE="$LOG/job.pid"; mkdir -p "$LOG"
DS=Dataset502_MSSEG_MSLesSeg; ID=502; CFGN=3d_fullres; TR=nnUNetTrainer
MODEL="$nnUNet_results/$DS/${TR}__nnUNetPlans__$CFGN/fold_all"
E9RUNS="$(python -c 'from msbench.config import paths; print(paths()["e9_runs"])')"

step() {   # step <name> <command...>
    local name="$1"; shift
    echo "=== $(date -Is) START $name"
    SECONDS=0
    "$@" >> "$LOG/$name.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then echo "=== $(date -Is) END $name: ok after $((SECONDS / 60)) min"
    else echo "=== $(date -Is) FAILED $name: exit $rc after $((SECONDS / 60)) min (see $LOG/$name.log); continuing"; fi
}
gpu() { python "$REPO/scripts/gpu_check.py" | sed 's/^/    /'; }
nnunet_prep() {
    python "$REPO/scripts/e9_steps.py" export-nnunet || return 1
    if [ -f "$nnUNet_preprocessed/$DS/nnUNetPlans.json" ] && [ -d "$nnUNet_preprocessed/$DS/nnUNetPlans_$CFGN" ]; then
        echo "already planned and preprocessed"; return 0; fi
    nnUNetv2_plan_and_preprocess -d $ID -c $CFGN --verify_dataset_integrity -np 8
}
nnunet_train() {
    [ -f "$MODEL/checkpoint_final.pth" ] && { echo "already trained"; return 0; }
    [ -f "$nnUNet_preprocessed/$DS/nnUNetPlans.json" ] || { echo "not preprocessed"; return 1; }
    nnUNetv2_train $ID $CFGN all -tr $TR --c
}
nnunet_predict() {
    [ -f "$MODEL/checkpoint_final.pth" ] || { echo "missing $MODEL/checkpoint_final.pth"; return 1; }
    local rc=0
    for s in msseg2016 isbi2015; do
        nnUNetv2_predict -i "$nnUNet_raw/Dataset501_MSSEG2016/imagesTs_$s" -o "$MSBENCH_RESULTS/E9_nnunet/test_predictions/$s" \
            -d $ID -c $CFGN -tr $TR -f all -chk checkpoint_final.pth --continue_prediction || rc=1
    done
    return $rc
}
e2c_train() {
    python "$REPO/scripts/cache_medsam_embeddings.py" --sets mslesseg_e9 --dry-run | tail -1 | grep -q "^84 already complete" \
        || { echo "MSLesSeg embedding cache incomplete"; return 1; }
    python "$REPO/scripts/e9_steps.py" e2c
}

if [ "${1:-}" = "--worker" ]; then
    set +e
    echo "=== $(date -Is) E9 job start (pid $$), commit $(git -C "$REPO" rev-parse --short HEAD), branch $(git -C "$REPO" branch --show-current)"
    step counts python "$REPO/scripts/e9_steps.py" counts
    step nnunet_prep nnunet_prep
    gpu; step nnunet_train nnunet_train
    gpu; step medsam_cache python "$REPO/scripts/cache_medsam_embeddings.py" --sets mslesseg_e9
    gpu; step e2c_train e2c_train
    gpu; step hardnet_train python "$REPO/scripts/e7_train.py" --model hardnet --fold 0 --dataset e9
    gpu; step predict_nnunet nnunet_predict
    step predict_e2c python "$REPO/scripts/e9_steps.py" predict-e2c
    step predict_hardnet python "$REPO/scripts/e9_steps.py" predict-hardnet
    step summary python "$REPO/scripts/e9_summary.py"
    echo "=== $(date -Is) E9 job done"
    exit 0
fi

free_gb=$(df -BG --output=avail "$MSBENCH_DATA_PROCESSED" | tail -1 | tr -dc 0-9)
echo "E9 plan ($DS, runs in $E9RUNS); free on the data disk: ${free_gb} GB (needs ~41 GB, require >= 100 GB)"
[ "$free_gb" -ge 100 ] || { echo "not enough free disk space" >&2; exit 1; }
if [ -f "$PIDFILE" ] && grep -q run_E9 "/proc/$(cat "$PIDFILE")/cmdline" 2>/dev/null; then
    echo "E9 job already running (pid $(cat "$PIDFILE"))" >&2; exit 1; fi
echo "  nnU-Net model:    $([ -f "$MODEL/checkpoint_final.pth" ] && echo done || echo 'to train (1000 epochs, ~9.5 h)')"
echo "  MedSAM cache:     $(python "$REPO/scripts/cache_medsam_embeddings.py" --sets mslesseg_e9 --dry-run | tail -1)"
echo "  E2c:              $([ -f "$E9RUNS/E9_E2c/checkpoint_final.pt" ] && echo done || echo 'to train (82,500 steps, ~55 min)')"
echo "  HarDNet:          $(ls "$E9RUNS"/results/*_hardnet_e9/ckpt/net_last_00299.pth >/dev/null 2>&1 && echo done || echo 'to train (29,700 iterations, ~2.8 h)')"
[ "${1:-}" = "--dry-run" ] && { echo "[dry-run] setsid nohup $0 --worker >> $LOG/job.log"; exit 0; }
python "$REPO/scripts/gpu_check.py" || { echo "GPU looks busy; not launching" >&2; exit 1; }
cd "$REPO"
setsid nohup "$0" --worker >> "$LOG/job.log" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "launched, pid $(cat "$PIDFILE"), log $LOG/job.log"
