#!/usr/bin/env bash
# E1: nnU-Net v2 3d_fullres, 5 patient-level folds on MSSEG-2016 training.
#
#   scripts/run_E1.sh                 # launch all unfinished folds
#   scripts/run_E1.sh --dry-run       # show what would run, launch nothing
#   scripts/run_E1.sh --folds "0 1"   # subset
#   scripts/run_E1.sh --parallel 2    # two folds at once on the GPU (default 1)
#   scripts/run_E1.sh --force         # launch even if the GPU looks busy
#
# Idempotent:
#   - a fold with validation/summary.json is finished and skipped;
#   - every other fold runs `nnUNetv2_train ... --c`, which starts fresh when
#     there is no checkpoint, resumes from checkpoint_latest, or reruns only
#     the validation when checkpoint_final exists;
#   - a fold whose tmux session is still alive is left alone.
# One tmux session per fold (msbench-E1-f<k>); log appended to logs/E1/fold<k>.log.
# With --parallel P the sessions start together but queue on P GPU-slot locks,
# so at most P folds train at once.
# Epochs come from configs/experiment.yaml (models.nnunet.epochs) and apply to
# every fold; change them only before fold 0 starts (CLAUDE.md).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO/scripts/env.sh"

FOLDS="0 1 2 3 4"
PARALLEL=1
DRY=0
FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --folds) FOLDS="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

EXP=E1
EPOCHS="$(python -c 'from msbench.config import load_config; print(load_config()["models"]["nnunet"]["epochs"])')"
if [ "$EPOCHS" = "1000" ]; then TRAINER=nnUNetTrainer; else TRAINER="nnUNetTrainer_${EPOCHS}epochs"; fi
python - "$TRAINER" <<'EOF'
import os, sys, nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
name = sys.argv[1]
if recursive_find_python_class(os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                               name, "nnunetv2.training.nnUNetTrainer") is None:
    sys.exit(f"nnU-Net has no trainer class {name}; pick epochs with an existing variant")
EOF

PRE="$nnUNet_preprocessed/$MSBENCH_NNUNET_DATASET"
OUT="$nnUNet_results/$MSBENCH_NNUNET_DATASET/${TRAINER}__nnUNetPlans__${MSBENCH_NNUNET_CONFIG}"
LOGDIR="$MSBENCH_LOGS/$EXP"
mkdir -p "$LOGDIR"

[ -f "$PRE/nnUNetPlans.json" ] || { echo "missing $PRE/nnUNetPlans.json; run plan_and_preprocess first" >&2; exit 1; }
[ -f "$PRE/splits_final.json" ] || { echo "missing $PRE/splits_final.json; run: python -m msbench.nnunet_export --splits" >&2; exit 1; }
cmp -s "$PRE/splits_final.json" "$REPO/experiments/E1/splits_final.json" \
    || { echo "splits_final.json differs from experiments/E1/splits_final.json; refusing" >&2; exit 1; }

echo "E1: $MSBENCH_NNUNET_DATASET $MSBENCH_NNUNET_CONFIG trainer=$TRAINER epochs=$EPOCHS folds=[$FOLDS] parallel=$PARALLEL"
echo "results: $OUT"

# tmux if installed (CLAUDE.md), otherwise setsid+nohup with a pidfile.
if command -v tmux >/dev/null; then LAUNCHER=tmux; else LAUNCHER=nohup; fi

running() {  # running <fold>
    local k="$1" pidf="$LOGDIR/fold$1.pid"
    if [ "$LAUNCHER" = tmux ] && tmux has-session -t "=msbench-$EXP-f$k" 2>/dev/null; then return 0; fi
    [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null \
        && grep -q nnunet_fold "/proc/$(cat "$pidf")/cmdline" 2>/dev/null
}

todo=()
for k in $FOLDS; do
    if [ -f "$OUT/fold_$k/validation/summary.json" ]; then
        echo "  fold $k: finished, skip"
    elif running "$k"; then
        echo "  fold $k: already running, skip"
    else
        state="new"
        [ -f "$OUT/fold_$k/checkpoint_latest.pth" ] && state="resume"
        [ -f "$OUT/fold_$k/checkpoint_final.pth" ] && state="validation only"
        echo "  fold $k: launch ($state)"
        todo+=("$k")
    fi
done
[ ${#todo[@]} -eq 0 ] && { echo "nothing to launch"; exit 0; }

if [ "$DRY" -eq 1 ]; then
    for i in "${!todo[@]}"; do
        k="${todo[$i]}"
        echo "[dry-run] $LAUNCHER msbench-$EXP-f$k: scripts/nnunet_fold.sh $k $TRAINER .gpu_slot$((i % PARALLEL)) $LOGDIR/fold$k.log --npz"
    done
    exit 0
fi

if [ "$FORCE" -eq 0 ]; then
    # nvidia-smi can fail on this machine (driver/library mismatch); gpu_check.py
    # uses torch + /proc to see whether anyone else holds the GPU.
    python "$REPO/scripts/gpu_check.py" || { echo "GPU looks busy; not launching (use --force to override)" >&2; exit 1; }
fi

for i in "${!todo[@]}"; do
    k="${todo[$i]}"
    sess="msbench-$EXP-f$k"
    log="$LOGDIR/fold$k.log"
    slot="$LOGDIR/.gpu_slot$((i % PARALLEL))"
    if [ "$LAUNCHER" = tmux ]; then
        tmux new-session -d -s "$sess" -c "$REPO" \
            "$REPO/scripts/nnunet_fold.sh" "$k" "$TRAINER" "$slot" "$log" --npz
        tmux list-panes -t "=$sess" -F '#{pane_pid}' > "$LOGDIR/fold$k.pid"
    else
        (cd "$REPO" && setsid nohup "$REPO/scripts/nnunet_fold.sh" "$k" "$TRAINER" "$slot" "$log" --npz \
            > /dev/null 2>&1 < /dev/null & echo $! > "$LOGDIR/fold$k.pid")
    fi
    echo "  launched $sess ($LAUNCHER, pid $(cat "$LOGDIR/fold$k.pid")) -> $log"
    sleep 2   # start in fold order so the slot queue is roughly ordered
done

cat <<EOF

Monitor:  tail -f $LOGDIR/fold<k>.log   |   python scripts/gpu_check.py
          (tmux: tmux ls / tmux attach -t msbench-$EXP-f<k>, detach Ctrl-b d)
Stop:     kill -- -\$(cat $LOGDIR/fold<k>.pid)     # whole process group of that fold
Resume:   re-run scripts/run_E1.sh (finished folds are skipped, others continue with --c)
EOF
