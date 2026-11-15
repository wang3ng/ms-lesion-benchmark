#!/usr/bin/env bash
# Train (or resume, or finish validation of) one nnU-Net fold. Called by
# scripts/run_<exp>.sh inside a tmux session; not meant to be run by hand.
#
#   nnunet_fold.sh <fold> <trainer> <gpu_slot_lockfile> <logfile> [extra nnUNetv2_train args...]
set -uo pipefail

FOLD="$1"; TRAINER="$2"; SLOT="$3"; LOG="$4"; shift 4
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# tmux starts from ~/.bashrc, which points nnUNet_* at a lab mate's folders.
# shellcheck disable=SC1091
source "$REPO/scripts/env.sh" || exit 1

exec >> "$LOG" 2>&1
echo "=== $(date -Is) fold $FOLD queued for GPU slot $(basename "$SLOT")"
exec 9> "$SLOT"
flock 9
echo "=== $(date -Is) fold $FOLD start: $MSBENCH_NNUNET_DATASET $MSBENCH_NNUNET_CONFIG -tr $TRAINER $*"
echo "=== host $(hostname), nnunetv2 $(python -c 'import importlib.metadata as m; print(m.version("nnunetv2"))'), torch $(python -c 'import torch; print(torch.__version__)')"
SECONDS=0
nnUNetv2_train "$MSBENCH_NNUNET_ID" "$MSBENCH_NNUNET_CONFIG" "$FOLD" -tr "$TRAINER" --c "$@"
rc=$?
echo "=== $(date -Is) fold $FOLD exit $rc after $((SECONDS / 3600))h$(((SECONDS % 3600) / 60))m"
exit $rc
