#!/usr/bin/env bash
# Train (or resume, or finish validation of) one nnU-Net fold, in the foreground.
# Called by scripts/run_<exp>.sh; not meant to be run by hand.
#
#   nnunet_fold.sh <fold> <trainer> <logfile> [extra nnUNetv2_train args...]
#
# `--c` is always passed: nnU-Net starts fresh when there is no checkpoint,
# resumes from checkpoint_latest, or only reruns validation when
# checkpoint_final exists.
set -uo pipefail

FOLD="$1"; TRAINER="$2"; LOG="$3"; shift 3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# ~/.bashrc points nnUNet_* at a lab mate's folders; always use ours.
# shellcheck disable=SC1091
source "$REPO/scripts/env.sh" || exit 1

{
    echo "=== $(date -Is) fold $FOLD start: $MSBENCH_NNUNET_DATASET $MSBENCH_NNUNET_CONFIG -tr $TRAINER $*"
    echo "=== host $(hostname), nnunetv2 $(python -c 'import importlib.metadata as m; print(m.version("nnunetv2"))'), torch $(python -c 'import torch; print(torch.__version__)')"
} >> "$LOG"
SECONDS=0
nnUNetv2_train "$MSBENCH_NNUNET_ID" "$MSBENCH_NNUNET_CONFIG" "$FOLD" -tr "$TRAINER" --c "$@" >> "$LOG" 2>&1
rc=$?
echo "=== $(date -Is) fold $FOLD exit $rc after $((SECONDS / 3600))h$(((SECONDS % 3600) / 60))m" >> "$LOG"
exit $rc
