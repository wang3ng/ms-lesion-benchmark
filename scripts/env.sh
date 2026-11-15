# Project environment. Source it, do not execute it:
#     source scripts/env.sh
#
# Activates the project .venv and points nnU-Net at OUR folders.
# ~/.bashrc on the lab machine exports nnUNet_* for a lab mate's datasets;
# this file overrides them, so every launcher (including commands run inside
# tmux, which re-reads ~/.bashrc) must source it.

_msbench_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -x "$_msbench_repo/.venv/bin/python" ]; then
    echo "env.sh: $_msbench_repo/.venv not found; see README for setup" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "$_msbench_repo/.venv/bin/activate"

# Paths, dataset ID and seed come from configs/experiment.yaml.
eval "$(python -m msbench.config --shell)" || { echo "env.sh: could not read config" >&2; return 1 2>/dev/null || exit 1; }

export MSBENCH_RESULTS="$_msbench_repo/results"

# Test sandbox (scripts only, never for real runs): MSBENCH_SANDBOX=<dir> sends
# nnU-Net results, results/ and logs/ into <dir>, and puts <dir>/bin first on
# PATH so a fake nnUNetv2_train can stand in for the GPU.
if [ -n "${MSBENCH_SANDBOX:-}" ]; then
    export nnUNet_results="$MSBENCH_SANDBOX/nnUNet_results"
    export MSBENCH_RESULTS="$MSBENCH_SANDBOX/results"
    export MSBENCH_LOGS="$MSBENCH_SANDBOX/logs"
    export PATH="$MSBENCH_SANDBOX/bin:$PATH"
fi

# Guard: never let nnU-Net write into another user's folders.
for _v in nnUNet_raw nnUNet_preprocessed nnUNet_results; do
    case "${!_v}" in
        "$(dirname "$_msbench_repo")"/*) ;;
        "${MSBENCH_SANDBOX:-/nonexistent}"/*) ;;
        *) echo "env.sh: $_v=${!_v} is outside the project; refusing" >&2; return 1 2>/dev/null || exit 1 ;;
    esac
    mkdir -p "${!_v}"
done
unset _v _msbench_repo

# nnU-Net's default of many dataloader workers; 24 cores are shared.
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-12}"
