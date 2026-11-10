#!/usr/bin/env bash
set -euo pipefail

if [ ! -x "$HOME/miniconda3/bin/conda" ]; then
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"

if ! conda env list | grep -q '^msbench '; then
  conda create -y -n msbench python=3.11
fi
conda activate msbench
python -V
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
pip install nnunetv2 nibabel SimpleITK tqdm pyyaml connected-components-3d
python -c "import nnunetv2; print('nnunetv2 ok')"
