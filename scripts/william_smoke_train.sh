#!/usr/bin/env bash
# Minimal 2D nnU-Net smoke train on a few synthetic cases if real data
# is not ready, or on converted MSLesSeg FLAIR slices when present.
set -euo pipefail

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msbench

ROOT="${HOME}/projects/ms-lesion-benchmark"
export nnUNet_raw="${ROOT}/data/nnUNet_raw"
export nnUNet_preprocessed="${ROOT}/data/nnUNet_preprocessed"
export nnUNet_results="${ROOT}/data/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

python - <<'PY'
"""Build Dataset999_Smoke: 8 tiny 2D FLAIR-like volumes for a GPU smoke train."""
from pathlib import Path
import json
import numpy as np
import SimpleITK as sitk

root = Path.home() / "projects/ms-lesion-benchmark"
ds = root / "data/nnUNet_raw/Dataset999_Smoke"
images = ds / "imagesTr"
labels = ds / "labelsTr"
images.mkdir(parents=True, exist_ok=True)
labels.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
# Small axial stacks so 6 GB VRAM is enough for 2d training.
shape = (32, 128, 128)  # z, y, x
spacing = (5.0, 1.0, 1.0)

for i in range(1, 9):
    case = f"smoke_{i:03d}"
    img = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    # A few small "lesions"
    for _ in range(3):
        z = int(rng.integers(4, shape[0] - 4))
        y = int(rng.integers(20, shape[1] - 20))
        x = int(rng.integers(20, shape[2] - 20))
        mask[z - 1 : z + 2, y - 2 : y + 3, x - 2 : x + 3] = 1
        img[mask == 1] += 2.5

    def save(arr, path, is_label=False):
        # SimpleITK expects x,y,z
        sitk_img = sitk.GetImageFromArray(arr)
        sitk_img.SetSpacing(spacing[::-1])
        if is_label:
            sitk_img = sitk.Cast(sitk_img, sitk.sitkUInt8)
        sitk.WriteImage(sitk_img, str(path))

    save(img, images / f"{case}_0000.nii.gz")
    save(mask, labels / f"{case}.nii.gz", is_label=True)

dataset = {
    "channel_names": {"0": "FLAIR"},
    "labels": {"background": 0, "lesion": 1},
    "numTraining": 8,
    "file_ending": ".nii.gz",
}
(ds / "dataset.json").write_text(json.dumps(dataset, indent=2))
print(f"Wrote {ds} with 8 cases")
PY

echo "Planning Dataset999_Smoke..."
nnUNetv2_plan_and_preprocess -d 999 -c 2d --verify_dataset_integrity

echo "Starting 5-epoch 2d smoke train (fold 0)..."
nnUNetv2_train 999 2d 0 -tr nnUNetTrainer_5epochs

echo "SMOKE_TRAIN_OK"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
