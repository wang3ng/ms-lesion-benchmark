# E1 — nnU-Net v2 3d_fullres on MSSEG-2016 (5 folds)

Status: **preprocessed, not launched** (2026-11-15).

## Setup

| Item | Value |
|---|---|
| Machine | shared lab RTX 4090 (24 GB), 24 CPU cores, 62 GB RAM |
| Environment | project `.venv`, Python 3.12.3, torch 2.11.0+cu130, nnunetv2 2.8.1 (`configs/requirements-lock-4090.txt`) |
| Dataset | `Dataset501_MSSEG2016`: 15 MSSEG training cases (5 Siemens Verio 3T, 5 Siemens Aera 1.5T, 5 Philips Ingenia 3T) |
| Inputs | FLAIR, T1, T2 (release-preprocessed; channels 0/1/2), z-score per channel within the nonzero mask |
| Label | MSSEG consensus (`lesion.nii.gz`) |
| Plans | default `nnUNetPlans`, `3d_fullres`: PlainConvUNet, 6 stages, patch 96×160×160, batch 2, spacing 1.1×0.744×0.744 mm ([nnUNetPlans.json](nnUNetPlans.json)) |
| Folds | patient-level, stratified by scanner (one patient per scanner per fold), seed 20261011 ([splits_final.json](splits_final.json)) |
| Trainer | `nnUNetTrainer` (1000 epochs) unless reduced uniformly, see below |
| Launcher | `scripts/run_E1.sh` → `scripts/nnunet_fold.sh`; logs `logs/E1/fold<k>.log` |

Reproduce from raw data:

```bash
source scripts/env.sh
python -m msbench.data.convert msseg2016
python -m msbench.nnunet_export
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres -np 8 --verify_dataset_integrity   # 43 s
python -m msbench.nnunet_export --splits
scripts/run_E1.sh
```

## Epoch budget

`nnUNetTrainerBenchmark_5epochs`, fold 0, 2026-11-15 (torch.compile on, GPU otherwise idle):
first epoch 68 s (includes compilation), then **32.4 s/epoch** (250 train + 50 val iterations).

| Epochs | Per fold | 5 folds, sequential |
|---|---|---|
| 1000 (default) | ≈ 9.0 h | ≈ 45 h (+ validation ≈ minutes per fold) |
| 500 | ≈ 4.5 h | ≈ 23 h |
| 250 | ≈ 2.3 h | ≈ 11 h |

_Chosen: to be filled when launched._

## Results

_To be filled after training: wall time per fold, validation Dice / lesion F1 per fold._
