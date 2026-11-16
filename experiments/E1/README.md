# E1 — nnU-Net v2 3d_fullres on MSSEG-2016 (5 folds)

Status: **folds 0–1 finished and evaluated (2026-11-16 07:20); folds 2–4 training** (one job, started 2026-11-16 08:28, expected to end ≈ 2026-11-17 12:00).

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
| Launcher | `scripts/run_E1.sh --folds "0 1"`: one background job, folds in order, `scripts/evaluate.py` after each; logs `logs/E1/job.log`, `logs/E1/fold<k>.log` |
| Evaluation | `results/E1/fold<k>_cases.csv` (per case: Dice, lesion P/R/F1, FP lesions, size-bin recall/F1, count and volume); summary: `scripts/midpoint_summary.py` → `results/midpoint/` |

Reproduce from raw data:

```bash
source scripts/env.sh
python -m msbench.data.convert msseg2016
python -m msbench.nnunet_export
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres -np 8 --verify_dataset_integrity   # 43 s
python -m msbench.nnunet_export --splits
scripts/run_E1.sh --folds "0 1"     # then later: scripts/run_E1.sh --folds "2 3 4"
python scripts/midpoint_summary.py
python scripts/e1_summary.py       # pools all evaluated folds -> results/E1/summary/
```

## Epoch budget

`nnUNetTrainerBenchmark_5epochs`, fold 0, 2026-11-15 (torch.compile on, GPU otherwise idle):
first epoch 68 s (includes compilation), then **32.4 s/epoch** (250 train + 50 val iterations).

| Epochs | Per fold | 5 folds, sequential |
|---|---|---|
| 1000 (default) | ≈ 9.0 h | ≈ 45 h (+ validation ≈ minutes per fold) |
| 500 | ≈ 4.5 h | ≈ 23 h |
| 250 | ≈ 2.3 h | ≈ 11 h |

**Chosen: 1000 epochs (nnU-Net default), unchanged for all folds** (decided 2026-11-15 before any fold started).

## Results

Job 1 (folds 0, 1), commit `f2fccd5`, 1000 epochs, RTX 4090 with no other compute job running.
Validation = the 3 held-out patients of each fold (one per scanner), MSSEG consensus reference.

| fold | wall time | s / epoch | nnU-Net val Dice | our Dice (mean of 3) | lesion F1 (mean of 3) | FP lesions / scan |
|---|---|---|---|---|---|---|
| 0 | 9 h 12 min | 32.4 | 0.753 | 0.753 | 0.682 | 5.0 |
| 1 | 9 h 11 min | 32.3 | 0.762 | 0.762 | 0.810 | 6.3 |

Midpoint report (folds 0–1, 6 patients): [`results/midpoint/summary.md`](../../results/midpoint/summary.md).
Pooled cross-validation over all 15 patients (mean ± SD, bootstrap 95% CI, recall by size bin):
`python scripts/e1_summary.py` -> `results/E1/summary/summary.md`, to be filled in here after fold 4.
