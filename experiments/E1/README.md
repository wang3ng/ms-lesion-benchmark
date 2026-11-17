# E1 — nnU-Net v2 3d_fullres on MSSEG-2016 (5 folds)

Status: **done.** All 5 folds trained for 1000 epochs and evaluated (last fold 2026-11-17 12:02). 5-fold ensemble predictions on the evaluation sets generated 2026-11-17, **not scored yet**.

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
| Launcher | `scripts/run_E1.sh --folds "0 1"`, then `--folds "2 3 4"`: one background job each, folds in order, `scripts/evaluate.py` after each; logs `logs/E1/job.log`, `logs/E1/fold<k>.log` |
| Evaluation | `results/E1/fold<k>_cases.csv` (per case: Dice, lesion P/R/F1, FP lesions, size-bin recall/F1, count and volume); CV report: `scripts/e1_summary.py` → [`results/E1/cv_summary.md`](../../results/E1/cv_summary.md) |
| Test inference | `scripts/predict_E1_test.sh`: folds 0–4 ensemble (`checkpoint_final`, default mirroring TTA) → `results/E1/test_predictions/{msseg2016,isbi2015}/` (not committed) |

Reproduce from raw data:

```bash
source scripts/env.sh
python -m msbench.data.convert msseg2016
python -m msbench.nnunet_export
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres -np 8 --verify_dataset_integrity   # 43 s
python -m msbench.nnunet_export --splits
scripts/run_E1.sh --folds "0 1"     # then later: scripts/run_E1.sh --folds "2 3 4"
python scripts/midpoint_summary.py
python scripts/e1_summary.py       # pools all evaluated folds -> results/E1/cv_summary.md
scripts/predict_E1_test.sh         # ensemble inference on MSSEG test + ISBI (predictions only)
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

Two jobs (folds 0–1, then 2–4), both started from commit `f2fccd5`, 1000 epochs,
RTX 4090 with no other compute job running. Validation = the 3 held-out patients of each fold (one per
scanner), MSSEG consensus reference. Total training ≈ 46 GPU-hours.

| fold | wall time | s / epoch | nnU-Net val Dice | our Dice (mean of 3) | lesion F1 (mean of 3) | FP lesions / scan |
|---|---|---|---|---|---|---|
| 0 | 9 h 12 min | 32.4 | 0.753 | 0.753 | 0.682 | 5.0 |
| 1 | 9 h 10 min | 32.3 | 0.762 | 0.762 | 0.810 | 6.3 |
| 2 | 9 h 11 min | 32.4 | 0.838 | 0.838 | 0.629 | 6.7 |
| 3 | 9 h 10 min | 32.3 | 0.783 | 0.783 | 0.720 | 7.3 |
| 4 | 9 h 11 min | 32.3 | 0.700 | 0.700 | 0.761 | 5.3 |

Pooled over the 15 patients (mean ± SD, bootstrap 95% CI of the mean;
full report [`results/E1/cv_summary.md`](../../results/E1/cv_summary.md)):

| metric | value |
|---|---|
| Dice | 0.767 ± 0.082 [0.728, 0.807] |
| lesion precision | 0.806 ± 0.098 [0.755, 0.851] |
| lesion recall | 0.683 ± 0.185 [0.590, 0.770] |
| lesion F1 | 0.721 ± 0.128 [0.657, 0.783] |
| FP lesions / scan | 6.1 ± 4.6 [4.0, 8.5] |
| pooled recall <20 / 20–100 / ≥100 mm³ | 0.46 (58/127) / 0.72 (174/243) / 0.86 (147/171) |

Missed lesions are mostly small and mostly untouched by the prediction (148 of 162 missed lesions have
no predicted voxel). 63 of 92 false-positive lesions (68%) are < 20 mm³; Philips Ingenia has the most
FPs (10.6 / scan vs 3.2 Verio, 4.6 Aera).

Out-of-fold predictions for all 15 training patients (input for E2b): `$nnUNet_results/Dataset501_MSSEG2016/
nnUNetTrainer__nnUNetPlans__3d_fullres/fold_<k>/validation/<case_id>.{nii.gz,npz}`, copied to
`results/E1/oof_predictions/`; list in [`results/E1/tables/oof_predictions.csv`](../../results/E1/tables/oof_predictions.csv).

Midpoint report (folds 0–1, 6 patients): [`results/midpoint/summary.md`](../../results/midpoint/summary.md).
