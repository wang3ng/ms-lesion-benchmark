# E9 — data scaling with MSSEG + MSLesSeg (EXPLORATORY, added after the results freeze; MSLesSeg labels follow a different annotation protocol)

Branch `e9-data-scaling` (from `e7-ms-sam-less`, which provides the HarDNet adapter). Nothing frozen is changed.
Report: [`results/E9/summary.md`](../../results/E9/summary.md).

## Training set (`msbench.e9.training_cases`, counts in `results/E9/training_set.json`)

| | scans | patients | scanners |
|---|---|---|---|
| MSSEG-2016 training | 15 | 15 | Siemens Verio 3T 5, Siemens Aera 1.5T 5, Philips Ingenia 3T 5 |
| MSLesSeg training (kept) | 84 | 49 | Philips Ingenia 64, Philips Achieva dStream 20 |
| **combined** | **99** | **64** | no GE |
| MSLesSeg excluded (per scan) | 9 | 8 | unknown 5, GE Signa HDxt 2, GE Signa Explorer 1, GE Optima MR450w 1 |

GE and unknown-scanner MSLesSeg scans are excluded per scan, so the GE test set stays unseen (4 patients keep their
other timepoints). MSLesSeg: 1 mm MNI-space, skull-stripped, FLAIR/T1/T2, no brain mask (MedSAM uses FLAIR != 0, as
for ISBI). The MSLesSeg test release is not used. All 99 scans checked: one grid per case, binary non-empty labels.

## Models (one per method, no cross-validation) — each method follows its own standard recipe

- **nnU-Net 3d_fullres**: dataset 502 (`Dataset502_MSSEG_MSLesSeg`), automatic planning on the combined data,
  default trainer, 1000 epochs (standard 250 iterations per epoch, so fewer updates per scan than E1), fold `all`.
- **MedSAM E2c**: same recipe, loss (Dice on lesion boxes + BCE) and collapse check; new embedding cache for the 84
  MSLesSeg scans (12,815 slices, ~27 GB); budget scaled to keep the same updates per scan as the 12-scan CV
  folds: 10,000 × 99/12 = 82,500 steps.
- **HarDNet Segmentor** (E7 baseline): same recipe and deviations as E7 (FLAIR/T1/T2, each scan 8× per epoch,
  last epoch, 300 epochs), so iterations scale with the data (29,700) and updates per scan stay the same.

MedSAM and HarDNet budgets keep the same updates per scan; nnU-Net keeps its standard 1000 epochs (fewer updates
per scan).

## Comparison

Each E9 model is scored on the frozen test sets with the frozen scripts' rules and paired with its frozen counterpart
(E9 nnU-Net vs E1, E9 E2c vs E2c, E9 HarDNet vs E7 HarDNet); Holm within each comparison and test set. The frozen
counterparts are 5-fold ensembles of 12-patient models, E9 models are single models. ISBI is resampled by patient
(5 patients) and is a less external test for E9: MSLesSeg, like ISBI, is Philips data in 1 mm MNI space.
`msseg_test_c07_p08` (empty consensus) is reported separately.

## Pre-launch checks (2026-11-20)

Integrity of all 99 training scans; one MSLesSeg embedding cached (157 slices, 8 s) and 60 E2c steps on an
MSSEG + MSLesSeg pair (no error, 0.08 s/step with 2 workers); HarDNet 1 epoch on the full split (99 iterations,
0.28 s/it, 1.3 min incl. caching 99 scans). Estimate: ~13.5 h total (nnU-Net ~9.5 h, cache 12 min, E2c ~55 min,
HarDNet ~2.3 h, test inference ~30 min).

## Run (2026-11-20 → 21, `scripts/run_E9.sh`)

| step | result | time |
|---|---|---|
| nnU-Net plan + preprocess | automatic plan: 1 mm isotropic, patch 128³, batch 2 (E1: 1.1 × 0.74 × 0.74 mm, 96 × 160 × 160) | 1 min |
| nnU-Net training (1000 epochs, 27 s/epoch, incl. fold-`all` final validation) | ok | 7 h 57 min |
| MedSAM cache (84 MSLesSeg scans) | ok, ~27 GB | 11 min |
| E2c (82,500 steps) | ok; all 81 collapse checks passed | 49 min |
| HarDNet (29,700 iterations) | ok | 2 h 25 min |
| test predictions (3 single models, 59 scans each) | ok | 8 min |
| summary | failed once on a bug in `e9_summary.py` (a column named `empty` collided with pandas' `DataFrame.empty`); fixed and rerun | 5 min |

## Results (exploratory; details, CIs and Holm-adjusted tests in `results/E9/summary.md`)

| Lesion F1 | nnU-Net: frozen E1 → E9 | MedSAM E2c: frozen → E9 | HarDNet: E7 → E9 |
|---|---|---|---|
| MSSEG in-distribution | 0.725 → 0.726 | 0.235 → 0.256 | 0.611 → 0.415 (Holm p < 0.001) |
| MSSEG GE | 0.648 → 0.695 | 0.154 → 0.147 | 0.550 → 0.446 |
| ISBI rater 1 / 2 | 0.649 → 0.622 / 0.616 → 0.607 | 0.079 → 0.146 / 0.076 → 0.127 | 0.511 → 0.373 / 0.450 → 0.331 |

- **nnU-Net**: one model on 99 scans matches the 5-fold ensemble of 12-patient models (MSSEG in-distribution lesion F1
  +0.001; Dice −0.013, Holm p 0.017); on GE small-lesion recall 0.13 → 0.27 and lesion F1 +0.05 (not significant,
  8 scans).
- **MedSAM E2c**: essentially unchanged on MSSEG (still ~80 FP per scan); better on ISBI (Dice +0.12/+0.13,
  not significant with 5 patients; ISBI is the domain closest to MSLesSeg).
- **HarDNet**: clearly worse. Diagnosis (no tuning done): it under-segments MSSEG scans — median predicted/reference
  volume 0.36 vs 0.89 for the E7 models — and small-lesion recall collapses (0.25 → 0.06). Its training loss ends
  where E7's does (0.36 vs 0.37), so the model fits the combined data, 85 % of which is 1 mm MNI-space MSLesSeg with
  its own label protocol; nnU-Net (own resampling and normalisation) does not show this.
