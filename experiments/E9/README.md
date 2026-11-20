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
