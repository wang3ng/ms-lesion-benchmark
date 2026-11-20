# E8 — targeted fix selected by the pre-registered rule: zoomed tiles

Status (2026-11-19): **cross-validation done** (15 MSSEG training patients, E1's 5 folds). Comparison with E1,
E2a, E2b, E2c: [`results/E2/cv_summary.md`](../../results/E2/cv_summary.md). Test-set ensemble predictions
ready, not scored: [`experiments/test_stage/README.md`](../test_stage/README.md).

## Selection

Rule fixed in `configs/experiment.yaml` before any MedSAM training, applied to the cross-validation recalls
(no test data): recall(E2a, < 20 mm³) − recall(E2c, < 20 mm³) = 0.6692 − 0.1504 = 0.5188 ≥ 0.15 →
**zoomed tiles**.

## Setup (everything else identical to E2c)

| Item | Value |
|---|---|
| Boxes | E2c's whole-brain box (slice embedding) + E2c's 4 × 4 tiles (25 % overlap, inside the brain bounding box), tiles without brain voxels skipped |
| Zoom | each tile crop re-embedded on its own keep-aspect 1024 canvas (longer physical side → 1024 px, ≈ 4× the E2c canvas); prompt = whole crop |
| Embeddings | MSSEG training tiles cached: 48,015 tiles, 29.2 GPU-min, 94 GiB at `/home/kirby/msbench_e8_cache` (NVMe, mode 700; **delete after the project**, `TODO.md`). Test tiles computed on the fly (checked identical to the cache) |
| Training | 10,000 steps × 16, AdamW 1e-4, BCE on every box + Dice on lesion boxes, 50 % lesion boxes, collapse check; 6.2 min per fold |
| Inference | mean probability over covering boxes > 0.5 (as E2c) |

Collapse check: all 5 folds passed all 8 checks (lesion-box loss 0.16–0.18× its start; lesion tiles
non-empty 99.6–100 %; lesion-free tiles with some output 67–72 % on training batches).

## Cross-validation results (15 patients)

| | E1 | E2c | E8 | E8 − E2c [95% CI], Holm p |
|---|---|---|---|---|
| Dice | 0.767 | 0.457 | 0.513 | +0.056 [0.024, 0.087], 0.030 |
| lesion F1 | 0.693 | 0.210 | 0.296 | +0.086 [0.049, 0.126], 0.009 |
| lesion recall | 0.687 | 0.434 | 0.502 | +0.067 [0.017, 0.118], 0.16 |
| FP lesions / scan | 8.3 | 113.1 | 62.9 | −50.3 [−67.5, −32.7], 0.003 |
| pooled recall < 20 mm³ | 0.474 | 0.150 | 0.188 | +0.038 [0.000, 0.090] |
| pooled recall 20–100 mm³ | 0.714 | 0.349 | 0.440 | +0.091 [0.038, 0.150] |
| pooled recall ≥ 100 mm³ | 0.866 | 0.669 | 0.657 | −0.012 [−0.126, 0.099] |
| FP / scan < 20 mm³ | 5.5 | 94.9 | 50.0 | |

Zooming roughly halves E2c's false positives and helps medium lesions, but barely changes detection of
lesions < 20 mm³ (the problem the rule targeted). E8 stays far below E1 on every metric (lesion F1 −0.40,
Holm p < 0.001).

## Exploratory: predicted-IoU filter (post-hoc, decided after seeing E2c results)

Thresholds by nested CV inside each fold's training patients (0.30–0.475 for E8, 0.325–0.40 for E2c):
E8 lesion F1 0.296 → 0.298, FP / scan 62.9 → 61.7; E2c lesion F1 0.210 → 0.226, FP / scan 113.1 → 110.3.
Small effects; details in `results/E2/cv_summary.md`. CV only, not applied to the test sets.
