# E2 — MedSAM box conditions on the E1 cross-validation folds

Status (2026-11-17): **E2a done and valid; E2c ran but its training collapsed (see below) — under
review. E2b and E8 not started.** Comparison report: [`results/E2/cv_summary.md`](../../results/E2/cv_summary.md)
(`python scripts/e2_summary.py --note experiments/E2/e2c_collapse_note.md`).

## Setup

| Item | Value |
|---|---|
| Model | MedSAM ViT-B (`medsam_vit_b.pth`, bowang-lab, 2023-08), `segment-anything` 1.0 |
| Inputs | axial slices with brain; FLAIR/T1/T2 as the 3 channels, each clipped to its 0.5–99.5 brain percentile → [0, 1]; keep-aspect 1024 × 1024 canvas |
| Embeddings | cached once (frozen encoder): 74 volumes, 14,180 slices, 13.3 GPU-min, 28 GB (`scripts/cache_medsam_embeddings.py`) |
| Trained | prompt encoder + mask decoder only, separately per condition and fold |
| Folds | E1's 5 patient-level folds (`experiments/E1/splits_final.json`); validation = 3 patients per fold |
| Training | 10,000 steps × 16 boxes, AdamW 1e-4 (wd 0.01, 200 warm-up), bf16; loss = per-box soft Dice (squared) + BCE on the 1024 mask, target = consensus inside the box; last checkpoint |
| E2a boxes | one per 2D 8-connected consensus component per slice, sides +U{0..5} voxels |
| E2c boxes | per brain slice: brain bounding box + 4 × 4 tiles (25 % overlap), empty tiles included; training batches 50 % boxes with lesion voxels |
| Inference | per pixel: mean sigmoid over the boxes covering it, > 0.5; slices stacked to 3D |
| Scoring | `scripts/evaluate.py` (Anima-port rule) |
| Runtime | 5.3–5.5 min per fold incl. prediction; whole E2a + E2c job 58 min on the RTX 4090 (2.5 GB GPU memory) |

## Results (15 patients, mean ± SD)

| | E1 nnU-Net | E2a oracle boxes | E2c brain box + tiles |
|---|---|---|---|
| Dice | 0.767 ± 0.082 | 0.757 ± 0.032 | 0.170 ± 0.233 |
| lesion F1 | 0.693 ± 0.133 | 0.840 ± 0.117 | 0.067 ± 0.082 |
| lesion recall | 0.687 ± 0.183 | 0.905 ± 0.093 | 0.139 ± 0.208 |
| FP lesions / scan | 8.3 ± 4.8 | 12.7 ± 14.6 | 41.8 ± 70.8 |
| pooled recall < 20 mm³ | 0.474 (63/133) | 0.669 (89/133) | 0.075 (10/133) |

E2a (boxes from the reference: an upper bound on tracing, not a fair comparison) traces medium and large
lesions almost perfectly (pooled recall 0.96 and 0.99) and recovers two thirds of lesions < 20 mm³ even
when told where they are; 92 % of its false positives are < 20 mm³ fragments inside the oracle boxes.

## E2c: training collapse (defect of our setup)

- Loss: every E2c fold plateaus at ≈ 0.49 from step 1,000 to 10,000 (fold 0–4 final: 0.43, 0.46, 0.46,
  0.39, 0.48); E2a falls steadily to 0.15–0.18.
- 0.5 is the loss of the constant "empty mask" solution: per-box soft Dice with smoothing 1e-5 is ≈ 0 on
  an empty tile only if the output is (almost) exactly empty and ≈ 1 for any visible output, and ≈ 1 on
  the lesion boxes if nothing is predicted; with 50 % empty boxes the mean is ≈ 0.5 (BCE adds little).
- Predictions agree: folds 1 and 2 predict essentially nothing for all six patients (0–4 tiny components),
  fold 0 lands just above the 0.5 threshold and floods (145–224 FP per patient), folds 3–4 in between.
- Cause: MedSAM's loss assumes every box contains the target; we combined it with empty tiles. E2a has no
  empty boxes and is not affected.

### Options (need a decision before any rerun; ≈ 30 GPU-min for E2c, 5 folds)

1. **Recommended:** keep BCE on every box, compute Dice only on boxes whose target is not empty. Standard
   for segmentation with empty samples; leaves E2a's training identical (all its boxes contain lesions),
   so E2a does not need a rerun. Applies unchanged to E2b and E8.
2. Batch-level Dice (one Dice over all boxes in the batch) + BCE. Also avoids the empty-box blow-up, but
   changes E2a's loss too (E2a would be rerun for consistency).
3. Drop empty tiles from training. Contradicts the protocol ("include empty tiles so MedSAM learns to
   output an empty mask") and would inflate E2c false positives.

This would be a fix chosen from the training-fold behaviour of E2c (no test data involved) and will be
recorded in the config and the report as a post-hoc correction.

## E8 decision rule (as configured), on the current numbers

recall(E2a, < 20 mm³) − recall(E2c, < 20 mm³) = 0.669 − 0.075 = 0.594 ≥ 0.15 → **zoomed tiles**.
Not to be acted on until E2c is fixed: with a collapsed E2c the gap is inflated by construction.
