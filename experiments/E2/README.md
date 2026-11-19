# E2 — MedSAM box conditions on the E1 cross-validation folds

Status (2026-11-19): **E2a, E2b and E2c done (cross-validation, 15 MSSEG training patients). E2c was rerun
with a corrected loss after its first run collapsed; the first run is kept as a documented failure. E8 not
started (awaiting review).** Report: [`results/E2/cv_summary.md`](../../results/E2/cv_summary.md)
(`python scripts/e2_summary.py --note experiments/E2/e2c_collapse_note.md`).

## Setup

| Item | Value |
|---|---|
| Model | MedSAM ViT-B (`medsam_vit_b.pth`, bowang-lab, 2023-08), `segment-anything` 1.0 |
| Inputs | axial slices with brain; FLAIR/T1/T2 as the 3 channels, each clipped to its 0.5–99.5 brain percentile → [0, 1]; keep-aspect 1024 × 1024 canvas |
| Embeddings | cached once (frozen encoder): 74 volumes, 14,180 slices, 13.3 GPU-min, 28 GB (`scripts/cache_medsam_embeddings.py`) |
| Trained | prompt encoder + mask decoder only, separately per condition and fold |
| Folds | E1's 5 patient-level folds (`experiments/E1/splits_final.json`); validation = 3 patients per fold |
| Training | 10,000 steps × 16 boxes, AdamW 1e-4 (wd 0.01, 200 warm-up), bf16; target = consensus inside the box; last checkpoint |
| Loss | BCE on every box + per-box soft Dice (squared) on boxes containing lesion voxels (`finetune.dice_on: lesion_boxes`); see "Post-hoc correction" |
| Collapse check | training batches only: lesion-box loss vs steps 1–100 (must be ≤ 0.8×) and share of lesion boxes with non-empty output (must be ≥ 0.10), at step 3,000 and every 1,000 after, pooled and per box kind (`finetune.collapse_check`) |
| E2a boxes | one per 2D 8-connected consensus component per slice, sides +U{0..5} voxels |
| E2b boxes | same, from the E1 out-of-fold prediction of each patient (training and validation patients; `experiments/E2b/PLAN.md`) |
| E2c boxes | per brain slice: brain bounding box + 4 × 4 tiles (25 % overlap), empty tiles included; training batches 50 % boxes with lesion voxels |
| Inference | per pixel: mean sigmoid over the boxes covering it, > 0.5; slices stacked to 3D |
| Scoring | `scripts/evaluate.py` (Anima-port rule) |
| Runtime | ≈ 6 min per fold incl. prediction (RTX 4090, 2.5 GB); E2a + first E2c 58 min (2026-11-17); E2c rerun + E2b 63 min (2026-11-19) |

## Results (15 patients, mean ± SD)

| | E1 nnU-Net | E2a oracle boxes | E2b E1 boxes | E2c brain box + tiles |
|---|---|---|---|---|
| Dice | 0.767 ± 0.082 | 0.757 ± 0.032 | 0.683 ± 0.073 | 0.457 ± 0.149 |
| lesion F1 | 0.693 ± 0.133 | 0.840 ± 0.117 | 0.619 ± 0.147 | 0.210 ± 0.103 |
| lesion recall | 0.687 ± 0.183 | 0.905 ± 0.093 | 0.659 ± 0.194 | 0.434 ± 0.150 |
| FP lesions / scan | 8.3 ± 4.8 | 12.7 ± 14.6 | 20.0 ± 16.4 | 113.1 ± 38.7 |
| pooled recall < 20 mm³ | 0.474 (63/133) | 0.669 (89/133) | 0.406 (54/133) | 0.150 (20/133) |
| pooled recall 20–100 mm³ | 0.714 | 0.964 | 0.698 | 0.349 |
| pooled recall ≥ 100 mm³ | 0.866 | 0.994 | 0.866 | 0.669 |

- **E2c (fair head-to-head) is far below E1** on every metric: Dice −0.31 [−0.36, −0.26], lesion F1 −0.48,
  +105 FP lesions per scan (Holm p ≤ 0.004). It trains properly now (all folds pass every collapse check;
  lesion-box loss falls to ≈ 0.28× its start). But on training patients it still outputs something in
  41–53 % of lesion-free tiles, and 84 % of its 1,697 FPs are < 20 mm³. BCE averaged over a 1024² mask barely
  penalises small blobs in empty tiles. It still misses 113/133 lesions < 20 mm³ (109 with no predicted
  voxel at all).
- **E2a** (boxes from the reference: an upper bound on tracing, not a fair comparison) traces medium and
  large lesions almost perfectly and recovers two thirds of lesions < 20 mm³.
- **E2b does not improve on E1**: Dice −0.084 [−0.096, −0.070] (Holm p 0.001), more FP lesions (20.0 vs 8.3,
  Holm p 0.013), recall similar (bounded by E1's). On training patients it keeps output in ≈ 92 % of the
  boxes around E1 false positives, so it does not clean up E1's false positives. Its extra FPs are mostly
  < 20 mm³ fragments (76 %).

## Post-hoc correction (first E2c run collapsed)

The first E2c run (2026-11-17) used per-box Dice + BCE on **every** box. It collapsed on the training folds:
the tiles learned to output nothing (fold 1 final checkpoint: 5 % of lesion-containing tiles non-empty on
training patients, folds 2/4: 0 % / 11 %), while the brain box over-segmented. Averaging probabilities over
covering boxes then gave near-empty (folds 1, 2, 4) or flooded (fold 0) validation predictions. On
2026-11-19, after that run and from training-fold behaviour only (no test data), the loss was changed to
BCE on every box + Dice only on boxes with lesion voxels. The same loss is used for E2a, E2b, E2c and E8. E2a's loss
is identical under both settings (every E2a box contains lesion voxels; tested) and it was not rerun.

My first diagnosis, "the ≈ 0.5 loss plateau = an empty output for every box", was wrong in detail: only the tiles
collapsed. A 3,000-step replay of the old loss on fold 1 still had 58 % of lesion boxes non-empty at step 3,000.
For that reason the collapse check is repeated to the end of training and applied to the brain box and the
tiles separately; a single pooled check at step 3,000 would have passed the first run. Details, first-run
numbers and box diagnostics: appendix of `results/E2/cv_summary.md`; first-run outputs kept under
`<medsam_runs>/E2c_run1`, `results/E2c_run1`, `logs/E2c_run1`.

## E8 decision rule (as configured), on the corrected E2c

recall(E2a, < 20 mm³) − recall(E2c, < 20 mm³) = 0.6692 − 0.1504 = **0.5188 ≥ 0.15 → zoomed tiles**
(paired cluster bootstrap 95% CI of the difference [0.381, 0.628]). Same selection as on the first run
(0.5940), so the collapse did not change the decision. Not started; awaiting review.
