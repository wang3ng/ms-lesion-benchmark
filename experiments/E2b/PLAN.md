# E2b — MedSAM with boxes from E1 detections (plan, not run)

Question: given nnU-Net's detections as prompts, does MedSAM trace lesions better than nnU-Net, and
does it drop some of nnU-Net's false positives? E2b cannot find lesions that E1 missed (no box), so
its lesion recall is bounded by E1's; the comparison is about outlines (Dice, per-lesion Dice of
detected lesions) and precision.

## Boxes

- Source: an E1 prediction, split into 2D 8-connected components per axial slice (canonical
  orientation, same as E2a). One box per component, each side pushed out by U{0..5} voxels
  (`boxes.nnunet` in `configs/experiment.yaml`); training draws new jitter per sample, inference
  uses a seeded draw. Implemented: `msbench.medsam.train` condition `nnunet`.
- **Training patients of fold k**: their E1 out-of-fold masks, `results/E1/oof_predictions/`
  (each made by the E1 fold model that did not see that patient). Using OOF rather than in-sample
  E1 predictions matters: in-sample masks are near-perfect and would teach MedSAM to trust boxes
  far more than it should at test time.
- **Validation patients of fold k**: their E1 OOF mask, which comes from E1 fold k, i.e. the model
  trained on exactly the 12 patients that train E2b fold k. Clean.
- Known caveat (standard for stacking on OOF predictions): the OOF mask of a *training* patient of
  fold k was produced by an E1 model whose training set contained fold k's validation patients.
  Information about them reaches E2b only through the box positions on other patients; we accept
  this and note it. The strict alternative (nested E1 folds) costs 20 more nnU-Net runs (≈ 180 GPU-h).
- **Test sets** (later, after the test protocol is unlocked): boxes from the E1 5-fold ensemble
  predictions in `results/E1/test_predictions/`; the 5 E2b fold models each predict, probabilities
  averaged (mirrors the nnU-Net ensemble). To be written into the config before any test scoring.

## Training and inference

Identical to E2a/E2c (`models.medsam.finetune`, `.inference`): prompt encoder + mask decoder only,
4000 steps × 16, AdamW 1e-4, Dice + BCE on the consensus restricted to the box, 50% boxes with lesion
voxels per batch (E1 false-positive components give the empty boxes), last checkpoint, mean
probability over covering boxes > 0.5. Same 5 folds as E1.

## Evaluation

`scripts/evaluate.py --exp E2b --fold k --pred-dir <medsam_runs>/E2b/fold_k/validation` (Anima-port
rule), then pooled CV like `scripts/e1_summary.py`. Extra analyses: (1) paired Wilcoxon vs E1 per
patient (Dice, precision, F1); (2) Dice per detected lesion, E2b vs E1, by size bin; (3) share of
E1 false positives that E2b turns into empty masks.

## Run (after OK)

```bash
scripts/run_E2.sh --exps E2b --dry-run
scripts/run_E2.sh --exps E2b
```
Estimated GPU time: similar to E2a (see the E2a/E2c benchmark in `experiments/E2/README.md`).
