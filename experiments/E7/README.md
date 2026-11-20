# E7 — MS-SAM-LESS and its HarDNet Segmentor baseline (EXPLORATORY, added after the results freeze)

Branch `e7-ms-sam-less` (from tag `results-frozen-v3`). Nothing on the frozen branch or tags is changed.
Report: [`results/E7/summary.md`](../../results/E7/summary.md).

## Method and code

- MS-SAMLess (Proietto Salanitri et al., *Pattern Recognit. Lett.* 2026): phase 1 trains a HarDNet-85 prompt
  learner that feeds dense prompts to a frozen SAM ViT-B; phase 2 drops SAM and trains a light ConvUpsample
  aggregator on the phase-1 backbone (SAM-free at inference). Baseline: the same HardNetSegmentor network from
  ImageNet weights, without phase 1.
- Code: github.com/perceivelab/MS-SAM-LESS @ `dde835c95aa2978030830515cace8479bc0a8eac` (MIT), cloned to
  `third_party/MS-SAM-LESS` (git-ignored) and used **unmodified**. Our adapter: `src/msbench/e7/` (split file,
  FLAIR/T1/T2 loader, x8 repeats, no-op wandb/TensorBoard, verified weights, inference mapped back to the case
  grid), `scripts/e7_train.py`, `scripts/e7_predict.py`, `scripts/run_E7.sh`, `scripts/e7_summary.py`;
  tests `tests/test_e7_adapter.py`.
- Weights (outside git, `paths.sam_vit_b` / `paths.hardnet85`): Meta SAM ViT-B `sam_vit_b_01ec64.pth`
  (md5 01ec64d2…, sha256 ec2df627…); HarDNet-85 ImageNet `hardnet85-a28faa00.pth` from ping-chao.com
  (sha256 a28faa00…). Both load as plain tensors with `weights_only=True` and match the architectures strictly;
  their code's download call is replaced by a loader that only serves the verified HarDNet file.

## Settings (configs/experiment.yaml `e7`)

Published (README) hyperparameters: SGD lr 0.01 (momentum 0), weight decay 1e-4, batch 8, gradient accumulation
2, DiceCE, 2 output channels, HarDNet-85, AMP, seed 42, RAS axial slices. Phase 1: 200 epochs, SAM ViT-B, 1024 px
SAM input / 512 px prompt-learner input, 85 % / 15 % lesion / background slice sampling. Phase 2: 100 epochs,
512 px, ConvUpsample head, backbone fine-tuned, initialised from phase 1, 75 % / 25 % sampling (their defaults).
Baseline: as phase 2 but ImageNet-initialised, 300 epochs.

## Deviations from the paper (approved 2026-11-20)

1. Inputs: FLAIR, T1, T2 stacked as three channels (each z-scored and scaled to [0, 1]); paper: FLAIR only.
2. Each training scan 8x per epoch (12 scans per fold -> 12 updates per epoch, ~2,400 phase-1 updates), to roughly
   match the updates of their MSLesSeg setup.
3. Last-epoch checkpoint, no per-epoch validation and no checkpoint selection on validation patients (paper: best
   validation IoU), as E1/E2.
4. HarDNet baseline: phase 1 + phase 2 = 300 epochs at x8, so the comparison isolates SAM's guidance.
5. Inference also resampled to 1 mm (their training resamples, their inference does not; MSSEG voxels are
   anisotropic) and mapped back to each case's grid (round trip of the reference masks: Dice 0.96 Verio,
   0.95 GE, 1.00 ISBI — the 1 mm processing of the method itself loses some sub-millimetre detail).
6. Test sets: 5-fold mean-probability ensemble (paper: single model).
7. Engineering only: all training scans cached in RAM (cache_rate 1.0); wandb/TensorBoard replaced by no-op
   stand-ins; slice-image logging off; Python 3.12 / torch 2.11 / MONAI 1.6.1 (paper: 3.8 / 1.13 / 1.3).

## Smoke test (2026-11-20, fold 0, 2 epochs each)

Phase 1 0.62 s per update (SAM forward at 1024 px), phase 2 / baseline 0.33 s per update; phase 2 loads the
phase-1 backbone and decoder (only its new head is initialised); predictions 7 s per case per model. After
2 epochs the phase-2 model already gives lesion voxels a higher foreground probability than background
(0.27 vs 0.20 on a training case). Estimated total: ~4.5 GPU-h training + ~0.5 h predictions.
