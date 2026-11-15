# CLAUDE.md — MS Lesion Benchmark (nnU-Net vs MedSAM)

Guidance for Claude Code working in this repository. Read this before making changes.

## Project in one paragraph

Course project (JHU Computer Vision, final report due **2026-12-07**). We test whether a medical foundation model (MedSAM) can *find* and *outline* small multiple sclerosis lesions as well as a specialist model (nnU-Net v2), including on an unseen scanner. The submitted proposal is the source of truth for the design; where `README.md`, `DEVELOPMENT_PLAN.md` or `configs/experiment.yaml` disagree with this file, this file reflects the newer design.

## Experiment design

| ID | What | Notes |
|---|---|---|
| E1 | nnU-Net v2 `3d_fullres`, 5 folds | Specialist baseline. Final numbers must come from 3D on the RTX 4090 |
| E2a | MedSAM + oracle boxes (from ground-truth lesions) | Upper bound on tracing; never the headline number |
| E2b | MedSAM + boxes around E1 detections | Does MedSAM refine nnU-Net's outlines? Depends on E1 |
| E2c | MedSAM + independent boxes: one whole-brain box per slice, and overlapping tiles inside the brain mask (include lesion-free tiles) | The only fair head-to-head with E1 |
| E8 | One targeted fix chosen from E2a vs E2c: smaller zoomed-in tiles (detection problem) or small-lesion-weighted loss (tracing problem) | Choose by a rule written down before looking at test data |
| E6 | Lesion count / volume agreement for E1 and the best MedSAM condition | MAE + Bland–Altman |

Each MedSAM box condition gets its **own fine-tune**, using the same box type at training and test time. MedSAM image encoder is frozen; only prompt encoder + mask decoder are trained. **Precompute and cache image embeddings** once per slice.

## Fixed decisions

- **Inputs:** FLAIR, T1, T2 only (shared by MSSEG, ISBI, MSLesSeg). Do not use T1-Gd or PD.
- **Seed:** 20261011. **Folds:** 5, patient-level. Never let slices or timepoints of one patient cross folds.
- **Test settings:** MSSEG in-distribution test (Siemens/Philips), MSSEG GE Discovery (8 unseen-scanner exams, report per case), ISBI 2015 (21 labeled scans, score each rater separately).
- **Metrics (one shared implementation in `src/msbench/metrics`):** voxel Dice; lesion-wise precision / recall / F1 under the MSSEG rule; F1 within lesion-volume bins whose edges are fixed from MSSEG **training-set** counts; false-positive lesions per scan; lesion count and volume (mL) agreement.
- **Statistics:** paired Wilcoxon signed-rank across patients, bootstrap 95% CIs.
- **Connectivity for lesion components:** pending verification against `animaSegPerfAnalyzer`; do not change the constant without updating the tests and noting it in the commit message.
- Test data are not used for any choice (thresholds, bins, epochs, fix selection).

## Data rules (data use agreements apply)

- Raw and processed data live **outside git** (see `data/README.md`); never commit images, masks, checkpoints or nnU-Net folders.
- Do not print, cat, or dump image/mask arrays or patient metadata into the conversation. Inspect data through summary statistics (shapes, spacing, voxel counts, lesion counts) only.
- Do not copy data to any cloud or shared location.

## Running long jobs

- This is a **shared lab GPU**. Run `nvidia-smi` before launching; do not kill other users' processes.
- Launch training in `tmux` (session name `msbench-<exp>-f<fold>`) or with `nohup`, logging to `logs/<exp>/fold<k>.log`. Never run multi-hour training in the foreground of a Claude session.
- nnU-Net resume: `nnUNetv2_train <ID> 3d_fullres <fold> --c`.
- Write one launcher script per experiment under `scripts/run_<exp>.sh`; it must be idempotent (skip finished folds, resume unfinished ones).
- After a job finishes, write a short summary to `experiments/<exp>/README.md` (config, epochs, wall time, GPU, validation metrics).
- If default 1000 epochs is too slow, reduce epochs **uniformly for all folds** and record it.

## Code conventions

- Python 3.11, package code in `src/msbench/`, tests in `tests/` (run `pytest -q` before committing).
- Configuration in `configs/experiment.yaml`; scripts read it rather than hard-coding paths or seeds.
- Work on a feature branch and open a PR; do not push directly to `main`. Four people share this repo.
- Keep commits small and messages descriptive.
