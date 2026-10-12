# Development plan

This plan turns the repository brief into a sequence of work packages.
Each package has an output and an exit test. Later packages do not start
because a model "looks promising"; they start because the previous exit
test passed.

The scientific target is a table, not a new architecture. nnU-Net and
MedSAM are scored on the same masks, with the same lesion-wise F1, under
a scanner split that was defined by the MSSEG-2016 challenge. Quantitative
count and volume agreement is a separate table on the same predictions.

## 1. Claims we will and will not make

We will claim only what the splits support:

- A difference between voxel Dice and lesion-wise F1, overall and inside
  the pre-specified punctate bin (equivalent-sphere diameter under 3 mm).
- A difference between in-scanner MSSEG test cases and the GE Discovery 3T
  cases that were never in the training set.
- Agreement between manual and automatic lesion count and lesion volume
  on those held-out cases.
- A transfer result between MSSEG and ISBI 2015, labeled as cross-dataset
  and single-direction unless both directions are run.

We will not claim clinical readiness, a new state of the art on the closed
MSSEG leaderboard, or a scanner effect estimated from MS3SEG alone (one
Toshiba scanner). Oracle-box MedSAM numbers are an upper bound. They are
never placed in the same column as fully automatic nnU-Net without a
label.

If Shanoir never releases the 38-case test set, the GE comparison is
reported as blocked. Open-data results still ship. They are not rewritten
as the cross-scanner study.

## 2. Hypotheses

These are stated so a negative result is still a result.

- **H1.** Lesion-wise F1, and especially punctate F1, will separate the
  models by more than voxel Dice does. Dice can stay high when small
  lesions are missed and large lesions dominate the voxel count.
- **H2.** Both models will lose more detection F1 than Dice when moving
  from seen MSSEG scanners to GE Discovery 3T. The foundation model is
  not assumed to win this gap; that is the measurement.
- **H3.** Rank correlation of lesion volume will be high even when count
  correlation is moderate, because missed punctate lesions move the count
  and barely move the volume.
- **H4 (secondary).** A model trained on MSSEG preprocessed data will lose
  more F1 on ISBI 2015 than on held-out MSSEG cases from seen scanners.
  The reverse direction is reported if compute remains.

H2 is testable only with the official GE cases. H1, H3, and the open-data
half of the project are testable on MSLesSeg or MS3SEG.

## 3. Decisions already made

| Decision | Choice | Reason |
| --- | --- | --- |
| Primary cohort | MSSEG-2016 official train/test | GE Discovery is a real unseen scanner, not a random fold. |
| Development cohort | MS3SEG NIfTI first, MSLesSeg if we implement MS-SAM-LESS | MS3SEG is ~2 GB and needs no approval. MSLesSeg matches a published prompt-learning baseline. |
| Transfer cohort | ISBI 2015 training subjects | Longitudinal, two raters, different site, one Philips 3T. |
| nnU-Net config | `3d_fullres` on 3D cohorts, `2d` on MS3SEG | Thick-slice 5.7 mm stacks do not support a 3D kernel. |
| MedSAM interface | Automatic prompts; oracle boxes on the side | Manual boxes would make the comparison semi-automatic. |
| MedSAM geometry | 2D slices, stacked to 3D before metrics | Published MedSAM is 2D. Scoring in 3D keeps F1 comparable to nnU-Net. |
| Ground truth on MSSEG | LOP STAPLE consensus shipped with the challenge | Re-fusing the seven raters is out of scope. |
| Ground truth on ISBI | Rater 1 for training; score both raters at test | Two-rater disagreement is a ceiling, not something to average away silently. |
| MS3SEG target | Abnormal WMH only | Normal WMH and ventricles are not MS lesions. |
| Size bin | 3 mm equivalent diameter, fixed now | Changing the cut after seeing scores is not allowed. |
| Preprocessing | Use each dataset's released preprocessed images | A private registration pipeline would confound scanner effects. |

Anything in this table changes only by a short note in `experiments/`
explaining why, dated before the affected run.

## 4. Work packages

### WP0 — Environment and data access

**Output.** A GPU environment that imports `torch` and `nnunetv2`, plus a
`data/raw/<id>/SOURCE.txt` for every archive actually downloaded.

**Tasks.**

1. Create the venv in the README and confirm `nvidia-smi` sees the GPU.
2. Request MSSEG-2016 on Shanoir the same day, because approval is the
   long pole. Do not block WP1–WP3 on that request.
3. Download public archives with `python scripts/download_datasets.py`
   (ISBI 2015, MSLesSeg, and 3D-MR-MS). URLs are in
   `configs/downloads.yaml`. Do not commit the zip files.
4. Download MS3SEG preprocessed NIfTI from its Figshare record if that
   fallback is still the first training cohort.
5. Download ISBI 2015 training and, when disk allows, the 2.2 GB test
   archive from the IACL links in the manifest.

**Exit.** `torch.cuda.is_available()` is true on the training machine.
At least one open dataset is unpacked under `data/raw/`. The Shanoir
request is filed, even if it is still pending.

### WP1 — Common case layout

**Output.** `data/processed/<dataset>/<case_id>/` as specified in
`data/README.md`, and a CSV index with columns
`case_id, dataset, subject_id, timepoint, scanner, site, spacing_mm`.

**Tasks.**

1. Write one converter per dataset. MSSEG maps the four scanners onto the
   names in `configs/experiment.yaml`. ISBI keeps subject and timepoint.
   MS3SEG reads the abnormal-WMH mask and ignores ventricle and normal-WMH
   masks. MSLesSeg follows the official split file when present.
2. Check that image and mask share shape, affine, and orientation. Fail
   the case rather than resampling quietly inside the converter.
3. Write a one-page QC note: case count, spacing, foreground voxel
   fraction, number of connected components, and any case dropped.

**Exit.** Case counts match the release notes (MS3SEG 100, MSLesSeg 75 or
the 53-patient training split, MSSEG 15 or 53 depending on what Shanoir
delivered, ISBI 5 subjects). A second person, or a fresh shell, can
rebuild the index from raw data with one command.

### WP2 — Metrics, before any training

**Output.** `src/msbench` functions for voxel Dice, lesion-wise detection
F1, punctate F1, lesion count, and lesion volume in mL, with tests on
synthetic blobs.

**Detection rule.** Follow MSSEG-2016 (Commowick et al., 2018):

- Components are 26-connected in 3D.
- A reference lesion is a true positive detection only if overlap with
  the prediction is sufficient **and** the predicted component that
  creates the overlap is not much larger than the reference lesion.
- The symmetric count on predicted components supplies the precision term.
- Sensitivity = detected reference lesions / reference lesions.
  Precision = accepted predicted lesions / predicted lesions.
  F1 is the harmonic mean. An empty reference and an empty prediction
  yield F1 = 1. An empty reference and a non-empty prediction yield F1 = 0.

The numeric overlap fraction and the size-ratio cutoff are copied from
the challenge paper into the test file as named constants. If the paper
and the challenge website disagree, the website's evaluation description
wins, and the choice is written down in the test module.

**Synthetic cases the tests must include.**

- Identical masks: Dice 1, F1 1, count error 0, volume error 0.
- Both empty: Dice 1, F1 1.
- One voxel lesion deleted: Dice may stay high on a large companion
  lesion; punctate F1 must fall.
- One predicted blob covering several reference lesions: those reference
  lesions must not all count as detected if the size-ratio rule fails.
- Volume scales with voxel spacing, not with voxel count. A 1 mm isotropic
  voxel and a 0.9 × 0.9 × 5.73 mm MS3SEG-like voxel of the same label
  count are different millilitres.

**Exit.** `pytest` passes. No network weight is loaded in this package.

### WP3 — nnU-Net on the open cohort

**Output.** Experiment `E1-nnunet-open`. Five-fold nnU-Net on MS3SEG
(`2d`, abnormal WMH) or on MSLesSeg (`3d_fullres`), using the dataset
fingerprint the planner produces.

**Tasks.**

1. Export `processed/` into `data/nnUNet_raw` with the channel order
   FLAIR, then T1, T2, PD, T1-Gd when the file exists. Missing optional
   channels are omitted, not zero-filled.
2. Run fingerprinting, planning, and preprocessing. Save the generated
   plans file into the experiment directory.
3. Train all five folds, or fewer only if the machine cannot finish a
   fold in a weekend, in which case document the reduced plan before
   training starts.
4. Predict the held-out folds, stack nothing (nnU-Net is already 3D or
   native 2D), and score with WP2.

**Exit.** A CSV of per-case Dice, F1, punctate F1, count, and volume,
plus the plans file and the nnU-Net version. This is the number MedSAM
has to beat under the same split.

### WP4 — MedSAM on the same open split

**Output.** Experiment `E2-medsam-open`, same cases and same metrics as
WP3.

**Prompt policy (automatic).** Freeze this rule in the experiment config
before training:

1. Run the frozen MedSAM image encoder on each axial slice.
2. Build prompts without looking at the test mask. The default policy is
   a coarse foreground proposal from a frozen copy of the WP3 nnU-Net
   (boxes around predicted components, plus a negative point in
   background). That uses nnU-Net as a prompter, so it is a pipeline, and
   the write-up must say so.
3. A second automatic policy, trained prompt embeddings in the style of
   MS-SAM-LESS, may replace step 2. It must not use the test mask either.
4. Fine-tune the prompt encoder and mask decoder. Update the image
   encoder only if 24 GB VRAM still has headroom after a decoder-only run.
5. Stack slice masks into a 3D volume on the original grid. Remove
   components smaller than one voxel. Then call the WP2 metrics.

**Oracle condition.** Boxes from reference components, same decoder. Report
in a separate column named `medsam_oracle_box`.

**Exit.** The automatic-prompt row and the nnU-Net row share a per-case
CSV joined on `case_id`. Oracle numbers are present and labeled.

### WP5 — MSSEG in-scanner and cross-scanner

**Output.** Experiments `E3-msseg-inscanner` and `E4-msseg-ge`.

This package waits on the 38-case test set. If only 15 training cases
arrive, stop after a sanity fine-tune and do not estimate a scanner
effect from random splits of those 15.

**When the full set is available.**

1. Train nnU-Net `3d_fullres` and the WP4 MedSAM pipeline on the 15
   official training cases (Siemens Verio, Siemens Aera, Philips
   Ingenia). Do not add the 8 GE cases "just for balance."
2. Split the 38 test cases in the metric script, not by retraining:
   - Seen-scanner test: Siemens and Philips cases in the official test set.
   - Unseen-scanner test: the 8 GE Discovery cases.
3. Report Dice, F1, and punctate F1 as mean and case-level spread for
   each cell. With n = 8 on GE, also list every GE case. A single mean
   can hide one failed scan.
4. Optional, and only after E3/E4 exist: leave-one-scanner-out on the
   scanners that do have training data, as a sensitivity analysis. It
   does not replace the GE hold-out.

**Exit.** A 2 × 2 table (model × seen/unseen) for Dice, F1, and punctate
F1, with per-case rows archived next to it. The plans file shows GE
never entered training.

### WP6 — Cross-dataset transfer

**Output.** Experiment `E5-transfer`.

Train on MSSEG (WP5 weights if they exist, otherwise skip to ISBI-trained
models scored on the open cohort). Evaluate on ISBI 2015 training
timepoints without any ISBI fine-tune. Score against each rater. The gap
between rater 1 and rater 2 on the same scan is the disagreement ceiling;
a model that lands inside that gap on Dice is not automatically a success
on punctate F1.

If compute remains, train on ISBI and test on MSSEG seen-scanner cases.
Direction is part of the result name (`msseg_to_isbi`, `isbi_to_msseg`).

**Exit.** A table with both raters and the same metrics as WP2. No
weight update on the target dataset.

### WP7 — Laboratory quantities

**Output.** Experiment `E6-biomarkers` on the predictions from WP4–WP6.
No new training.

For every held-out case compute manual vs predicted:

- lesion count (26-connected components surviving the same minimum size
  the detection metric uses)
- total lesion volume in mL

Then:

- Pearson r, Spearman ρ, ICC, MAE
- Bland–Altman plot with mean difference and 95% limits of agreement
- the same statistics inside the punctate subset, for count only (volume
  of that subset is a secondary line)

ISBI only: within-subject change between consecutive timepoints. The
question is whether the automatic count changes in the same direction as
the manual count, not whether either one is a clinical endpoint.

**Exit.** One agreement table per test cohort and two Bland–Altman
figures (count, volume) for the primary test set. Cases with an empty
manual mask are included and marked.

### WP8 — Optional robustness extensions

Start only if WP3 and WP4 are done and the calendar still has room.
Pick at most one.

- **MS-SAM-LESS on MSLesSeg**, using the authors' training split, then
  the WP2 metrics so the number is comparable to E1/E2. Their published
  score is a reference, not a substitute for our metric code.
- **TTIN** applied at inference to the best nnU-Net checkpoint, evaluated
  on the GE split if it exists, otherwise on the transfer split. No
  retraining. If F1 does not move, record the negative and stop.

**Exit.** Either a delta against the matched baseline under WP2, or a
written decision that the extension was not run.

## 5. Experiment matrix

| ID | WP | Train | Test | Blocks the write-up? |
| --- | --- | --- | --- | --- |
| E0 | WP2 | synthetic | synthetic | Yes |
| E1 | WP3 | MS3SEG or MSLesSeg | held-out folds | Yes, as the open baseline |
| E2 | WP4 | same as E1 | same as E1 | Yes |
| E3 | WP5 | MSSEG 15 | MSSEG test, seen scanners | Only the scanner claim |
| E4 | WP5 | MSSEG 15 | GE Discovery, 8 cases | Only the scanner claim |
| E5 | WP6 | one cohort | the other, no fine-tune | Secondary |
| E6 | WP7 | — | predictions from E2–E5 | Yes for the quantitative claim |
| E7 | WP8 | optional | matched to E1 or E4 | No |

## 6. Suggested timeline

Assume part-time work on one 24 GB GPU. Shift dates; do not reorder past
a failed exit test. Week 0 is the week Shanoir is requested.

| Week | Work | Done when |
| --- | --- | --- |
| 0 | WP0 | GPU imports torch; MS3SEG unpacked; Shanoir ticket filed |
| 1 | WP1 | Index CSV, QC note, counts match the release |
| 2 | WP2 | Pytest covers the five synthetic cases |
| 3–4 | WP3 | E1 CSV and plans file stored |
| 5–6 | WP4 | E2 joined to E1 on `case_id` |
| 7 | Buffer / MSSEG arrives | Converters extended; if no test set, freeze the open-data report |
| 8–9 | WP5 | E3 and E4 table, or an explicit block note |
| 10 | WP6 | One transfer direction scored against both ISBI raters |
| 11 | WP7 | Agreement table and Bland–Altman figures |
| 12 | Write-up, WP8 only if 8–11 were early | README status section points at result files |

A 12 GB GPU adds time in weeks 3–6 and 8–9. Prefer finishing five folds of
one configuration over starting a second architecture early.

## 7. Risks

| Risk | Likelihood | Response |
| --- | --- | --- |
| MSSEG test set stays closed | Medium | Open-data track is the deliverable. Do not invent a GE split from MS3SEG. |
| 12 GB VRAM cannot hold `3d_fullres` | Medium | Accept the planner's reduced batch or patch. Record it. Do not switch to a custom U-Net. |
| MedSAM automatic prompts are just nnU-Net boxes | High, by design of the default policy | Report the pipeline honestly. Add the MS-SAM-LESS prompt learner if the oracle gap shows that prompting, not the mask decoder, is the limit. |
| n = 8 GE cases | Certain | Publish per-case scores. Do not dress a mean as a precise vendor effect. |
| ISBI has only five training subjects | Certain | Use it as a transfer target or a small-source train set, never as the primary benchmark. |
| Dice looks fine while punctate F1 is poor | This is the point of H1 | Lead the write-up with the detection table. |
| Disk fills with nnU-Net caches | Medium | Delete preprocessing for a finished cohort after predictions are exported. |
| Seven-rater consensus hides a lesion all raters doubted | Known MSSEG issue | Keep the consensus. Mention inter-rater spread from the 2018 paper as context, do not re-annotate. |

## 8. Reproducibility checklist

Every experiment directory that is allowed into the write-up contains:

- the config (copy of `configs/experiment.yaml` plus run-specific fields)
- git commit of this repo, and commit or version of nnU-Net or MedSAM
- the exact command line
- GPU name, VRAM, and PyTorch/CUDA versions
- the plans file or MedSAM hyper-parameters (epochs, learning rate, what was frozen)
- per-case metric CSV from WP2 code
- a one-paragraph note of anything that departed from this plan

Random seed is `20261011` unless a run records a different one.

## 9. Implementation order inside the repo

The first code to add, and nothing else until it exists:

1. `src/msbench/metrics/` — Dice, components, detection F1, volume.
2. `tests/test_metrics.py` — the synthetic cases in WP2.
3. `src/msbench/data/` — one converter and the index writer.
4. `src/msbench/nnunet_export.py` — `processed/` to `nnUNet_raw`.
5. `src/msbench/medsam_infer.py` — slice, prompt, stack, then metrics.

Training stays a thin wrapper around `nnUNetv2_train` and the MedSAM
fine-tune script. Re-implementing either trainer is out of scope.

## 10. Definition of done

The project is complete for the course write-up when:

- E0, E1, and E2 exist and share the WP2 metric implementation.
- E6 has been run on the E2 predictions (count and volume agreement on
  the open cohort).
- Either E3 and E4 exist, or `experiments/E4-msseg-ge/BLOCKED.md` explains
  that the official unseen-scanner cases were not available.
- The README status section links those artifacts.
- No MRI, mask, or checkpoint is in the git history.

E5 and E7 improve the paper. They are not required to close the project.
