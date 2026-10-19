# Project Proposal

**Benchmarking nnU-Net against MedSAM for Multiple Sclerosis Lesion Segmentation**

**Repository:** [wang3ng/ms-lesion-benchmark](https://github.com/wang3ng/ms-lesion-benchmark)  
**Operational plan:** [`DEVELOPMENT_PLAN.md`](../DEVELOPMENT_PLAN.md)  
**Protocol:** [`configs/experiment.yaml`](../configs/experiment.yaml)

## Abstract

Multiple sclerosis (MS) lesion segmentation is usually ranked by voxel-wise Dice. That score can look acceptable while a model misses the small, scattered lesions that change a lesion count, or while it fails on a scanner it never saw in training. This project benchmarks nnU-Net, a self-configuring convolutional baseline, against MedSAM, a medical vision foundation model, on those two failure modes. The primary cohort is MSSEG-2016: 53 multi-contrast scans from four scanners, with a seven-expert consensus and a GE Discovery 3T arm held out of training. ISBI 2015 supplies a cross-dataset transfer test. Public fallbacks (MSLesSeg, MS3SEG, and 3D-MR-MS) let the pipeline start before Shanoir access to the MSSEG test set. MedSAM is scored only with automatic prompts, then stacked back to a 3D mask, so the comparison is fully automatic. An oracle-box condition is reported separately as an upper bound. On the same predictions we test whether lesion count and lesion volume agree with the manual masks. The product is a locked metric table, not a new architecture. Success is a completed open-data comparison plus either the official GE hold-out or a written block if those cases are unavailable.

## 1. Problem

MS follow-up depends on knowing how many lesions are present and how much tissue they occupy. Manual outlines are slow and variable, which is why automatic segmentation is attractive. Global overlap is a poor summary of that job. A large lesion can dominate Dice while punctate lesions, often only a few voxels, are missed. Those missed lesions barely move the volume and can change the count. A second, separate problem is scanner shift. Intensity, resolution, and contrast differ across vendors even when the protocol is nominally the same. A model tuned on one site can look strong in cross-validation and then drop on an unseen scanner.

Foundation models such as MedSAM are trained across many medical tasks and might be expected to handle that shift better than a specialist network. Specialist models such as nnU-Net are still the usual reference for biomedical segmentation, and they are fully automatic and three-dimensional. MedSAM, as published, is two-dimensional and needs a prompt. A comparison that hands MedSAM a box drawn from the ground truth answers a different question from the one a lab would ask. This proposal fixes the comparison so both systems produce a 3D mask with no manual prompt, then measures detection of small lesions, behavior on an unseen scanner, and agreement of the quantitative readouts.

## 2. Aims

**Aim 1. Detection versus overlap.** Measure whether lesion-wise F1, including a pre-specified punctate bin, ranks nnU-Net and MedSAM differently from voxel Dice.

**Aim 2. Cross-scanner generalization.** On MSSEG-2016, compare error on scanners present in training with error on GE Discovery 3T, which has no training cases. Report Dice and detection F1 side by side.

**Aim 3. Quantitative agreement.** On held-out scans, test whether lesion count and lesion volume computed from the automatic masks reproduce the same quantities from the manual masks.

**Secondary aim.** Score one cross-dataset transfer direction, MSSEG to ISBI 2015 or the reverse, with no fine-tuning on the target set. Optional extensions (learnable dense prompts, or test-time instance normalization) start only after Aims 1 and the open-data half of Aim 3 exist.

### Hypotheses

- **H1.** Punctate lesion-wise F1 will separate the two models by more than voxel Dice does, because Dice is dominated by large lesions.
- **H2.** Both models will lose more detection F1 than Dice when moving from seen MSSEG scanners to GE Discovery 3T. Which model loses less is the measurement, not an assumption.
- **H3.** Volume will agree with the manual volume more closely than count agrees with the manual count, because missed punctate lesions move the count and barely move the volume.
- **H4.** A model trained on MSSEG will lose more F1 on ISBI 2015 than on held-out MSSEG cases from scanners it has seen.

A negative result on any hypothesis is a valid outcome. Thresholds and the punctate cut are fixed before any model is scored.

## 3. Data

Imaging stays out of git. Collaborators download into `data/raw/` using `scripts/download_datasets.py`. Sources and licenses are in `configs/downloads.yaml`.

| Cohort | Role | What it provides |
| --- | --- | --- |
| MSSEG-2016 | Primary | 53 patients, four scanners (Siemens Verio 3T, Siemens Aera 1.5T, Philips Ingenia 3T, GE Discovery 3T), three manufacturers. Train 15 / test 38. GE Discovery is 0 train / 8 test. Seven expert masks fused by LOP STAPLE. Contrasts: 3D FLAIR, 3D T1, 3D T1 post-gadolinium, axial PD/T2. Access via Shanoir. |
| ISBI 2015 | Transfer | 5 training subjects, mean 4.4 timepoints, and 14 test subjects, all on one 3T Philips scanner. T1, T2, PD, FLAIR. Two raters. Direct zip files from the IACL challenge page. |
| MSLesSeg | Open fallback | 75 patients / 115 scans (T1, T2, FLAIR); the published training split is 53 patients. Figshare, about 1.37 GB preprocessed. Preferred if the prompt-learning extension is run. |
| MS3SEG | Open fallback | 100 patients, one Toshiba Vantage 1.5T, thick-slice axial FLAIR (~5.7 mm). Tri-mask labels; we use abnormal white-matter hyperintensity only. NIfTI portion about 2 GB. |
| 3D-MR-MS | Additional public cohort | 30 patients with consensus lesion masks, six zip files, about 4.5 GB. License CC BY-NC-ND, so the files stay local. |

The GE comparison requires the 38-case MSSEG test set. If that release stays closed, Aim 2 is reported as blocked. Aims 1 and 3 still run on the open cohorts. MS3SEG is a single vendor, so it cannot define the scanner effect.

We train on each dataset's released preprocessed images. A private registration pipeline would mix scanner differences with our own preprocessing.

## 4. Methods

**nnU-Net v2** is the specialist baseline. The planner chooses resampling, normalization, patch size, and schedule from the dataset fingerprint. Volumetric cohorts use `3d_fullres`. MS3SEG uses `2d` because a 5.7 mm slice spacing does not support a 3D kernel. We record the plans file and do not hand-edit the architecture to chase a metric.

**MedSAM** is the foundation-model arm. The published model is a ViT image encoder with a prompt encoder and a mask decoder, trained on a large medical image-mask corpus, and it operates on 2D slices. The primary condition fine-tunes the prompt encoder and mask decoder. The image encoder is updated only if a 24 GB GPU still has headroom after a decoder-only run. Prompts are automatic: no test-mask boxes. The default prompt is a box around components from a frozen nnU-Net prediction, plus a background point. That makes MedSAM a second stage on top of nnU-Net, and the write-up will say so. Slice masks are stacked to the original 3D grid before any metric is computed. A separate `medsam_oracle_box` column uses boxes from the reference lesions and is an upper bound, not the headline number.

**Optional, after the baseline table exists.** MS-SAM-LESS learns dense prompts with SAM present only at training time, then replaces SAM with a small convolutional head. It was developed on MSLesSeg. Test-time instance normalization, as used in UNISELF, refreshes feature statistics per test volume with no retraining. At most one of these is added, and only if the calendar allows.

External code stays in its own checkout. This repository owns conversion, the prompting rule, the metrics, and the tables.

## 5. Evaluation

All scores use the same 3D masks and the same code path. A reference lesion is punctate when its equivalent-sphere diameter is under 3 mm. That cut is not changed after scores are seen.

| Measure | Role |
| --- | --- |
| Voxel Dice | Global overlap. Required, and not sufficient. |
| Lesion-wise detection F1 | MSSEG-2016 rule: 26-connected components; a reference lesion counts as detected only with sufficient overlap and without a disproportionately large predicted component. |
| Punctate F1 | Detection F1 on reference lesions under 3 mm. |
| Count and volume (mL) | Pearson and Spearman correlation, ICC, MAE, and Bland–Altman limits of agreement against the manual masks. |

Empty masks are defined. Both empty gives Dice 1 and F1 1. An empty reference with a non-empty prediction gives Dice 0 and F1 0. Overlap and size-ratio constants are copied from the MSSEG evaluation description into unit tests on synthetic blobs before any training run.

On ISBI, predictions are scored against each rater. The gap between raters is the disagreement ceiling, not a number to average away. For longitudinal pairs, we check whether the automatic count moves in the same direction as the manual count between consecutive timepoints. That is a sanity check, not a clinical endpoint.

Reports stratify by scanner and by lesion size. The GE arm has eight cases, so every GE case is listed next to the mean.

## 6. Experiment order

| ID | Question | Required for the write-up |
| --- | --- | --- |
| E0 | Do the metrics match hand-built volumes? | Yes |
| E1 | nnU-Net on MSLesSeg or MS3SEG | Yes |
| E2 | MedSAM, automatic prompts, same split as E1 | Yes |
| E3 | MSSEG test cases from scanners seen in training | Only the scanner claim |
| E4 | GE Discovery 3T | Only the scanner claim |
| E5 | Cross-dataset transfer, no fine-tune | Secondary |
| E6 | Count and volume agreement on E2, and on E3–E5 if they exist | Yes for Aim 3 |
| E7 | Prompt learning or test-time normalization | No |

E1 and E2 share a per-case CSV joined on case id. E4 is not replaced by a random split of the 15 MSSEG training cases.

## 7. Resources

Training runs on a Linux GPU node or WSL2. A 12 GB GPU can force batch size 1 or a smaller nnU-Net patch; 24 GB matches the planner's preferred footprint and leaves room to fine-tune MedSAM's decoder. Disk is 50 GB if one cohort is preprocessed at a time, and about 100 GB if MSSEG, ISBI, one fallback, caches, and both model families stay on disk together. Eight CPU cores and 32 GB of RAM cover fingerprinting and resampling.

Checkpoints and NIfTI files are gitignored. Each experiment directory that enters the write-up stores the config, the command, the GPU and library versions, the plans or MedSAM hyperparameters, and the per-case metric CSV.

## 8. Timeline

Part-time work on one GPU. The order is fixed; dates can slip. Week 0 is the week the MSSEG access request is filed.

| Weeks | Work | Exit |
| --- | --- | --- |
| 0 | Environment, Shanoir request, one public archive unpacked | GPU import works; ticket filed |
| 1 | Common NIfTI layout and case index | Counts match the release notes |
| 2 | Metrics and synthetic tests | Pytest passes with no network weights |
| 3–4 | nnU-Net on the open cohort | E1 table and plans file |
| 5–6 | MedSAM on the same split | E2 joined to E1 |
| 7 | MSSEG arrives, or the open-data report is frozen | Block note if the test set is absent |
| 8–9 | In-scanner and GE evaluation | E3 and E4, or the block note |
| 10 | One transfer direction | Scored against both ISBI raters |
| 11 | Count and volume agreement | Bland–Altman figures |
| 12 | Write-up; E7 only if earlier weeks finished early | README points at the result files |

## 9. Risks and scope

| Risk | Response |
| --- | --- |
| MSSEG test set stays closed | Ship the open-data comparison. Do not invent a GE split from another cohort. |
| 12 GB is tight for `3d_fullres` | Accept the planner's reduced batch or patch and record it. |
| Automatic MedSAM prompts are nnU-Net boxes | Label the pipeline as such. Use the oracle gap to see whether prompting, rather than the decoder, is the limit. |
| Eight GE cases | Publish per-case scores. A mean is not a precise vendor effect. |
| Five ISBI training subjects | Use ISBI as a transfer set, not as the primary benchmark. |
| 3D-MR-MS license is CC BY-NC-ND | Download for local use. Do not commit or rehost the archives. |

Out of scope: a new network, a clinical device claim, re-annotation of the seven MSSEG raters, and a private preprocessing stack. Oracle-box MedSAM is never placed in the same column as fully automatic nnU-Net without a label.

## 10. Deliverables

1. Metric code with tests for Dice, detection F1, punctate F1, count, and volume.
2. Per-case tables for E1 and E2, plus E6 agreement on those predictions.
3. Either the E3/E4 scanner table or `experiments/E4-msseg-ge/BLOCKED.md`.
4. A short write-up that leads with detection and punctate F1, then Dice, then count and volume.
5. Configs, commands, and version pins sufficient for someone else to rerun a finished experiment.

E5 and E7 improve the study. They are not required to close it.

## References

- Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods* 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z
- Ma J, He Y, Li F, Han L, You C, Wang B. Segment anything in medical images. *Nature Communications* 15, 654 (2024). https://doi.org/10.1038/s41467-024-44824-z
- Proietto Salanitri F, et al. SAM-guided prompt learning for multiple sclerosis lesion segmentation. *Pattern Recognition Letters* 199, 205–211 (2026). https://doi.org/10.1016/j.patrec.2025.11.018
- Commowick O, et al. Objective evaluation of multiple sclerosis lesion segmentation using a data management and processing infrastructure. *Scientific Reports* (2018). https://doi.org/10.1038/s41598-018-31911-7
- Commowick O, et al. Multiple sclerosis lesions segmentation from multiple experts: The MICCAI 2016 challenge dataset. *NeuroImage* 244, 118589 (2021). https://doi.org/10.1016/j.neuroimage.2021.118589
- Carass A, et al. Longitudinal multiple sclerosis lesion segmentation: Resource and challenge. *NeuroImage* 148, 77–102 (2017).
- Guarnera F, et al. MSLesSeg: baseline and benchmarking of a new multiple sclerosis lesion segmentation dataset. *Scientific Data* (2025). https://doi.org/10.1038/s41597-025-05250-y
- Bawil MB, et al. A multiple sclerosis MRI dataset with tri-mask annotations for lesion segmentation. *Scientific Data* (2026). https://doi.org/10.1038/s41597-026-07184-5
- Lesjak Ž, Galimzianova A, Špiclin Ž. 3D-MR-MS database. https://lit.fe.uni-lj.si/en/research/resources/3D-MR-MS/
