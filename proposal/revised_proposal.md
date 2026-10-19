# Project Full Proposal

## Can a Medical Foundation Model Find Small MS Lesions? Benchmarking and Diagnosing MedSAM against nnU-Net Across Scanners

*This proposal is organized under the four rubric criteria. The week-by-week schedule and experiment IDs are in DEVELOPMENT_PLAN.md.*

---

## 1. Problem Area

Multiple sclerosis (MS) is monitored with MRI. The numbers clinicians use are how many white-matter lesions a patient has and how much volume they occupy, and both still come mostly from manual outlines. Manual outlining is slow, and trained readers disagree: on the MSSEG-2016 cohort, seven experts outlined the same scans and differed substantially in both overlap and lesion detection.

Automatic segmentation is the obvious remedy, but two failures limit its use:

- **Small lesions.** Voxel-wise Dice, the usual score, is dominated by large lesions. A model can score well on Dice while missing the tiny, scattered lesions that change the lesion count but barely change the volume.
- **Unseen scanners.** The same nominal protocol looks different on Siemens, Philips, and GE hardware. A model that looks strong at one site can fail at another, and multi-center trials only work if the measurement survives a change of scanner.

General-purpose foundation models such as MedSAM promise to transfer across domains with little task-specific data, which makes them attractive for exactly these problems. However, they are built to *outline* an object someone points at, not to *search* a brain for dozens of tiny lesions. Whether they can find small MS lesions on their own, and whether that holds on a scanner they have never seen, is untested.

This project asks: **can a medical foundation model find and outline small MS lesions as well as a specialist model, including on an unseen scanner? If not, is its weakness in finding lesions or in tracing them, and can a simple change fix it?**

---

## 2. Context

**Challenge results.** The MICCAI 2016 MSSEG challenge scored thirteen automatic pipelines against a seven-expert consensus. The best automatic method still trailed the best expert on both delineation (segmentation scores on the order of 0.59 versus about 0.78) and lesion detection (on the order of 0.49 versus about 0.89), and performance dropped further on images from a scanner withheld from training (Commowick et al., 2018). The ISBI 2015 longitudinal challenge reached a similar conclusion on a single-scanner cohort scored against two raters (Carass et al., 2017).

**Specialist models.** Since then, nnU-Net (Isensee et al., 2021) has become the standard specialist baseline. It is a self-configuring convolutional network that chooses its own preprocessing, patch size, and architecture from the data, and it is the default strong result across biomedical segmentation, including MS lesions.

**Evaluation beyond Dice.** Recent work shows why Dice alone is not enough for MS. *Beyond Dice* (Basit et al., 2026) compared nnU-Net, SegResNet, and SwinUNETR on a 24-subject MSSEG evaluation cohort containing 1,172 lesions. Lesion-level metrics separated models that Dice did not: nnU-Net improved subject-mean lesion recall over SegResNet, while SwinUNETR produced about 10.7 more false-positive lesions per scan than nnU-Net. That study covers only specialist CNN and transformer models; it does not include foundation models or an unseen-scanner test.

**Foundation models.** SAM (Kirillov et al., 2023) is a promptable segmenter: a heavy image encoder, a light prompt encoder, and a light mask decoder, trained on over a billion masks from 11 million images. In zero-shot tests across 23 datasets it often matched or beat a strong interactive segmentation baseline, but the authors also noted its weaknesses: it misses fine structures, produces small disconnected fragments, and trails domain-specific tools within a specialty. MedSAM (Ma et al., 2024) fine-tuned SAM on about 1.6 million medical image–mask pairs across 10 modalities and reported better accuracy than modality-specific specialists. However, MedSAM is a 2D, box-prompted model: it outlines what a box points at, and it was not designed to search for dozens of punctate lesions. An MS-specific adaptation, MS-SAM-LESS (Proietto Salanitri et al., 2026), avoids this by using SAM only to learn dense prompts during training and replacing it with a small convolutional head at inference, with gains on the public MSLesSeg set. Broader benchmarks of SAM-family models against nnU-Net on brain tumors (e.g., arXiv:2506.20786) find that nnU-Net remains a very strong reference.

**The gap.** No study has (a) separated a foundation model's ability to *find* MS lesions from its ability to *trace* them, (b) compared it fairly with nnU-Net when neither model is given the lesion locations, and (c) done so with small-lesion detection metrics on the official unseen-scanner split. A key design issue in prior foundation-model pipelines is that boxes often come from another detector or from the ground truth, which hides the foundation model's own detection ability.

---

## 3. Data

All data are multi-contrast brain MRI in NIfTI format with expert lesion masks. Each team member applies for access individually, as the data use agreements require, and no scans are committed to the repository.

| Dataset | Cases | Contrasts | Labels | Role |
|---|---|---|---|---|
| **MSSEG-2016** | 53 exams (15 train / 38 test); 4 scanners. GE Discovery 3T appears only in test (8 exams) | 3D FLAIR, 3D T1, 3D T1-Gd, PD/T2 | 7 individual experts + consensus | **Primary**: training, in-distribution test, unseen-scanner test |
| **ISBI 2015** | 5 train subjects (21 scans, ~4.4 timepoints each) + 14 test subjects; one 3T Philips scanner | T1 (MPRAGE), T2, PD, FLAIR | 2 raters | **Cross-dataset transfer** test |
| **MSLesSeg** | 75 patients, 115 scans (53 patients in the training split) | T1, T2, FLAIR | Expert-checked masks | **Fallback** if MSSEG test access is delayed |

**MSSEG-2016.** The scanners are Siemens Verio 3T, Siemens Aera 1.5T, Philips Ingenia 3T, and GE Discovery 3T. Training contains 5 exams from each of the first three; GE is withheld entirely from training, which provides a built-in unseen-scanner test. We train and score against the published consensus mask, and use the seven individual expert masks to measure human inter-rater variability as a reference for model errors. Access is through the Shanoir platform, with the Shifts benchmark release on Zenodo (Malinin et al., 2022) as a backup route. Test-set access may initially be limited to the 15 training cases.

**ISBI 2015.** Data are downloaded from the organizers' page at JHU (IACL). The official online leaderboard is currently unavailable, so we use the 21 labeled training scans as an external test set and score against each rater separately with the organizers' released evaluation code.

**Preprocessing and contrast matching.** All volumes are skull-stripped, bias-corrected, co-registered to FLAIR, and resampled to 1 mm isotropic. Because T1-Gd exists only in MSSEG, every model used for cross-dataset transfer is trained only on the contrasts shared across datasets (FLAIR, T1, T2).

**Optional extras** (only if time allows): MS3SEG (100 patients, thick 5.7 mm slices) and 3D-MR-MS (30 patients).

**Total:** roughly 70 to 190 labeled scans, depending on which releases are approved.

---

## 4. Proposed Solution

### 4.1 Models

**nnU-Net v2 (specialist baseline).** We use the 3D full-resolution configuration with the planner's own preprocessing and patch size. It sees the whole volume and decides on its own where every lesion is.

**MedSAM (foundation model).** MedSAM is fine-tuned on the same training cases with the image encoder frozen and the prompt encoder and mask decoder trained. Inputs are 2D axial slices from the co-registered volumes, with the three input channels set to FLAIR, T1, and T2. Slice predictions are stacked into a 3D mask before scoring. Both models use the same cross-validation folds, and we report training time and GPU budget for each.

### 4.2 Three-part study design

MedSAM needs a box telling it where to look, so where the boxes come from determines what we measure. We run MedSAM under three box conditions and fine-tune a separate model for each, using the same type of box during training and testing.

| MedSAM condition | Where the boxes come from | Fair comparison with nnU-Net? | Question it answers |
|---|---|---|---|
| **Oracle boxes** | The true lesions (upper bound only) | No, MedSAM is given the answer | How well can MedSAM *trace* lesions with perfect guidance? |
| **nnU-Net boxes** | nnU-Net's detected lesions | No, its recall is capped by nnU-Net | Does MedSAM *improve nnU-Net's outlines*? |
| **Independent boxes** | Whole-brain box per slice, and overlapping tiles (e.g., 4×4) inside the brain mask, including lesion-free tiles | **Yes**, both models work unaided | Can MedSAM *find* lesions on its own? |

**Part 1: Head-to-head.** nnU-Net versus MedSAM with independent boxes. This is the only fair test of which model is better.

**Part 2: Diagnosis.** Comparing the three MedSAM conditions locates the weakness:
- Good oracle results but poor independent results mean MedSAM traces well but cannot find small lesions (a **detection** problem).
- Poor results even with oracle boxes mean MedSAM cannot trace small lesions (a **tracing** problem).

**Part 3: Targeted fix.** Based on the diagnosis, we test one simple, pre-specified improvement:
- *Detection problem:* smaller, zoomed-in tiles, so tiny lesions occupy more pixels after MedSAM resizes its input.
- *Tracing problem:* a fine-tuning loss weighted toward small lesions, so small misses count more during training.

### 4.3 Evaluation

All models are scored with one shared implementation:

- **Voxel Dice**, for comparability with prior work.
- **Lesion-wise detection F1** under the MSSEG rule (26-connected components, minimum overlap, and rejection of predicted blobs far larger than the reference lesion).
- **Size-stratified detection F1.** Lesions are grouped into volume bins (e.g., under 20 mm³, 20–100 mm³, over 100 mm³). Bin edges are fixed in advance from lesion counts in the training data only, to ensure each bin has enough lesions.
- **False-positive lesions per scan.**
- **Lesion count and total volume agreement** with the reference: mean absolute error and Bland–Altman plots, with ICC reported as descriptive.
- **Human reference:** the spread among the seven MSSEG experts, scored with the same metrics.

**Test settings:** in-distribution (Siemens/Philips MSSEG test cases), unseen scanner (8 GE cases), and cross-dataset (ISBI 2015).

**Statistics:** 5-fold patient-level cross-validation on the training set for model selection; paired per-patient comparisons between models (Wilcoxon signed-rank test); and 95% bootstrap confidence intervals for all metrics. All metrics, size bins, and thresholds are fixed before test data are examined.

### 4.4 Expected problems and mitigations

| Problem | Mitigation |
|---|---|
| MSSEG test set (including all 8 GE cases) may stay behind access approval | Apply now via Shanoir and Zenodo; if blocked, report the scanner result as pending and use ISBI 2015 and MSLesSeg for the remaining analyses |
| Few lesions in the smallest size bin | Count lesions per bin in the training data before fixing bin edges; merge bins if needed |
| MedSAM with independent boxes may perform poorly | This is still a valid finding about foundation models; Parts 2 and 3 then explain and address it |
| Many tiles contain no lesion, which may cause false positives | Include lesion-free tiles in fine-tuning so MedSAM learns to output an empty mask |
| Only 8 GE test cases | Report every case individually, with confidence intervals, and treat the vendor effect as exploratory |
| Annotation styles differ across datasets | Interpret ISBI results as combined scanner and labeling shift, not a pure scanner effect; score each rater separately |
| Limited GPU memory (12 GB) | Allow nnU-Net to use a smaller patch; MedSAM works slice by slice with a frozen encoder |
| ISBI online leaderboard is unavailable | Use the labeled training subjects as the external test set, scored with the organizers' evaluation code |

### 4.5 Expected outcome

Even if the targeted fix does not help, the study produces a complete result: a fair comparison of a specialist and a foundation model on small-lesion detection and unseen-scanner robustness, plus a clear diagnosis of whether foundation models fail at finding or at tracing MS lesions. If the fix works, the conclusion becomes: *foundation models miss small MS lesions mainly because of X, and a simple change Y recovers much of the gap.*

---

## References

1. Basit A, Hassan N, Shafique M. Beyond Dice: Clinically structured bias analysis of multiple sclerosis lesion segmentation models. *Proceedings of Machine Learning Research* 340 (2026). https://proceedings.mlr.press/v340/basit26a.html
2. Carass A, et al. Longitudinal multiple sclerosis lesion segmentation: Resource and challenge. *NeuroImage* 148, 77–102 (2017).
3. Carass A, et al. Longitudinal multiple sclerosis lesion segmentation data resource. *Data in Brief* 12, 346–350 (2017).
4. Commowick O, et al. Objective evaluation of multiple sclerosis lesion segmentation using a data management and processing infrastructure. *Scientific Reports* 8, 13650 (2018). https://doi.org/10.1038/s41598-018-31911-7
5. Commowick O, et al. Multiple sclerosis lesions segmentation from multiple experts: The MICCAI 2016 challenge dataset. *NeuroImage* 244, 118589 (2021). https://doi.org/10.1016/j.neuroimage.2021.118589
6. Guarnera F, et al. MSLesSeg. *Scientific Data* (2025). https://doi.org/10.1038/s41597-025-05250-y
7. Isensee F, et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods* 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z
8. Kirillov A, et al. Segment Anything. arXiv:2304.02643 (2023).
9. Ma J, et al. Segment anything in medical images. *Nature Communications* 15, 654 (2024). https://doi.org/10.1038/s41467-024-44824-z
10. Malinin A, et al. Shifts 2.0: Extending the dataset of real distributional shifts. arXiv:2206.15407 (2022).
11. Proietto Salanitri F, et al. SAM-guided prompt learning for multiple sclerosis lesion segmentation. *Pattern Recognition Letters* 199, 205–211 (2026). https://doi.org/10.1016/j.patrec.2025.11.018
12. Benchmarking SAM-based models and nnU-Net for brain tumor segmentation. arXiv:2506.20786 (2025).
