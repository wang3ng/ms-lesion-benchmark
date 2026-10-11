# MS Lesion Benchmark: nnU-Net vs MedSAM

This repository benchmarks a specialized medical segmentation framework,
**nnU-Net**, against the medical vision foundation model **MedSAM** on
multiple sclerosis (MS) white-matter lesions.

The comparison is not a leaderboard on global voxel overlap alone. MS
lesions are small, scattered, and sensitive to the scanner that acquired
the scan. A model can post a respectable Dice score while missing the
punctate lesions that change a lesion count, or while collapsing on a
vendor it never saw in training. This project measures those two failure
modes directly, then checks whether the masks still reproduce the manual
quantitative readouts a lab would actually report: lesion count and lesion
volume.

No models have been trained in this commit. The tree is the protocol,
the data map, and the order of work. The schedule and exit criteria are
in [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md). The machine-readable
version of the protocol is [`configs/experiment.yaml`](configs/experiment.yaml).

## Questions

1. On lesions small enough that a few voxels decide detection, does
   lesion-wise F1 rank nnU-Net and MedSAM differently from voxel Dice?
2. When the test scanner is absent from training, how large is the drop
   in detection F1 relative to the drop in Dice, and is that drop the
   same for both models?
3. On held-out scans, do counts and volumes computed from the automatic
   masks agree with the same quantities computed from manual masks?

A fourth, secondary question is whether either model survives a dataset
shift (MSSEG-trained weights scored on ISBI 2015, and the reverse) without
retraining. Extensions that add learnable dense prompts or test-time
instance normalization are in scope only after the primary comparison
exists.

## Methods

**nnU-Net** (Isensee et al., Nature Methods, 2021) is the specialized
baseline. It is a self-configuring CNN: given a dataset fingerprint it
chooses resampling, intensity normalization, patch size, topology, and a
training schedule. We will use nnU-Net v2. Volumetric cohorts
(MSSEG-2016, ISBI 2015, MSLesSeg) use `3d_fullres`. MS3SEG is thick-slice
axial FLAIR (about 5.7 mm), so that cohort uses the `2d` configuration.
Fingerprints are recorded with each run; we will not hand-tune an
architecture to chase a metric.

**MedSAM** (Ma et al., Nature Communications, 2024) is a SAM variant with
a ViT image encoder and medical-image pretraining. The published model is
2D and prompt-driven. nnU-Net is fully automatic and 3D. Those are
different interfaces, so the primary comparison does not use hand-drawn
boxes. MedSAM is fine-tuned on its prompt encoder and mask decoder, run
slice-wise with an automatic prompting rule, and stacked back into a 3D
mask before any metric is computed. An oracle-box condition, in which
prompts come from the ground-truth lesion extents, is reported separately
as an upper bound. It is not the headline number.

**Optional extensions**, after the baseline table exists:

- **MS-SAM-LESS** (Proietto Salanitri et al., Pattern Recognition Letters,
  2026) uses SAM only while training a dense prompt learner, then replaces
  SAM at inference with a small convolutional aggregator. It was developed
  on MSLesSeg, which makes it a natural add-on for the open-data track.
- **Test-time instance normalization (TTIN)**, as used in UNISELF, refreshes
  normalization statistics on each test volume. It is a cheap probe of
  whether domain shift in this benchmark is partly a feature-statistics
  problem. It is not a third full training program.

Code for both external projects stays in their own checkouts. This repo
owns data conversion, prompting policy, metrics, and the result tables.

## Evaluation

All scores are computed on 3D masks in the processed dataset grid, after
MedSAM slices have been stacked. Metrics are stratified by scanner and by
lesion size. The size cut is fixed before any model is scored: a reference
lesion is **punctate** when its equivalent-sphere diameter is under 3 mm.

| Measure | What it answers |
| --- | --- |
| Voxel Dice | Global overlap. Necessary, not sufficient. |
| Lesion-wise detection F1 | Whether each lesion object was found, using the MSSEG-2016 detection rule (sufficient overlap, and the overlapping predicted component must not be disproportionately large). |
| Punctate-lesion F1 | The same F1 restricted to reference lesions under 3 mm equivalent diameter. |
| Lesion count and volume (mL) | Downstream lab quantities. Agreement with manual masks via Pearson and Spearman correlation, ICC, Bland–Altman, and MAE. |

Empty scans are defined, not dropped. If both masks are empty, Dice is 1.
If the reference is empty and the prediction is not, Dice is 0 and every
predicted component is a false positive. MSSEG itself substitutes count
and volume when a consensus mask cannot be formed; we will follow that
rule on MSSEG cases that have no consensus and document how many cases
that affects.

Exact overlap and component-size thresholds for detection F1 will be taken
from Commowick et al. (Scientific Reports, 2018) and locked in a unit test
against hand-built volumes. They will not be adjusted after models are
scored.

## Data

Imaging is not in git. Download instructions and the on-disk layout are
in [`data/README.md`](data/README.md).

### Primary: MSSEG-2016

53 MS patients, harmonized OFSEP-style protocol, four scanners at three
French sites. Seven experts delineated FLAIR-hyperintense lesions on every
case (with T2 as a control). The challenge ground truth is a LOP STAPLE
consensus of those seven masks, not a single rater.

The public description says four scanners from three manufacturers
(Siemens, Philips, GE), not four vendors. The GE Discovery 3T arm was
withheld from training on purpose. That split is the cross-scanner test.

| Scanner | Site | Field | Train | Test |
| --- | --- | --- | --- | --- |
| Siemens Verio | Rennes | 3T | 5 | 10 |
| Siemens Aera | Lyon | 1.5T | 5 | 10 |
| Philips Ingenia | Lyon | 3T | 5 | 10 |
| GE Discovery | Bordeaux | 3T | 0 | 8 |
| **Total** | | | **15** | **38** |

Contrasts: 3D FLAIR, 3D T1, 3D T1 post-gadolinium, and an axial dual-echo
PD/T2. Raw and preprocessed (denoised, registered, brain-extracted,
bias-corrected) releases both exist. We train on the preprocessed release
so that scanner effects are not confounded with a home-grown registration
pipeline, and we keep the raw release for QC.

Access is through Shanoir after registration
([data page](https://portal.fli-iam.irisa.fr/msseg-challenge/english-msseg-data/)).
Training (15 cases) has been the reliably downloadable portion. The 38-case
test set, including all 8 GE scans, is required before any cross-scanner
claim is written up. Cite Commowick et al., NeuroImage 2021
(10.1016/j.neuroimage.2021.118589) and the challenge analysis in Scientific
Reports 2018 (10.1038/s41598-018-31911-7).

### Secondary: ISBI 2015 longitudinal challenge

Five training subjects (mean 4.4 timepoints, about one year apart) and
fourteen test subjects, all acquired on one 3T Philips scanner at Johns
Hopkins. Contrasts: T1 MPRAGE, T2, PD, and FLAIR. Two expert raters marked
lesions. The cohort is the wrong tool for a scanner study and the right
tool for cross-dataset transfer and for a longitudinal sanity check
(counts and volumes should not jump arbitrarily between timepoints of the
same subject). Cite Carass et al., NeuroImage 2017.

### Fallback, used first if MSSEG access is slow

These two sets are public and do not wait on Shanoir. They are for
pipeline bring-up and for a result we can compute immediately. They do
not replace the GE hold-out.

**MSLesSeg** (Guarnera et al., Scientific Data, 2025;
[Figshare 10.6084/m9.figshare.27919209](https://doi.org/10.6084/m9.figshare.27919209)).
The full release is 75 patients and 115 scans (T1, T2, FLAIR), registered
to MNI152 at 182 × 218 × 182. The challenge training split is 53 patients
(50 RRMS, 3 SPMS). That training split is the "53 clinical cases" in the
project brief. Masks were drawn on FLAIR and checked by an MS neurologist.
MS-SAM-LESS publishes against this split, so it is the preferred open set
when the extension is in play.

**MS3SEG** (Bawil et al., Scientific Data, 2026;
[Figshare 10.6084/m9.figshare.30393475](https://doi.org/10.6084/m9.figshare.30393475)).
100 patients from one Toshiba Vantage 1.5T scanner. The preprocessed NIfTI
portion is about 2 GB; raw DICOM is larger (about 4 GB in the data paper).
Axial T2-FLAIR spacing is about 0.90 × 0.90 × 5.73 mm, so this is a 2D
clinical stack, not an isotropic 3D research protocol. Tri-mask labels
separate ventricles, normal-appearing white-matter hyperintensities, and
abnormal (MS) hyperintensities. We train and score the abnormal class
only. The single Toshiba vendor makes MS3SEG a useful external domain for
models trained on Siemens/Philips/GE data, and a poor place to *define*
cross-scanner performance.

## Compute and storage

Training assumes a Linux GPU node (native or WSL2). Windows is enough for
metrics, conversion, and documentation.

| Resource | Target | Why |
| --- | --- | --- |
| GPU | 12–24 GB VRAM (RTX 3090, RTX 4090, or a similar cluster GPU) | nnU-Net `3d_fullres` on brain MRI. 12 GB may force batch size 1 or a smaller patch; 24 GB matches the planner's preferred footprint. MedSAM ViT-B fine-tuning of the decoder fits the same range when the image encoder is frozen or updated at a low rate. |
| Disk | 50 GB minimum, 100 GB if MSSEG, ISBI, one fallback, preprocessed caches, and both model families stay on disk together | The 30–50 GB brief covers one cohort plus checkpoints. nnU-Net's preprocessed cache is often several times the raw NIfTI. |
| CPU / RAM | 8 cores, 32 GB RAM | Resampling and the nnU-Net fingerprint. |

Keep one preprocessed cohort at a time if the disk budget is 50 GB.
Checkpoints for unused folds can be deleted after the fold's metrics are
exported.

## Repository layout

```
configs/experiment.yaml   protocol (splits, metrics, model names)
data/README.md             download and on-disk layout; volumes are gitignored
experiments/               one folder per run, configs only
results/                   reviewed tables and figures
src/msbench/               conversion, metrics, and runners (started empty)
DEVELOPMENT_PLAN.md        phases, exit criteria, risks
```

## Environment

Python 3.10 or newer. Install a CUDA build of PyTorch that matches the
driver before nnU-Net.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
# then, on the GPU machine:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[train]"
```

Set the nnU-Net paths at the start of every training shell so caches do
not land in `$HOME`:

```bash
export nnUNet_raw="$PWD/data/nnUNet_raw"
export nnUNet_preprocessed="$PWD/data/nnUNet_preprocessed"
export nnUNet_results="$PWD/data/nnUNet_results"
```

MedSAM weights and the MS-SAM-LESS checkout are external. Record the
commit hash of each in the experiment folder. Do not vendor their trees
into this repo.

## Status

This commit is the scaffold: protocol, citations, and directory contract.
Metric code, dataset converters, and training entry points are the first
implementation tasks in the development plan. Do not cite numbers from
this repository until an experiment directory contains a config, a command
log, and a table produced by the locked metric code.

## References

- Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a
  self-configuring method for deep learning-based biomedical image
  segmentation. *Nature Methods* 18, 203–211 (2021).
  https://doi.org/10.1038/s41592-020-01008-z
  Code: https://github.com/MIC-DKFZ/nnUNet
- Ma J, He Y, Li F, Han L, You C, Wang B. Segment anything in medical
  images. *Nature Communications* 15, 654 (2024).
  https://doi.org/10.1038/s41467-024-44824-z
  Code: https://github.com/bowang-lab/MedSAM
- Proietto Salanitri F, Bellitto G, Calcagno S, Bagci U, Spampinato C,
  Pennisi M. SAM-guided prompt learning for multiple sclerosis lesion
  segmentation. *Pattern Recognition Letters* 199, 205–211 (2026).
  https://doi.org/10.1016/j.patrec.2025.11.018
  Code: https://github.com/perceivelab/MS-SAM-LESS
- Commowick O, et al. Objective evaluation of multiple sclerosis lesion
  segmentation using a data management and processing infrastructure.
  *Scientific Reports* 8, 13650 (2018).
  https://doi.org/10.1038/s41598-018-31911-7
- Commowick O, et al. Multiple sclerosis lesions segmentation from
  multiple experts: The MICCAI 2016 challenge dataset. *NeuroImage* 244,
  118589 (2021). https://doi.org/10.1016/j.neuroimage.2021.118589
- Carass A, et al. Longitudinal multiple sclerosis lesion segmentation:
  Resource and challenge. *NeuroImage* 148, 77–102 (2017).
- Guarnera F, et al. MSLesSeg: baseline and benchmarking of a new multiple
  sclerosis lesion segmentation dataset. *Scientific Data* (2025).
  https://doi.org/10.1038/s41597-025-05250-y
- Bawil MB, et al. A multiple sclerosis MRI dataset with tri-mask
  annotations for lesion segmentation (MS3SEG). *Scientific Data* (2026).
  https://doi.org/10.1038/s41597-026-07184-5

UNISELF, which introduces the TTIN adaptation listed in the brief, is
described at https://pmc.ncbi.nlm.nih.gov/articles/PMC12754720/. Confirm
the bibliographic record before citing it in a paper; the method is in
scope here only as an optional robustness add-on.

## Data use

MSSEG-2016 and ISBI 2015 are challenge datasets with their own access and
citation rules. MSLesSeg and MS3SEG are released under CC BY 4.0. This
code repository does not relicense any of them. Masks and volumes stay
out of git. This project produces technical segmentation metrics. It does
not produce a diagnostic device, a clinical claim, or a substitute for
radiologist review.
