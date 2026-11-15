# Data

Imaging stays on local disk. This directory is the only place raw volumes,
preprocessed caches, and nnU-Net working folders should live, and those
paths are gitignored.

Expected layout after a dataset is unpacked:

```
data/
  raw/<dataset_id>/          # files as released
  interim/<dataset_id>/      # renamed NIfTI, channel index, QC notes
  processed/<dataset_id>/    # common layout consumed by evaluation
  nnUNet_raw/
  nnUNet_preprocessed/
  nnUNet_results/
```

**Where the data actually live.** Paths are set in `configs/experiment.yaml`
(`paths:`), relative to the repo root. On the lab machine raw data are in
`CV_MS_proj/data/raw/`, the common layout in `CV_MS_proj/data/processed/`,
and our nnU-Net folders in `CV_MS_proj/nnunet/` (exported by
`source scripts/env.sh`, which overrides the lab mate's `nnUNet_*` in
`~/.bashrc`).

### Common case layout (`python -m msbench.data.convert`)

```
processed/
  index.csv            one row per case (scanner, split, shape, spacing, primary-reference stats)
  lesion_stats.csv     one row per (case, reference mask): voxels, lesion count, volume mL
  <dataset>/<case_id>/
    flair.nii.gz t1.nii.gz t2.nii.gz   FLAIR/T1/T2 only (CLAUDE.md); release's preprocessed images
    lesion.nii.gz                      primary reference: MSSEG consensus, MSLesSeg mask
    lesion_rater<k>.nii.gz             MSSEG experts 1-7, ISBI raters 1-2
    brain_mask.nii.gz                  MSSEG only
    meta.json                          scanner, site, subject, timepoint, sources, header fixes, stats
```

- ISBI has no consensus, so it has no `lesion.nii.gz`; score each rater.
- `release_split` is the release's own split. MSSEG `test` holds the GE
  (`ge_discovery_3t`, center 03) unseen-scanner cases. ISBI cases come from
  the release's `train` folder but are our external test set.
- `subject_id` is the cross-validation unit (ISBI/MSLesSeg timepoints share it).
- All images and masks are on the FLAIR grid. Cases whose grids differ fail;
  affine differences < 0.01 mm (one MSSEG test case) are fixed by writing the
  FLAIR header, and listed under `header_fixes` in `meta.json`.
- Masks are uint8 {0,1} and carry the FLAIR qform **and** sform verbatim.

Do not download data into the repository root. The download script writes
`data/raw/<dataset_id>/SOURCE.txt` with the URL and the UTC time. That file
is gitignored with the rest of `data/raw/`.

## Do not commit the archives

Keep the zip files and the unpacked volumes out of git. ISBI training is
787 MB and the test archive is 2.2 GB. MSLesSeg's preprocessed zip is about
1.37 GB. 3D-MR-MS is six parts, about 4.3 GB, under CC BY-NC-ND, which does
not allow redistributing the files. A collaborator clones this repo and
runs the script; they do not pull the scans from git history.

```bash
python scripts/download_datasets.py
python scripts/download_datasets.py isbi2015 mslesseg
python scripts/download_datasets.py 3d-mr-ms
python scripts/download_datasets.py mslesseg --include-optional
```

URLs and expected byte counts are in `configs/downloads.yaml`. A finished
file whose size matches the manifest is skipped, so the command is safe to
re-run.

## Official links

### ISBI 2015

Challenge page: https://iacl.ece.jhu.edu/index.php/MSChallenge

| Archive | URL | Size on the IACL page |
| --- | --- | --- |
| Training | https://iacl.ece.jhu.edu/~aaron/data/training_final_v4.zip | 787 MB |
| Test | https://iacl.ece.jhu.edu/~aaron/data/testdata_website_2016-03-24.zip | 2.2 GB |

### MSLesSeg

Landing page:
https://springernature.figshare.com/articles/dataset/MSLesSeg_baseline_and_benchmarking_of_a_new_Multiple_Sclerosis_Lesion_Segmentation_dataset/27919209

DOI: https://doi.org/10.6084/m9.figshare.27919209

The script fetches the preprocessed `MSLesSeg Dataset.zip` (Figshare file
52771814, 1,473,284,755 bytes). `MSLesSeg_RAW.zip` is optional.

### 3D-MR-MS

Resource page: https://lit.fe.uni-lj.si/en/research/resources/3D-MR-MS/

Authors listed there: Žiga Lesjak, Alfiia Galimzianova, Žiga Špiclin.
License: CC BY-NC-ND. Thirty patients in six zip files, plus a short PDF
description. Direct file URLs are the `./data/research/resources/3D-MR-MS/`
links on that page, recorded in the manifest.

## What to fetch first

1. **MS3SEG NIfTI** if the goal is a same-day baseline. About 2 GB of
   preprocessed NIfTI, no approval step. Figshare:
   https://doi.org/10.6084/m9.figshare.30393475
2. **MSLesSeg** if the goal is the published MS-SAM-LESS split. Training
   subset is 53 patients; the full Figshare release is 75 patients / 115
   scans. https://doi.org/10.6084/m9.figshare.27919209
3. **MSSEG-2016** for the primary cross-scanner study. Register on Shanoir
   and request the MSSEG-2016 study:
   https://portal.fli-iam.irisa.fr/msseg-challenge/english-msseg-data/
   The public portal has at times released training only (15 cases). The
   test set (38 cases), including the GE scanner absent from training, is
   required for the primary claim.
4. **ISBI 2015 training** (5 subjects) for cross-dataset transfer.
   https://www.nitrc.org/projects/longitudinal_ms/

Dataset facts, splits, and citations are in the repository README.
