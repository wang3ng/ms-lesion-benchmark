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

`processed/<dataset_id>/<case_id>/` should contain one NIfTI per channel
(`flair.nii.gz`, and `t1.nii.gz`, `t2.nii.gz`, `pd.nii.gz`, `t1gd.nii.gz`
when that dataset has them), `lesion.nii.gz` for the consensus or chosen
rater mask, and `meta.json` with scanner, site, subject, and timepoint.

Do not download data into the repository root. Record the release URL, the
date, and any access ticket in `data/raw/<dataset_id>/SOURCE.txt` (that file
is gitignored with the rest of `data/raw/`).

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
