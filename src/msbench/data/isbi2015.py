"""ISBI 2015 longitudinal MS challenge converter (labelled training subjects).

Release layout::

    training_final_v4/training/trainingNN/
        preprocessed/trainingNN_TT_{flair,mprage,t2,pd}_pp.nii
        masks/trainingNN_TT_mask{1,2}.nii

Only the 5 training subjects (21 scans) have masks; the challenge test set
does not. There is no consensus: each rater is kept separately
(lesion_rater1, lesion_rater2) and scored separately. T1 is the MPRAGE.
All scans were acquired on one Philips 3T scanner (Carass et al., 2017).
Timepoints of one subject share a subject_id so they never cross folds.
"""

from __future__ import annotations

import re
from pathlib import Path

from .common import CaseSpec

DATASET = "isbi2015"
SCANNER = "philips_3t"
SITE = "jhu"


def discover(raw_root: Path) -> list[CaseSpec]:
    root = Path(raw_root) / DATASET / "training_final_v4" / "training"
    specs = []
    for flair in sorted(root.glob("training*/preprocessed/training*_*_flair_pp.nii")):
        m = re.fullmatch(r"(training\d+)_(\d+)_flair_pp\.nii", flair.name)
        subject, tp = m.group(1), m.group(2)
        pre, masks = flair.parent, flair.parent.parent / "masks"
        stem = f"{subject}_{tp}"
        specs.append(CaseSpec(
            case_id=f"isbi_{stem}",
            dataset=DATASET,
            release_split="train",
            subject_id=f"isbi_{subject}",
            timepoint=tp,
            center="",
            scanner=SCANNER,
            site=SITE,
            images={
                "flair": flair,
                "t1": pre / f"{stem}_mprage_pp.nii",
                "t2": pre / f"{stem}_t2_pp.nii",
            },
            masks={
                "lesion_rater1": masks / f"{stem}_mask1.nii",
                "lesion_rater2": masks / f"{stem}_mask2.nii",
            },
            primary_reference="",
        ))
    return specs
