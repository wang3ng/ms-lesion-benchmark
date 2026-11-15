"""MSSEG-2016 (MICCAI challenge) converter.

Release layout::

    MSSEG-{Training,Testing}/{Training,Testing}/Center_XX/Patient_YY/
        Preprocessed_Data/{FLAIR,T1,T2,DP,GADO}_preprocessed.nii.gz
        Masks/{Consensus,Brain_Mask,ManualSegmentation_1..7}.nii.gz

Centers and scanners come from the release README. Center 03 (GE Discovery)
appears only in the test set; it is the unseen-scanner setting.
Patient numbers restart in each split, so the split is part of the case ID.
"""

from __future__ import annotations

import re
from pathlib import Path

from .common import CaseSpec

DATASET = "msseg2016"
N_EXPERTS = 7

CENTERS = {
    "01": ("siemens_verio_3t", "rennes"),
    "03": ("ge_discovery_3t", "bordeaux"),
    "07": ("siemens_aera_1p5t", "lyon"),
    "08": ("philips_ingenia_3t", "lyon"),
}
SPLITS = {"Training": "train", "Testing": "test"}


def discover(raw_root: Path) -> list[CaseSpec]:
    root = Path(raw_root) / DATASET
    specs = []
    for release, split in SPLITS.items():
        for case_dir in sorted(root.glob(f"MSSEG-{release}/{release}/Center_*/Patient_*")):
            center = re.fullmatch(r"Center_(\d+)", case_dir.parent.name).group(1)
            patient = re.fullmatch(r"Patient_(\d+)", case_dir.name).group(1)
            if center not in CENTERS:
                raise ValueError(f"unknown MSSEG center {center} in {case_dir}")
            scanner, site = CENTERS[center]
            case_id = f"msseg_{split}_c{center}_p{patient}"
            pre, masks = case_dir / "Preprocessed_Data", case_dir / "Masks"
            specs.append(CaseSpec(
                case_id=case_id,
                dataset=DATASET,
                release_split=split,
                subject_id=case_id,           # one exam per patient
                timepoint="",
                center=center,
                scanner=scanner,
                site=site,
                images={
                    "flair": pre / "FLAIR_preprocessed.nii.gz",
                    "t1": pre / "T1_preprocessed.nii.gz",
                    "t2": pre / "T2_preprocessed.nii.gz",
                },
                masks={
                    "lesion": masks / "Consensus.nii.gz",
                    **{f"lesion_rater{k}": masks / f"ManualSegmentation_{k}.nii.gz"
                       for k in range(1, N_EXPERTS + 1)},
                    "brain_mask": masks / "Brain_Mask.nii.gz",
                },
                primary_reference="lesion",
            ))
    return specs
