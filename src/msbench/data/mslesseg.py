"""MSLesSeg (Figshare 27919209) converter. Development dataset only.

Release layout (preprocessed, MNI 1 mm)::

    MSLesSeg Dataset/train/P<n>/T<k>/P<n>_T<k>_{FLAIR,T1,T2,MASK}.nii.gz
    MSLesSeg Dataset/test/P<n>/P<n>_{FLAIR,T1,T2,MASK}.nii.gz
    MSLesSeg Dataset/info_dataset/patient_scanners_info/patient_scanners_info_FLAIR.csv

The train/test folders are the official split. Note that the timepoint
folder "T1" is unrelated to the T1-weighted image. Scanner per scan is read
from the FLAIR scanner table (Patient, Timepoint, MANUFACTURER, SCANNER).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .common import CaseSpec

DATASET = "mslesseg"
RELEASE_DIR = Path("27919209") / "MSLesSeg Dataset"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def _scanner_table(root: Path) -> dict[tuple[str, str], str]:
    csv = root / "info_dataset" / "patient_scanners_info" / "patient_scanners_info_FLAIR.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv, sep=";", dtype=str).dropna(subset=["Patient", "Timepoint"])
    table = {}
    for p, t, manu, model in zip(df.Patient, df.Timepoint, df.MANUFACTURER, df.SCANNER):
        manu, model = str(manu), str(model)
        if manu.strip().upper() in ("NONE", "NAN", ""):
            name = "unknown"
        else:
            vendor = _norm(manu).split("_")[0]          # "Philips Medical Systems" -> philips
            name = f"{vendor}_{_norm(model)}"
        table[(p.strip(), t.strip())] = name
    return table


def discover(raw_root: Path) -> list[CaseSpec]:
    root = Path(raw_root) / DATASET / RELEASE_DIR
    scanners = _scanner_table(root)
    found: list[tuple[str, str, str, Path]] = []  # split, patient, timepoint, file stem
    for flair in sorted(root.glob("train/P*/T*/P*_T*_FLAIR.nii.gz")):
        found.append(("train", flair.parent.parent.name, flair.parent.name,
                      flair.parent / flair.name[: -len("_FLAIR.nii.gz")]))
    for flair in sorted(root.glob("test/P*/P*_FLAIR.nii.gz")):
        patient = flair.parent.name
        tps = sorted(t for (p, t) in scanners if p == patient)
        tp = tps[0] if len(tps) == 1 else ""
        found.append(("test", patient, tp, flair.parent / flair.name[: -len("_FLAIR.nii.gz")]))

    specs = []
    for split, patient, tp, stem in found:
        case_id = f"mslesseg_{patient}" + (f"_{tp}" if tp else "")
        specs.append(CaseSpec(
            case_id=case_id,
            dataset=DATASET,
            release_split=split,
            subject_id=f"mslesseg_{patient}",
            timepoint=tp,
            center="",
            scanner=scanners.get((patient, tp), "unknown"),
            site="",
            images={
                "flair": Path(f"{stem}_FLAIR.nii.gz"),
                "t1": Path(f"{stem}_T1.nii.gz"),
                "t2": Path(f"{stem}_T2.nii.gz"),
            },
            masks={"lesion": Path(f"{stem}_MASK.nii.gz")},
            primary_reference="lesion",
        ))
    return specs
