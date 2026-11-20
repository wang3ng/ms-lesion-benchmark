"""E9 (exploratory, after the results freeze): combined MSSEG + MSLesSeg training set and its exports."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from msbench.config import paths

CHANNELS = ("flair", "t1", "t2")


def training_cases(cfg: dict) -> pd.DataFrame:
    """MSSEG training (15) + MSLesSeg training scans not from GE or unknown scanners (per scan)."""
    ts = cfg["e9"]["training_set"]
    index = pd.read_csv(paths(cfg)["data_processed"] / "index.csv", dtype=str)
    ms = index[(index.dataset == "msseg2016") & (index.release_split == ts["msseg2016"])]
    ml = index[(index.dataset == "mslesseg") & (index.release_split == ts["mslesseg"]["split"])]
    bad = ml.scanner.isin(ts["mslesseg"]["exclude_scanner"]) | ml.scanner.apply(
        lambda s: any(str(s).startswith(p) for p in ts["mslesseg"]["exclude_scanner_prefix"]))
    out = pd.concat([ms, ml[~bad]]).sort_values("case_id").reset_index(drop=True)
    assert not out.scanner.str.startswith("ge_").any() and (out.scanner != "unknown").all()
    test = index[((index.dataset == "msseg2016") & (index.release_split == "test")) | (index.dataset == "isbi2015")]
    assert not set(out.case_id) & set(test.case_id), "training overlaps a test set"
    return out


def excluded_cases(cfg: dict) -> pd.DataFrame:
    ts = cfg["e9"]["training_set"]
    index = pd.read_csv(paths(cfg)["data_processed"] / "index.csv", dtype=str)
    ml = index[(index.dataset == "mslesseg") & (index.release_split == ts["mslesseg"]["split"])]
    return ml[~ml.case_id.isin(training_cases(cfg).case_id)]


def medsam_steps(cfg: dict, n_scans: int) -> int:
    m = cfg["e9"]["medsam_e2c"]
    return int(round(m["reference_steps"] * n_scans / m["reference_scans"]))


def export_nnunet(cfg: dict, raw_root: Path) -> Path:
    """nnUNet_raw/Dataset502_...: imagesTr (FLAIR/T1/T2 = _0000/_0001/_0002) + labelsTr (lesion). Idempotent."""
    nn = cfg["e9"]["nnunet"]
    out = raw_root / nn["dataset_name"]
    tr = training_cases(cfg)
    dj = out / "dataset.json"
    if dj.exists() and json.loads(dj.read_text())["numTraining"] == len(tr):
        return out
    if out.exists():
        shutil.rmtree(out)
    (out / "imagesTr").mkdir(parents=True)
    (out / "labelsTr").mkdir()
    processed = paths(cfg)["data_processed"]
    for c, rel in zip(tr.case_id, tr.path):
        for i, ch in enumerate(CHANNELS):
            shutil.copyfile(processed / rel / f"{ch}.nii.gz", out / "imagesTr" / f"{c}_{i:04d}.nii.gz")
        shutil.copyfile(processed / rel / "lesion.nii.gz", out / "labelsTr" / f"{c}.nii.gz")
    dj.write_text(json.dumps({
        "name": nn["dataset_name"],
        "description": "E9 (exploratory): MSSEG-2016 training (consensus) + MSLesSeg training (non-GE, known scanners), FLAIR/T1/T2",
        "channel_names": {"0": "FLAIR", "1": "T1", "2": "T2"}, "labels": {"background": 0, "lesion": 1},
        "numTraining": len(tr), "file_ending": ".nii.gz"}, indent=2))
    return out


def hardnet_split(cfg: dict, out: Path) -> Path:
    """Split file for MS-SAM-LESS: fold0 = all combined scans for training, no validation."""
    processed = paths(cfg)["data_processed"]
    tr = training_cases(cfg)
    entries = [{**{m: str(processed / r / f"{m}.nii.gz") for m in CHANNELS}, "mask": str(processed / r / "lesion.nii.gz"),
                "case_id": c} for c, r in zip(tr.case_id, tr.path)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fold0": {"train": entries, "val": []}, "test": []}, indent=1))
    return out
