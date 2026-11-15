"""Export MSSEG-2016 training cases from processed/ to nnU-Net raw format.

    source scripts/env.sh
    python -m msbench.nnunet_export            # writes Dataset501_MSSEG2016
    python -m msbench.nnunet_export --splits   # after preprocessing: write splits_final.json

Channels: 0 FLAIR, 1 T1, 2 T2 (z-score normalisation in nnU-Net).
Label: the MSSEG consensus (lesion.nii.gz). Only release_split == train is
exported; test cases are never seen by nnU-Net planning or training.
Folds: patient-level, stratified by scanner, seed from the config. They are
written to nnUNet_preprocessed/<dataset>/splits_final.json (nnU-Net uses an
existing file instead of its own default split) and copied to experiments/E1/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from msbench.config import REPO_ROOT, load_config, nnunet_dataset_name, paths
from msbench.data import CHANNELS
from msbench.splits import patient_kfold

CHANNEL_NAMES = {"flair": "FLAIR", "t1": "T1", "t2": "T2"}


def _nnunet_dir(var: str) -> Path:
    try:
        return Path(os.environ[var])
    except KeyError:
        raise SystemExit(f"{var} is not set; run `source scripts/env.sh` first") from None


def training_cases(cfg: dict) -> pd.DataFrame:
    index = pd.read_csv(paths(cfg)["data_processed"] / "index.csv", dtype=str)
    ds = cfg["primary_dataset"]
    tr = index[(index.dataset == ds) & (index.release_split == "train")].sort_values("case_id")
    allowed = set(cfg["splits"][ds]["train_scanners"])
    bad = set(tr.scanner) - allowed
    if bad:
        raise SystemExit(f"training cases from unexpected scanners {bad}")
    return tr


def export(cfg: dict, overwrite: bool = False) -> Path:
    assert tuple(cfg["channels"]) == CHANNELS, "channel order must match the config"
    out = _nnunet_dir("nnUNet_raw") / nnunet_dataset_name(cfg)
    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to rebuild")
        shutil.rmtree(out)
    (out / "imagesTr").mkdir(parents=True)
    (out / "labelsTr").mkdir()

    processed = paths(cfg)["data_processed"]
    tr = training_cases(cfg)
    for case_id, rel in zip(tr.case_id, tr.path):
        src = processed / rel
        for i, ch in enumerate(CHANNELS):
            shutil.copyfile(src / f"{ch}.nii.gz", out / "imagesTr" / f"{case_id}_{i:04d}.nii.gz")
        shutil.copyfile(src / "lesion.nii.gz", out / "labelsTr" / f"{case_id}.nii.gz")

    dataset_json = {
        "name": nnunet_dataset_name(cfg),
        "description": "MSSEG-2016 training set (consensus labels), FLAIR/T1/T2, msbench export",
        "channel_names": {str(i): CHANNEL_NAMES[ch] for i, ch in enumerate(CHANNELS)},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(tr),
        "file_ending": ".nii.gz",
    }
    (out / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"exported {len(tr)} cases to {out}")
    print(tr.groupby("scanner").size().to_string())
    return out


def write_splits(cfg: dict) -> Path:
    tr = training_cases(cfg)
    n_folds = cfg["statistics"]["cross_validation"]["folds"]
    splits = patient_kfold(tr.case_id.tolist(), tr.subject_id.tolist(), tr.scanner.tolist(),
                           n_folds, cfg["seed"])
    pre = _nnunet_dir("nnUNet_preprocessed") / nnunet_dataset_name(cfg)
    if not pre.exists():
        raise SystemExit(f"{pre} missing; run nnUNetv2_plan_and_preprocess first")
    text = json.dumps(splits, indent=2)
    (pre / "splits_final.json").write_text(text)
    rec = REPO_ROOT / "experiments" / "E1"
    rec.mkdir(parents=True, exist_ok=True)
    (rec / "splits_final.json").write_text(text)
    scanner = dict(zip(tr.case_id, tr.scanner))
    for k, s in enumerate(splits):
        print(f"fold {k}: val = " + ", ".join(f"{c} ({scanner[c]})" for c in s["val"]))
    return pre / "splits_final.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", action="store_true", help="write splits_final.json only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.splits:
        write_splits(cfg)
    else:
        export(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
