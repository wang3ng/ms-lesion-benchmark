#!/usr/bin/env python3
"""Small real-data nnU-Net 2D train on MSLesSeg (subset) for william 6GB GPU."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path.home() / "projects/ms-lesion-benchmark"
RAW = ROOT / "data/raw/mslesseg"
ZIP_PATH = RAW / "MSLesSeg Dataset.zip"
ZIP_URL = "https://ndownloader.figshare.com/files/52771814"
ZIP_BYTES = 1473284755
NNUNET_RAW = ROOT / "data/nnUNet_raw"
DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID:03d}_MSLesSegSmall"
N_CASES = 12  # keep small for 6GB + short wall time
UA = "msbench-small-train/0.1"


def download_zip() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size == ZIP_BYTES:
        print(f"skip download ({ZIP_PATH.name})")
        return
    print(f"downloading {ZIP_URL} -> {ZIP_PATH}")
    partial = ZIP_PATH.with_suffix(".zip.partial")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": UA}
    if 0 < existing < ZIP_BYTES:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(ZIP_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        status = getattr(resp, "status", 200)
        mode = "ab"
        if status == 200 and existing:
            mode = "wb"
            existing = 0
        done = existing
        with open(partial, mode) as out:
            while True:
                block = resp.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if done % (50 * 1024 * 1024) < 1024 * 1024:
                    print(f"  {done / 1e6:.0f}/{ZIP_BYTES / 1e6:.0f} MB", flush=True)
    if partial.stat().st_size != ZIP_BYTES:
        raise SystemExit(f"size mismatch: {partial.stat().st_size} != {ZIP_BYTES}")
    partial.replace(ZIP_PATH)
    print("download complete")


def extract_zip() -> Path:
    extract_dir = RAW / "extracted"
    marker = extract_dir / ".extracted_ok"
    if marker.exists():
        print("skip extract")
        return extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok\n")
    return extract_dir


def find_case_pairs(extract_dir: Path) -> list[tuple[Path, Path]]:
    """Return (flair, mask) pairs. Prefer train/ over test/."""
    nii = [p for p in extract_dir.rglob("*.nii*") if p.is_file()]
    by_case: dict[str, dict[str, Path]] = {}
    for p in nii:
        name = p.name.lower()
        # MSLesSeg: P12_FLAIR.nii.gz / P12_MASK.nii.gz
        stem = p.name.split(".")[0]
        if "_" not in stem:
            continue
        case_id, modality = stem.split("_", 1)
        modality = modality.lower()
        slot = by_case.setdefault(case_id, {})
        if "flair" in modality:
            slot["flair"] = p
        elif "mask" in modality or "seg" in modality or "label" in modality:
            slot["mask"] = p

    pairs = []
    for case_id, slot in sorted(by_case.items()):
        if "flair" in slot and "mask" in slot:
            pairs.append((slot["flair"], slot["mask"]))

    def prefer_train(item: tuple[Path, Path]) -> tuple[int, str]:
        flair = str(item[0]).lower()
        # train first, then validation-like, then test
        rank = 0 if "/train/" in flair.replace("\\", "/") else (
            1 if "/val" in flair.replace("\\", "/") else 2
        )
        return (rank, flair)

    pairs.sort(key=prefer_train)
    print(f"found {len(pairs)} flair/mask pairs under {extract_dir}")
    if pairs:
        print("example:", pairs[0][0], "->", pairs[0][1])
    else:
        print("sample nifti paths:")
        for p in nii[:30]:
            print(" ", p.relative_to(extract_dir))
    return pairs


def convert_subset(pairs: list[tuple[Path, Path]]) -> None:
    ds = NNUNET_RAW / DATASET_NAME
    images = ds / "imagesTr"
    labels = ds / "labelsTr"
    if ds.exists():
        shutil.rmtree(ds)
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    used = pairs[:N_CASES]
    for i, (flair, mask) in enumerate(used, start=1):
        case = f"mslesseg_{i:03d}"
        img = nib.load(str(flair))
        seg = nib.load(str(mask))
        img_data = np.asanyarray(img.dataobj)
        seg_data = np.asanyarray(seg.dataobj)
        if img_data.ndim == 4:
            img_data = img_data[..., 0]
        if seg_data.ndim == 4:
            seg_data = seg_data[..., 0]
        if img_data.shape != seg_data.shape:
            raise SystemExit(
                f"shape mismatch {case}: img {img_data.shape} seg {seg_data.shape}"
            )
        # binary lesion
        seg_bin = (seg_data > 0).astype(np.uint8)
        # nnU-Net expects RAS-ish via nifti; keep affine from flair
        nib.save(nib.Nifti1Image(img_data.astype(np.float32), img.affine), images / f"{case}_0000.nii.gz")
        nib.save(nib.Nifti1Image(seg_bin, img.affine), labels / f"{case}.nii.gz")
        print(f"converted {case}: shape={img_data.shape} fg={int(seg_bin.sum())}")

    dataset = {
        "channel_names": {"0": "FLAIR"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(used),
        "file_ending": ".nii.gz",
    }
    (ds / "dataset.json").write_text(json.dumps(dataset, indent=2))
    print(f"wrote {ds} with {len(used)} cases")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(ROOT / "data/nnUNet_preprocessed")
    env["nnUNet_results"] = str(ROOT / "data/nnUNet_results")
    subprocess.check_call(cmd, env=env)


def main() -> int:
    download_zip()
    extract_dir = extract_zip()
    pairs = find_case_pairs(extract_dir)
    if len(pairs) < 4:
        raise SystemExit("need at least 4 cases for a small train")
    convert_subset(pairs)
    run(
        [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            str(DATASET_ID),
            "-c",
            "2d",
            "--verify_dataset_integrity",
        ]
    )
    # 20 epochs is still a small train; fits overnight-ish but should be minutes on tiny 2d
    run(
        [
            "nnUNetv2_train",
            str(DATASET_ID),
            "2d",
            "0",
            "-tr",
            "nnUNetTrainer_20epochs",
        ]
    )
    # print summary from log-ish results folder
    results = (
        ROOT
        / "data/nnUNet_results"
        / DATASET_NAME
        / "nnUNetTrainer_20epochs__nnUNetPlans__2d"
        / "fold_0"
    )
    summary = results / "summary.json"
    if summary.exists():
        print("==== summary.json ====")
        print(summary.read_text()[:4000])
    else:
        # fallback: look for validation metrics file
        for p in results.rglob("*.json"):
            if "summary" in p.name.lower() or "validation" in p.name.lower():
                print("====", p, "====")
                print(p.read_text()[:4000])
    print("SMALL_TRAIN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
