"""Common case layout shared by all dataset converters.

Output for each case::

    <processed>/<dataset>/<case_id>/
        flair.nii.gz  t1.nii.gz  t2.nii.gz      # FLAIR/T1/T2 only (CLAUDE.md)
        lesion.nii.gz                           # primary reference, if the dataset has one
        lesion_rater<k>.nii.gz                  # individual raters / experts
        brain_mask.nii.gz                       # when the release provides one
        meta.json

Images are the release's preprocessed images on their native grid. Every
image and mask must share the FLAIR grid; a case whose grid differs is
failed, never silently resampled. Affines that differ by less than
``AFFINE_TOL_MM`` (float rounding in the release) are replaced by the FLAIR
affine and the fix is recorded in meta.json.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from msbench.metrics.components import label_lesions, voxel_volume_mm3

CHANNELS = ("flair", "t1", "t2")
AFFINE_TOL_MM = 1e-2
CONVERTER_VERSION = 2  # 2: copy FLAIR qform+sform verbatim into masks

INDEX_COLUMNS = [
    "case_id", "dataset", "release_split", "subject_id", "timepoint", "center",
    "scanner", "site", "shape", "spacing_mm", "orientation", "references",
    "primary_reference", "lesion_voxels", "n_lesions", "lesion_volume_ml", "path",
]


class ConversionError(RuntimeError):
    pass


@dataclass
class CaseSpec:
    case_id: str
    dataset: str
    release_split: str            # split as named by the release (train / test)
    subject_id: str               # patient-level unit for cross-validation folds
    timepoint: str
    center: str
    scanner: str
    site: str
    images: dict[str, Path]       # keys: CHANNELS
    masks: dict[str, Path]        # keys: lesion, lesion_rater<k>, brain_mask
    primary_reference: str = "lesion"   # "" when the dataset has no single reference
    notes: list[str] = field(default_factory=list)


def _same_grid(img: nib.Nifti1Image, ref: nib.Nifti1Image, tol: float) -> bool:
    return img.shape == ref.shape and np.allclose(img.affine, ref.affine, atol=tol, rtol=0)


def _with_ref_header(arr: np.ndarray, ref: nib.Nifti1Image, dtype) -> nib.Nifti1Image:
    """New image on the FLAIR grid with the FLAIR's qform AND sform copied verbatim.

    nibabel prefers the sform, SimpleITK (nnU-Net) the qform; a few release
    files have qform != sform, so both must match the FLAIR for image and
    mask to agree under either reader.
    """
    out = nib.Nifti1Image(np.asarray(arr, dtype=dtype), ref.affine)
    q, qc = ref.header.get_qform(coded=True)
    s, sc = ref.header.get_sform(coded=True)
    out.set_qform(ref.header.get_qform() if q is None else q, code=int(qc))
    out.set_sform(ref.header.get_sform() if s is None else s, code=int(sc))
    out.header.set_xyzt_units(*ref.header.get_xyzt_units())
    return out


def _load_3d(path: Path) -> nib.Nifti1Image:
    img = nib.load(str(path))
    if len(img.shape) == 4 and img.shape[3] == 1:
        img = img.slicer[..., 0]
    if len(img.shape) != 3:
        raise ConversionError(f"{path.name}: expected 3D, got shape {img.shape}")
    return img


def _write_image(src: Path, dst: Path, ref: nib.Nifti1Image, fixes: list[str], name: str) -> None:
    img = _load_3d(src)
    if not _same_grid(img, ref, AFFINE_TOL_MM):
        raise ConversionError(
            f"{name}: grid differs from FLAIR (shape {img.shape} vs {ref.shape}, "
            f"max affine diff {np.abs(img.affine - ref.affine).max():.4g} mm)"
        )
    exact = np.allclose(img.affine, ref.affine, atol=1e-6, rtol=0)
    if exact and src.name.endswith(".nii.gz") and len(nib.load(str(src)).shape) == 3:
        shutil.copyfile(src, dst)  # byte-exact copy of the release file
        return
    if not exact:
        fixes.append(f"{name}: affine within {AFFINE_TOL_MM} mm of FLAIR; FLAIR affine written")
    data = np.asanyarray(img.dataobj)
    dtype = np.float32 if data.dtype == np.float64 else data.dtype  # scaled ints load as float64
    nib.save(_with_ref_header(data, ref, dtype), str(dst))


def _write_mask(src: Path, dst: Path, ref: nib.Nifti1Image, fixes: list[str], name: str) -> np.ndarray:
    img = _load_3d(src)
    if not _same_grid(img, ref, AFFINE_TOL_MM):
        raise ConversionError(
            f"{name}: grid differs from FLAIR (shape {img.shape} vs {ref.shape}, "
            f"max affine diff {np.abs(img.affine - ref.affine).max():.4g} mm)"
        )
    if not np.allclose(img.affine, ref.affine, atol=1e-6, rtol=0):
        fixes.append(f"{name}: affine within {AFFINE_TOL_MM} mm of FLAIR; FLAIR affine written")
    data = np.asanyarray(img.dataobj)
    values = np.unique(data)
    if not np.all(np.isin(values, (0, 1))):
        nonzero = values[values != 0]
        if len(nonzero) == 1:
            fixes.append(f"{name}: binary mask stored as {{0, {nonzero[0]:g}}}; mapped to {{0, 1}}")
        else:
            raise ConversionError(f"{name}: not a binary mask ({len(values)} distinct values)")
    mask = (data != 0).astype(np.uint8)
    nib.save(_with_ref_header(mask, ref, np.uint8), str(dst))
    return mask


def mask_stats(mask: np.ndarray, spacing) -> dict:
    """Summary statistics only (counts and volumes; no voxel data)."""
    _, n = label_lesions(mask, spacing)
    vox = int(mask.sum())
    return {
        "lesion_voxels": vox,
        "n_lesions": n,
        "lesion_volume_ml": round(vox * voxel_volume_mm3(spacing) / 1000.0, 4),
    }


def convert_case(spec: CaseSpec, out_root: Path, raw_root: Path, overwrite: bool = False) -> dict:
    """Write one case in the common layout and return its meta dict.

    Idempotent: a case with an up-to-date meta.json is skipped.
    """
    out_dir = out_root / spec.dataset / spec.case_id
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text())
        if meta.get("converter_version") == CONVERTER_VERSION:
            return meta

    missing = [str(p) for p in [*spec.images.values(), *spec.masks.values()] if not Path(p).exists()]
    if set(spec.images) != set(CHANNELS):
        raise ConversionError(f"{spec.case_id}: channels {sorted(spec.images)} != {CHANNELS}")
    if missing:
        raise ConversionError(f"{spec.case_id}: missing files {missing}")

    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    ref = _load_3d(spec.images["flair"])
    spacing = [float(z) for z in ref.header.get_zooms()[:3]]
    fixes: list[str] = []
    try:
        for ch in CHANNELS:
            _write_image(Path(spec.images[ch]), tmp_dir / f"{ch}.nii.gz", ref, fixes, ch)
        stats = {}
        for name, src in spec.masks.items():
            mask = _write_mask(Path(src), tmp_dir / f"{name}.nii.gz", ref, fixes, name)
            if name.startswith("lesion"):
                stats[name] = mask_stats(mask, spacing)
    except ConversionError as e:
        shutil.rmtree(tmp_dir)
        raise ConversionError(f"{spec.case_id}: {e}") from None

    d = asdict(spec)
    d.pop("images"), d.pop("masks")
    meta = {
        **d,
        "shape": list(ref.shape),
        "spacing_mm": [round(s, 4) for s in spacing],
        "orientation": "".join(nib.aff2axcodes(ref.affine)),
        "references": sorted(stats),
        "stats": stats,
        "header_fixes": fixes,
        "sources": {
            k: str(Path(v).relative_to(raw_root))
            for k, v in {**spec.images, **spec.masks}.items()
        },
        "converter_version": CONVERTER_VERSION,
    }
    (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    return meta


def build_index(out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild index.csv and lesion_stats.csv from every meta.json under out_root."""
    rows, stat_rows = [], []
    for meta_path in sorted(out_root.glob("*/*/meta.json")):
        m = json.loads(meta_path.read_text())
        prim = m["stats"].get(m["primary_reference"], {}) if m["primary_reference"] else {}
        rows.append({
            **{k: m[k] for k in ("case_id", "dataset", "release_split", "subject_id", "timepoint",
                                 "center", "scanner", "site", "orientation", "primary_reference")},
            "shape": "x".join(map(str, m["shape"])),
            "spacing_mm": "x".join(f"{s:g}" for s in m["spacing_mm"]),
            "references": ";".join(m["references"]),
            "lesion_voxels": prim.get("lesion_voxels"),
            "n_lesions": prim.get("n_lesions"),
            "lesion_volume_ml": prim.get("lesion_volume_ml"),
            "path": str(meta_path.parent.relative_to(out_root)),
        })
        for ref_name, s in m["stats"].items():
            stat_rows.append({"case_id": m["case_id"], "dataset": m["dataset"],
                              "reference": ref_name, **s})
    index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    stats = pd.DataFrame(stat_rows, columns=["case_id", "dataset", "reference",
                                             "lesion_voxels", "n_lesions", "lesion_volume_ml"])
    out_root.mkdir(parents=True, exist_ok=True)
    index.to_csv(out_root / "index.csv", index=False)
    stats.to_csv(out_root / "lesion_stats.csv", index=False)
    return index, stats
