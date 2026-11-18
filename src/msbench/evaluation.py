"""Score predicted masks against a reference with the shared msbench.metrics.

One function for every model and condition (E1, E2a-c, E8), so all
results tables use the same metric code, lesion rule and size bins.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import pandas as pd

from msbench.config import load_config, paths
from msbench.metrics import bin_names, evaluate_case, lesion_records

# Column order of the per-case CSV (size-bin columns are appended per bin).
BASE_COLUMNS = [
    "case_id", "exp", "fold", "dataset", "scanner", "reference",
    "dice", "lesion_precision", "lesion_recall", "lesion_f1", "fp_lesions",
    "n_ref_lesions", "n_pred_lesions", "ref_volume_ml", "pred_volume_ml",
    "ref_voxels", "pred_voxels", "affine_max_diff_mm",
]
BIN_FIELDS = ["recall", "f1", "precision", "n_ref", "detected", "n_pred"]


def detection_rule(cfg: dict) -> dict:
    """Lesion rule parameters from the config (connectivity, min size, alpha, beta, gamma)."""
    r = cfg["metrics"]["lesion_detection"]
    return {k: r[k] for k in ("connectivity", "min_lesion_mm3", "alpha", "beta", "gamma")}


def size_bins(cfg: dict) -> list[float]:
    bins = cfg["metrics"]["size_bins_mm3"]
    if not bins:
        raise ValueError("metrics.size_bins_mm3 is not set in configs/experiment.yaml")
    return [float(b) for b in bins]


def load_mask(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    return data > 0, img.affine, tuple(float(z) for z in img.header.get_zooms()[:3])


def evaluate_predictions(
    pred_dir: Path,
    case_ids: Sequence[str],
    exp: str,
    fold: int | None = None,
    reference: str = "lesion",
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Per-case metrics for <pred_dir>/<case_id>.nii.gz vs processed/<case>/<reference>.nii.gz."""
    cfg = cfg or load_config()
    processed = paths(cfg)["data_processed"]
    index = pd.read_csv(processed / "index.csv", dtype=str).set_index("case_id")
    rule, bins = detection_rule(cfg), size_bins(cfg)

    missing = [c for c in case_ids if not (Path(pred_dir) / f"{c}.nii.gz").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} prediction(s) missing in {pred_dir}: {missing}")
    unknown = [c for c in case_ids if c not in index.index]
    if unknown:
        raise KeyError(f"cases not in index.csv: {unknown}")

    rows = []
    for case_id in case_ids:
        meta = index.loc[case_id]
        ref, ref_aff, spacing = load_mask(processed / meta["path"] / f"{reference}.nii.gz")
        pred, pred_aff, _ = load_mask(Path(pred_dir) / f"{case_id}.nii.gz")
        if pred.shape != ref.shape:
            raise ValueError(f"{case_id}: prediction shape {pred.shape} != reference {ref.shape}")
        # Same voxel grid is what matters; header drift is recorded, not fatal.
        aff_diff = float(np.abs(pred_aff - ref_aff).max())
        if aff_diff > 1e-2:
            warnings.warn(f"{case_id}: prediction affine differs from reference by {aff_diff:.3g} mm")
        row = {
            "case_id": case_id, "exp": exp, "fold": fold,
            "dataset": meta["dataset"], "scanner": meta["scanner"], "reference": reference,
            "ref_voxels": int(ref.sum()), "pred_voxels": int(pred.sum()),
            "affine_max_diff_mm": round(aff_diff, 6),
            **evaluate_case(pred, ref, spacing, bins, **rule),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    bin_cols = [c for c in df.columns if c not in BASE_COLUMNS]
    return df[BASE_COLUMNS + bin_cols]


def lesion_tables(
    pred_dir: Path,
    case_ids: Sequence[str],
    reference: str = "lesion",
    cfg: dict | None = None,
    **extra,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per reference lesion and one per predicted component (error analysis).

    Reference rows: case_id, dataset, scanner, volume_mm3, bin, detected, covered_frac.
    Predicted rows: case_id, dataset, scanner, volume_mm3, bin, true_positive, overlap_frac.
    Same lesion rule and bins as evaluate_predictions; `extra` columns (e.g. fold=k) are added.
    """
    cfg = cfg or load_config()
    processed = paths(cfg)["data_processed"]
    index = pd.read_csv(processed / "index.csv", dtype=str).set_index("case_id")
    rule, bins = detection_rule(cfg), size_bins(cfg)
    names = bin_names(bins)
    ref_rows, pred_rows = [], []
    for case_id in case_ids:
        meta = index.loc[case_id]
        ref, _, spacing = load_mask(processed / meta["path"] / f"{reference}.nii.gz")
        pred, _, _ = load_mask(Path(pred_dir) / f"{case_id}.nii.gz")
        r = lesion_records(pred, ref, spacing, **rule)
        base = {"case_id": case_id, "dataset": meta["dataset"], "scanner": meta["scanner"], **extra}
        for v, d, c in zip(r["ref_volume_mm3"], r["ref_detected"], r["ref_covered_frac"]):
            ref_rows.append({**base, "volume_mm3": float(v), "bin": names[int(np.digitize(v, bins))],
                             "detected": bool(d), "covered_frac": float(c)})
        for v, t, o in zip(r["pred_volume_mm3"], r["pred_true_positive"], r["pred_overlap_frac"]):
            pred_rows.append({**base, "volume_mm3": float(v), "bin": names[int(np.digitize(v, bins))],
                              "true_positive": bool(t), "overlap_frac": float(o)})
    cols_r = ["case_id", "dataset", "scanner", *extra, "volume_mm3", "bin", "detected", "covered_frac"]
    cols_p = ["case_id", "dataset", "scanner", *extra, "volume_mm3", "bin", "true_positive", "overlap_frac"]
    return pd.DataFrame(ref_rows, columns=cols_r), pd.DataFrame(pred_rows, columns=cols_p)
