"""Segmentation-quality metrics (added after the results freeze; detection rule unchanged).

Lesion-wise metrics use the same lesions and matching as `lesion_detection` (Anima-port rule:
6-connected components > min_lesion_mm3, alpha/beta/gamma test). For each DETECTED reference lesion
R_i, its prediction P_i is the union of the predicted components that overlap R_i (so a predicted
component that merges several lesions is compared with each of them). Then:

  lesion Dice  = 2 |R_i ∩ P_i| / (|R_i| + |P_i|)
  lesion HD95  = max(95th pct of d(∂R_i → ∂P_i), 95th pct of d(∂P_i → ∂R_i))   (mm; MONAI convention)
  lesion ASD   = mean of all those surface distances, both directions                (mm, symmetric)

Surfaces are the mask voxels with a 6-neighbour outside the mask; distances use the voxel spacing.
Missed reference lesions have no value (NaN): these metrics are conditional on detection.

Whole-mask metrics on the raw binary masks (as whole-mask Dice): HD95 (NaN if either mask is empty;
a single distant false positive can dominate it) and IoU = |A ∩ B| / |A ∪ B| (1 if both are empty).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage

from .components import DEFAULT_CONNECTIVITY, DEFAULT_MIN_LESION_MM3, component_volumes_mm3, label_lesions
from .detection import DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA, _match, _overlap_table, bin_names

_FACE = ndimage.generate_binary_structure(3, 1)


def _surface(m: np.ndarray) -> np.ndarray:
    return m & ~ndimage.binary_erosion(m, structure=_FACE, border_value=0)


def surface_distances(a: np.ndarray, b: np.ndarray, spacing: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Distances (mm) from each surface voxel of a to the surface of b, and from b to a. Both non-empty."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    idx = np.argwhere(a | b)
    lo, hi = np.maximum(idx.min(0) - 1, 0), idx.max(0) + 2   # crop to the union (+1 voxel)
    sl = tuple(slice(int(x), int(y)) for x, y in zip(lo, hi))
    a, b = a[sl], b[sl]
    sa, sb = _surface(a), _surface(b)
    da = ndimage.distance_transform_edt(~sb, sampling=spacing)[sa]
    db = ndimage.distance_transform_edt(~sa, sampling=spacing)[sb]
    return da, db


def hd95(a: np.ndarray, b: np.ndarray, spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> float:
    """95th-percentile symmetric Hausdorff distance (mm); NaN if either mask is empty."""
    if not np.any(a) or not np.any(b):
        return float("nan")
    da, db = surface_distances(a, b, spacing)
    return float(max(np.percentile(da, 95), np.percentile(db, 95)))


def average_surface_distance(a: np.ndarray, b: np.ndarray, spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> float:
    """Symmetric average surface distance (mm); NaN if either mask is empty."""
    if not np.any(a) or not np.any(b):
        return float("nan")
    da, db = surface_distances(a, b, spacing)
    return float(np.concatenate([da, db]).mean())


def iou(pred: np.ndarray, ref: np.ndarray, empty_both: float = 1.0) -> float:
    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    union = int((pred | ref).sum())
    return float(empty_both) if union == 0 else float((pred & ref).sum() / union)


def lesion_segmentation_quality(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    connectivity: int = DEFAULT_CONNECTIVITY,
    min_lesion_mm3: float = DEFAULT_MIN_LESION_MM3,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> dict[str, np.ndarray]:
    """Per reference lesion: volume_mm3, detected, dice, hd95, asd (NaN when not detected)."""
    if np.shape(pred) != np.shape(ref):
        raise ValueError(f"shape mismatch: {np.shape(pred)} vs {np.shape(ref)}")
    ref_lab, n_ref = label_lesions(ref, spacing, connectivity, min_lesion_mm3)
    pred_lab, n_pred = label_lesions(pred, spacing, connectivity, min_lesion_mm3)
    detected, _ = _match(ref_lab, n_ref, pred_lab, n_pred, alpha, beta, gamma)
    table = _overlap_table(ref_lab, n_ref, pred_lab, n_pred)
    out = {k: np.full(n_ref, np.nan) for k in ("dice", "hd95", "asd")}
    objs = ndimage.find_objects(ref_lab)
    pobjs = ndimage.find_objects(pred_lab) if n_pred else []
    for i in np.flatnonzero(detected):
        js = np.flatnonzero(table[i + 1, 1:] > 0) + 1
        # bounding box of the lesion and its overlapping predicted components
        boxes = [objs[i]] + [pobjs[j - 1] for j in js]
        lo = [min(b[d].start for b in boxes) for d in range(3)]
        hi = [max(b[d].stop for b in boxes) for d in range(3)]
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        r = ref_lab[sl] == i + 1
        p = np.isin(pred_lab[sl], js)
        out["dice"][i] = 2 * (r & p).sum() / (r.sum() + p.sum())
        da, db = surface_distances(r, p, spacing)
        out["hd95"][i] = max(np.percentile(da, 95), np.percentile(db, 95))
        out["asd"][i] = np.concatenate([da, db]).mean()
    return {"volume_mm3": component_volumes_mm3(ref_lab, n_ref, spacing), "detected": detected, **out}


def segmentation_quality(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing: Sequence[float],
    bin_edges_mm3: Sequence[float],
    **rule,
) -> tuple[dict, dict[str, np.ndarray]]:
    """One case: (flat row of per-case values, per-lesion records).

    Row: lesion_dice / lesion_hd95 / lesion_asd = mean over the detected reference lesions (NaN if
    none), the same per size bin (reference lesion volume), n_detected(_bin), mask_hd95, mask_iou.
    """
    rec = lesion_segmentation_quality(pred, ref, spacing, **rule)
    names = bin_names(list(bin_edges_mm3))
    b = np.digitize(rec["volume_mm3"], list(bin_edges_mm3))
    det = rec["detected"]

    def mean(x, sel):
        return float(np.mean(x[sel])) if sel.any() else float("nan")

    row = {"n_detected": int(det.sum())}
    for m in ("dice", "hd95", "asd"):
        row[f"lesion_{m}"] = mean(rec[m], det)
    for k, name in enumerate(names):
        sel = det & (b == k)
        row[f"n_detected_{name}"] = int(sel.sum())
        for m in ("dice", "hd95", "asd"):
            row[f"lesion_{m}_{name}"] = mean(rec[m], sel)
    row["mask_hd95"] = hd95(pred, ref, spacing)
    row["mask_iou"] = iou(pred, ref)
    rec["bin"] = np.array([names[k] for k in b], dtype=object)
    return row, rec
