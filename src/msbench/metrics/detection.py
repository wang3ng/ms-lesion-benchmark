"""Lesion-wise detection metrics (precision, recall, F1), overall and by size bin.

Port of the lesion detection rule of animaSegPerfAnalyzer (Anima 4.2,
SegPerfCAnalyzer::getDetectionMarks / getTruePositiveLesions /
falsePositiveRatioTester), the tool used for the MSSEG-2016 challenge
(Commowick et al., Sci Rep 2018). Defaults are the command-line defaults of
animaSegPerfAnalyzer: -x alpha 0.10, -y beta 0.70, -z gamma 0.65, -v 3 mm^3.

Both masks are split into 6-connected components; components of volume
<= min_lesion_mm3 are removed (they become background). With the overlap table
O[i, j] = voxels in reference component i and predicted component j (index 0 =
background):

* Reference lesion i is DETECTED if
  (a) the fraction of i covered by predicted components is > alpha, and
  (b) going through the predicted components that overlap i in decreasing order
      of overlap, until they account for >= gamma of the covered voxels, none of
      them has more than beta of its own volume outside the reference mask.
* Predicted component j is a TRUE POSITIVE under the same test with the roles of
  the masks swapped: > alpha of j lies inside reference lesions, and none of the
  reference lesions that make up the first gamma of that overlap is more than
  beta uncovered by the prediction.

Recall = detected reference lesions / reference lesions (Anima SensL).
Precision = true-positive predicted components / predicted components (PPVL).
Empty masks: no reference lesion and no predicted lesion -> 1 for all three
(project convention; Anima returns NaN); reference lesions but no prediction ->
precision 0, recall 0, F1 0 (as Anima); prediction but no reference lesion ->
precision 0, recall NaN, F1 0 (as Anima).

Verified against the animaSegPerfAnalyzer 4.2 binary: see scripts/anima_check.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .components import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_MIN_LESION_MM3,
    component_volumes_mm3,
    label_lesions,
)

DEFAULT_ALPHA = 0.10  # -x MinOverlapRatio: min covered fraction (strictly greater)
DEFAULT_BETA = 0.70   # -y MaxFalsePositiveRatio: max fraction of an overlapping component outside
DEFAULT_GAMMA = 0.65  # -z MaxFalsePositiveRatioModerator: share of the overlap that is checked


@dataclass
class DetectionResult:
    n_ref: int
    n_pred: int
    tp_ref: int  # detected reference lesions
    tp_pred: int  # predicted components that overlap the reference
    fp_pred: int  # false-positive predicted components
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict:
        return asdict(self)


def _prf(tp_ref: int, n_ref: int, tp_pred: int, n_pred: int) -> tuple[float, float, float]:
    """(precision, recall, F1) with Anima's conventions, except both-empty -> 1 (project choice)."""
    if n_ref == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0
    precision = tp_pred / n_pred if n_pred else 0.0
    recall = tp_ref / n_ref if n_ref else float("nan")
    if n_ref == 0 or n_pred == 0:
        return precision, recall, 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _overlap_table(ref_lab: np.ndarray, n_ref: int, pred_lab: np.ndarray, n_pred: int) -> np.ndarray:
    """O[i, j]: voxels with reference label i and predicted label j (0 = background)."""
    fg = (ref_lab > 0) | (pred_lab > 0)
    table = np.zeros((n_ref + 1, n_pred + 1), dtype=np.int64)
    np.add.at(table, (ref_lab[fg], pred_lab[fg]), 1)
    table[0, 0] = ref_lab.size - int(fg.sum())
    return table


def _anima_true_positives(table: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """SegPerfCAnalyzer::getTruePositiveLesions for the rows 1..n of `table` (bool per row)."""
    n_rows, n_cols = table.shape
    out = np.zeros(n_rows - 1, dtype=bool)
    if n_rows < 2 or n_cols < 2:  # Anima only counts when both images have lesions
        return out
    col_sum = table.sum(axis=0)       # whole size of each column component (incl. background row)
    for i in range(1, n_rows):
        row = table[i, 1:]
        tp = int(row.sum())
        fn = int(table[i, 0])
        if tp / (tp + fn) <= alpha:
            continue
        # falsePositiveRatioTester: columns by decreasing overlap with row i.
        order = np.argsort(-row, kind="stable") + 1
        weight, exited = 0.0, False
        for j in order:
            if weight >= gamma:
                break
            if table[0, j] / col_sum[j] > beta:
                exited = True
                break
            weight += table[i, j] / tp
        out[i - 1] = not exited
    return out


def _match(
    ref_lab: np.ndarray,
    n_ref: int,
    pred_lab: np.ndarray,
    n_pred: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean arrays: detected[ref_i], true_positive[pred_j] (Anima rule)."""
    table = _overlap_table(ref_lab, n_ref, pred_lab, n_pred)
    detected = _anima_true_positives(table, alpha, beta, gamma)
    true_pos = _anima_true_positives(table.T, alpha, beta, gamma)
    return detected, true_pos


def lesion_detection(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    connectivity: int = DEFAULT_CONNECTIVITY,
    min_lesion_mm3: float = DEFAULT_MIN_LESION_MM3,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> DetectionResult:
    """Lesion-wise precision, recall and F1 for one case."""
    if np.shape(pred) != np.shape(ref):
        raise ValueError(f"shape mismatch: {np.shape(pred)} vs {np.shape(ref)}")
    ref_lab, n_ref = label_lesions(ref, spacing, connectivity, min_lesion_mm3)
    pred_lab, n_pred = label_lesions(pred, spacing, connectivity, min_lesion_mm3)
    detected, true_pos = _match(ref_lab, n_ref, pred_lab, n_pred, alpha, beta, gamma)
    tp_ref, tp_pred = int(detected.sum()), int(true_pos.sum())
    p, r, f = _prf(tp_ref, n_ref, tp_pred, n_pred)
    return DetectionResult(n_ref, n_pred, tp_ref, tp_pred, n_pred - tp_pred, p, r, f)


def lesion_records(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    connectivity: int = DEFAULT_CONNECTIVITY,
    min_lesion_mm3: float = DEFAULT_MIN_LESION_MM3,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> dict[str, np.ndarray]:
    """Per-lesion outcome under the same rule as lesion_detection (for error analysis).

    Returns arrays, one entry per reference lesion:
      ref_volume_mm3, ref_detected, ref_covered_frac (fraction of the lesion covered by any
      predicted component, before the beta/gamma test)
    and one entry per predicted component:
      pred_volume_mm3, pred_true_positive, pred_overlap_frac (fraction inside the reference mask).
    Counts agree with lesion_detection / lesion_detection_by_size by construction.
    """
    if np.shape(pred) != np.shape(ref):
        raise ValueError(f"shape mismatch: {np.shape(pred)} vs {np.shape(ref)}")
    ref_lab, n_ref = label_lesions(ref, spacing, connectivity, min_lesion_mm3)
    pred_lab, n_pred = label_lesions(pred, spacing, connectivity, min_lesion_mm3)
    detected, true_pos = _match(ref_lab, n_ref, pred_lab, n_pred, alpha, beta, gamma)
    ref_sizes = np.bincount(ref_lab.ravel(), minlength=n_ref + 1)[1:]
    pred_sizes = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)[1:]
    ref_cov = np.bincount(ref_lab[pred_lab > 0].ravel(), minlength=n_ref + 1)[1:]
    pred_in = np.bincount(pred_lab[ref_lab > 0].ravel(), minlength=n_pred + 1)[1:]
    return {
        "ref_volume_mm3": component_volumes_mm3(ref_lab, n_ref, spacing),
        "ref_detected": detected,
        "ref_covered_frac": ref_cov / np.maximum(ref_sizes, 1),
        "pred_volume_mm3": component_volumes_mm3(pred_lab, n_pred, spacing),
        "pred_true_positive": true_pos,
        "pred_overlap_frac": pred_in / np.maximum(pred_sizes, 1),
    }


def lesion_detection_by_size(
    pred: np.ndarray,
    ref: np.ndarray,
    bin_edges_mm3: Sequence[float],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    connectivity: int = DEFAULT_CONNECTIVITY,
    min_lesion_mm3: float = DEFAULT_MIN_LESION_MM3,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> dict[str, DetectionResult]:
    """Detection metrics within lesion-volume bins.

    ``bin_edges_mm3`` are the inner edges, e.g. [20, 100] gives bins
    "<20", "20-100" and ">=100". Reference lesions are binned by their own
    volume (recall); predicted components by theirs (precision).
    Bin edges must be fixed from TRAINING-set lesion counts only.
    """
    edges = list(bin_edges_mm3)
    if edges != sorted(edges):
        raise ValueError("bin edges must be increasing")
    names = bin_names(edges)
    ref_lab, n_ref = label_lesions(ref, spacing, connectivity, min_lesion_mm3)
    pred_lab, n_pred = label_lesions(pred, spacing, connectivity, min_lesion_mm3)
    detected, true_pos = _match(ref_lab, n_ref, pred_lab, n_pred, alpha, beta, gamma)
    ref_bin = np.digitize(component_volumes_mm3(ref_lab, n_ref, spacing), edges)
    pred_bin = np.digitize(component_volumes_mm3(pred_lab, n_pred, spacing), edges)

    out: dict[str, DetectionResult] = {}
    for b, name in enumerate(names):
        r_sel, p_sel = ref_bin == b, pred_bin == b
        nr, npd = int(r_sel.sum()), int(p_sel.sum())
        tr, tp = int(detected[r_sel].sum()), int(true_pos[p_sel].sum())
        p, r, f = _prf(tr, nr, tp, npd)
        out[name] = DetectionResult(nr, npd, tr, tp, npd - tp, p, r, f)
    return out


def bin_names(edges: Sequence[float]) -> list[str]:
    if not edges:
        return ["all"]
    fmt = lambda x: f"{x:g}"  # noqa: E731
    names = [f"<{fmt(edges[0])}"]
    names += [f"{fmt(a)}-{fmt(b)}" for a, b in zip(edges[:-1], edges[1:])]
    names.append(f">={fmt(edges[-1])}")
    return names


def lesion_volume_histogram(
    masks_and_spacings: Sequence[tuple[np.ndarray, Sequence[float]]],
    candidate_edges_mm3: Sequence[float],
    connectivity: int = DEFAULT_CONNECTIVITY,
    min_lesion_mm3: float = DEFAULT_MIN_LESION_MM3,
) -> dict[str, int]:
    """Count reference lesions per volume bin across TRAINING masks.

    Use this to choose bin edges so that each bin has enough lesions,
    before any model is evaluated.
    """
    counts = np.zeros(len(candidate_edges_mm3) + 1, dtype=int)
    for mask, spacing in masks_and_spacings:
        lab, n = label_lesions(mask, spacing, connectivity, min_lesion_mm3)
        vols = component_volumes_mm3(lab, n, spacing)
        counts += np.bincount(np.digitize(vols, candidate_edges_mm3), minlength=len(counts))
    return dict(zip(bin_names(list(candidate_edges_mm3)), counts.tolist()))
