"""Lesion-wise detection metrics (precision, recall, F1), overall and by size bin.

Matching rule (modelled on the MSSEG-2016 evaluation, Commowick et al. 2018):

* A reference lesion is DETECTED if the predicted components overlapping it
  cover at least ``alpha`` of its volume, counting only predicted components
  that are "not much larger" than the lesion, i.e. whose volume lying outside
  the reference mask is at most ``gamma`` of their own volume.
* A predicted component is a TRUE POSITIVE if at least ``alpha`` of its volume
  overlaps the reference mask; otherwise it is a false positive.

Recall = detected reference lesions / reference lesions.
Precision = true-positive predicted components / predicted components.

The threshold values below are placeholders taken from the project plan.
They MUST be checked against animaSegPerfAnalyzer on a few cases before
final results, and changed only together with the tests.
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

DEFAULT_ALPHA = 0.10  # minimum overlap fraction
DEFAULT_GAMMA = 0.65  # maximum fraction of a predicted component outside the reference


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
    # Convention: if both masks have no lesions, the case is perfect (1.0).
    if n_ref == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0
    recall = tp_ref / n_ref if n_ref else float("nan")
    precision = tp_pred / n_pred if n_pred else float("nan")
    if n_ref == 0:  # only false positives
        return 0.0, recall, 0.0
    if n_pred == 0:  # everything missed
        return precision, 0.0, 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _match(
    ref_lab: np.ndarray,
    n_ref: int,
    pred_lab: np.ndarray,
    n_pred: int,
    alpha: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean arrays: detected[ref_i], true_positive[pred_j]."""
    ref_sizes = np.bincount(ref_lab.ravel(), minlength=n_ref + 1)
    pred_sizes = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)
    ref_any = ref_lab > 0
    pred_any = pred_lab > 0

    # Overlap table between every reference lesion and predicted component.
    both = ref_any & pred_any
    pairs = np.stack([ref_lab[both], pred_lab[both]], axis=1)
    overlap = np.zeros((n_ref + 1, n_pred + 1), dtype=np.int64)
    if len(pairs):
        np.add.at(overlap, (pairs[:, 0], pairs[:, 1]), 1)

    # Fraction of each predicted component lying outside the reference mask.
    pred_in_ref = np.bincount(pred_lab[ref_any].ravel(), minlength=n_pred + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        outside_frac = 1.0 - pred_in_ref / np.maximum(pred_sizes, 1)
    not_too_large = outside_frac <= gamma
    not_too_large[0] = False

    detected = np.zeros(n_ref, dtype=bool)
    for i in range(1, n_ref + 1):
        covered = overlap[i, not_too_large].sum()
        detected[i - 1] = covered >= alpha * ref_sizes[i]

    true_pos = np.zeros(n_pred, dtype=bool)
    for j in range(1, n_pred + 1):
        true_pos[j - 1] = pred_in_ref[j] >= alpha * pred_sizes[j]
    return detected, true_pos


def lesion_detection(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    connectivity: int = DEFAULT_CONNECTIVITY,
    min_lesion_mm3: float = DEFAULT_MIN_LESION_MM3,
    alpha: float = DEFAULT_ALPHA,
    gamma: float = DEFAULT_GAMMA,
) -> DetectionResult:
    """Lesion-wise precision, recall and F1 for one case."""
    if np.shape(pred) != np.shape(ref):
        raise ValueError(f"shape mismatch: {np.shape(pred)} vs {np.shape(ref)}")
    ref_lab, n_ref = label_lesions(ref, spacing, connectivity, min_lesion_mm3)
    pred_lab, n_pred = label_lesions(pred, spacing, connectivity, min_lesion_mm3)
    detected, true_pos = _match(ref_lab, n_ref, pred_lab, n_pred, alpha, gamma)
    tp_ref, tp_pred = int(detected.sum()), int(true_pos.sum())
    p, r, f = _prf(tp_ref, n_ref, tp_pred, n_pred)
    return DetectionResult(n_ref, n_pred, tp_ref, tp_pred, n_pred - tp_pred, p, r, f)


def lesion_detection_by_size(
    pred: np.ndarray,
    ref: np.ndarray,
    bin_edges_mm3: Sequence[float],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    connectivity: int = DEFAULT_CONNECTIVITY,
    min_lesion_mm3: float = DEFAULT_MIN_LESION_MM3,
    alpha: float = DEFAULT_ALPHA,
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
    detected, true_pos = _match(ref_lab, n_ref, pred_lab, n_pred, alpha, gamma)
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
