"""Shared metric implementation used for every model and condition."""

from typing import Sequence

import numpy as np

from .components import DEFAULT_CONNECTIVITY, DEFAULT_MIN_LESION_MM3, label_lesions
from .detection import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DetectionResult,
    bin_names,
    lesion_detection,
    lesion_detection_by_size,
    lesion_records,
    lesion_volume_histogram,
)
from .overlap import dice, lesion_count, lesion_volume_ml
from .segquality import (
    average_surface_distance,
    hd95,
    iou,
    lesion_segmentation_quality,
    segmentation_quality,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    "DEFAULT_CONNECTIVITY",
    "DEFAULT_GAMMA",
    "DEFAULT_MIN_LESION_MM3",
    "DetectionResult",
    "average_surface_distance",
    "bin_names",
    "dice",
    "evaluate_case",
    "hd95",
    "iou",
    "label_lesions",
    "lesion_count",
    "lesion_detection",
    "lesion_detection_by_size",
    "lesion_records",
    "lesion_segmentation_quality",
    "lesion_volume_histogram",
    "lesion_volume_ml",
    "segmentation_quality",
]


def evaluate_case(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing: Sequence[float],
    bin_edges_mm3: Sequence[float],
    **rule,
) -> dict:
    """All per-case metrics in one flat dict (one row of the results table)."""
    det = lesion_detection(pred, ref, spacing, **rule)
    row = {
        "dice": dice(pred, ref),
        "lesion_precision": det.precision,
        "lesion_recall": det.recall,
        "lesion_f1": det.f1,
        "fp_lesions": det.fp_pred,
        "n_ref_lesions": det.n_ref,
        "n_pred_lesions": det.n_pred,
        "ref_volume_ml": lesion_volume_ml(ref, spacing, **_cc(rule)),
        "pred_volume_ml": lesion_volume_ml(pred, spacing, **_cc(rule)),
    }
    for name, res in lesion_detection_by_size(pred, ref, bin_edges_mm3, spacing, **rule).items():
        row[f"f1_{name}"] = res.f1
        row[f"recall_{name}"] = res.recall
        row[f"precision_{name}"] = res.precision
        row[f"n_ref_{name}"] = res.n_ref
        row[f"detected_{name}"] = res.tp_ref   # for pooled (lesion-level) recall per bin
        row[f"n_pred_{name}"] = res.n_pred
    return row


def _cc(rule: dict) -> dict:
    return {k: v for k, v in rule.items() if k in ("connectivity", "min_lesion_mm3")}
