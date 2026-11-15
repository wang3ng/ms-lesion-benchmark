"""Voxel overlap and quantitative biomarkers (lesion count, lesion volume)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .components import (
    DEFAULT_CONNECTIVITY,
    DEFAULT_MIN_LESION_MM3,
    label_lesions,
    voxel_volume_mm3,
)


def dice(pred: np.ndarray, ref: np.ndarray, empty_both: float = 1.0) -> float:
    """Voxel-wise Dice. Returns ``empty_both`` when both masks are empty."""
    pred = np.asarray(pred).astype(bool)
    ref = np.asarray(ref).astype(bool)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {ref.shape}")
    denom = pred.sum() + ref.sum()
    if denom == 0:
        return float(empty_both)
    return float(2.0 * np.logical_and(pred, ref).sum() / denom)


def lesion_count(
    mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    connectivity: int = DEFAULT_CONNECTIVITY,
    min_lesion_mm3: float = DEFAULT_MIN_LESION_MM3,
) -> int:
    return label_lesions(mask, spacing, connectivity, min_lesion_mm3)[1]


def lesion_volume_ml(
    mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    connectivity: int = DEFAULT_CONNECTIVITY,
    min_lesion_mm3: float = DEFAULT_MIN_LESION_MM3,
) -> float:
    """Total lesion volume in mL, after the same min-size filter as the count."""
    labels, _ = label_lesions(mask, spacing, connectivity, min_lesion_mm3)
    return float((labels > 0).sum() * voxel_volume_mm3(spacing) / 1000.0)
