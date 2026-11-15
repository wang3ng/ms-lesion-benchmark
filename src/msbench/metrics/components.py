"""Connected-component helpers shared by all lesion-level metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage

# Neighbourhood used to group voxels into lesions.
# 26 matches the submitted proposal. The MSSEG-2016 evaluation tool
# (animaSegPerfAnalyzer) must be checked before the final results; change
# this value only together with tests/test_metrics.py and the report.
DEFAULT_CONNECTIVITY = 26

# Components smaller than this (in mm^3) are ignored, as in the MSSEG-2016
# evaluation. Verify against animaSegPerfAnalyzer before final results.
DEFAULT_MIN_LESION_MM3 = 3.0


def _structure(connectivity: int) -> np.ndarray:
    rank_by_conn = {6: 1, 18: 2, 26: 3}
    if connectivity not in rank_by_conn:
        raise ValueError(f"connectivity must be 6, 18 or 26, got {connectivity}")
    return ndimage.generate_binary_structure(3, rank_by_conn[connectivity])


def voxel_volume_mm3(spacing: Sequence[float]) -> float:
    if len(spacing) != 3:
        raise ValueError("spacing must have 3 values (mm)")
    return float(np.prod(spacing))


def label_lesions(
    mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    connectivity: int = DEFAULT_CONNECTIVITY,
    min_lesion_mm3: float = DEFAULT_MIN_LESION_MM3,
) -> tuple[np.ndarray, int]:
    """Label lesions in a binary 3D mask, dropping components below min size.

    Returns (labels, n) where labels are consecutive integers 1..n.
    """
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 3:
        raise ValueError(f"expected a 3D mask, got shape {mask.shape}")
    labels, n = ndimage.label(mask, structure=_structure(connectivity))
    if n == 0:
        return labels.astype(np.int32), 0
    sizes = np.bincount(labels.ravel())[1:]
    min_voxels = min_lesion_mm3 / voxel_volume_mm3(spacing)
    keep = np.flatnonzero(sizes >= min_voxels) + 1
    relabel = np.zeros(n + 1, dtype=np.int32)
    relabel[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    return relabel[labels], int(len(keep))


def component_volumes_mm3(labels: np.ndarray, n: int, spacing: Sequence[float]) -> np.ndarray:
    """Volume of each labelled component 1..n in mm^3."""
    if n == 0:
        return np.zeros(0)
    counts = np.bincount(labels.ravel(), minlength=n + 1)[1 : n + 1]
    return counts * voxel_volume_mm3(spacing)
