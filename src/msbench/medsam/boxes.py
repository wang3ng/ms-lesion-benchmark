"""Box prompts for the MedSAM conditions, in voxel units of a canonical axial slice.

Boxes are (x0, y0, x1, y1) with x along canonical axis 1 (columns) and y along
axis 0 (rows), half-open in voxels; SliceGeometry.box_to_1024 maps them to the
1024 canvas.

* E2a oracle:      one box per 2D (8-connected) lesion component of the reference
                   in that slice, each side pushed out by U{0..jitter} voxels.
* E2b nnunet:      the same, from the components of an E1 prediction (out-of-fold
                   for training patients, ensemble for test cases).
* E2c independent: per brain slice, the brain bounding box plus a grid of
                   overlapping tiles inside it (lesion-free tiles included).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

_EIGHT = np.ones((3, 3), dtype=bool)


def component_boxes(mask2d: np.ndarray) -> np.ndarray:
    """Tight boxes (n, 4) around 8-connected components of a 2D mask."""
    lab, n = ndimage.label(mask2d, structure=_EIGHT)
    boxes = []
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        ys, xs = sl
        boxes.append([xs.start, ys.start, xs.stop, ys.stop])
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def jitter_boxes(boxes: np.ndarray, shape: tuple[int, int], max_px: int, rng: np.random.Generator) -> np.ndarray:
    """Push every side of every box outwards by an integer in [0, max_px] voxels, clipped to the slice."""
    if len(boxes) == 0 or max_px <= 0:
        return boxes.copy()
    d = rng.integers(0, max_px + 1, size=boxes.shape)
    out = boxes + d * np.array([-1, -1, 1, 1])
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, shape[1])
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, shape[0])
    return out


def oracle_boxes(ref2d: np.ndarray, max_jitter: int, rng: np.random.Generator) -> np.ndarray:
    return jitter_boxes(component_boxes(ref2d), ref2d.shape, max_jitter, rng)


def independent_boxes(brain2d: np.ndarray, grid: tuple[int, int] = (4, 4), overlap: float = 0.25) -> np.ndarray:
    """Brain bounding box + grid[0] x grid[1] tiles covering it with the given fractional overlap."""
    if not brain2d.any():
        return np.zeros((0, 4))
    ys, xs = np.nonzero(brain2d)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    boxes = [[x0, y0, x1, y1]]

    def edges(lo, hi, n):
        # n tiles of width w with step w * (1 - overlap) covering [lo, hi)
        w = (hi - lo) / (n - (n - 1) * overlap) if n > 1 else hi - lo
        starts = [lo + k * w * (1 - overlap) for k in range(n)]
        return [(s, min(hi, s + w)) for s in starts]

    for ya, yb in edges(y0, y1, grid[0]):
        for xa, xb in edges(x0, x1, grid[1]):
            boxes.append([np.floor(xa), np.floor(ya), np.ceil(xb), np.ceil(yb)])
    return np.asarray(boxes, dtype=float)


def box_mask(box: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    m[max(y0, 0):min(y1, shape[0]), max(x0, 0):min(x1, shape[1])] = True
    return m
