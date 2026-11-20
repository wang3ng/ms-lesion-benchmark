"""E8 zoomed tiles: each E2c tile is cropped from the slice and re-embedded on its own 1024 canvas.

Tiles are exactly E2c's (`boxes.independent.tiles`: grid inside the brain bounding box, fractional
overlap), restricted to tiles that contain brain voxels (the others cannot hold lesions and are never
cached or predicted). A tile crop keeps its physical aspect ratio: its longer physical side becomes
1024 px (zoom = 1024 px / tile extent, about 4x the E2c canvas for a 4 x 4 grid). The prompt for a
zoomed tile is the whole crop. Embeddings: cached for MSSEG training cases
(scripts/cache_e8_tiles.py), computed on the fly for test cases (`encode_tiles`).
"""

from __future__ import annotations

import numpy as np
import torch

from msbench.medsam.boxes import box_mask, independent_boxes
from msbench.medsam.slices import SliceGeometry


def zoom_tiles(brain2d: np.ndarray, grid: tuple[int, int], overlap: float) -> np.ndarray:
    """(n, 4) integer tiles (x0, y0, x1, y1) of E2c's grid that contain brain voxels, clipped to the slice."""
    boxes = independent_boxes(brain2d, grid, overlap)[1:]
    h, w = brain2d.shape
    out = []
    for b in boxes:
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 > x0 and y1 > y0 and brain2d[y0:y1, x0:x1].any():
            out.append((x0, y0, x1, y1))
    return np.asarray(out, dtype=int).reshape(-1, 4)


def tile_geometry(tile: np.ndarray, spacing2d: tuple[float, float]) -> SliceGeometry:
    x0, y0, x1, y1 = (int(v) for v in tile)
    return SliceGeometry.from_slice((y1 - y0, x1 - x0), spacing2d)


def full_box_1024(geom: SliceGeometry) -> np.ndarray:
    """Prompt box covering the whole crop, in 1024 px."""
    return geom.box_to_1024(np.array([0, 0, geom.shape[1], geom.shape[0]], float))


def tile_canvases(image2d: torch.Tensor, tiles: np.ndarray, spacing2d: tuple[float, float]) -> torch.Tensor:
    """(3, H, W) slice -> (n, 3, 1024, 1024) zoomed tile canvases."""
    out = []
    for t in tiles:
        x0, y0, x1, y1 = (int(v) for v in t)
        out.append(tile_geometry(t, spacing2d).image_to_1024(image2d[:, y0:y1, x0:x1]))
    return torch.stack(out) if out else image2d.new_zeros((0, 3, 1024, 1024))


@torch.no_grad()
def encode_tiles(encoder, image2d: torch.Tensor, tiles: np.ndarray, spacing2d: tuple[float, float],
                 batch: int = 8) -> torch.Tensor:
    """Frozen-encoder embeddings (n, 256, 64, 64) float16 of the zoomed tiles of one slice."""
    embs = []
    for i in range(0, len(tiles), batch):
        x = tile_canvases(image2d, tiles[i:i + batch], spacing2d)
        with torch.autocast("cuda", dtype=torch.float16):
            embs.append(encoder(x).half())
    return torch.cat(embs) if embs else torch.zeros((0, 256, 64, 64), dtype=torch.float16, device=image2d.device)


def paste_tile(prob_crop: torch.Tensor, tile: np.ndarray, shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Place a (h, w) crop probability into a slice-sized map; returns (prob map, coverage mask)."""
    x0, y0, x1, y1 = (int(v) for v in tile)
    full = prob_crop.new_zeros(shape)
    full[y0:y1, x0:x1] = prob_crop
    cov = prob_crop.new_zeros(shape)
    cov[y0:y1, x0:x1] = 1
    return full, cov


def same_region_as_box_mask(tile: np.ndarray, shape: tuple[int, int]) -> bool:
    """A tile crop covers exactly box_mask(tile) (used in tests)."""
    m = np.zeros(shape, bool)
    x0, y0, x1, y1 = (int(v) for v in tile)
    m[y0:y1, x0:x1] = True
    return bool((m == box_mask(tile, shape)).all())
