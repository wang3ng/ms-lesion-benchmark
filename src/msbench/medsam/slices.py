"""Axial slices for MedSAM: orientation, intensity normalisation and 1024x1024 geometry.

Every volume is put in a canonical voxel order with the axial (S-I) axis last
(`to_canonical` / `from_canonical`, lossless axis flips/permutations only).
Channels are FLAIR, T1, T2 -> the three SAM input channels. Each channel is
clipped to its 0.5-99.5 percentile inside the brain and scaled to [0, 1]
(MedSAM feeds [0, 1] images straight into the image encoder).

Slices are mapped to 1024x1024 keeping their physical aspect ratio: the longer
physical side becomes 1024 px, the other is scaled by the same mm/px and the
image is zero-padded at the bottom/right. `SliceGeometry` records this so that
boxes can be mapped in and masks mapped back to the original voxel grid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

CHANNELS = ("flair", "t1", "t2")
SIDE = 1024


# ---------------------------------------------------------------- orientation
def canonical_ornt(affine: np.ndarray) -> np.ndarray:
    """Orientation transform that brings the array to RAS-like order (axial axis last)."""
    return nib.orientations.ornt_transform(nib.io_orientation(affine), nib.orientations.axcodes2ornt("RAS"))


def to_canonical(arr: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return nib.orientations.apply_orientation(arr, canonical_ornt(affine))


def from_canonical(arr: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Inverse of to_canonical for an array in canonical order."""
    inv = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt("RAS"), nib.io_orientation(affine))
    return nib.orientations.apply_orientation(arr, inv)


def canonical_spacing(img: nib.Nifti1Image) -> tuple[float, float, float]:
    """Voxel size (mm) along the canonical axes."""
    ornt = canonical_ornt(img.affine)
    zooms = np.asarray(img.header.get_zooms()[:3], float)
    out = np.zeros(3)
    for src_axis, (dst_axis, _flip) in enumerate(ornt):
        out[int(dst_axis)] = zooms[src_axis]
    return tuple(float(z) for z in out)


# ---------------------------------------------------------------- volumes
@dataclass
class Volume:
    image: np.ndarray            # (3, X, Y, Z) float32 in [0, 1], canonical order
    brain: np.ndarray            # (X, Y, Z) bool
    spacing: tuple[float, float, float]
    affine: np.ndarray           # original affine (to invert the orientation)
    shape: tuple[int, int, int]  # original array shape
    clip: dict                   # channel -> [low, high] intensity percentiles used


def load_volume(case_dir: Path, channels: tuple[str, ...] = CHANNELS) -> Volume:
    ref = nib.load(str(case_dir / f"{channels[0]}.nii.gz"))
    bm_path = case_dir / "brain_mask.nii.gz"
    if bm_path.exists():
        brain = np.asanyarray(nib.load(str(bm_path)).dataobj) > 0
    else:  # e.g. ISBI: released skull-stripped, no separate mask
        brain = np.asanyarray(ref.dataobj) != 0
    brain = to_canonical(brain, ref.affine)
    chans, clip = [], {}
    for ch in channels:
        img = nib.load(str(case_dir / f"{ch}.nii.gz"))
        if img.shape != ref.shape:
            raise ValueError(f"{case_dir.name}: {ch} shape {img.shape} != {ref.shape}")
        a = to_canonical(np.asanyarray(img.dataobj).astype(np.float32), ref.affine)
        vals = a[brain] if brain.any() else a.ravel()
        lo, hi = np.percentile(vals, [0.5, 99.5])
        chans.append(np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32))
        clip[ch] = [float(lo), float(hi)]
    return Volume(np.stack(chans), brain, canonical_spacing(ref), ref.affine, tuple(ref.shape), clip)


def brain_slices(vol: Volume) -> np.ndarray:
    """Indices (canonical axial axis) of slices that contain brain."""
    return np.flatnonzero(vol.brain.any(axis=(0, 1)))


# ---------------------------------------------------------------- 1024 geometry
@dataclass
class SliceGeometry:
    shape: tuple[int, int]       # in-plane voxel shape (canonical axes 0, 1)
    scaled: tuple[int, int]      # size inside the 1024 canvas
    px_mm: float                 # mm per 1024-pixel

    @classmethod
    def from_slice(cls, shape: tuple[int, int], spacing: tuple[float, float]) -> "SliceGeometry":
        ext = (shape[0] * spacing[0], shape[1] * spacing[1])
        px_mm = max(ext) / SIDE
        scaled = (min(SIDE, int(round(ext[0] / px_mm))), min(SIDE, int(round(ext[1] / px_mm))))
        return cls(tuple(shape), scaled, px_mm)

    def to_dict(self) -> dict:
        return asdict(self)

    def image_to_1024(self, img: torch.Tensor) -> torch.Tensor:
        """(C, H, W) or (N, C, H, W) slice(s) -> zero-padded (.., C, 1024, 1024), bilinear."""
        squeeze = img.dim() == 3
        x = img[None] if squeeze else img
        x = F.interpolate(x, size=self.scaled, mode="bilinear", align_corners=False)
        x = F.pad(x, (0, SIDE - self.scaled[1], 0, SIDE - self.scaled[0]))
        return x[0] if squeeze else x

    def mask_to_1024(self, mask: np.ndarray) -> np.ndarray:
        """Binary (H, W) mask -> (1024, 1024) bool, nearest neighbour (for training targets)."""
        t = torch.from_numpy(mask.astype(np.float32))[None, None]
        t = F.interpolate(t, size=self.scaled, mode="nearest")
        t = F.pad(t, (0, SIDE - self.scaled[1], 0, SIDE - self.scaled[0]))
        return t[0, 0].numpy() > 0.5

    def logits_from_1024(self, logits: torch.Tensor) -> torch.Tensor:
        """(N, 1, 1024, 1024) or (N, 1, 256, 256) logits -> (N, 1, H, W) on the voxel grid."""
        if logits.shape[-1] != SIDE:
            logits = F.interpolate(logits, size=(SIDE, SIDE), mode="bilinear", align_corners=False)
        logits = logits[..., : self.scaled[0], : self.scaled[1]]
        return F.interpolate(logits, size=self.shape, mode="bilinear", align_corners=False)

    def box_to_1024(self, box_vox: np.ndarray) -> np.ndarray:
        """Box (x0, y0, x1, y1) in voxel units of the slice (x = axis 1, y = axis 0) -> 1024 px."""
        sy = self.scaled[0] / self.shape[0]
        sx = self.scaled[1] / self.shape[1]
        b = np.asarray(box_vox, float)
        return np.array([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy])
