"""E7 inference: their preprocessing + 1 mm resampling, slice-wise HarDNet segmentor, mapped back to the case grid.

Forward (as their training pipeline): FLAIR/T1/T2 -> z-score, scale to [0, 1] (each) -> 1 mm isotropic ->
RAS -> stack. Each axial slice (3, X, Y) -> their ResizeWithRatio (longest in-plane side -> dim, the other
int(side * dim / longest)) -> symmetric zero pad to dim x dim -> network -> sigmoid, channel 1.
Back: resize to the canvas, remove the padding, resize to (X, Y) -> probability volume on the 1 mm RAS grid
-> trilinear resampling onto the case's own voxel grid through the two affines. Ensembles average the
1 mm probability volumes of the fold models before mapping back; final mask = probability > 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from msbench.e7 import MODALITIES


@dataclass
class Canvas:
    xy: tuple[int, int]      # in-plane size of the 1 mm slice
    resized: tuple[int, int]
    pad: tuple[int, int]     # leading zero padding on each axis
    dim: int

    @classmethod
    def of(cls, xy: tuple[int, int], dim: int) -> "Canvas":
        h, w = xy
        if h > w:
            nh, nw = dim, int(w * (dim / h))
        elif w > h:
            nh, nw = int(h * (dim / w)), dim
        else:
            nh, nw = dim, dim
        return cls(tuple(xy), (nh, nw), ((dim - nh) // 2, (dim - nw) // 2), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, X, Y) -> (N, C, dim, dim)."""
        x = F.interpolate(x, size=self.resized, mode="bilinear", align_corners=False)
        (ph, pw), (nh, nw) = self.pad, self.resized
        return F.pad(x, (pw, self.dim - nw - pw, ph, self.dim - nh - ph))

    def backward(self, p: torch.Tensor) -> torch.Tensor:
        """(N, C, h, w) network output (any resolution) -> (N, C, X, Y)."""
        p = F.interpolate(p, size=(self.dim, self.dim), mode="bilinear", align_corners=False)
        (ph, pw), (nh, nw) = self.pad, self.resized
        return F.interpolate(p[..., ph:ph + nh, pw:pw + nw], size=self.xy, mode="bilinear", align_corners=False)


def preprocess(paths: dict, keys=MODALITIES, interp: str = "bilinear"):
    """-> (image (C, X, Y, Z) float32 numpy on the 1 mm RAS grid, A1 (4x4), A0 (4x4), original shape)."""
    from monai.transforms import (Compose, ConcatItemsd, EnsureChannelFirstD, LoadImageD, NormalizeIntensityD,
                                  OrientationD, ScaleIntensityD, SpacingD)
    keys = list(keys)
    t = [LoadImageD(keys), EnsureChannelFirstD(keys=keys)]
    if interp == "bilinear":
        t += [NormalizeIntensityD(keys=keys), ScaleIntensityD(keys=keys)]
    t += [SpacingD(keys, pixdim=(1., 1., 1.), mode=interp), OrientationD(keys, axcodes="RAS"),
          ConcatItemsd(keys, name="image")]
    d = Compose(t)({k: str(paths[k]) for k in keys})
    img = d["image"]
    a0 = np.asarray(d[keys[0]].meta["original_affine"], float)
    shape = tuple(int(s) for s in d[keys[0]].meta["spatial_shape"])
    return img.numpy().astype(np.float32), np.asarray(img.affine, float), a0, shape


def to_original(prob: np.ndarray, a1: np.ndarray, a0: np.ndarray, shape: tuple) -> np.ndarray:
    """Trilinear resampling of a volume on grid a1 onto the original grid (a0, shape)."""
    m = np.linalg.inv(a1) @ a0
    out = np.zeros(shape, np.float32)
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    for k in range(shape[2]):
        ijk = np.stack([ii.ravel(), jj.ravel(), np.full(ii.size, k), np.ones(ii.size)])
        c = (m @ ijk)[:3]
        out[:, :, k] = ndimage.map_coordinates(prob, c, order=1, mode="constant", cval=0.0).reshape(shape[:2])
    return out


@torch.no_grad()
def predict_volume(models: list, image: np.ndarray, dim: int, batch: int = 16, device: str = "cuda") -> np.ndarray:
    """Mean over models of the foreground probability, (X, Y, Z) on the 1 mm RAS grid."""
    c, x, y, z = image.shape
    cv = Canvas.of((x, y), dim)
    vol = torch.from_numpy(image).to(device)
    out = torch.zeros((x, y, z), device=device)
    for k0 in range(0, z, batch):
        sl = vol[..., k0:k0 + batch].permute(3, 0, 1, 2)            # (n, C, X, Y)
        canvas = cv.forward(sl)
        acc = torch.zeros((sl.shape[0], 1, x, y), device=device)
        for m in models:
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = m(canvas)
            acc += cv.backward(torch.sigmoid(logits.float())[:, 1:2])
        out[..., k0:k0 + batch] = (acc[:, 0] / len(models)).permute(1, 2, 0)
    return out.cpu().numpy()


def load_segmentor(ckpt, dim: int, device: str = "cuda"):
    """Their HardNetSegmentor (ConvUpsample head) with our trained weights (strict)."""
    import sys
    from msbench.e7 import UPSTREAM
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from models.segmentor_net import HardNetSegmentor
    from msbench.config import paths
    from msbench.e7.runtime import verified_hub
    # Their HarDNet only truncates itself to the segmentation backbone when pretrained=True, so build it that
    # way (verified local ImageNet file, weights_only) and then overwrite every weight with our checkpoint.
    hub = torch.hub.load_state_dict_from_url
    torch.hub.load_state_dict_from_url = verified_hub(paths()["hardnet85"])
    try:
        args = {"out_channels": 2, "depth_wise": 0, "order": 85, "pretrained": True, "upsample_factor_list": [2, 2],
                "decoder_channels": [], "Idim": dim}
        m = HardNetSegmentor(args, segmentor="ConvUpsample")
    finally:
        torch.hub.load_state_dict_from_url = hub
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    return m.to(device).eval()
