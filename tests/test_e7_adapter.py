import importlib.util

import nibabel as nib
import numpy as np
import pytest
import torch

pytest.importorskip("monai")
from msbench.e7 import UPSTREAM  # noqa: E402
from msbench.e7.predict import Canvas, preprocess, to_original  # noqa: E402


def _case(tmp_path):
    """Anisotropic, non-RAS (LPS) volume with a ball 'lesion' and three modalities."""
    shape = (90, 80, 30)
    aff = np.array([[-0.5, 0, 0, 40.0], [0, -0.6, 0, 30.0], [0, 0, 1.3, -20.0], [0, 0, 0, 1]])
    zz, yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    ball = ((zz - 50) * 0.5) ** 2 + ((yy - 35) * 0.6) ** 2 + ((xx - 14) * 1.3) ** 2 < 6.0 ** 2
    rng = np.random.default_rng(0)
    paths = {}
    for k in ("flair", "t1", "t2"):
        paths[k] = tmp_path / f"{k}.nii.gz"
        nib.save(nib.Nifti1Image((rng.random(shape) + 3 * ball).astype(np.float32), aff), paths[k])
    paths["mask"] = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(ball.astype(np.uint8), aff), paths["mask"])
    return paths, ball, aff


def test_canvas_shapes_and_roundtrip():
    cv = Canvas.of((180, 216), 512)
    assert cv.resized == (426, 512) and cv.pad == (43, 0)
    x = torch.rand(2, 3, 180, 216)
    back = cv.backward(cv.forward(x))
    assert back.shape == x.shape and torch.allclose(back.mean(), x.mean(), atol=1e-2)


def test_geometry_roundtrip_to_original_grid(tmp_path):
    paths, ball, aff = _case(tmp_path)
    img, a1, a0, shape = preprocess(paths)
    assert img.shape[0] == 3 and shape == ball.shape and np.allclose(a0, aff)
    assert np.allclose(np.abs(np.diag(a1)[:3]), 1.0)           # 1 mm, RAS
    mk, a1m, a0m, _ = preprocess({"mask": paths["mask"]}, keys=("mask",), interp="nearest")
    assert np.allclose(a1m, a1)
    vol = torch.from_numpy(mk[0])
    cv = Canvas.of(vol.shape[:2], 512)
    sl = vol.permute(2, 0, 1)[:, None].float()                     # identity "network" through the canvas
    prob = cv.backward(cv.forward(sl))[:, 0].permute(1, 2, 0).numpy()
    back = to_original(prob, a1, a0, shape) > 0.5
    dice = 2 * (back & ball).sum() / (back.sum() + ball.sum())
    assert dice > 0.9, dice


@pytest.mark.skipif(not (UPSTREAM / "dataset" / "tfs.py").exists(), reason="third_party/MS-SAM-LESS not cloned")
def test_canvas_matches_their_resize_and_pad():
    import sys
    sys.path.insert(0, str(UPSTREAM))
    from monai.transforms import ResizeWithPadOrCropd
    from dataset.tfs import ResizeWithRatio
    x = torch.rand(3, 180, 216, 1)
    d = ResizeWithRatio(keys=["image"], image_size=512, mode="trilinear")({"image": x})
    d = ResizeWithPadOrCropd(keys=["image"], spatial_size=(512, 512, -1), mode="constant")(d)
    ours = Canvas.of((180, 216), 512).forward(x[..., 0][None])[0]
    assert torch.allclose(torch.as_tensor(d["image"])[..., 0], ours, atol=1e-5)
