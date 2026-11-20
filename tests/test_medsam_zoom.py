import numpy as np
import torch

from msbench.medsam.boxes import independent_boxes
from msbench.medsam.zoom import full_box_1024, paste_tile, same_region_as_box_mask, tile_canvases, tile_geometry, zoom_tiles


def _brain():
    b = np.zeros((120, 100), bool)
    yy, xx = np.mgrid[:120, :100]
    b[((yy - 60) / 50) ** 2 + ((xx - 50) / 40) ** 2 < 1] = True
    return b


def test_zoom_tiles_are_e2c_tiles_with_brain():
    b = _brain()
    tiles = zoom_tiles(b, (4, 4), 0.25)
    e2c = independent_boxes(b, (4, 4), 0.25)[1:]
    assert 0 < len(tiles) <= 16
    e2c_int = {tuple(int(round(v)) for v in t) for t in e2c}
    assert all(tuple(t) in e2c_int for t in tiles)
    for t in tiles:
        assert b[t[1]:t[3], t[0]:t[2]].any() and same_region_as_box_mask(t, b.shape)
    two = np.zeros_like(b); two[0:10, 0:10] = True; two[110:120, 90:100] = True   # brain in two corners
    t2 = zoom_tiles(two, (4, 4), 0.25)
    assert len(t2) == 2 and len(independent_boxes(two, (4, 4), 0.25)) == 17     # E2c keeps all 16 tiles


def test_tile_geometry_zooms_and_keeps_aspect():
    t = np.array([10, 20, 40, 35])                           # 30 wide, 15 high
    g = tile_geometry(t, (1.0, 1.0))
    assert g.shape == (15, 30) and g.scaled == (512, 1024)
    assert np.allclose(full_box_1024(g), [0, 0, 1024, 512])
    g2 = tile_geometry(t, (2.0, 1.0))                        # anisotropic: 30 mm x 30 mm
    assert g2.scaled == (1024, 1024)


def test_canvas_and_paste_roundtrip():
    img = torch.rand(3, 120, 100)
    t = np.array([[10, 20, 40, 35]])
    c = tile_canvases(img, t, (1.0, 1.0))
    assert c.shape == (1, 3, 1024, 1024) and c[0, :, 512:].abs().sum() == 0   # zero padding below
    full, cov = paste_tile(torch.ones(15, 30), t[0], (120, 100))
    assert full.sum() == 450 and cov[20:35, 10:40].all() and cov.sum() == 450
