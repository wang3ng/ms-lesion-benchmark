import numpy as np
import nibabel as nib
import torch

from msbench.medsam.slices import SIDE, SliceGeometry, from_canonical, to_canonical


def test_canonical_roundtrip_lps_and_oblique_order():
    rng = np.random.default_rng(0)
    a = rng.random((5, 7, 9))
    for aff in [np.diag([-1.0, -1.0, 1.0, 1.0]),                       # LPS
                np.array([[0, 0, 1.2, 0], [0, -0.5, 0, 0], [-0.5, 0, 0, 0], [0, 0, 0, 1.0]])]:
        c = to_canonical(a, aff)
        assert np.array_equal(from_canonical(c, aff), a)
        # axial (S) axis is last after canonicalisation
        assert "".join(nib.aff2axcodes(aff))  # sanity


def test_geometry_keeps_aspect_and_maps_back():
    g = SliceGeometry.from_slice((144, 512), (1.2, 0.5))    # 172.8 mm x 256 mm
    assert g.scaled == (691, 1024)
    m = np.zeros((144, 512), bool); m[70:74, 250:262] = True
    m1024 = g.mask_to_1024(m)
    assert m1024.shape == (SIDE, SIDE) and not m1024[700:].any()
    logits = torch.from_numpy(np.where(m1024, 10.0, -10.0)).float()[None, None]
    back = g.logits_from_1024(logits)[0, 0].numpy() > 0
    assert back.shape == (144, 512) and (back & m).sum() / m.sum() > 0.9
    box = g.box_to_1024(np.array([250, 70, 262, 74]))
    assert np.allclose(box, [500, 70 * 691 / 144, 524, 74 * 691 / 144])


def test_decode_matches_sam_for_one_image():
    from segment_anything import sam_model_registry
    from msbench.medsam.model import decode
    torch.manual_seed(0)
    m = sam_model_registry["vit_b"](checkpoint=None).eval()
    emb = torch.randn(1, 256, 64, 64)
    boxes = torch.tensor([[100.0, 120.0, 300.0, 330.0], [500.0, 500.0, 700.0, 640.0]])
    with torch.no_grad():
        sp, de = m.prompt_encoder(points=None, boxes=boxes[:, None, :], masks=None)
        ref, _ = m.mask_decoder(image_embeddings=emb, image_pe=m.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sp, dense_prompt_embeddings=de, multimask_output=False)
        ours = decode(m, emb.expand(2, -1, -1, -1), boxes)
    assert torch.allclose(ours, ref, atol=1e-5)
