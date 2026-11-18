import numpy as np

from msbench.medsam.boxes import box_mask, component_boxes, independent_boxes, jitter_boxes, oracle_boxes


def test_component_boxes_8_connected():
    m = np.zeros((20, 30), bool)
    m[2:4, 3:6] = True
    m[4, 6] = True            # diagonal neighbour -> same component
    m[10:12, 20:25] = True
    b = component_boxes(m)
    assert sorted(b.tolist()) == [[3, 2, 7, 5], [20, 10, 25, 12]]


def test_jitter_only_grows_and_stays_inside():
    rng = np.random.default_rng(0)
    b = np.array([[3.0, 2.0, 7.0, 5.0]])
    for _ in range(50):
        j = jitter_boxes(b, (20, 30), 5, rng)[0]
        assert j[0] <= 3 and j[1] <= 2 and j[2] >= 7 and j[3] >= 5
        assert j[0] >= 0 and j[1] >= 0 and j[2] <= 30 and j[3] <= 20
    assert oracle_boxes(np.zeros((5, 5), bool), 5, rng).shape == (0, 4)


def test_independent_boxes_cover_brain_with_overlap():
    brain = np.zeros((100, 120), bool)
    brain[10:90, 20:110] = True
    b = independent_boxes(brain, (4, 4), 0.25)
    assert len(b) == 17 and b[0].tolist() == [20, 10, 110, 90]
    cover = np.zeros_like(brain, int)
    for box in b[1:]:
        cover += box_mask(box, brain.shape)
    assert (cover[brain] >= 1).all()                     # tiles cover the whole brain box
    assert (cover[brain] >= 2).mean() > 0.3               # and overlap
    assert independent_boxes(np.zeros((5, 5), bool)).shape == (0, 4)
