import math

import numpy as np

from msbench.metrics import (
    average_surface_distance, hd95, iou, lesion_detection, lesion_segmentation_quality, segmentation_quality,
)


def cube(shape, lo, size):
    m = np.zeros(shape, bool)
    m[lo[0]:lo[0] + size, lo[1]:lo[1] + size, lo[2]:lo[2] + size] = True
    return m


def test_identical_masks_zero_distance():
    a = cube((20, 20, 20), (5, 5, 5), 6)
    assert hd95(a, a) == 0 and average_surface_distance(a, a) == 0


def test_shift_and_spacing():
    a = cube((30, 20, 20), (5, 5, 5), 6)
    b = cube((30, 20, 20), (7, 5, 5), 6)       # shifted by 2 voxels along axis 0
    assert math.isclose(hd95(a, b), 2.0)
    assert math.isclose(hd95(a, b, (2.0, 1.0, 1.0)), 4.0)   # 2 voxels x 2 mm
    assert 0 < average_surface_distance(a, b) < 2.0


def test_empty_masks():
    a = cube((10, 10, 10), (2, 2, 2), 3)
    z = np.zeros_like(a)
    assert math.isnan(hd95(a, z)) and math.isnan(average_surface_distance(z, a))
    assert iou(z, z) == 1.0 and iou(a, z) == 0.0


def test_iou_value():
    a = cube((10, 10, 10), (0, 0, 0), 4)       # 64 voxels
    b = cube((10, 10, 10), (2, 0, 0), 4)       # overlap 2*4*4 = 32
    assert math.isclose(iou(a, b), 32 / 96)


def test_lesion_dice_detected_only_and_merged_prediction():
    shape = (40, 40, 40)
    r1, r2, r3 = cube(shape, (5, 5, 5), 4), cube(shape, (5, 12, 5), 4), cube(shape, (30, 30, 30), 3)
    ref = r1 | r2 | r3
    pred = np.zeros(shape, bool)
    pred[5:9, 5:16, 5:9] = True                # one component covering r1, r2 and the gap between
    rec = lesion_segmentation_quality(pred, ref)
    det = lesion_detection(pred, ref)
    assert rec["detected"].sum() == det.tp_ref == 2          # r3 missed
    assert np.isnan(rec["dice"][~rec["detected"]]).all()
    # each lesion is compared with the whole merged component: 2*64 / (64 + 176)
    assert np.allclose(rec["dice"][rec["detected"]], 128 / 240)
    exact = lesion_segmentation_quality(ref, ref)
    assert np.allclose(exact["dice"], 1) and np.allclose(exact["hd95"], 0) and np.allclose(exact["asd"], 0)


def test_case_row_bins_and_distant_false_positive():
    shape = (60, 40, 40)
    ref = cube(shape, (5, 5, 5), 5)            # 125 mm3 -> ">=100"
    pred = ref.copy()
    far = cube(shape, (50, 30, 30), 3)         # distant false positive (27 mm3)
    row, rec = segmentation_quality(pred | far, ref, (1.0, 1.0, 1.0), [20, 100])
    assert row["n_detected"] == 1 and row["n_detected_>=100"] == 1 and np.isnan(row["lesion_dice_<20"])
    assert row["lesion_dice"] == 1.0 and row["lesion_hd95"] == 0.0     # lesion-wise: unaffected
    assert row["mask_hd95"] > 40                                       # whole mask: dominated by the FP
    assert math.isclose(row["mask_iou"], 125 / 152)
    assert list(rec["bin"]) == [">=100"]
