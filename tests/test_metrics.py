import math

import numpy as np
import pytest

from msbench.metrics import (
    dice,
    evaluate_case,
    label_lesions,
    lesion_count,
    lesion_detection,
    lesion_detection_by_size,
    lesion_volume_histogram,
    lesion_volume_ml,
)
from msbench.stats import (
    bland_altman,
    bootstrap_ci,
    bootstrap_ratio_ci,
    paired_bootstrap_diff_ci,
    paired_wilcoxon,
)

SHAPE = (40, 40, 40)


def cube(mask, z, y, x, r):
    mask[z - r : z + r + 1, y - r : y + r + 1, x - r : x + r + 1] = 1
    return mask


def empty():
    return np.zeros(SHAPE, dtype=np.uint8)


# ---------- Dice ----------

def test_dice_identical_and_disjoint():
    a = cube(empty(), 10, 10, 10, 2)
    b = cube(empty(), 30, 30, 30, 2)
    assert dice(a, a) == 1.0
    assert dice(a, b) == 0.0


def test_dice_both_empty_is_one():
    assert dice(empty(), empty()) == 1.0


# ---------- Components ----------

def test_connectivity_changes_lesion_count():
    m = empty()
    m[10, 10, 10] = 1
    m[11, 11, 11] = 1  # touches only at a corner
    assert label_lesions(m, min_lesion_mm3=0, connectivity=26)[1] == 1
    assert label_lesions(m, min_lesion_mm3=0, connectivity=6)[1] == 2


def test_min_size_filter_uses_spacing():
    m = empty()
    m[5, 5, 5] = 1  # one voxel
    assert lesion_count(m, spacing=(1, 1, 1), min_lesion_mm3=3) == 0
    assert lesion_count(m, spacing=(2, 2, 2), min_lesion_mm3=3) == 1  # 8 mm^3


def test_volume_ml():
    m = cube(empty(), 20, 20, 20, 2)  # 5^3 = 125 voxels
    assert lesion_volume_ml(m, spacing=(2, 1, 1)) == pytest.approx(0.25)


# ---------- Detection ----------

def test_perfect_detection():
    ref = cube(cube(empty(), 10, 10, 10, 1), 30, 30, 30, 2)
    res = lesion_detection(ref, ref)
    assert (res.n_ref, res.tp_ref, res.fp_pred) == (2, 2, 0)
    assert res.f1 == 1.0


def test_missed_small_lesion_lowers_recall_not_dice_much():
    ref = cube(cube(empty(), 10, 10, 10, 6), 32, 32, 32, 1)  # big + small
    pred = cube(empty(), 10, 10, 10, 6)  # only big
    res = lesion_detection(pred, ref)
    assert res.recall == 0.5 and res.precision == 1.0
    assert dice(pred, ref) > 0.98  # Dice barely notices


def test_false_positive_component():
    ref = cube(empty(), 10, 10, 10, 2)
    pred = cube(cube(empty(), 10, 10, 10, 2), 30, 30, 30, 2)
    res = lesion_detection(pred, ref)
    assert res.recall == 1.0 and res.precision == 0.5 and res.fp_pred == 1


def test_oversized_prediction_does_not_count_as_detection():
    ref = cube(empty(), 20, 20, 20, 1)  # 27 voxels
    pred = cube(empty(), 20, 20, 20, 8)  # huge blob covering it
    res = lesion_detection(pred, ref)
    assert res.recall == 0.0  # mostly outside the reference -> rejected


def test_empty_cases():
    assert lesion_detection(empty(), empty()).f1 == 1.0
    ref = cube(empty(), 10, 10, 10, 2)
    assert lesion_detection(empty(), ref).f1 == 0.0
    assert lesion_detection(ref, empty()).f1 == 0.0


def test_detection_by_size_bins():
    ref = cube(cube(empty(), 10, 10, 10, 6), 32, 32, 32, 1)  # 2197 and 27 voxels
    pred = cube(empty(), 10, 10, 10, 6)
    out = lesion_detection_by_size(pred, ref, bin_edges_mm3=[20, 100])
    assert list(out) == ["<20", "20-100", ">=100"]
    assert out["20-100"].n_ref == 1 and out["20-100"].recall == 0.0
    assert out[">=100"].recall == 1.0
    assert out["<20"].n_ref == 0 and out["<20"].f1 == 1.0


def test_volume_histogram():
    m = cube(cube(empty(), 10, 10, 10, 6), 32, 32, 32, 1)
    hist = lesion_volume_histogram([(m, (1, 1, 1)), (m, (1, 1, 1))], [20, 100])
    assert hist == {"<20": 0, "20-100": 2, ">=100": 2}


def test_evaluate_case_row():
    ref = cube(cube(empty(), 10, 10, 10, 6), 32, 32, 32, 1)
    row = evaluate_case(ref, ref, (1, 1, 1), [20, 100])
    assert row["dice"] == 1.0 and row["lesion_f1"] == 1.0 and row["fp_lesions"] == 0
    assert "f1_20-100" in row


# ---------- Stats ----------

def test_wilcoxon_and_bootstrap():
    a = np.array([0.8, 0.7, 0.9, 0.85, 0.75, 0.8, 0.9, 0.7])
    b = a - 0.1
    w = paired_wilcoxon(a, b)
    assert w["median_diff"] == pytest.approx(0.1) and w["p"] < 0.05
    ci = paired_bootstrap_diff_ci(a, b, n_boot=500)
    assert ci["low"] == pytest.approx(0.1) and ci["high"] == pytest.approx(0.1)
    one = bootstrap_ci(a, n_boot=500)
    assert one["low"] <= one["estimate"] <= one["high"]


def test_bootstrap_ratio_ci_pools_whole_patients():
    detected = [5, 0, 9, 2]
    reference = [10, 0, 10, 4]   # second patient has no lesion in this bin
    ci = bootstrap_ratio_ci(detected, reference, n_boot=2000)
    assert ci["estimate"] == pytest.approx(16 / 24) and ci["n"] == 4
    assert 0.5 <= ci["low"] <= ci["estimate"] <= ci["high"] <= 0.9
    same = bootstrap_ratio_ci([3, 6, 9], [3, 6, 9], n_boot=500)
    assert same["low"] == same["high"] == 1.0
    assert math.isnan(bootstrap_ratio_ci([0, 0], [0, 0])["estimate"])


def test_stats_drop_nan_pairs():
    w = paired_wilcoxon([1.0, math.nan, 0.5], [0.5, 0.2, math.nan])
    assert w["n"] == 1


def test_bland_altman():
    ba = bland_altman([10, 12, 14], [9, 11, 13])
    assert ba["bias"] == pytest.approx(1.0) and ba["mae"] == pytest.approx(1.0)
