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
    lesion_records,
    lesion_volume_histogram,
    lesion_volume_ml,
)
from msbench.stats import (
    bland_altman,
    bootstrap_ci,
    bootstrap_ratio_ci,
    paired_bootstrap_diff_ci,
    paired_bootstrap_ratio_diff_ci,
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


def test_lesion_records_match_detection_counts():
    ref = cube(cube(cube(empty(), 10, 10, 10, 6), 32, 32, 32, 1), 10, 32, 32, 2)  # 2197, 27, 125 voxels
    pred = cube(cube(empty(), 10, 10, 10, 6), 30, 10, 10, 1)  # hits the big lesion + one isolated FP
    rec = lesion_records(pred, ref)
    det = lesion_detection(pred, ref)
    assert rec["ref_detected"].sum() == det.tp_ref and len(rec["ref_detected"]) == det.n_ref == 3
    assert (~rec["pred_true_positive"]).sum() == det.fp_pred == 1
    assert sorted(rec["ref_volume_mm3"]) == [27, 125, 2197]
    fp = ~rec["pred_true_positive"]
    assert rec["pred_volume_mm3"][fp][0] == 27 and rec["pred_overlap_frac"][fp][0] == 0
    missed = ~rec["ref_detected"]
    assert np.all(rec["ref_covered_frac"][missed] == 0)
    by = lesion_detection_by_size(pred, ref, bin_edges_mm3=[20, 100])
    assert by["20-100"].n_ref == ((rec["ref_volume_mm3"] >= 20) & (rec["ref_volume_mm3"] < 100)).sum()


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


def test_paired_bootstrap_ratio_diff():
    den = [10, 4, 0, 6]
    a, b = [8, 3, 0, 5], [5, 2, 0, 3]
    ci = paired_bootstrap_ratio_diff_ci(a, b, den, n_boot=2000)
    assert ci["estimate"] == pytest.approx((16 - 10) / 20)
    assert 0 < ci["low"] <= ci["estimate"] <= ci["high"]
    same = paired_bootstrap_ratio_diff_ci(a, a, den, n_boot=200)
    assert same["low"] == same["high"] == 0


def test_stats_drop_nan_pairs():
    w = paired_wilcoxon([1.0, math.nan, 0.5], [0.5, 0.2, math.nan])
    assert w["n"] == 1


def test_bland_altman():
    ba = bland_altman([10, 12, 14], [9, 11, 13])
    assert ba["bias"] == pytest.approx(1.0) and ba["mae"] == pytest.approx(1.0)


# ---------- Anima port (animaSegPerfAnalyzer 4.2 semantics) ----------

def box(mask, z0, z1, y0, y1, x0, x1):
    mask[z0:z1, y0:y1, x0:x1] = 1
    return mask


def test_anima_defaults():
    from msbench.metrics import DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_CONNECTIVITY, DEFAULT_GAMMA
    assert (DEFAULT_CONNECTIVITY, DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA) == (6, 0.10, 0.70, 0.65)


def test_anima_default_connectivity_is_face():
    m = empty()
    box(m, 5, 8, 5, 8, 5, 8)
    box(m, 8, 11, 8, 11, 8, 11)  # touches the first block only at a corner
    assert lesion_count(m) == 2


def test_anima_min_size_is_strict():
    from msbench.metrics.components import min_lesion_voxels
    assert min_lesion_voxels(3.0, (1, 1, 1)) == 4          # exactly 3 mm^3 is dropped
    assert min_lesion_voxels(3.0, (0.7, 0.744, 0.744)) == 8
    three, four = box(empty(), 5, 6, 5, 6, 5, 8), box(empty(), 5, 6, 5, 6, 5, 9)
    assert lesion_count(three) == 0 and lesion_count(four) == 1


def test_anima_alpha_is_strict():
    ref = box(empty(), 5, 9, 5, 10, 5, 7)                  # 40 voxels
    exactly_10 = empty()
    exactly_10[5, 5, 5:7] = 1; exactly_10[6, 5, 5:7] = 1    # 4 voxels inside = 10 %
    more = exactly_10.copy(); more[7, 5, 5] = 1              # 5 voxels = 12.5 %
    assert lesion_detection(exactly_10, ref).tp_ref == 0
    assert lesion_detection(more, ref).tp_ref == 1


def test_anima_beta_rejects_oversized_component():
    ref = cube(empty(), 20, 20, 20, 1)                       # 27 voxels
    pred = cube(empty(), 20, 20, 20, 4)                      # 729 voxels, 96 % outside
    det = lesion_detection(pred, ref)
    assert det.tp_ref == 0 and det.fp_pred == 1


def test_anima_gamma_only_checks_the_main_overlaps():
    ref = box(empty(), 10, 12, 10, 20, 10, 20)               # 200 voxels
    good = box(empty(), 10, 12, 10, 17, 10, 20)              # 140 voxels, all inside (70 % of overlap)
    bad = box(empty(), 10, 12, 18, 20, 10, 20)               # 40 inside ...
    bad = box(bad, 10, 12, 21, 39, 10, 20)                   # ... + 360 outside, not connected to `good`
    bad[10:12, 20, 10:20] = 1                                # bridge stays outside the reference
    pred = np.maximum(good, bad)
    rec = lesion_records(pred, ref)
    assert rec["ref_detected"].tolist() == [True]            # first 65 % of overlap comes from `good`
    swapped = box(empty(), 10, 12, 10, 13, 10, 20)           # good part now only 60 voxels (30 %)
    swapped = np.maximum(swapped, box(empty(), 10, 12, 14, 39, 10, 20))  # bad part: 120 in, 380 out
    assert lesion_records(swapped, ref)["ref_detected"].tolist() == [False]


def test_anima_precision_uses_the_swapped_test():
    ref = box(empty(), 5, 15, 5, 15, 5, 15)                  # 1000 voxels
    pred = box(empty(), 5, 15, 5, 15, 5, 7)                  # 200 voxels, all inside -> covers 20 %
    det = lesion_detection(pred, ref)
    assert det.tp_ref == 1                                   # 20 % > alpha, component 0 % outside
    assert det.tp_pred == 0 and det.precision == 0.0         # the lesion is 80 % uncovered > beta


def test_anima_empty_conventions():
    ref = cube(empty(), 20, 20, 20, 2)
    none = lesion_detection(empty(), ref)
    assert (none.precision, none.recall, none.f1) == (0.0, 0.0, 0.0)
    fp_only = lesion_detection(ref, empty())
    assert fp_only.precision == 0.0 and math.isnan(fp_only.recall) and fp_only.f1 == 0.0


def test_cluster_mean_ci_resamples_patients():
    """score_test's patient bootstrap: with one scan per patient it equals the plain scan bootstrap."""
    import importlib.util
    from pathlib import Path
    from msbench.stats import bootstrap_ci
    spec = importlib.util.spec_from_file_location("score_test", Path(__file__).parents[1] / "scripts" / "score_test.py")
    st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
    rng = np.random.default_rng(1)
    x = rng.random(12)
    kw = {"n_boot": 2000, "level": 0.95, "seed": 3}
    r = st.mean_ci(x, np.arange(12).astype(str), kw)
    b = bootstrap_ci(x, **kw)
    assert np.isclose(r["mean"], x.mean()) and np.isclose(r["low"], b["low"]) and np.isclose(r["high"], b["high"])
    # clustered: 3 patients x 4 scans; every resample keeps whole patients -> only 10 distinct means
    cl = np.repeat(["a", "b", "c"], 4)
    s, n = st._agg(x, cl)
    assert n.tolist() == [4, 4, 4] and np.isclose(s.sum(), x.sum())
    p = st.paired(x, x - 0.1, cl, kw)
    assert np.isclose(p["diff"], 0.1) and p["n_patients"] == 3 and np.isclose(p["low"], 0.1)
