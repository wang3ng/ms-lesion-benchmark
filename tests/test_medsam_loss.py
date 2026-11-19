import math

import torch
from monai.losses import DiceLoss

from msbench.medsam.train import WindowStats, box_loss_terms, box_nonempty, collapse_verdict, combine_loss

DICE = DiceLoss(sigmoid=True, squared_pred=True, reduction="none")


def _batch(seed=0, empty=(1, 3)):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(4, 1, 16, 16, generator=g)
    tgt = (torch.rand(4, 1, 16, 16, generator=g) > 0.7).float()
    for i in empty:
        tgt[i] = 0
    return logits, tgt


def test_all_boxes_equals_first_run_loss():
    logits, tgt = _batch()
    dice, bce, lesion = box_loss_terms(logits, tgt, DICE)
    old = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")(logits, tgt) + \
        torch.nn.BCEWithLogitsLoss()(logits, tgt)
    assert torch.allclose(combine_loss(dice, bce, lesion, "all_boxes"), old, atol=1e-6)
    assert lesion.tolist() == [True, False, True, False]


def test_lesion_boxes_drops_dice_of_empty_targets_only():
    logits, tgt = _batch()
    dice, bce, lesion = box_loss_terms(logits, tgt, DICE)
    new = combine_loss(dice, bce, lesion, "lesion_boxes")
    expected = dice[[0, 2]].mean() + torch.nn.BCEWithLogitsLoss()(logits, tgt)
    assert torch.allclose(new, expected, atol=1e-6)


def test_identical_when_every_box_has_lesion():  # E2a: correction does not change its loss
    logits, tgt = _batch(empty=())
    dice, bce, lesion = box_loss_terms(logits, tgt, DICE)
    assert lesion.all()
    assert torch.equal(combine_loss(dice, bce, lesion, "lesion_boxes"), combine_loss(dice, bce, lesion, "all_boxes"))


def test_no_lesion_box_in_batch_gives_bce_only_and_grad():
    logits, tgt = _batch(empty=(0, 1, 2, 3))
    logits.requires_grad_(True)
    dice, bce, lesion = box_loss_terms(logits, tgt, DICE)
    loss = combine_loss(dice, bce, lesion, "lesion_boxes")
    assert torch.allclose(loss, bce.mean())
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_empty_output_is_the_old_optimum_but_not_the_new_one():
    """The first-run failure: with half empty boxes, an all-empty output scores ~0.5 under all_boxes."""
    tgt = torch.zeros(2, 1, 256, 256)   # small lesion in a large box, as for tiles
    tgt[0, 0, 10:14, 10:14] = 1
    empty_out = torch.full_like(tgt, -20.0)
    dice, bce, lesion = box_loss_terms(empty_out, tgt, DICE)
    assert math.isclose(float(combine_loss(dice, bce, lesion, "all_boxes")), 0.5, abs_tol=0.02)
    assert float(combine_loss(dice, bce, lesion, "lesion_boxes")) > 0.95


def test_box_nonempty_only_looks_inside_box():
    logits = torch.full((2, 1, 64, 64), -5.0)
    logits[0, 0, 40, 40] = 3.0     # outside its box
    logits[1, 0, 12, 12] = 3.0     # inside its box
    boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0], [5.0, 5.0, 20.5, 20.5]])
    assert box_nonempty(logits, boxes).tolist() == [False, True]


def test_window_stats_and_verdict():
    w = WindowStats()
    dice = torch.tensor([0.9, 0.2, 0.4, 0.1])
    bce = torch.tensor([0.1, 0.05, 0.1, 0.01])
    lesion = torch.tensor([True, False, True, False])
    nonempty = torch.tensor([True, True, False, False])
    w.add(torch.tensor(0.8), dice, bce, lesion, nonempty, "lesion_boxes")
    r = w.result()
    assert math.isclose(r["lesion_loss"], (1.0 + 0.5) / 2, rel_tol=1e-6)
    assert math.isclose(r["empty_loss"], (0.05 + 0.01) / 2, rel_tol=1e-6)   # no Dice on empty boxes
    assert r["lesion_nonempty"] == 0.5 and r["empty_nonempty"] == 0.5
    cc = {"max_lesion_loss_ratio": 0.8, "min_lesion_nonempty": 0.10}
    start = {"lesion_loss": 1.0, "lesion_nonempty": 0.9}
    assert collapse_verdict(start, {"lesion_loss": 0.5, "lesion_nonempty": 0.6}, cc)["passed"]
    v = collapse_verdict(start, {"lesion_loss": 0.9, "lesion_nonempty": 0.6}, cc)
    assert not v["passed"] and len(v["reasons"]) == 1
    v = collapse_verdict(start, {"lesion_loss": 0.5, "lesion_nonempty": 0.02}, cc)
    assert not v["passed"] and "non-empty" in v["reasons"][0]
    assert not collapse_verdict(start, {"lesion_loss": float("nan"), "lesion_nonempty": float("nan")}, cc)["passed"]


def test_verdict_catches_collapsed_tiles_hidden_by_brain_box():
    """First-run pattern: brain box non-empty, tiles empty; pooled non-empty fraction looks fine."""
    w = WindowStats()
    n = 40
    lesion = torch.ones(2 * n, dtype=torch.bool)
    kind = torch.tensor([1] * n + [2] * n)               # brain boxes, tiles
    nonempty = torch.tensor([True] * n + [False] * (n - 2) + [True] * 2)
    dice, bce = torch.full((2 * n,), 0.5), torch.full((2 * n,), 0.01)
    w.add(torch.tensor(0.5), dice, bce, lesion, nonempty, "lesion_boxes", kind)
    r = w.result()
    assert set(r["by_kind"]) == {"brain", "tile"} and r["by_kind"]["tile"]["lesion_nonempty"] == 0.05
    start = {"lesion_loss": 1.0, "by_kind": {"brain": {"n_lesion_boxes": 40, "lesion_loss": 1.0},
                                             "tile": {"n_lesion_boxes": 40, "lesion_loss": 1.0}}}
    cc = {"max_lesion_loss_ratio": 0.8, "min_lesion_nonempty": 0.10, "min_boxes_per_kind": 30}
    v = collapse_verdict(start, r, cc)
    assert r["lesion_nonempty"] > 0.5 and not v["passed"]
    assert v["reasons"] == [x for x in v["reasons"] if x.startswith("tile")]
