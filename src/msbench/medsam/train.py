"""MedSAM fine-tuning on cached embeddings and slice-wise 3D inference.

One run = one box condition (E2a oracle, E2b nnunet, E2c independent) x one
fold. Only the prompt encoder and mask decoder are trained; the image encoder
is represented by the cached embeddings (scripts/cache_medsam_embeddings.py).
Settings: configs/experiment.yaml models.medsam.finetune / .inference, boxes.*.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss

from msbench.config import load_config, paths
from msbench.medsam.boxes import box_mask, component_boxes, independent_boxes, jitter_boxes
from msbench.medsam.model import decode, load_medsam
from msbench.medsam.slices import SIDE, SliceGeometry, from_canonical, to_canonical

CONDITIONS = {"E2a": "oracle", "E2b": "nnunet", "E2c": "independent"}
BOX_KINDS = ("component", "brain", "tile")  # E2a/E2b component boxes; E2c brain box and tiles


# ---------------------------------------------------------------- per-case data
@dataclass
class CaseData:
    case_id: str
    dataset: str
    emb_path: Path
    slices: np.ndarray                    # canonical axial indices of the cached slices
    geom: SliceGeometry
    affine: np.ndarray
    original_shape: tuple
    ref: np.ndarray | None = None         # canonical reference mask (training cases)
    brain: np.ndarray | None = None       # canonical brain mask
    box_source: np.ndarray | None = None  # canonical mask used for E2b boxes
    _emb: np.ndarray | None = field(default=None, repr=False)

    @property
    def emb(self) -> np.ndarray:
        if self._emb is None:  # opened lazily (per DataLoader worker)
            self._emb = np.load(self.emb_path, mmap_mode="r")
        return self._emb


def _canonical_mask(path: Path, affine: np.ndarray) -> np.ndarray:
    img = nib.load(str(path))
    a = np.asanyarray(img.dataobj)
    if a.ndim == 4:
        a = a[..., 0]
    return to_canonical(a > 0, affine)


def load_case(case_id: str, cfg: dict, with_ref: bool, box_source_dir: Path | None = None) -> CaseData:
    p = paths(cfg)
    import pandas as pd
    index = pd.read_csv(p["data_processed"] / "index.csv", dtype=str).set_index("case_id")
    meta_row = index.loc[case_id]
    cache_dir = p["medsam_cache"] / meta_row["dataset"]
    meta = json.loads((cache_dir / f"{case_id}.json").read_text())
    g = meta["geometry"]
    affine = np.asarray(meta["affine"])
    case_dir = p["data_processed"] / meta_row["path"]
    cd = CaseData(case_id, meta_row["dataset"], cache_dir / f"{case_id}.npy", np.asarray(meta["slices"]),
                  SliceGeometry(tuple(g["shape"]), tuple(g["scaled"]), g["px_mm"]), affine,
                  tuple(meta["original_shape"]))
    bm = case_dir / "brain_mask.nii.gz"
    cd.brain = _canonical_mask(bm if bm.exists() else case_dir / "flair.nii.gz", affine)
    if with_ref:
        cd.ref = _canonical_mask(case_dir / "lesion.nii.gz", affine)
    if box_source_dir is not None:
        cd.box_source = _canonical_mask(box_source_dir / f"{case_id}.nii.gz", affine)
    return cd


# ---------------------------------------------------------------- boxes per slice
def slice_boxes(cd: CaseData, k: int, condition: str, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Boxes (voxel units) for cached slice k of a case under a condition."""
    z = int(cd.slices[k])
    shape = cd.geom.shape
    if condition == "independent":
        t = cfg["boxes"]["independent"]["tiles"]
        return independent_boxes(cd.brain[:, :, z], tuple(t["grid"]), float(t["overlap"]))
    src = cd.ref if condition == "oracle" else cd.box_source
    if src is None:
        raise ValueError(f"{cd.case_id}: no mask for {condition} boxes")
    jit = int(cfg["boxes"][condition]["jitter_px"][1])
    return jitter_boxes(component_boxes(src[:, :, z]), shape, jit, rng)


# ---------------------------------------------------------------- training
class BoxSampler:
    """Draws (case, slice, box, target) training samples for one condition."""

    def __init__(self, cases: list[CaseData], condition: str, cfg: dict, seed: int):
        self.cases, self.condition, self.cfg = cases, condition, cfg
        self.rng = np.random.default_rng(seed)
        self.pos_frac = float(cfg["models"]["medsam"]["finetune"]["positive_fraction"])
        # Pools of (case index, slice index, box index) with/without reference voxels in the box.
        self.pos, self.neg = [], []
        self._static_boxes = {}
        for ci, cd in enumerate(cases):
            for k, z in enumerate(cd.slices):
                if condition == "oracle":
                    if cd.ref[:, :, z].any():
                        self.pos.append((ci, k, -1))
                    continue
                boxes = slice_boxes(cd, k, condition, cfg, np.random.default_rng(0))
                if condition == "nnunet":  # jitter is redrawn per sample
                    boxes = component_boxes(cd.box_source[:, :, z])
                self._static_boxes[(ci, k)] = boxes
                for bi, b in enumerate(boxes):
                    has = cd.ref[:, :, z][box_mask(b, cd.geom.shape)].any()
                    (self.pos if has else self.neg).append((ci, k, bi))
        if not self.pos:
            raise ValueError("no positive training boxes")

    def counts(self) -> dict:
        return {"positive_boxes_or_slices": len(self.pos), "negative_boxes": len(self.neg)}

    def draw(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """One sample: embedding (256,64,64) float32, box in 1024 px (4,), target (1024,1024) bool,
        box kind (BOX_KINDS index: component box, E2c brain box or E2c tile)."""
        use_neg = self.condition != "oracle" and self.neg and self.rng.random() >= self.pos_frac
        ci, k, bi = (self.neg if use_neg else self.pos)[self.rng.integers(len(self.neg if use_neg else self.pos))]
        cd = self.cases[ci]
        z = int(cd.slices[k])
        ref2d = cd.ref[:, :, z]
        if self.condition == "oracle":
            boxes = slice_boxes(cd, k, "oracle", self.cfg, self.rng)
            box = boxes[self.rng.integers(len(boxes))]
        else:
            box = self._static_boxes[(ci, k)][bi]
            if self.condition == "nnunet":
                jit = int(self.cfg["boxes"]["nnunet"]["jitter_px"][1])
                box = jitter_boxes(box[None], cd.geom.shape, jit, self.rng)[0]
        target = cd.geom.mask_to_1024(ref2d & box_mask(box, cd.geom.shape))
        emb = np.asarray(cd.emb[k], dtype=np.float32)
        kind = 0 if self.condition != "independent" else (1 if bi == 0 else 2)  # box 0 = brain box
        return emb, cd.geom.box_to_1024(box).astype(np.float32), target, kind


class SampleStream(torch.utils.data.IterableDataset):
    def __init__(self, sampler_args: tuple, n: int):
        self.sampler_args, self.n = sampler_args, n

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        cases, cond, cfg, seed = self.sampler_args
        s = BoxSampler(cases, cond, cfg, seed * 1000 + wid)
        for _ in range(wid, self.n, nw):
            yield s.draw()


def trainable_state(model) -> dict:
    return {"prompt_encoder": model.prompt_encoder.state_dict(), "mask_decoder": model.mask_decoder.state_dict()}


def load_trainable(model, state: dict) -> None:
    model.prompt_encoder.load_state_dict(state["prompt_encoder"])
    model.mask_decoder.load_state_dict(state["mask_decoder"])


DICE_ON = ("lesion_boxes", "all_boxes")


def box_loss_terms(logits: torch.Tensor, tgt: torch.Tensor, dice_fn) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-box terms for logits/targets of shape (B, 1, H, W): (dice (B,), bce (B,), has_lesion (B,) bool).

    dice_fn is monai DiceLoss(sigmoid=True, squared_pred=True, reduction="none").
    """
    b = logits.shape[0]
    dice = dice_fn(logits, tgt).reshape(b)
    bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none").reshape(b, -1).mean(1)
    lesion = tgt.reshape(b, -1).amax(1) > 0
    return dice, bce, lesion


def combine_loss(dice: torch.Tensor, bce: torch.Tensor, lesion: torch.Tensor, dice_on: str) -> torch.Tensor:
    """Training loss: BCE averaged over every box + per-box Dice averaged over the boxes in `dice_on`.

    "all_boxes" equals monai DiceLoss(reduction="mean") + BCEWithLogitsLoss() (the first E2c run);
    "lesion_boxes" leaves boxes with an empty target out of the Dice mean (Dice of an empty target is
    ~0 only for an exactly empty output, which pushes every box towards empty). Both give the same
    loss when every box contains lesion voxels (E2a).
    """
    if dice_on == "all_boxes":
        d = dice.mean()
    elif dice_on == "lesion_boxes":
        d = dice[lesion].mean() if bool(lesion.any()) else dice.sum() * 0.0
    else:
        raise ValueError(f"dice_on must be one of {DICE_ON}")
    return d + bce.mean()


def box_nonempty(logits: torch.Tensor, boxes_1024: torch.Tensor) -> torch.Tensor:
    """(B,) bool: any logit > 0 inside the box (the region used at inference), logits (B, 1, H, W)."""
    _, _, h, w = logits.shape
    ys = torch.arange(h, device=logits.device)[None, :, None]
    xs = torch.arange(w, device=logits.device)[None, None, :]
    x0, y0, x1, y1 = (boxes_1024[:, i, None, None] for i in range(4))
    inside = (xs >= x0.floor()) & (xs < x1.ceil()) & (ys >= y0.floor()) & (ys < y1.ceil())
    return ((logits[:, 0] > 0) & inside).flatten(1).any(1)


class WindowStats:
    """Accumulates the per-box monitoring numbers over one logging window."""

    KEYS = ("loss", "lesion_loss", "lesion_dice", "lesion_nonempty", "empty_loss", "empty_nonempty")

    def __init__(self):
        self.sums = {k: 0.0 for k in self.KEYS}
        self.n = {"steps": 0, "lesion": 0, "empty": 0}
        self.kind = {}  # kind -> [n lesion boxes, lesion loss sum, non-empty sum]

    def add(self, loss, dice, bce, lesion, nonempty, dice_on, kind=None):
        per_box = bce + dice * (lesion if dice_on == "lesion_boxes" else torch.ones_like(lesion)).float()
        nl, ne = int(lesion.sum()), int((~lesion).sum())
        if kind is not None:
            for kd in kind[lesion].unique().tolist():
                sel = lesion & (kind == kd)
                acc = self.kind.setdefault(BOX_KINDS[kd], [0, 0.0, 0.0])
                acc[0] += int(sel.sum())
                acc[1] += float(per_box[sel].sum())
                acc[2] += float(nonempty[sel].sum())
        self.n["steps"] += 1
        self.sums["loss"] += float(loss)
        if nl:
            self.n["lesion"] += nl
            self.sums["lesion_loss"] += float(per_box[lesion].sum())
            self.sums["lesion_dice"] += float(dice[lesion].sum())
            self.sums["lesion_nonempty"] += float(nonempty[lesion].sum())
        if ne:
            self.n["empty"] += ne
            self.sums["empty_loss"] += float(per_box[~lesion].sum())
            self.sums["empty_nonempty"] += float(nonempty[~lesion].sum())

    def result(self) -> dict:
        nan = float("nan")
        out = {"loss": self.sums["loss"] / max(self.n["steps"], 1)}
        for k in ("lesion_loss", "lesion_dice", "lesion_nonempty"):
            out[k] = self.sums[k] / self.n["lesion"] if self.n["lesion"] else nan
        for k in ("empty_loss", "empty_nonempty"):
            out[k] = self.sums[k] / self.n["empty"] if self.n["empty"] else nan
        out.update(n_lesion_boxes=self.n["lesion"], n_empty_boxes=self.n["empty"])
        if len(self.kind) > 1:  # E2c: brain box and tiles separately
            out["by_kind"] = {k: {"n_lesion_boxes": n, "lesion_loss": l / n, "lesion_nonempty": e / n}
                              for k, (n, l, e) in sorted(self.kind.items())}
        return out


def collapse_verdict(start: dict, check: dict, cc: dict) -> dict:
    """Training-only collapse check (models.medsam.finetune.collapse_check). passed=False -> stop the fold.

    Both criteria are applied to all lesion boxes together and, for E2c, to the brain box and the tiles
    separately (box kinds with at least min_boxes_per_kind lesion boxes in both windows): pooled numbers
    can hide collapsed tiles behind an over-segmenting brain box (first E2c run).
    """
    groups = [("lesion boxes", start, check)]
    for k, c in check.get("by_kind", {}).items():
        s = start.get("by_kind", {}).get(k)
        mn = int(cc.get("min_boxes_per_kind", 30))
        if s and s["n_lesion_boxes"] >= mn and c["n_lesion_boxes"] >= mn:
            groups.append((f"{k} lesion boxes", s, c))
    reasons, ratios = [], {}
    for name, s, c in groups:
        ratio = c["lesion_loss"] / s["lesion_loss"]
        ratios[name] = ratio
        if not ratio <= float(cc["max_lesion_loss_ratio"]):
            reasons.append(f"{name}: loss {c['lesion_loss']:.4f} is {ratio:.2f} x its start value "
                           f"{s['lesion_loss']:.4f} (needs <= {cc['max_lesion_loss_ratio']})")
        if not c["lesion_nonempty"] >= float(cc["min_lesion_nonempty"]):
            reasons.append(f"{name}: only {c['lesion_nonempty']:.3f} get a non-empty prediction "
                           f"(needs >= {cc['min_lesion_nonempty']})")
    return {"passed": not reasons, "reasons": reasons, "lesion_loss_ratio": ratios["lesion boxes"],
            "lesion_loss_ratios": ratios, "start": start, "check": check, "criteria": cc}


def train(cases: list[CaseData], condition: str, cfg: dict, run_dir: Path, seed: int,
          steps: int | None = None, log=print, num_workers: int = 8, dice_on: str | None = None) -> dict:
    """Train (or resume). Returns {"steps", "seconds", "s_per_step", "collapse_check", "collapsed"}.

    If the collapse check fails at collapse_check.at_step, training stops there (no checkpoint_final.pt)
    and collapsed=True. Per-window statistics go to <run_dir>/train_log.jsonl and the check result to
    <run_dir>/collapse_check.json.
    """
    ft = cfg["models"]["medsam"]["finetune"]
    steps = int(steps or ft["steps"])
    bs = int(ft["batch_size"])
    dice_on = dice_on or ft["dice_on"]
    cc = ft["collapse_check"]
    at_step, win, every = int(cc["at_step"]), int(cc["window"]), int(cc["repeat_every"])
    torch.manual_seed(seed)
    model = load_medsam(cfg)
    model.image_encoder = None  # embeddings are cached; frees ~360 MB of GPU memory
    params = list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters())
    opt = torch.optim.AdamW(params, lr=float(ft["lr"]), weight_decay=float(ft["weight_decay"]))
    warm = int(ft["warmup_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    dice_fn = DiceLoss(sigmoid=True, squared_pred=True, reduction="none")

    start, start_stats = 0, None
    latest = run_dir / "checkpoint_latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        load_trainable(model, ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        start, start_stats = int(ck["step"]), ck.get("start_stats")
        log(f"resumed from step {start}")
    model.prompt_encoder.train(), model.mask_decoder.train()
    log(f"loss: Dice on {dice_on} + BCE on every box; collapse check at step {at_step} "
        f"(window {win}, lesion-loss ratio <= {cc['max_lesion_loss_ratio']}, "
        f"lesion non-empty >= {cc['min_lesion_nonempty']})")

    loader = torch.utils.data.DataLoader(SampleStream((cases, condition, cfg, seed + start), (steps - start) * bs),
                                         batch_size=bs, num_workers=num_workers, pin_memory=True,
                                         persistent_workers=False, prefetch_factor=4 if num_workers else None)
    t0, step = time.time(), start
    win_stats = WindowStats()
    verdict = None
    cf = run_dir / "collapse_check.json"
    history = json.loads(cf.read_text()).get("history", []) if cf.exists() else []
    jl = (run_dir / "train_log.jsonl").open("a")
    try:
        for emb, box, tgt, kind in loader:
            emb, box = emb.cuda(non_blocking=True), box.cuda(non_blocking=True)
            tgt = tgt.cuda(non_blocking=True)[:, None].float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                low = decode(model, emb, box)
            logits = F.interpolate(low.float(), size=(SIDE, SIDE), mode="bilinear", align_corners=False)
            dice, bce, lesion = box_loss_terms(logits, tgt, dice_fn)
            loss = combine_loss(dice, bce, lesion, dice_on)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(), sched.step()
            step += 1
            with torch.no_grad():
                win_stats.add(loss.detach(), dice.detach(), bce.detach(), lesion,
                              box_nonempty(logits.detach(), box), dice_on, kind.to(lesion.device))
            if step % win == 0 or step == steps:
                r = win_stats.result()
                win_stats = WindowStats()
                if step == win and start == 0:
                    start_stats = r
                jl.write(json.dumps({"step": step, **r, "lr": sched.get_last_lr()[0]}) + "\n")
                jl.flush()
                log(f"step {step}/{steps} loss {r['loss']:.4f} | lesion boxes: loss {r['lesion_loss']:.4f} "
                    f"(dice {r['lesion_dice']:.4f}) non-empty {r['lesion_nonempty']:.3f} | empty boxes: loss "
                    f"{r['empty_loss']:.4f} non-empty {r['empty_nonempty']:.3f} | lr {sched.get_last_lr()[0]:.2e} "
                    f"{(time.time() - t0) / (step - start):.3f} s/step"
                    + "".join(f" | {k} lesion boxes: loss {v['lesion_loss']:.4f} non-empty {v['lesion_nonempty']:.3f}"
                              for k, v in r.get("by_kind", {}).items()))
                due = step >= at_step and ((step - at_step) % every == 0 or step == steps)
                if due and start_stats is not None:
                    verdict = collapse_verdict(start_stats, r, cc)
                    verdict.update(step=step, dice_on=dice_on, condition=condition)
                    history.append({"step": step, "passed": verdict["passed"], "reasons": verdict["reasons"],
                                    "lesion_loss_ratios": verdict["lesion_loss_ratios"],
                                    "lesion_nonempty": r["lesion_nonempty"],
                                    **({"by_kind_nonempty": {k: v["lesion_nonempty"] for k, v in r["by_kind"].items()}}
                                       if "by_kind" in r else {})})
                    verdict["history"] = history
                    (run_dir / "collapse_check.json").write_text(json.dumps(verdict, indent=2))
                    log(f"collapse check at step {step}: {'PASSED' if verdict['passed'] else 'FAILED'} "
                        f"(lesion-loss ratios {', '.join(f'{k} {v:.3f}' for k, v in verdict['lesion_loss_ratios'].items())}"
                        f"; lesion non-empty {r['lesion_nonempty']:.3f})"
                        + "".join(f"\n    {x}" for x in verdict["reasons"]))
            if step % 1000 == 0 or step == steps or (verdict is not None and not verdict["passed"]):
                torch.save({"model": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step, "start_stats": start_stats}, latest)
            if verdict is not None and not verdict["passed"]:
                break
    finally:
        jl.close()
    collapsed = verdict is not None and not verdict["passed"]
    if not collapsed:
        torch.save(trainable_state(model), run_dir / "checkpoint_final.pt")
    return {"steps": step, "seconds": time.time() - t0, "s_per_step": (time.time() - t0) / max(step - start, 1),
            "collapse_check": verdict, "collapsed": collapsed}


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict_case(model, cd: CaseData, condition: str, cfg: dict, seed: int) -> np.ndarray:
    """Binary mask on the ORIGINAL voxel grid of the case."""
    inf = cfg["models"]["medsam"]["inference"]
    rng = np.random.default_rng(seed)
    H, W = cd.geom.shape
    pred = np.zeros(cd.brain.shape, dtype=bool)
    bs = int(inf["batch_size"])
    for k, z in enumerate(cd.slices):
        boxes = slice_boxes(cd, k, condition, cfg, rng)
        if len(boxes) == 0:
            continue
        emb = torch.from_numpy(np.asarray(cd.emb[k], dtype=np.float32)).cuda()[None]
        prob_sum = torch.zeros((H, W), device="cuda")
        cover = torch.zeros((H, W), device="cuda")
        for i in range(0, len(boxes), bs):
            bb = boxes[i:i + bs]
            b1024 = torch.from_numpy(np.stack([cd.geom.box_to_1024(b) for b in bb])).float().cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                low = decode(model, emb.expand(len(bb), -1, -1, -1), b1024)
            prob = torch.sigmoid(cd.geom.logits_from_1024(low.float()))[:, 0]
            for b, pr in zip(bb, prob):
                m = torch.from_numpy(box_mask(b, (H, W))).cuda()
                prob_sum += pr * m
                cover += m
        mean = torch.where(cover > 0, prob_sum / cover.clamp(min=1), torch.zeros_like(prob_sum))
        pred[:, :, z] = (mean > float(inf["threshold"])).cpu().numpy()
    return from_canonical(pred, cd.affine)


def save_prediction(mask: np.ndarray, case_id: str, cfg: dict, out_dir: Path) -> Path:
    """Write with the FLAIR header of the case (same grid as the reference masks)."""
    import pandas as pd
    p = paths(cfg)
    index = pd.read_csv(p["data_processed"] / "index.csv", dtype=str).set_index("case_id")
    ref = nib.load(str(p["data_processed"] / index.loc[case_id, "path"] / "flair.nii.gz"))
    hdr = ref.header.copy()
    hdr.set_data_dtype(np.uint8)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"{case_id}.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), ref.affine, hdr), f)
    return f
