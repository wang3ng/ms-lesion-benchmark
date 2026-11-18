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

    def draw(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One sample: embedding (256,64,64) float32, box in 1024 px (4,), target (1024,1024) bool."""
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
        return emb, cd.geom.box_to_1024(box).astype(np.float32), target


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


def train(cases: list[CaseData], condition: str, cfg: dict, run_dir: Path, seed: int,
          steps: int | None = None, log=print, num_workers: int = 8) -> dict:
    ft = cfg["models"]["medsam"]["finetune"]
    steps = int(steps or ft["steps"])
    bs = int(ft["batch_size"])
    torch.manual_seed(seed)
    model = load_medsam(cfg)
    model.image_encoder = None  # embeddings are cached; frees ~360 MB of GPU memory
    params = list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters())
    opt = torch.optim.AdamW(params, lr=float(ft["lr"]), weight_decay=float(ft["weight_decay"]))
    warm = int(ft["warmup_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    bce = torch.nn.BCEWithLogitsLoss()

    start = 0
    latest = run_dir / "checkpoint_latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        load_trainable(model, ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        start = int(ck["step"])
        log(f"resumed from step {start}")
    model.prompt_encoder.train(), model.mask_decoder.train()

    loader = torch.utils.data.DataLoader(SampleStream((cases, condition, cfg, seed + start), (steps - start) * bs),
                                         batch_size=bs, num_workers=num_workers, pin_memory=True,
                                         persistent_workers=False, prefetch_factor=4 if num_workers else None)
    t0, run_loss, step = time.time(), 0.0, start
    for emb, box, tgt in loader:
        emb, box = emb.cuda(non_blocking=True), box.cuda(non_blocking=True)
        tgt = tgt.cuda(non_blocking=True)[:, None].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            low = decode(model, emb, box)
        logits = F.interpolate(low.float(), size=(SIDE, SIDE), mode="bilinear", align_corners=False)
        loss = dice_loss(logits, tgt) + bce(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(), sched.step()
        step += 1
        run_loss += loss.item()
        if step % 100 == 0 or step == steps:
            el = time.time() - t0
            log(f"step {step}/{steps} loss {run_loss / 100:.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"{el / (step - start):.3f} s/step")
            run_loss = 0.0
        if step % 1000 == 0 or step == steps:
            torch.save({"model": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step}, latest)
    torch.save(trainable_state(model), run_dir / "checkpoint_final.pt")
    return {"steps": steps, "seconds": time.time() - t0, "s_per_step": (time.time() - t0) / max(step - start, 1)}


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
