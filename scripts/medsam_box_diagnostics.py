"""Per-box-type behaviour of a trained MedSAM fold model on TRAINING patients (diagnosis only).

    source scripts/env.sh
    python scripts/medsam_box_diagnostics.py --exp E2c --folds 0 1 2 3 4
    python scripts/medsam_box_diagnostics.py --exp E2c_run1 --folds 0 1   # the first, collapsed E2c run

For the first --cases training patients of each fold (never validation or test patients), with the
fold's final checkpoint and the condition's own boxes: share of lesion-containing boxes with a
non-empty prediction (prob > 0.5 inside the box), split by box kind (E2c: brain box vs tiles), the
same for lesion-free boxes, and lesion-voxel recall / false-positive voxels per patient for the
configured combination (mean probability over covering boxes) and, for E2c, for the brain box alone
and the tiles alone. Used to explain failures; not used to choose any setting.
Output: results/E2/tables/box_diagnostics_<exp>.csv (summary numbers only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from msbench.config import REPO_ROOT, load_config, paths
from msbench.medsam.boxes import box_mask
from msbench.medsam.model import decode, load_medsam
from msbench.medsam.train import CONDITIONS, load_case, load_trainable, slice_boxes


@torch.no_grad()
def diagnose(exp: str, fold: int, n_cases: int, cfg: dict) -> dict:
    base = exp.split("_")[0]
    cond = CONDITIONS[base]
    split = json.loads((REPO_ROOT / "experiments" / "E1" / "splits_final.json").read_text())[fold]
    oof = Path(os.environ.get("MSBENCH_RESULTS", REPO_ROOT / "results")) / "E1" / "oof_predictions"
    model = load_medsam(cfg)
    model.image_encoder = None
    load_trainable(model, torch.load(paths(cfg)["medsam_runs"] / exp / f"fold_{fold}" / "checkpoint_final.pt",
                                     weights_only=False))
    model.eval()
    rng = np.random.default_rng(int(cfg["seed"]) + 100 * fold)
    kinds = ("brain", "tile") if cond == "independent" else ("box",)
    ne = {k: [0, 0] for k in kinds}            # lesion boxes: [non-empty, total]
    ne_empty = [0, 0]                          # lesion-free boxes (E2c tiles / E2b E1-FP boxes)
    rules = ["mean_all"] + (["brain_only", "tiles_only"] if cond == "independent" else [])
    tp = {r: 0 for r in rules}
    fp = {r: 0 for r in rules}
    n_ref = 0
    cases = split["train"][:n_cases]
    for c in cases:
        cd = load_case(c, cfg, with_ref=True, box_source_dir=oof if cond == "nnunet" else None)
        H, W = cd.geom.shape
        for k, z in enumerate(cd.slices):
            ref = cd.ref[:, :, z]
            n_ref += int(ref.sum())
            boxes = slice_boxes(cd, k, cond, cfg, rng)
            if len(boxes) == 0:
                continue
            emb = torch.from_numpy(np.asarray(cd.emb[k], dtype=np.float32)).cuda()[None]
            b1024 = torch.from_numpy(np.stack([cd.geom.box_to_1024(b) for b in boxes])).float().cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                low = decode(model, emb.expand(len(boxes), -1, -1, -1), b1024)
            prob = torch.sigmoid(cd.geom.logits_from_1024(low.float()))[:, 0].cpu().numpy()
            masks = np.stack([box_mask(b, (H, W)) for b in boxes])
            pin = prob * masks
            for i in range(len(boxes)):
                nonempty = bool((pin[i] > 0.5).any())
                if (ref & masks[i]).any():
                    kd = kinds[0] if cond != "independent" else ("brain" if i == 0 else "tile")
                    ne[kd][0] += nonempty
                    ne[kd][1] += 1
                elif cond != "independent" or i > 0:
                    ne_empty[0] += nonempty
                    ne_empty[1] += 1
            maps = {"mean_all": (pin.sum(0), masks.sum(0))}
            if cond == "independent":
                maps["brain_only"] = (pin[0], masks[0].astype(int))
                maps["tiles_only"] = (pin[1:].sum(0), masks[1:].sum(0))
            for r, (num, cov) in maps.items():
                pr = np.where(cov > 0, num / np.maximum(cov, 1), 0) > 0.5
                tp[r] += int((pr & ref).sum())
                fp[r] += int((pr & ~ref).sum())
    row = {"model": exp, "fold": fold, "training patients": len(cases)}
    for kd, (a, n) in ne.items():
        row[f"{kd} lesion boxes non-empty"] = a / n if n else np.nan
    row["lesion-free boxes non-empty"] = ne_empty[0] / ne_empty[1] if ne_empty[1] else np.nan
    for r in rules:
        row[f"voxel recall ({r})"] = tp[r] / max(n_ref, 1)
        row[f"FP voxels / patient ({r})"] = fp[r] / len(cases)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cases", type=int, default=4, help="training patients per fold (default 4)")
    args = ap.parse_args(argv)
    cfg = load_config()
    rows = [diagnose(args.exp, k, args.cases, cfg) for k in args.folds]
    df = pd.DataFrame(rows)
    out = Path(os.environ.get("MSBENCH_RESULTS", REPO_ROOT / "results")) / "E2" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"box_diagnostics_{args.exp}.csv"
    df.to_csv(f, index=False, float_format="%.4f")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
