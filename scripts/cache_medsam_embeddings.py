"""Cache frozen MedSAM image-encoder embeddings, one file per case (GPU).

    source scripts/env.sh
    python scripts/cache_medsam_embeddings.py                       # MSSEG train + test, ISBI
    python scripts/cache_medsam_embeddings.py --cases msseg_train_c01_p01 --dry-run

For each case: FLAIR/T1/T2 -> canonical orientation -> slices containing brain ->
1024x1024 keep-aspect canvas (msbench.medsam.slices) -> MedSAM ViT-B encoder
(fp16 autocast) -> <medsam_cache>/<dataset>/<case_id>.npy, float16, shape
(n_slices, 256, 64, 64), plus <case_id>.json with the slice indices, geometry,
orientation and intensity clipping needed to map boxes in and masks back.
The .json is written last, so a case is complete iff its .json exists; re-running
skips complete cases. Images only; no label is read. Embeddings stay outside git.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from msbench.config import load_config, paths
from msbench.medsam.model import checkpoint_path, load_medsam
from msbench.medsam.slices import SliceGeometry, brain_slices, load_volume

SETS = {
    "msseg2016_train": lambda ix: ix[(ix.dataset == "msseg2016") & (ix.release_split == "train")],
    "msseg2016_test": lambda ix: ix[(ix.dataset == "msseg2016") & (ix.release_split == "test")],
    "isbi2015": lambda ix: ix[ix.dataset == "isbi2015"],
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--cases", nargs="+", help="only these case ids")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="list work, encode nothing")
    args = ap.parse_args(argv)

    cfg = load_config()
    p = paths(cfg)
    index = pd.read_csv(p["data_processed"] / "index.csv", dtype=str)
    todo = pd.concat([SETS[s](index) for s in args.sets]).sort_values("case_id")
    if args.cases:
        todo = todo[todo.case_id.isin(args.cases)]
    cache = p["medsam_cache"]
    print(f"{len(todo)} cases -> {cache}; checkpoint {checkpoint_path(cfg)}", flush=True)
    if args.dry_run:
        done = sum((cache / r.dataset / f"{r.case_id}.json").exists() for r in todo.itertuples())
        print(f"{done} already complete")
        return 0

    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_medsam(cfg)
    encoder = model.image_encoder
    t_all, n_slices_all = time.time(), 0
    for r in todo.itertuples():
        out_dir = cache / r.dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_f, emb_f = out_dir / f"{r.case_id}.json", out_dir / f"{r.case_id}.npy"
        if meta_f.exists():
            continue
        t0 = time.time()
        vol = load_volume(p["data_processed"] / r.path)
        zs = brain_slices(vol)
        geom = SliceGeometry.from_slice(vol.brain.shape[:2], vol.spacing[:2])
        emb = np.lib.format.open_memmap(emb_f, mode="w+", dtype=np.float16, shape=(len(zs), 256, 64, 64))
        img = torch.from_numpy(vol.image).cuda()            # (3, X, Y, Z)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(zs), args.batch):
                z = torch.as_tensor(zs[i:i + args.batch], device="cuda")
                x = img[..., z].permute(3, 0, 1, 2)              # (B, 3, X, Y)
                x = geom.image_to_1024(x)
                emb[i:i + len(z)] = encoder(x).float().cpu().numpy().astype(np.float16)
        emb.flush()
        del emb
        meta = {
            "case_id": r.case_id, "dataset": r.dataset, "release_split": r.release_split, "scanner": r.scanner,
            "slices": zs.tolist(), "canonical_shape": list(vol.brain.shape), "original_shape": list(vol.shape),
            "affine": vol.affine.tolist(), "spacing_canonical": list(vol.spacing), "geometry": geom.to_dict(),
            "clip": vol.clip, "dtype": "float16", "encoder": "medsam_vit_b", "seconds": round(time.time() - t0, 1),
        }
        meta_f.write_text(json.dumps(meta))
        n_slices_all += len(zs)
        print(f"{r.case_id}: {len(zs)} slices in {time.time() - t0:.1f} s "
              f"({emb_f.stat().st_size / 2**20:.0f} MiB)", flush=True)
    dt = time.time() - t_all
    print(f"done: {n_slices_all} slices in {dt / 60:.1f} min"
          + (f" ({n_slices_all / dt:.1f} slices/s)" if n_slices_all else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
