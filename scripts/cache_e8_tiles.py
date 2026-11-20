"""Cache frozen MedSAM embeddings of the E8 zoomed tiles for the MSSEG TRAINING cases (GPU).

    source scripts/env.sh
    python scripts/cache_e8_tiles.py --dry-run     # tile counts, disk and GPU-time estimate, free space
    python scripts/cache_e8_tiles.py

Per case: same volume, normalisation and brain slices as the slice cache (scripts/cache_medsam_embeddings.py),
then E2c's tiles with brain voxels (msbench.medsam.zoom), each cropped and put on its own keep-aspect 1024
canvas -> MedSAM ViT-B encoder (fp16) -> <medsam_e8_cache>/<dataset>/<case_id>.npy (n_tiles, 256, 64, 64)
float16 + <case_id>.json (rows: slice index k into the slice cache, z, tile x0 y0 x1 y1). The .json is
written last (complete iff it exists); re-running skips complete cases. Test sets are never cached: their
tile embeddings are computed on the fly at inference. The cache root is created with mode 700 and lives
outside git (paths.medsam_e8_cache, NVMe; delete it when the project ends, see TODO.md).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

from msbench.config import load_config, paths
from msbench.medsam.model import load_medsam
from msbench.medsam.slices import brain_slices, load_volume
from msbench.medsam.zoom import encode_tiles, zoom_tiles

EMB_BYTES = 256 * 64 * 64 * 2
S_PER_EMB = 13.3 * 60 / 14180   # measured for the slice cache (same encoder, same canvas size)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-free-after-gb", type=float, default=600.0)
    args = ap.parse_args(argv)
    cfg = load_config()
    p = paths(cfg)
    t = cfg["boxes"]["independent"]["tiles"]
    grid, overlap = tuple(t["grid"]), float(t["overlap"])
    index = pd.read_csv(p["data_processed"] / "index.csv", dtype=str)
    todo = index[(index.dataset == "msseg2016") & (index.release_split == "train")].sort_values("case_id")
    if args.cases:
        todo = todo[todo.case_id.isin(args.cases)]
    root = p["medsam_e8_cache"]

    plans, n_total = [], 0
    for r in todo.itertuples():
        meta_f = root / r.dataset / f"{r.case_id}.json"
        if meta_f.exists():
            continue
        plans.append(r)
    print(f"{len(todo)} MSSEG training cases, {len(todo) - len(plans)} complete, {len(plans)} to do -> {root}")

    if args.dry_run or plans:
        # tile counts need the brain masks; computed here so the estimate is exact
        counts = {}
        for r in plans:
            vol = load_volume(p["data_processed"] / r.path)
            counts[r.case_id] = sum(len(zoom_tiles(vol.brain[:, :, z], grid, overlap)) for z in brain_slices(vol))
        n_total = sum(counts.values())
        need = n_total * EMB_BYTES / 1e9
        probe = root if root.exists() else root.parent
        free = shutil.disk_usage(probe).free / 1e9
        print(f"tiles: {n_total}; disk {need:.1f} GB; est. GPU {n_total * S_PER_EMB / 60:.0f} min; "
              f"free on {probe}: {free:.0f} GB -> {free - need:.0f} GB after")
        if args.dry_run:
            return 0
        if free - need < args.min_free_after_gb:
            print(f"refusing: less than {args.min_free_after_gb:.0f} GB would remain free", file=sys.stderr)
            return 1

    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    torch.backends.cuda.matmul.allow_tf32 = True
    encoder = load_medsam(cfg).image_encoder
    t_all, n_all = time.time(), 0
    for r in plans:
        out = root / r.dataset
        out.mkdir(parents=True, exist_ok=True)
        os.chmod(out, 0o700)
        t0 = time.time()
        vol = load_volume(p["data_processed"] / r.path)
        zs = brain_slices(vol)
        slice_meta = json.loads((p["medsam_cache"] / r.dataset / f"{r.case_id}.json").read_text())
        if list(map(int, zs)) != slice_meta["slices"]:
            raise RuntimeError(f"{r.case_id}: brain slices differ from the slice cache")
        rows = [(k, int(z), *map(int, tl)) for k, z in enumerate(zs)
                for tl in zoom_tiles(vol.brain[:, :, z], grid, overlap)]
        emb_f = out / f"{r.case_id}.npy"
        emb = np.lib.format.open_memmap(emb_f, mode="w+", dtype=np.float16, shape=(len(rows), 256, 64, 64))
        img = torch.from_numpy(vol.image).cuda()
        i = 0
        for k, z in enumerate(zs):
            sel = [j for j in range(i, len(rows)) if rows[j][0] == k]
            if not sel:
                continue
            tiles = np.array([rows[j][2:] for j in sel])
            e = encode_tiles(encoder, img[..., int(z)], tiles, vol.spacing[:2], args.batch)
            emb[sel[0]:sel[-1] + 1] = e.cpu().numpy()
            i = sel[-1] + 1
        emb.flush()
        del emb
        meta = {"case_id": r.case_id, "dataset": r.dataset, "grid": list(grid), "overlap": overlap,
                "columns": ["k", "z", "x0", "y0", "x1", "y1"], "rows": rows, "spacing_2d": list(vol.spacing[:2]),
                "dtype": "float16", "encoder": "medsam_vit_b", "seconds": round(time.time() - t0, 1)}
        (out / f"{r.case_id}.json").write_text(json.dumps(meta))
        n_all += len(rows)
        print(f"{r.case_id}: {len(rows)} tiles in {time.time() - t0:.0f} s ({emb_f.stat().st_size / 2**30:.1f} GiB)",
              flush=True)
    dt = time.time() - t_all
    print(f"done: {n_all} tiles in {dt / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
