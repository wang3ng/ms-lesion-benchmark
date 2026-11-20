"""5-fold-ensemble MedSAM predictions on the test sets (MSSEG 2016 test incl. GE, ISBI 2015). NO SCORING.

    source scripts/env.sh
    python scripts/predict_test_medsam.py --exp E2c                # both test sets
    python scripts/predict_test_medsam.py --exp E8 --sets isbi2015

Ensemble: the 5 cross-validation fold models of the condition (<medsam_runs>/<exp>/fold_k/
checkpoint_final.pt); per pixel, the mean-over-covering-boxes probability of each model is averaged over the
5 models and thresholded at 0.5 (as the nnU-Net 5-fold ensemble of E1). Boxes are drawn once per slice,
seeded, and shared by the 5 models.
  E2a  ground-truth boxes (upper bound): MSSEG test from the consensus; ISBI once per rater (boxes from
       rater k, written to isbi2015/lesion_raterk/), to be scored against the same rater.
  E2b  boxes from the E1 5-fold-ensemble test predictions (results/E1/test_predictions/<dataset>/).
  E2c  brain box + 4 x 4 tiles.
  E8   brain box + zoomed tiles; tile embeddings computed on the fly (test sets are not cached).
Output: results/<exp>/test_predictions/<dataset>[/<reference>]/<case_id>.nii.gz + manifest.csv. Only E2a
reads a reference mask (to place its boxes); no metric is computed here. Re-running skips existing files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from msbench.config import REPO_ROOT, load_config, paths
from msbench.medsam.model import load_medsam
from msbench.medsam.train import CONDITIONS, load_case, load_trainable, predict, save_prediction

RESULTS = Path(os.environ.get("MSBENCH_RESULTS", REPO_ROOT / "results"))
SETS = {"msseg2016_test": ("msseg2016", "test"), "isbi2015": ("isbi2015", None)}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, choices=list(CONDITIONS))
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    args = ap.parse_args(argv)
    cfg = load_config()
    cond = CONDITIONS[args.exp]
    p = paths(cfg)
    index = pd.read_csv(p["data_processed"] / "index.csv", dtype=str)
    seed = int(cfg["seed"])
    models = []
    for k in range(5):
        ck = p["medsam_runs"] / args.exp / f"fold_{k}" / "checkpoint_final.pt"
        if not ck.exists():
            raise SystemExit(f"missing fold model {ck}")
        m = load_medsam(cfg)
        m.image_encoder = None
        load_trainable(m, torch.load(ck, weights_only=False))
        models.append(m.eval())
    encoder = load_medsam(cfg).image_encoder.eval() if cond == "zoomed" else None
    out_root = RESULTS / args.exp / "test_predictions"
    manifest = []
    for s in args.sets:
        ds, split = SETS[s]
        cases = index[(index.dataset == ds) & ((index.release_split == split) if split else True)]
        for r in cases.sort_values("case_id").itertuples():
            refs = [None]
            if cond == "oracle":
                refs = ["lesion"] if ds == "msseg2016" else r.references.split(";")
            for ref_name in refs:
                out_dir = out_root / ds / (ref_name if (cond == "oracle" and ds == "isbi2015") else "")
                f = out_dir / f"{r.case_id}.nii.gz"
                manifest.append({"case_id": r.case_id, "dataset": ds, "release_split": r.release_split,
                                 "scanner": r.scanner, "boxes_from": ref_name or {"nnunet": "E1 ensemble",
                                 "independent": "brain box + tiles", "zoomed": "brain box + zoomed tiles"}[cond],
                                 "prediction": str(f.relative_to(out_root)), "models": 5})
                if f.exists():
                    continue
                t0 = time.time()
                box_dir = RESULTS / "E1" / "test_predictions" / ds if cond == "nnunet" else None
                cd = load_case(r.case_id, cfg, with_ref=cond == "oracle", box_source_dir=box_dir,
                               zoom=cond == "zoomed", ref_name=ref_name or "lesion")
                mask = predict(models, cd, cond, cfg, seed, encoder=encoder)["mask"]
                save_prediction(mask, r.case_id, cfg, out_dir)
                log(f"{args.exp} {ds} {r.case_id}{' (' + ref_name + ')' if ref_name and ds == 'isbi2015' else ''}: "
                    f"{time.time() - t0:.0f} s")
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(out_root / "manifest.csv", index=False)
    (out_root / "README.txt").write_text(
        f"{args.exp} 5-fold-ensemble test predictions (scripts/predict_test_medsam.py). Not scored.\n")
    log(f"{args.exp}: {len(manifest)} test predictions in {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
