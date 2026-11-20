"""E7 (exploratory, after the results freeze): predictions of MS-SAMLess (phase 2) and the HarDNet baseline.

    source scripts/env.sh
    python scripts/e7_predict.py --exp E7 --cv            # each fold model -> its 3 validation patients
    python scripts/e7_predict.py --exp E7 --test          # 5-fold ensemble -> MSSEG test + ISBI
    python scripts/e7_predict.py --exp E7_hardnet --cv --test

E7 = MS-SAMLess (phase-2 network, SAM-free at inference); E7_hardnet = HarDNet Segmentor baseline. Last-epoch
checkpoints only. Preprocessing/back-mapping: msbench.e7.predict (1 mm at inference, mapped back to the case
grid). CV -> <e7_runs>/<exp>/fold_<k>/validation/<case>.nii.gz (score with scripts/evaluate.py);
test -> results/<exp>/test_predictions/<dataset>/<case>.nii.gz + manifest.csv (no scoring here).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from msbench.config import REPO_ROOT, load_config, paths
from msbench.e7 import MODALITIES
from msbench.e7.predict import load_segmentor, predict_volume, preprocess, to_original
from msbench.medsam.train import save_prediction

RESULTS = Path(os.environ.get("MSBENCH_RESULTS", REPO_ROOT / "results"))
NAMES = {"E7": ("samless_p2", "phase2"), "E7_hardnet": ("hardnet", "hardnet_baseline")}


def ckpt(cfg, exp: str, fold: int, smoke: bool = False) -> Path:
    stem, part = NAMES[exp]
    name = f"{stem}_fold{fold}" + ("_smoke" if smoke else "")
    epochs = 2 if smoke else int(cfg["e7"][part]["epochs"])
    hits = sorted(paths(cfg)["e7_runs"].glob(f"results/*_{name}/ckpt/net_last_{epochs - 1:05d}.pth"))
    if not hits:
        raise SystemExit(f"missing checkpoint for {name}")
    return hits[-1]


def predict_case(models, case_id: str, index, cfg) -> "np.ndarray":
    d = paths(cfg)["data_processed"] / index.loc[case_id, "path"]
    img, a1, a0, shape = preprocess({m: d / f"{m}.nii.gz" for m in MODALITIES})
    dim = int(cfg["e7"]["phase2"]["image_dim"])
    prob = predict_volume(models, img, dim)
    return to_original(prob, a1, a0, shape) > float(cfg["e7"]["common"]["threshold"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, choices=list(NAMES))
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="fold-0 smoke checkpoint, first validation case only")
    args = ap.parse_args(argv)
    cfg = load_config()
    p = paths(cfg)
    index = pd.read_csv(p["data_processed"] / "index.csv", dtype=str).set_index("case_id")
    splits = json.loads((REPO_ROOT / "experiments" / "E1" / "splits_final.json").read_text())
    dim = int(cfg["e7"]["phase2"]["image_dim"])
    if args.smoke:
        m = load_segmentor(ckpt(cfg, args.exp, 0, smoke=True), dim)
        c = splits[0]["val"][0]
        t = time.time()
        mask = predict_case([m], c, index, cfg)
        out = p["e7_runs"] / "smoke" / args.exp
        save_prediction(mask, c, cfg, out)
        print(json.dumps({"case": c, "seconds": round(time.time() - t, 1), "pred_voxels": int(mask.sum()),
                          "file": str(out / f"{c}.nii.gz")}))
        return 0
    if args.cv:
        for k, sp in enumerate(splits):
            out = p["e7_runs"] / args.exp / f"fold_{k}" / "validation"
            todo = [c for c in sp["val"] if not (out / f"{c}.nii.gz").exists()]
            if not todo:
                continue
            m = load_segmentor(ckpt(cfg, args.exp, k), dim)
            for c in todo:
                t = time.time()
                save_prediction(predict_case([m], c, index, cfg), c, cfg, out)
                print(f"{args.exp} fold {k} {c}: {time.time() - t:.0f} s", flush=True)
    if args.test:
        models = [load_segmentor(ckpt(cfg, args.exp, k), dim) for k in range(5)]
        root = RESULTS / args.exp / "test_predictions"
        rows = []
        cases = index[((index.dataset == "msseg2016") & (index.release_split == "test")) | (index.dataset == "isbi2015")]
        for c in sorted(cases.index):
            ds = cases.loc[c, "dataset"]
            f = root / ds / f"{c}.nii.gz"
            rows.append({"case_id": c, "dataset": ds, "scanner": cases.loc[c, "scanner"],
                         "prediction": f"{ds}/{c}.nii.gz", "models": 5})
            if f.exists():
                continue
            t = time.time()
            save_prediction(predict_case(models, c, index, cfg), c, cfg, root / ds)
            print(f"{args.exp} test {c}: {time.time() - t:.0f} s", flush=True)
        pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
        (root / "README.txt").write_text(f"{args.exp} 5-fold-ensemble test predictions (exploratory, E7). "
                                         "Scored only by scripts/e7_summary.py.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
