"""EXPLORATORY (post-hoc, decided after seeing E2c results): MedSAM predicted-IoU mask filter, nested CV.

    source scripts/env.sh
    python scripts/iou_filter_analysis.py predict --exp E2c        # GPU: inner models + level volumes
    python scripts/iou_filter_analysis.py select --exp E2c         # CPU: choose thresholds, score outer folds

Rule (configs/experiment.yaml analysis.iou_filter): a box whose MedSAM predicted-IoU score is below t
gets an empty mask (it still counts in the per-pixel coverage of the mean-probability combination).

Nested CV, per outer fold k of E1's 5 folds:
  * inner CV over fold k's 12 TRAINING patients only (4 inner folds of 3, one patient per scanner where
    possible): an inner model per inner fold, same recipe as the outer models (10,000 steps, same loss,
    same collapse check), trained on 9 patients, predicts the 3 held out;
  * t_k = threshold with the highest mean per-patient lesion F1 over those 12 inner held-out
    predictions (ties -> lowest threshold; "no filter" is a candidate);
  * the outer fold-k model (already trained) predicts fold k's 3 validation patients, filtered with t_k.
The validation patients are never used to choose t_k. Not applied to any test set.

predict writes, per model, uint8 "level" volumes (msbench.medsam.train.predict): voxel value v = number of
thresholds at which the voxel is still positive, so one file holds every threshold. Inner runs:
<medsam_runs>/<exp>_iou_inner/fold_k/inner_j/; outer levels: <medsam_runs>/<exp>/fold_k/iou_levels/.
select writes <medsam_runs>/<exp>/fold_k/validation_iou/ masks, results/<exp>_iou/fold<k>_cases.csv
(scripts/evaluate.py) and results/E2/tables/iou_filter_{selection,curves}.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from msbench.config import REPO_ROOT, load_config, paths
from msbench.evaluation import detection_rule
from msbench.medsam.model import load_medsam
from msbench.medsam.train import CONDITIONS, iou_levels, load_case, load_trainable, predict, save_prediction, train

RESULTS = Path(os.environ.get("MSBENCH_RESULTS", REPO_ROOT / "results"))


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def inner_folds(train_ids: list[str], n: int, seed: int, cfg: dict) -> list[list[str]]:
    """n patient-level inner folds; patients of each scanner spread round-robin (seeded shuffle)."""
    index = pd.read_csv(paths(cfg)["data_processed"] / "index.csv", dtype=str).set_index("case_id")
    rng = np.random.default_rng(seed)
    order = []
    for sc in sorted(set(index.loc[train_ids, "scanner"])):
        ids = sorted(c for c in train_ids if index.loc[c, "scanner"] == sc)
        order += [ids[i] for i in rng.permutation(len(ids))]
    folds = [[] for _ in range(n)]
    for i, c in enumerate(order):
        folds[i % n].append(c)
    return [sorted(f) for f in folds]


def save_levels(vol: np.ndarray, case_id: str, cfg: dict, out_dir: Path) -> None:
    save_prediction(vol, case_id, cfg, out_dir)  # uint8 values kept (FLAIR header, uint8)


def _model(cfg, ckpt: Path):
    m = load_medsam(cfg)
    m.image_encoder = None
    load_trainable(m, torch.load(ckpt, weights_only=False))
    return m.eval()


def cmd_predict(args, cfg) -> int:
    cond = CONDITIONS[args.exp]
    zoom = cond == "zoomed"
    ic = cfg["analysis"]["iou_filter"]
    levels = iou_levels(cfg)
    splits = json.loads((REPO_ROOT / "experiments" / "E1" / "splits_final.json").read_text())
    runs = paths(cfg)["medsam_runs"]
    status = 0
    for k in args.folds:
        sp = splits[k]
        seed = int(cfg["seed"]) + 100 * k
        # outer model -> level volumes of the outer validation patients
        outer = runs / args.exp / f"fold_{k}"
        lv = outer / "iou_levels"
        if not all((lv / f"{c}.nii.gz").exists() for c in sp["val"]):
            m = _model(cfg, outer / "checkpoint_final.pt")
            for c in sp["val"]:
                cd = load_case(c, cfg, with_ref=False, zoom=zoom)
                save_levels(predict([m], cd, cond, cfg, seed, levels=levels)["levels"], c, cfg, lv)
            log(f"{args.exp} fold {k}: outer validation levels written")
        folds = inner_folds(sp["train"], int(ic["inner_folds"]), seed, cfg)
        for j, held in enumerate(folds):
            run_dir = runs / f"{args.exp}_iou_inner" / f"fold_{k}" / f"inner_{j}"
            run_dir.mkdir(parents=True, exist_ok=True)
            tr = sorted(set(sp["train"]) - set(held))
            (run_dir / "split.json").write_text(json.dumps({"train": tr, "held_out": held}, indent=2))
            out = run_dir / "levels"
            if all((out / f"{c}.nii.gz").exists() for c in held):
                continue
            check = run_dir / "collapse_check.json"
            if check.exists() and not json.loads(check.read_text())["passed"]:
                log(f"{args.exp} fold {k} inner {j}: collapse check failed earlier; skipped")
                status = 3
                continue
            iseed = seed + 10 * (j + 1)
            if not (run_dir / "checkpoint_final.pt").exists():
                log(f"{args.exp} fold {k} inner {j}: train on {len(tr)} patients, hold out {held}")
                cases = [load_case(c, cfg, with_ref=True, zoom=zoom) for c in tr]
                info = train(cases, cond, cfg, run_dir, iseed, log=log, num_workers=args.workers)
                del cases
                if info["collapsed"]:
                    log(f"{args.exp} fold {k} inner {j}: COLLAPSE; no prediction")
                    status = 3
                    continue
            m = _model(cfg, run_dir / "checkpoint_final.pt")
            for c in held:
                cd = load_case(c, cfg, with_ref=False, zoom=zoom)
                save_levels(predict([m], cd, cond, cfg, iseed, levels=levels)["levels"], c, cfg, out)
            log(f"{args.exp} fold {k} inner {j}: levels written for {held}")
    return status


# ---------------------------------------------------------------- selection (CPU)
def _f1_curve(job) -> list[float]:
    """Lesion F1 at every level for one patient."""
    lev_path, ref_path, n_levels, rule = job
    from msbench.metrics import lesion_detection
    lev = np.asanyarray(nib.load(lev_path).dataobj)
    img = nib.load(ref_path)
    ref = np.asanyarray(img.dataobj) > 0
    sp = tuple(float(z) for z in img.header.get_zooms()[:3])
    return [lesion_detection(lev > j, ref, sp, **rule).f1 for j in range(n_levels)]


def cmd_select(args, cfg) -> int:
    levels = iou_levels(cfg)
    splits = json.loads((REPO_ROOT / "experiments" / "E1" / "splits_final.json").read_text())
    runs = paths(cfg)["medsam_runs"]
    index = pd.read_csv(paths(cfg)["data_processed"] / "index.csv", dtype=str).set_index("case_id")
    rule = detection_rule(cfg)
    ref = lambda c: str(paths(cfg)["data_processed"] / index.loc[c, "path"] / "lesion.nii.gz")  # noqa: E731
    tables = RESULTS / "E2" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    sel_rows, curve_rows = [], []
    for k in args.folds:
        jobs, ids = [], []
        for j in range(int(cfg["analysis"]["iou_filter"]["inner_folds"])):
            d = runs / f"{args.exp}_iou_inner" / f"fold_{k}" / f"inner_{j}"
            held = json.loads((d / "split.json").read_text())["held_out"]
            for c in held:
                f = d / "levels" / f"{c}.nii.gz"
                if not f.exists():
                    raise SystemExit(f"missing inner prediction {f}")
                jobs.append((str(f), ref(c), len(levels), rule))
                ids.append(c)
        with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as ex:
            curves = np.array(list(ex.map(_f1_curve, jobs)))           # (patients, levels)
        mean_f1 = np.nanmean(curves, axis=0)
        best = int(np.flatnonzero(mean_f1 == np.nanmax(mean_f1))[0])
        for j, t in enumerate(levels):
            curve_rows.append({"model": args.exp, "fold": k, "level": j, "threshold": t,
                               "inner_mean_lesion_f1": mean_f1[j]})
        sel_rows.append({"model": args.exp, "fold": k, "inner_patients": len(ids), "level": best,
                         "threshold": levels[best], "inner_f1_no_filter": mean_f1[0],
                         "inner_f1_selected": mean_f1[best]})
        log(f"{args.exp} fold {k}: threshold {levels[best]} (inner mean F1 {mean_f1[0]:.3f} -> {mean_f1[best]:.3f})")
        # apply to the outer validation patients
        outer = runs / args.exp / f"fold_{k}"
        vdir = outer / "validation_iou"
        mismatch = 0
        for c in splits[k]["val"]:
            img = nib.load(str(outer / "iou_levels" / f"{c}.nii.gz"))
            lev = np.asanyarray(img.dataobj)
            base = np.asanyarray(nib.load(str(outer / "validation" / f"{c}.nii.gz")).dataobj) > 0
            mismatch += int(((lev > 0) != base).sum())
            save_prediction((lev > best).astype(np.uint8), c, cfg, vdir)
        sel_rows[-1]["unfiltered_mismatch_voxels"] = mismatch   # level 0 must reproduce the saved prediction
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"), "--exp", f"{args.exp}_iou",
                        "--fold", str(k), "--pred-dir", str(vdir), "--cases", *splits[k]["val"]], check=True)
    for name, rows in (("iou_filter_selection", sel_rows), ("iou_filter_curves", curve_rows)):
        f = tables / f"{name}.csv"
        old = pd.read_csv(f) if f.exists() else pd.DataFrame()
        if len(old):
            old = old[~((old.model == args.exp) & old.fold.isin(args.folds))]
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True).sort_values(["model", "fold"]).to_csv(f, index=False)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["predict", "select"])
    ap.add_argument("--exp", required=True, choices=["E2c", "E8"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    cfg = load_config()
    return cmd_predict(args, cfg) if args.cmd == "predict" else cmd_select(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
