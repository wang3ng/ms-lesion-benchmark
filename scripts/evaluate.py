"""Score one fold of an experiment and save a per-case CSV.

    source scripts/env.sh
    python scripts/evaluate.py --exp E1 --fold 0
        -> results/E1/fold0_cases.csv  (+ fold0_meta.json)

E1 reads nnU-Net's validation predictions
(<nnUNet_results>/Dataset501_MSSEG2016/<trainer>__nnUNetPlans__3d_fullres/fold_<k>/validation)
and checks that every validation case of that fold (experiments/E1/splits_final.json)
has a prediction. Other experiments pass --pred-dir (and optionally --cases).

Columns: Dice; lesion precision / recall / F1; fp_lesions (false-positive
lesions in this scan); reference vs predicted lesion count and volume (mL);
per size bin (configs/experiment.yaml metrics.size_bins_mm3): recall_, f1_,
precision_, n_ref_, detected_, n_pred_. Lesion rule from the config.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from msbench.config import REPO_ROOT, load_config, nnunet_fold_dir
from msbench.evaluation import detection_rule, evaluate_predictions, size_bins


def _git_commit() -> str:
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="experiment ID, e.g. E1")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--pred-dir", type=Path, help="folder of <case_id>.nii.gz (default for E1: nnU-Net validation)")
    ap.add_argument("--cases", nargs="+", help="case IDs to score (default: fold's validation cases / all preds)")
    ap.add_argument("--reference", default="lesion", help="reference mask name (lesion, lesion_rater1, ...)")
    ap.add_argument("--results-dir", type=Path,
                    default=Path(os.environ.get("MSBENCH_RESULTS", REPO_ROOT / "results")))
    ap.add_argument("--nnunet-results", type=Path, help="override nnUNet_results (testing)")
    args = ap.parse_args(argv)

    cfg = load_config()
    nnunet_summary = None
    if args.pred_dir is None:
        if args.exp != "E1":
            ap.error("--pred-dir is required for experiments other than E1")
        fold_dir = nnunet_fold_dir(args.fold, cfg, args.nnunet_results)
        args.pred_dir = fold_dir / "validation"
        summary_json = args.pred_dir / "summary.json"
        if summary_json.exists():
            nnunet_summary = json.loads(summary_json.read_text()).get("foreground_mean")
    if args.cases is None:
        splits_file = REPO_ROOT / "experiments" / args.exp / "splits_final.json"
        if splits_file.exists():
            args.cases = json.loads(splits_file.read_text())[args.fold]["val"]
        else:
            args.cases = sorted(p.name[: -len(".nii.gz")] for p in args.pred_dir.glob("*.nii.gz"))

    df = evaluate_predictions(args.pred_dir, args.cases, args.exp, args.fold, args.reference, cfg)

    out_dir = args.results_dir / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fold{args.fold}_cases" if args.reference == "lesion" else f"fold{args.fold}_cases_{args.reference}"
    csv_path = out_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False, float_format="%.6g")
    meta = {
        "exp": args.exp, "fold": args.fold, "reference": args.reference,
        "pred_dir": str(args.pred_dir), "n_cases": len(df), "cases": list(df.case_id),
        "rule": detection_rule(cfg), "size_bins_mm3": size_bins(cfg),
        "nnunet_validation_summary_foreground_mean": nnunet_summary,
        "git_commit": _git_commit(), "created": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / f"{stem.replace('_cases', '_meta')}.json").write_text(json.dumps(meta, indent=2))

    cols = ["case_id", "scanner", "dice", "lesion_precision", "lesion_recall", "lesion_f1",
            "fp_lesions", "n_ref_lesions", "n_pred_lesions", "ref_volume_ml", "pred_volume_ml"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(df[cols].to_string(index=False))
        print("\nmean:", df[["dice", "lesion_precision", "lesion_recall", "lesion_f1"]].mean().round(3).to_dict())
    if nnunet_summary:
        print(f"nnU-Net's own validation Dice (foreground_mean): {nnunet_summary.get('Dice', float('nan')):.3f}")
    print(f"saved {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
