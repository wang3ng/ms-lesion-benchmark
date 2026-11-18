"""Check msbench.metrics against the real animaSegPerfAnalyzer on MSSEG training cases.

    source scripts/env.sh
    python scripts/anima_check.py --anima ../tools/anima-4.2/animaSegPerfAnalyzer
    python scripts/anima_check.py --anima ... --cases all     # all 15 training patients

For each case, the reference is the MSSEG consensus and the "tested" masks are
(1) the E1 out-of-fold nnU-Net prediction and (2) each of the 7 expert masks.
Both masks are written as uint8 with the reference header (Anima refuses any
origin/direction difference; voxels are unchanged). Anima runs with its default
detection parameters (-x 0.10 -y 0.70 -z 0.65 -v 3), the values in
configs/experiment.yaml. Reports lesion PPV / sensitivity / F1 and Dice from both
implementations, plus what the earlier 26-connected rule gave. Training cases only;
only summary numbers are printed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from msbench.config import REPO_ROOT, load_config, paths
from msbench.evaluation import detection_rule, load_mask
from msbench.metrics import dice, lesion_detection

DEFAULT_CASES = ["msseg_train_c01_p01", "msseg_train_c07_p01", "msseg_train_c08_p01"]  # Verio, Aera, Philips


def run_anima(anima: Path, pred: Path, ref: Path, out_base: Path, rule: dict) -> dict:
    cmd = [str(anima), "-i", str(pred), "-r", str(ref), "-o", str(out_base), "-l", "-s", "-X",
           "-x", str(rule["alpha"]), "-y", str(rule["beta"]), "-z", str(rule["gamma"]),
           "-v", str(rule["min_lesion_mm3"]), "-t", "8"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"anima failed ({res.returncode}): {res.stderr[-500:]}")
    root = ET.parse(f"{out_base}_global.xml").getroot()  # values rounded to 6 decimals
    vals = {m.get("name"): float(m.text) for m in root.iter("measure")}
    return vals


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anima", type=Path, required=True, help="path to animaSegPerfAnalyzer")
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES, help="case ids, or 'all'")
    ap.add_argument("--pred-dir", type=Path, default=Path(os.environ.get("MSBENCH_RESULTS", REPO_ROOT / "results"))
                    / "E1" / "oof_predictions")
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("MSBENCH_RESULTS", REPO_ROOT / "results"))
                    / "E1" / "tables" / "anima_check.csv")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args(argv)

    cfg = load_config()
    rule = detection_rule(cfg)
    old_rule = dict(rule, connectivity=26)
    processed = paths(cfg)["data_processed"]
    index = pd.read_csv(processed / "index.csv", dtype=str).set_index("case_id")
    cases = sorted(index[(index.dataset == "msseg2016") & (index.release_split == "train")].index) \
        if args.cases == ["all"] else args.cases
    if any(index.loc[c, "release_split"] != "train" for c in cases):
        raise SystemExit("only MSSEG training cases may be used here")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for case in cases:
            meta = index.loc[case]
            case_dir = processed / meta["path"]
            ref_img = nib.load(str(case_dir / "lesion.nii.gz"))
            ref, _, spacing = load_mask(case_dir / "lesion.nii.gz")
            tested = {"nnunet_oof": args.pred_dir / f"{case}.nii.gz"}
            tested.update({f"expert{k}": case_dir / f"lesion_rater{k}.nii.gz" for k in range(1, 8)})
            hdr = ref_img.header.copy()
            hdr.set_data_dtype(np.uint8)
            ref_f = tmp / f"{case}_ref.nii.gz"
            nib.save(nib.Nifti1Image(ref.astype(np.uint8), ref_img.affine, hdr), ref_f)
            for name, path in tested.items():
                if not path.exists():
                    continue
                pred, _, _ = load_mask(path)
                pred_f = tmp / f"{case}_{name}.nii.gz"
                nib.save(nib.Nifti1Image(pred.astype(np.uint8), ref_img.affine, hdr), pred_f)
                a = run_anima(args.anima, pred_f, ref_f, tmp / f"{case}_{name}", rule)
                ours = lesion_detection(pred, ref, spacing, **rule)
                old = lesion_detection(pred, ref, spacing, **old_rule)
                rows.append({
                    "case_id": case, "scanner": meta["scanner"], "tested": name,
                    "anima_ppvl": a["PPVL"], "ours_precision": ours.precision,
                    "anima_sensl": a["SensL"], "ours_recall": ours.recall,
                    "anima_f1": a["F1_score"], "ours_f1": ours.f1,
                    "anima_dice": a["Dice"], "ours_dice": dice(pred, ref),
                    "n_ref": ours.n_ref, "n_pred": ours.n_pred,
                    "old26_precision": old.precision, "old26_recall": old.recall, "old26_f1": old.f1,
                })
    df = pd.DataFrame(rows)
    pairs = [("anima_ppvl", "ours_precision"), ("anima_sensl", "ours_recall"),
             ("anima_f1", "ours_f1"), ("anima_dice", "ours_dice")]
    for a, o in pairs:
        df[f"diff_{o.split('_', 1)[1]}"] = (df[o] - df[a]).abs()
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    df["match"] = (df[diff_cols].fillna(0) <= args.tol).all(axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    show = ["case_id", "tested", "n_ref", "n_pred", "anima_sensl", "ours_recall", "anima_ppvl", "ours_precision",
            "anima_f1", "ours_f1", "old26_f1", "match"]
    print(df[show].round(4).to_string(index=False))
    print(f"\n{int(df.match.sum())} / {len(df)} comparisons match within {args.tol:g} "
          f"(max |diff|: " + ", ".join(f"{c[5:]} {df[c].max():.2e}" for c in diff_cols) + ")")
    print(f"old 26-connected rule vs Anima, mean |F1 diff|: {(df.old26_f1 - df.anima_f1).abs().mean():.4f}")
    print(f"wrote {args.out}")
    return 0 if df.match.all() else 1


if __name__ == "__main__":
    sys.exit(main())
