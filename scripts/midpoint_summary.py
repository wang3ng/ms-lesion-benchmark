"""Midpoint summary from whatever E1 folds exist so far.

    source scripts/env.sh
    python scripts/midpoint_summary.py            # -> results/midpoint/summary.md + figures/

Reads: data/processed/index.csv + lesion_stats.csv, MSSEG training consensus
masks (size bins), results/<exp>/fold<k>_cases.csv (scripts/evaluate.py),
nnU-Net fold folders (training logs, progress.png), ISBI rater masks (agreement).
Safe to run while training is still going: missing folds are reported as such.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from msbench.config import REPO_ROOT, load_config, nnunet_fold_dir, nnunet_trainer, paths  # noqa: E402
from msbench.metrics import bin_names, dice  # noqa: E402
from msbench.reporting import (  # noqa: E402
    INK, INK2, MAIN_METRICS, SCANNER_LABELS, SURFACE, fig_agreement_scatter, job_alive, mask_empty_bins,
    md_table, mean_sd, new_fig,
)
from msbench.summaries import bin_counts, nnunet_training_progress, training_lesion_volumes  # noqa: E402


# ---------------------------------------------------------------- data facts
def isbi_rater_agreement(processed: Path, index: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    isbi = index[index.dataset == "isbi2015"]
    rows = []
    s = stats[stats.dataset == "isbi2015"].set_index(["case_id", "reference"])
    for case_id, rel in zip(isbi.case_id, isbi.path):
        m1 = np.asanyarray(nib.load(str(processed / rel / "lesion_rater1.nii.gz")).dataobj) > 0
        m2 = np.asanyarray(nib.load(str(processed / rel / "lesion_rater2.nii.gz")).dataobj) > 0
        rows.append({
            "case_id": case_id,
            "rater1_lesions": int(s.loc[(case_id, "lesion_rater1"), "n_lesions"]),
            "rater2_lesions": int(s.loc[(case_id, "lesion_rater2"), "n_lesions"]),
            "rater1_ml": float(s.loc[(case_id, "lesion_rater1"), "lesion_volume_ml"]),
            "rater2_ml": float(s.loc[(case_id, "lesion_rater2"), "lesion_volume_ml"]),
            "inter_rater_dice": dice(m1, m2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def fig_recall_by_bin(cases: pd.DataFrame, names: list[str], out: Path) -> pd.DataFrame:
    """Pooled lesion-level recall per bin (bars) + per-fold pooled recall (dots)."""
    pooled = pd.DataFrame({
        "detected": [cases[f"detected_{b}"].sum() for b in names],
        "reference_lesions": [cases[f"n_ref_{b}"].sum() for b in names],
    }, index=names)
    pooled["recall"] = pooled.detected / pooled.reference_lesions.replace(0, np.nan)

    fig, ax = new_fig(6.4, 4.0)
    x = np.arange(len(names))
    ax.bar(x, pooled.recall.fillna(0), width=0.55, color="#2a78d6", edgecolor=SURFACE, linewidth=2,
           label="all evaluated folds (pooled)", zorder=2)
    folds = sorted(cases.fold.unique())
    for j, f in enumerate(folds):
        c = cases[cases.fold == f]
        r = [c[f"detected_{b}"].sum() / c[f"n_ref_{b}"].sum() if c[f"n_ref_{b}"].sum() else np.nan for b in names]
        ax.scatter(x + (j - (len(folds) - 1) / 2) * 0.12, r, s=46, color=INK, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, marker="os^Dv"[j % 5], label=f"fold {f}")
    # Values live in the tick labels so they never collide with the fold markers.
    ticks = [f"{b} mm³\nrecall {'–' if np.isnan(r) else f'{r:.2f}'} · n = {int(n)}"
             for b, r, n in zip(names, pooled.recall, pooled.reference_lesions)]
    ax.set_xticks(x, ticks, color=INK, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Lesion recall (detected / reference)", color=INK2, fontsize=9)
    ax.set_title("E1 validation: lesion recall by lesion volume", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return pooled


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="E1")
    ap.add_argument("--results-dir", type=Path,
                    default=Path(os.environ.get("MSBENCH_RESULTS", REPO_ROOT / "results")))
    ap.add_argument("--out", type=Path, default=None, help="default: <results-dir>/midpoint")
    ap.add_argument("--nnunet-results", type=Path, help="override nnUNet_results (testing)")
    args = ap.parse_args(argv)

    cfg = load_config()
    p = paths(cfg)
    processed = p["data_processed"]
    out = args.out or args.results_dir / "midpoint"
    figs = out / "figures"
    if figs.exists():
        shutil.rmtree(figs)
    figs.mkdir(parents=True)

    index = pd.read_csv(processed / "index.csv", dtype={"timepoint": str, "center": str})
    stats = pd.read_csv(processed / "lesion_stats.csv")
    edges = [float(b) for b in cfg["metrics"]["size_bins_mm3"]]
    names = bin_names(edges)
    n_folds = cfg["models"]["nnunet"]["folds"]
    epochs_target = cfg["models"]["nnunet"]["epochs"]
    rule = cfg["metrics"]["lesion_detection"]
    logdir = Path(os.environ.get("MSBENCH_LOGS", p["logs"])) / args.exp
    alive = job_alive(logdir)
    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    md: list[str] = []
    add = md.append
    add(f"# Midpoint summary — {args.exp} (nnU-Net v2 {cfg['models']['nnunet']['configuration']}, MSSEG-2016)\n")
    add(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} from commit `{commit}` by `scripts/midpoint_summary.py`. "
        f"Lesion rule: {rule['connectivity']}-connected components > {rule['min_lesion_mm3']} mm³, "
        f"α = {rule['alpha']}, β = {rule.get('beta', '–')}, γ = {rule['gamma']} "
        "(port of animaSegPerfAnalyzer 4.2). "
        f"Size bins (fixed from MSSEG training counts): {', '.join(names)} mm³. "
        "All metrics below are on **cross-validation folds of the 15 MSSEG training cases**; "
        "no test case has been scored.\n")

    # ---- training status
    add("## Training status\n")
    trows, eval_cases = [], []
    for k in range(n_folds):
        fold_dir = nnunet_fold_dir(k, cfg, args.nnunet_results)
        tp = nnunet_training_progress(fold_dir)
        status = tp.status
        if status == "incomplete":
            status = "running" if alive else "stopped (incomplete)"
        csv = args.results_dir / args.exp / f"fold{k}_cases.csv"
        if csv.exists():
            eval_cases.append(pd.read_csv(csv))
        trows.append({
            "fold": k, "status": status, "epochs": f"{tp.epochs_completed} / {epochs_target}",
            "train time (h)": round(tp.train_hours, 2),
            "s / epoch": round(tp.mean_epoch_s, 1) if tp.epochs_completed else float("nan"),
            "last pseudo-Dice": tp.last_pseudo_dice, "best EMA pseudo-Dice": tp.best_ema_pseudo_dice,
            "last log entry": tp.last_log_time or "–", "evaluated": "yes" if csv.exists() else "no",
        })
        png = fold_dir / "progress.png"
        if png.exists():
            suffix = "" if tp.status == "finished" else "_partial"
            shutil.copyfile(png, figs / f"fold{k}_progress{suffix}.png")
    add(md_table(pd.DataFrame(trows), index=False))
    add(f"\nTrainer `{nnunet_trainer(cfg)}` ({epochs_target} epochs, same for every fold). Train time = sum of "
        "nnU-Net's per-epoch times (250 training + 50 validation iterations each). Pseudo-Dice is nnU-Net's "
        "online estimate on validation patches, not the final metric. Background job "
        f"{'**running**' if alive else 'not running'} (`{logdir / 'job.pid'}`).\n")

    # ---- datasets
    add("## Datasets\n")
    ds = (index.groupby(["dataset", "release_split", "scanner"]).agg(
        cases=("case_id", "size"), subjects=("subject_id", "nunique")).reset_index())
    add(md_table(ds, index=False))
    add("\nMSSEG `test` holds the 8 GE Discovery cases (unseen scanner). ISBI scans come from the release's "
        "labelled `train` folder and are our external test set. MSLesSeg is for pipeline development only.\n")

    # ---- size bins
    add("## Lesion counts per size bin (MSSEG training consensus)\n")
    vols = training_lesion_volumes(cfg)
    by_sc = bin_counts(vols, edges, by="scanner")
    by_sc.loc["all"] = by_sc.sum()
    vox = vols.groupby("scanner").voxel_volume_mm3.median()
    by_sc["voxel mm³"] = vox.reindex(by_sc.index).round(2)
    by_sc["min lesion (voxels)"] = (rule["min_lesion_mm3"] / vox).reindex(by_sc.index).apply(np.ceil)
    add(md_table(by_sc.rename_axis("scanner"), floatfmt="{:g}"))
    add("")

    # ---- metrics
    add("## Validation metrics\n")
    if not eval_cases:
        add("_No fold has been evaluated yet (`results/{}/fold<k>_cases.csv` missing)._\n".format(args.exp))
        cases = None
    else:
        cases = mask_empty_bins(pd.concat(eval_cases, ignore_index=True), names)
        metric_cols = MAIN_METRICS + [f"recall_{b}" for b in names] + [f"f1_{b}" for b in names]
        tbl = {f"fold {f}": [mean_sd(cases[cases.fold == f][m]) for m in metric_cols]
               for f in sorted(cases.fold.unique())}
        tbl["pooled"] = [mean_sd(cases[m]) for m in metric_cols]
        add("Mean ± SD over validation cases. Size-bin rows: recall leaves out cases without reference lesions "
            "in that bin; F1 leaves out cases with no lesion of that size in either mask, so n can be smaller. "
            "`fp_lesions` = false-positive lesions per scan.\n")
        add(md_table(pd.DataFrame(tbl, index=metric_cols).rename_axis("metric")))
        pooled = fig_recall_by_bin(cases, names, figs / "recall_by_size_bin.png")
        add("\nLesion-level recall pooled over all evaluated cases (detected / reference lesions):\n")
        add(md_table(pooled.rename_axis("bin (mm³)").astype({"detected": int, "reference_lesions": int})))
        fig_agreement_scatter(cases, figs / "count_volume_scatter.png")

        add("\n### Per-case results\n")
        pc_cols = ["fold", "case_id", "scanner", "dice", "lesion_precision", "lesion_recall", "lesion_f1",
                   "fp_lesions", "n_ref_lesions", "n_pred_lesions", "ref_volume_ml", "pred_volume_ml"] + \
                  [f"recall_{b}" for b in names]
        add(md_table(cases[pc_cols].sort_values(["fold", "case_id"]), index=False))
        add("")

    # ---- figures
    add("## Figures\n")
    for f in sorted(figs.glob("*.png")):
        add(f"- [{f.name}](figures/{f.name})")
        add(f"\n  ![{f.stem}](figures/{f.name})\n")

    # ---- data facts
    add("## Data facts worth reporting\n")
    agree = isbi_rater_agreement(processed, index, stats)
    diff = (agree.rater2_lesions - agree.rater1_lesions)
    add(f"**ISBI 2015 rater disagreement.** Over {len(agree)} scans, rater 1 marks "
        f"{agree.rater1_lesions.median():.0f} lesions per scan (median) and rater 2 {agree.rater2_lesions.median():.0f}; "
        f"the per-scan count difference ranges {diff.min():+d} to {diff.max():+d} "
        f"(mean |difference| {diff.abs().mean():.1f} lesions). Inter-rater voxel Dice is "
        f"{agree.inter_rater_dice.mean():.3f} ± {agree.inter_rater_dice.std(ddof=1):.3f} "
        f"(range {agree.inter_rater_dice.min():.3f}–{agree.inter_rater_dice.max():.3f}). "
        "This is why each rater is scored separately.\n")
    add("<details><summary>Per-scan ISBI rater table</summary>\n")
    add(md_table(agree, index=False))
    add("\n</details>\n")

    ms = index[index.dataset == "msseg2016"]
    empty = ms[ms.n_lesions.fillna(0) == 0]
    for _, r in empty.iterrows():
        ex = stats[(stats.case_id == r.case_id) & stats.reference.str.startswith("lesion_rater")]
        counts = ", ".join(f"{int(n)}" for n in ex.sort_values("reference").n_lesions)
        add(f"**Empty consensus.** `{r.case_id}` ({r.release_split}, {SCANNER_LABELS.get(r.scanner, r.scanner)}) "
            f"has no lesion in the consensus; the 7 experts marked {counts} lesions. With our convention "
            "(empty reference and empty prediction → F1 = 1; any predicted lesion → F1 = 0, precision 0) this case "
            "rewards a model that predicts nothing, and its Dice is 1 or 0. Report it separately or note it.\n")

    small = by_sc[names[0]].drop("all")
    tot = int(small.sum())
    parts = ", ".join(f"{SCANNER_LABELS.get(s, s)} {int(n)} ({100 * n / tot:.0f}%)" for s, n in small.items())
    add(f"**Who contributes the smallest bin ({names[0]} mm³).** {tot} training lesions: {parts}. "
        f"Voxel volume differs {vox.max() / vox.min():.1f}× between scanners "
        f"({vox.min():.2f}–{vox.max():.2f} mm³, see table above), so a 20 mm³ lesion is "
        f"≈{20 / vox.get('siemens_aera_1p5t', np.nan):.0f} voxels on the Aera 1.5T but "
        f"≈{20 / vox.get('philips_ingenia_3t', np.nan):.0f} on the Philips. Small-lesion results are therefore "
        "largely Philips results; per-scanner size-bin numbers for the Aera are thin.\n")

    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'summary.md'}")
    for f in sorted(figs.glob("*.png")):
        print(f"      {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
