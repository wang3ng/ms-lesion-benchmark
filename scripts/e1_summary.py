"""E1 cross-validation summary: pools every evaluated fold (works on partial results).

    source scripts/env.sh
    python scripts/e1_summary.py        # -> results/E1/summary/

Reads results/E1/fold<k>_cases.csv (written by scripts/evaluate.py after each fold),
experiments/E1/splits_final.json (which patient belongs to which fold), the nnU-Net
fold folders (training status) and logs/E1/fold<k>.log (wall time). Reads no images.

Writes to results/E1/summary/:
  summary.md               status, CV metrics (mean ± SD, median, bootstrap 95% CI),
                           recall by size bin, count/volume agreement, per-patient table
  metrics.csv              one row per metric (patient-level statistics)
  recall_by_size_bin.csv   pooled lesion-level recall per bin, overall and per scanner
  cases.csv                all evaluated validation cases (one row per patient)
  figures/recall_by_size_bin.png, figures/patient_metrics.png, figures/count_volume_scatter.png

Each MSSEG training patient is in exactly one validation fold, so after all five
folds every patient has one out-of-fold prediction and n = 15. The bootstrap
resamples patients (10,000 resamples, seed from configs/experiment.yaml); for
pooled lesion recall whole patients are resampled with all their lesions.
With few patients the percentile bootstrap CI is too narrow; treat CIs on
partial results (n < 15) as indicative only.
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

import numpy as np
import pandas as pd

from msbench.config import REPO_ROOT, load_config, nnunet_fold_dir, nnunet_trainer, paths
from msbench.metrics import bin_names
from msbench.reporting import (
    INK, INK2, MAIN_METRICS, SCANNER_COLORS, SCANNER_LABELS, SURFACE, fig_agreement_scatter,
    fold_wall_time, job_alive, mask_empty_bins, md_table, new_fig, save_fig,
)
from msbench.stats import bland_altman, bootstrap_ci, bootstrap_ratio_ci
from msbench.summaries import nnunet_training_progress

BAR_NEUTRAL = "#d3d2cd"
METRIC_LABELS = {
    "dice": "Dice (voxel)", "lesion_precision": "Lesion precision", "lesion_recall": "Lesion recall",
    "lesion_f1": "Lesion F1", "fp_lesions": "FP lesions / scan",
}


# ---------------------------------------------------------------- loading
def load_folds(args, cfg, splits: list[dict], alive: bool) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str]]:
    """Training status table, the evaluated per-case frames, and warnings."""
    rows, frames, warnings = [], [], []
    logdir = Path(os.environ.get("MSBENCH_LOGS", paths(cfg)["logs"])) / args.exp
    epochs_target = cfg["models"]["nnunet"]["epochs"]
    for k, split in enumerate(splits):
        fold_dir = nnunet_fold_dir(k, cfg, args.nnunet_results)
        tp = nnunet_training_progress(fold_dir)
        status = tp.status
        if status == "incomplete":
            status = "running" if alive else "stopped (incomplete)"
        wall = fold_wall_time(logdir / f"fold{k}.log")
        csv = args.results_dir / args.exp / f"fold{k}_cases.csv"
        meta_f = csv.with_name(f"fold{k}_meta.json")
        nn_dice, evaluated = float("nan"), "no"
        if csv.exists():
            df = pd.read_csv(csv)
            df = df[df.reference == "lesion"]
            expected, got = set(split["val"]), set(df.case_id)
            if expected != got:
                warnings.append(f"fold {k}: evaluated cases {sorted(got)} differ from splits_final.json "
                                f"{sorted(expected)}; fold left out")
                evaluated = "case mismatch"
            else:
                frames.append(df)
                evaluated = "yes"
                summ = fold_dir / "validation" / "summary.json"
                if summ.exists() and summ.stat().st_mtime > csv.stat().st_mtime:
                    warnings.append(f"fold {k}: nnU-Net validation is newer than {csv.name}; "
                                    f"rerun scripts/evaluate.py --exp {args.exp} --fold {k}")
                    evaluated = "yes (stale)"
            if meta_f.exists():
                nn = json.loads(meta_f.read_text()).get("nnunet_validation_summary_foreground_mean") or {}
                nn_dice = float(nn.get("Dice", float("nan")))
        rows.append({
            "fold": k, "status": status, "epochs": f"{tp.epochs_completed} / {epochs_target}",
            "wall time (h)": wall["wall_h"], "train time (h)": tp.train_hours if tp.epochs_completed else float("nan"),
            "s / epoch": tp.mean_epoch_s if tp.epochs_completed else float("nan"),
            "nnU-Net val Dice": nn_dice, "evaluated": evaluated,
            "validation patients": ", ".join(c.replace("msseg_train_", "") for c in sorted(split["val"])),
        })
    return pd.DataFrame(rows), frames, warnings


# ---------------------------------------------------------------- statistics
def patient_stats(cases: pd.DataFrame, metrics: list[str], ci_kw: dict) -> pd.DataFrame:
    out = []
    for m in metrics:
        x = pd.to_numeric(cases[m], errors="coerce").dropna().to_numpy()
        ci = bootstrap_ci(x, **ci_kw)
        out.append({
            "metric": m, "n": len(x),
            "mean": x.mean() if len(x) else np.nan,
            "sd": x.std(ddof=1) if len(x) > 1 else np.nan,
            "median": np.median(x) if len(x) else np.nan,
            "ci_low": ci["low"] if len(x) > 1 else np.nan,
            "ci_high": ci["high"] if len(x) > 1 else np.nan,
        })
    return pd.DataFrame(out)


def pooled_bin_recall(cases: pd.DataFrame, names: list[str], ci_kw: dict) -> pd.DataFrame:
    """Lesion-level recall per bin (sum detected / sum reference), overall and per scanner."""
    rows = []
    groups = [("all", cases)] + [(s, cases[cases.scanner == s]) for s in SCANNER_COLORS if s in set(cases.scanner)]
    for group, c in groups:
        for b in names:
            det, ref = c[f"detected_{b}"].to_numpy(float), c[f"n_ref_{b}"].to_numpy(float)
            ci = bootstrap_ratio_ci(det, ref, **ci_kw) if len(c) > 1 else {"low": np.nan, "high": np.nan}
            rows.append({
                "scanner": group, "bin_mm3": b, "patients": len(c),
                "patients_with_lesions_in_bin": int((ref > 0).sum()),
                "detected": int(det.sum()), "reference_lesions": int(ref.sum()),
                "recall": det.sum() / ref.sum() if ref.sum() else np.nan,
                "ci_low": ci["low"], "ci_high": ci["high"],
            })
    return pd.DataFrame(rows)


def fmt_ci(r) -> str:
    if np.isnan(r["mean"]):
        return "–"
    s = f"{r['mean']:.3f}"
    if not np.isnan(r["sd"]):
        s += f" ± {r['sd']:.3f}"
    return s


def fmt_interval(lo, hi) -> str:
    return "–" if np.isnan(lo) else f"[{lo:.3f}, {hi:.3f}]"


# ---------------------------------------------------------------- figures
def fig_recall_by_bin(rec: pd.DataFrame, names: list[str], out: Path) -> None:
    """Pooled recall per bin with patient-bootstrap CI; per-scanner pooled recall as markers."""
    fig, ax = new_fig(6.8, 4.2)
    x = np.arange(len(names))
    allr = rec[rec.scanner == "all"].set_index("bin_mm3").loc[names]
    err = np.vstack([allr.recall - allr.ci_low, allr.ci_high - allr.recall]).clip(min=0)
    # Neutral bars: the scanner markers carry the colour (Verio blue would vanish on a blue bar).
    ax.bar(x, allr.recall.fillna(0), width=0.55, color=BAR_NEUTRAL, edgecolor=SURFACE, linewidth=2,
           label="all patients (pooled)", zorder=2)
    if not np.all(np.isnan(err)):
        ax.errorbar(x, allr.recall, yerr=np.nan_to_num(err), fmt="none", ecolor=INK, elinewidth=1.5,
                    capsize=4, zorder=3, label="95% CI (patient bootstrap)")
    scanners = [s for s in SCANNER_COLORS if s in set(rec.scanner)]
    for j, s in enumerate(scanners):
        r = rec[rec.scanner == s].set_index("bin_mm3").loc[names]
        ax.scatter(x + (j - (len(scanners) - 1) / 2) * 0.16, r.recall, s=48, color=SCANNER_COLORS[s],
                   edgecolor=SURFACE, linewidth=2, zorder=4, label=SCANNER_LABELS.get(s, s))
    ticks = [f"{b} mm³\nrecall {'–' if np.isnan(r) else f'{r:.2f}'} · n = {int(n)}"
             for b, r, n in zip(names, allr.recall, allr.reference_lesions)]
    ax.set_xticks(x, ticks, color=INK, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Lesion recall (detected / reference)", color=INK2, fontsize=9)
    n_pat = int(allr.patients.iloc[0])
    ax.set_title(f"E1 cross-validation ({n_pat} patients): lesion recall by volume",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left", ncol=2)
    save_fig(fig, out)


def fig_patient_metrics(cases: pd.DataFrame, stats: pd.DataFrame, out: Path) -> None:
    """One dot per patient (coloured by scanner) for the four 0–1 metrics; mean and 95% CI beside."""
    metrics = ["dice", "lesion_precision", "lesion_recall", "lesion_f1"]
    fig, ax = new_fig(6.8, 4.0)
    rng = np.random.default_rng(0)  # jitter only
    st = stats.set_index("metric")
    for i, m in enumerate(metrics):
        for s in [s for s in SCANNER_COLORS if s in set(cases.scanner)]:
            v = cases.loc[cases.scanner == s, m]
            ax.scatter(i - 0.12 + rng.uniform(-0.07, 0.07, len(v)), v, s=40, color=SCANNER_COLORS[s],
                       edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=SCANNER_LABELS.get(s, s) if i == 0 else None)
        r = st.loc[m]
        if not np.isnan(r.ci_low):
            ax.errorbar(i + 0.18, r["mean"], yerr=[[r["mean"] - r.ci_low], [r.ci_high - r["mean"]]], fmt="o",
                        color=INK, ms=6, elinewidth=1.5, capsize=4, zorder=3,
                        label="mean, 95% CI" if i == 0 else None)
        else:
            ax.scatter(i + 0.18, r["mean"], s=36, color=INK, zorder=3, label="mean" if i == 0 else None)
    ax.set_xticks(range(len(metrics)), [METRIC_LABELS[m] for m in metrics], color=INK, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"E1 cross-validation: per-patient metrics (n = {len(cases)})", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left", ncol=2)
    save_fig(fig, out)


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="E1")
    ap.add_argument("--results-dir", type=Path,
                    default=Path(os.environ.get("MSBENCH_RESULTS", REPO_ROOT / "results")))
    ap.add_argument("--out", type=Path, default=None, help="default: <results-dir>/<exp>/summary")
    ap.add_argument("--splits", type=Path, default=None, help="default: experiments/<exp>/splits_final.json")
    ap.add_argument("--nnunet-results", type=Path, help="override nnUNet_results (testing)")
    args = ap.parse_args(argv)

    cfg = load_config()
    splits = json.loads((args.splits or REPO_ROOT / "experiments" / args.exp / "splits_final.json").read_text())
    names = bin_names([float(b) for b in cfg["metrics"]["size_bins_mm3"]])
    ci_cfg = cfg["statistics"]["confidence_interval"]
    ci_kw = {"n_boot": int(ci_cfg["n_boot"]), "level": float(ci_cfg["level"]), "seed": int(cfg["seed"])}
    rule = cfg["metrics"]["lesion_detection"]
    epochs = cfg["models"]["nnunet"]["epochs"]
    logdir = Path(os.environ.get("MSBENCH_LOGS", paths(cfg)["logs"])) / args.exp
    alive = job_alive(logdir, marker=f"run_{args.exp}")
    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    out = args.out or args.results_dir / args.exp / "summary"
    figs = out / "figures"
    if figs.exists():
        shutil.rmtree(figs)
    figs.mkdir(parents=True)

    status, frames, warnings = load_folds(args, cfg, splits, alive)
    n_expected = sum(len(s["val"]) for s in splits)
    done_folds = [int(f.fold.iloc[0]) for f in frames]
    missing = [k for k in range(len(splits)) if k not in done_folds]

    md: list[str] = []
    add = md.append
    add(f"# {args.exp} cross-validation summary — nnU-Net v2 {cfg['models']['nnunet']['configuration']}, MSSEG-2016\n")
    add(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} from commit `{commit}` by `scripts/e1_summary.py`.\n")
    if not frames:
        add(f"> **No fold evaluated yet.** Expected `{args.results_dir / args.exp}/fold<k>_cases.csv`.\n")
    elif missing:
        add(f"> **PARTIAL RESULT: {sum(len(f) for f in frames)} of {n_expected} patients** (folds "
            f"{', '.join(map(str, done_folds))} evaluated; folds {', '.join(map(str, missing))} pending). "
            "Every number below will change when the remaining folds finish.\n")
    else:
        add(f"> **Complete: all {len(splits)} folds evaluated, {n_expected} of {n_expected} patients**, "
            "each scored once by the model that did not train on it.\n")
    for w in warnings:
        add(f"> ⚠ {w}\n")
    add(f"Trainer `{nnunet_trainer(cfg)}`, {epochs} epochs in every fold; patient-level folds stratified by "
        f"scanner (one patient per scanner per fold), seed {cfg['seed']}. Lesion rule: "
        f"{rule['connectivity']}-connected components ≥ {rule['min_lesion_mm3']} mm³, α = {rule['alpha']}, "
        f"γ = {rule['gamma']} (pending check against animaSegPerfAnalyzer). Size bins (from MSSEG training counts): "
        f"{', '.join(names)} mm³. Reference: MSSEG consensus. No test case is used here.\n")

    add("## Folds\n")
    add(md_table(status, index=False, floatfmt="{:.2f}"))
    add(f"\nWall time = nnU-Net run incl. final validation (from `logs/{args.exp}/fold<k>.log`); train time = sum of "
        "per-epoch times. nnU-Net val Dice = nnU-Net's own validation summary (foreground mean), for cross-checking "
        f"our Dice. Background job {'**running**' if alive else 'not running'}.\n")

    if not frames:
        (out / "summary.md").write_text("\n".join(md) + "\n")
        status.to_csv(out / "folds.csv", index=False)
        print(f"wrote {out / 'summary.md'} (no evaluated folds yet)")
        return 0

    cases = pd.concat(frames, ignore_index=True).sort_values(["fold", "case_id"]).reset_index(drop=True)
    cases = mask_empty_bins(cases, names)
    if cases.case_id.duplicated().any():
        raise SystemExit(f"patient(s) appear in more than one fold: {sorted(cases.case_id[cases.case_id.duplicated()])}")
    cases["count_error"] = cases.n_pred_lesions - cases.n_ref_lesions
    cases["volume_error_ml"] = cases.pred_volume_ml - cases.ref_volume_ml

    # ---- patient-level metrics
    metric_cols = MAIN_METRICS + [f"recall_{b}" for b in names] + [f"f1_{b}" for b in names]
    st = patient_stats(cases, metric_cols, ci_kw)
    add("## Cross-validation metrics (one value per patient)\n")
    add(f"Mean ± SD, median and percentile-bootstrap 95% CI of the mean over patients "
        f"({ci_kw['n_boot']:,} resamples). `recall_<bin>` only counts patients with at least one reference "
        "lesion in that bin; `f1_<bin>` also counts patients who have only false-positive lesions in that bin "
        "(F1 = 0) and leaves out patients with no lesion of that size in either mask, so n can differ "
        "(`msbench.reporting.mask_empty_bins`).\n")
    view = pd.DataFrame({
        "metric": st.metric, "n": st.n,
        "mean ± SD": [fmt_ci(r) for _, r in st.iterrows()],
        "median": st["median"],
        "95% CI": [fmt_interval(lo, hi) for lo, hi in zip(st.ci_low, st.ci_high)],
    })
    add(md_table(view, index=False))
    st.to_csv(out / "metrics.csv", index=False)

    add("\nPer-fold means (fold-to-fold spread; each fold has 3 patients, one per scanner):\n")
    pf = cases.groupby("fold")[MAIN_METRICS].mean().T
    pf.columns = [f"fold {c}" for c in pf.columns]
    add(md_table(pf.rename_axis("metric")))

    # ---- recall by size bin
    rec = pooled_bin_recall(cases, names, ci_kw)
    rec.to_csv(out / "recall_by_size_bin.csv", index=False)
    add("\n## Lesion recall by size bin (pooled lesions)\n")
    add("Detected / reference lesions summed over patients, with a cluster bootstrap 95% CI that resamples "
        "whole patients. The per-patient mean recall per bin is in the table above (`recall_<bin>`).\n")
    allr = rec[rec.scanner == "all"].drop(columns="scanner")
    allr["95% CI"] = [fmt_interval(lo, hi) for lo, hi in zip(allr.ci_low, allr.ci_high)]
    add(md_table(allr.drop(columns=["ci_low", "ci_high"]), index=False))
    add("\nPer scanner (detected / reference, recall):\n")
    sc = rec[rec.scanner != "all"].copy()
    sc["cell"] = [f"{d} / {n} ({'–' if np.isnan(r) else f'{r:.2f}'})"
                  for d, n, r in zip(sc.detected, sc.reference_lesions, sc.recall)]
    piv = sc.pivot(index="scanner", columns="bin_mm3", values="cell")[names]
    piv.index = [SCANNER_LABELS.get(s, s) for s in piv.index]
    add(md_table(piv.rename_axis("scanner")))

    # ---- agreement
    add("\n## Lesion count and volume agreement\n")
    arows = []
    for label, a, m, unit in [("lesion count", "n_pred_lesions", "n_ref_lesions", "lesions"),
                              ("lesion volume", "pred_volume_ml", "ref_volume_ml", "mL")]:
        ba = bland_altman(cases[a], cases[m])
        arows.append({"quantity": label, "unit": unit, "n": ba["n"], "bias (pred − ref)": ba["bias"],
                      "MAE": ba["mae"], "LoA low": ba["loa_low"], "LoA high": ba["loa_high"]})
    add(md_table(pd.DataFrame(arows), index=False))
    add("\nBias and 95% limits of agreement (Bland–Altman, bias ± 1.96 SD of the differences).\n")

    # ---- figures
    fig_recall_by_bin(rec, names, figs / "recall_by_size_bin.png")
    fig_patient_metrics(cases, st, figs / "patient_metrics.png")
    fig_agreement_scatter(cases, figs / "count_volume_scatter.png")
    add("## Figures\n")
    for f in sorted(figs.glob("*.png")):
        add(f"![{f.stem}](figures/{f.name})\n")

    # ---- per patient
    add("## Per-patient results\n")
    pc_cols = ["fold", "case_id", "scanner", "dice", "lesion_precision", "lesion_recall", "lesion_f1",
               "fp_lesions", "n_ref_lesions", "n_pred_lesions", "ref_volume_ml", "pred_volume_ml"] + \
              [f"recall_{b}" for b in names]
    add(md_table(cases[pc_cols], index=False))
    cases.to_csv(out / "cases.csv", index=False)
    status.to_csv(out / "folds.csv", index=False)

    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'summary.md'}  ({len(cases)}/{n_expected} patients, folds {done_folds})")
    for w in warnings:
        print(f"warning: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
