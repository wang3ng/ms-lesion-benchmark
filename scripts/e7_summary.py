"""E7 report (EXPLORATORY, added after the results freeze): MS-SAMLess and HarDNet baseline vs E1.

    source scripts/env.sh
    python scripts/e7_summary.py            # -> results/E7/summary.md (+ tables/, figures/)

Cross-validation: the 15 MSSEG training patients, each scored by the fold model that did not train on them
(E7 per-case tables from scripts/evaluate.py; E1 from results/E1). Test sets: the frozen MSSEG in-distribution,
MSSEG GE and ISBI (per rater) sets, scored exactly as scripts/score_test.py (same rule, bins, metrics; ISBI
resampled and tested by patient; msseg_test_c07_p08 reported separately). E1's frozen test tables are reused;
E7 test predictions are scored here (results/E7/tables/test_cases_*). Paired comparisons: E7 − E1 and
E7 − HarDNet baseline, Holm within each comparison and set. Nothing frozen is modified.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from msbench.config import REPO_ROOT, load_config, nnunet_fold_dir, paths
from msbench.evaluation import evaluate_predictions, lesion_tables
from msbench.metrics import bin_names
from msbench.reporting import INK, INK2, MAIN_METRICS, SURFACE, fmt_interval, md_table, new_fig, save_fig
from msbench.stats import bland_altman, bootstrap_ratio_ci, paired_bootstrap_ratio_diff_ci


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ST = _load("score_test")
RESULTS = Path(os.environ.get("MSBENCH_RESULTS", REPO_ROOT / "results"))
MODELS = ["E1", "E7", "E7_hardnet"]
LABELS = {"E1": "E1 nnU-Net", "E7": "E7 MS-SAMLess", "E7_hardnet": "E7 HarDNet Segmentor"}
COLORS = {"E1": "#2a78d6", "E7": "#008300", "E7_hardnet": "#6250d6"}   # palette slots 1, 6, 7
COMPARISONS = [("E7", "E1"), ("E7", "E7_hardnet")]
GROUPS = [("cv", "Cross-validation (15 MSSEG training patients)")] + [(k, s["label"]) for k, s in ST.SETS.items()]
SHORT = {"cv": "Cross-validation", **ST.SHORT}


def cv_pred_dir(exp: str, k: int, cfg) -> Path:
    return nnunet_fold_dir(k, cfg) / "validation" if exp == "E1" else paths(cfg)["e7_runs"] / exp / f"fold_{k}" / "validation"


def load_cv(cfg, tables: Path) -> tuple[dict, dict]:
    splits = json.loads((REPO_ROOT / "experiments" / "E1" / "splits_final.json").read_text())
    data, les = {}, {}
    for e in MODELS:
        frames, pl = [], []
        for k, sp in enumerate(splits):
            d = pd.read_csv(RESULTS / e / f"fold{k}_cases.csv")
            d = d[d.reference == "lesion"]
            if set(d.case_id) != set(sp["val"]):
                raise SystemExit(f"{e} fold {k}: cases differ from the split")
            frames.append(d.assign(fold=k))
            f = tables.parent / "lesions" / f"cv_pred_lesions_{e}_fold{k}.csv"
            if not f.exists():
                f.parent.mkdir(parents=True, exist_ok=True)
                lesion_tables(cv_pred_dir(e, k, cfg), sorted(sp["val"]), cfg=cfg)[1].to_csv(f, index=False)
            pl.append(pd.read_csv(f))
        data[e], les[e] = pd.concat(frames, ignore_index=True), pd.concat(pl, ignore_index=True)
    return data, les


def load_test(key: str, cfg, tables: Path, index) -> tuple[dict, dict]:
    s = ST.SETS[key]
    ids = list(ST.set_cases(s, index).case_id)
    data, les = {}, {}
    for e in MODELS:
        if e == "E1":   # frozen tables from scripts/score_test.py
            data[e] = pd.read_csv(RESULTS / "test" / "tables" / f"cases_E1_{key}.csv")
            les[e] = pd.read_csv(RESULTS / "test" / "lesions" / f"pred_lesions_E1_{key}.csv")
            continue
        fc, fl = tables / f"test_cases_{e}_{key}.csv", tables.parent / "lesions" / f"test_pred_lesions_{e}_{key}.csv"
        if not (fc.exists() and fl.exists()):
            fl.parent.mkdir(parents=True, exist_ok=True)
            evaluate_predictions(ST.pred_dir(e, s), ids, e, reference=s["ref"], cfg=cfg).to_csv(fc, index=False)
            lesion_tables(ST.pred_dir(e, s), ids, reference=s["ref"], cfg=cfg)[1].to_csv(fl, index=False)
        data[e], les[e] = pd.read_csv(fc), pd.read_csv(fl)
    return data, les


def fig_recall(rec: pd.DataFrame, names: list[str], out: Path) -> None:
    fig, axes = new_fig(4.0 * len(GROUPS), 4.2, ncols=len(GROUPS))
    for ax, (g, _) in zip(axes, GROUPS):
        x = np.arange(len(names))
        w = 0.8 / len(MODELS)
        r = rec[rec.group == g]
        for j, e in enumerate(MODELS):
            q = r[r.model == e].set_index("bin").loc[names]
            xs = x + (j - (len(MODELS) - 1) / 2) * w
            ax.bar(xs, q.recall.fillna(0), width=w, color=COLORS[e], edgecolor=SURFACE, linewidth=1.5, label=LABELS[e], zorder=2)
            err = np.vstack([q.recall - q.low, q.high - q.recall]).clip(min=0)
            ax.errorbar(xs, q.recall, yerr=np.nan_to_num(err), fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        n = r[r.model == "E1"].set_index("bin").loc[names].reference
        ax.set_xticks(x, [f"{b} mm³\nn = {int(v)}" for b, v in zip(names, n)], color=INK, fontsize=8)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(SHORT[g], color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("Lesion recall (pooled), 95% CI", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    save_fig(fig, out)


def fig_ba(data: dict, title: str, out: Path) -> None:
    import matplotlib.pyplot as plt
    from msbench.reporting import GRID, style_axes
    fig, axes = plt.subplots(2, len(MODELS), figsize=(3.8 * len(MODELS), 6.4), facecolor=SURFACE)
    for row, (pc, rc, nm) in enumerate([("n_pred_lesions", "n_ref_lesions", "lesion count"),
                                        ("pred_volume_ml", "ref_volume_ml", "volume (mL)")]):
        lim = max(np.abs(data[e][pc] - data[e][rc]).max() for e in MODELS) * 1.15 or 1
        for col, e in enumerate(MODELS):
            ax = axes[row, col]
            style_axes(ax)
            ba = bland_altman(data[e][pc], data[e][rc])
            ax.scatter(ba["means"], ba["diffs"], s=36, color=COLORS[e], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            for v, ls in ((ba["bias"], "-"), (ba["loa_low"], "--"), (ba["loa_high"], "--")):
                ax.axhline(v, color=INK2, linestyle=ls, linewidth=1, zorder=2)
            ax.axhline(0, color=GRID, linewidth=1.2, zorder=1)
            ax.set_ylim(-lim, lim)
            ax.set_title(f"{LABELS[e]}: {nm}", color=INK, fontsize=9, loc="left")
            ax.text(0.98, 0.04, f"bias {ba['bias']:.1f}\nLoA [{ba['loa_low']:.1f}, {ba['loa_high']:.1f}]",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK)
            if col == 0:
                ax.set_ylabel(f"predicted − reference {nm}", color=INK2, fontsize=8)
            if row == 1:
                ax.set_xlabel("mean of predicted and reference", color=INK2, fontsize=8)
    fig.suptitle(f"Bland–Altman, {title}", color=INK, fontsize=11, x=0.01, ha="left")
    save_fig(fig, out)


def main() -> int:
    cfg = load_config()
    names = bin_names([float(b) for b in cfg["metrics"]["size_bins_mm3"]])
    ci = cfg["statistics"]["confidence_interval"]
    ci_kw = {"n_boot": int(ci["n_boot"]), "level": float(ci["level"]), "seed": int(cfg["seed"])}
    index = pd.read_csv(paths(cfg)["data_processed"] / "index.csv", dtype=str)
    subj = index.set_index("case_id")["subject_id"]
    out = RESULTS / "E7"
    figs, tables = out / "figures", out / "tables"
    for d in (figs, tables):
        d.mkdir(parents=True, exist_ok=True)
    e7 = cfg["e7"]
    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    md: list[str] = []
    add = md.append
    add("# E7 — MS-SAMLess and HarDNet Segmentor vs E1 nnU-Net (exploratory, added after the results freeze)\n")
    add(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} from commit `{commit}` (branch `e7-ms-sam-less`) by "
        "`scripts/e7_summary.py`.\n")
    add("> **Exploratory, added after the results freeze.** Not part of the pre-specified comparison and not used to "
        "change anything in the frozen results (tags `results-frozen`, `-v2`, `-v3`). Same patients, folds, lesion "
        "rule, size bins, metrics and statistics as the frozen reports; E1's numbers are the frozen ones.\n")
    add("**What E7 is.** MS-SAMLess (Proietto Salanitri et al., *Pattern Recognit. Lett.* 2026; code "
        f"`perceivelab/MS-SAM-LESS` @ `{e7['upstream_commit'][:7]}`, used unmodified): a HarDNet-85 prompt learner "
        "trained with a frozen SAM ViT-B (phase 1), then a SAM-free ConvUpsample segmentor on that backbone "
        "(phase 2, the model scored here). **HarDNet Segmentor**: the same network trained from ImageNet weights "
        "without SAM, for the same total training (300 epochs), so E7 − HarDNet isolates SAM's guidance. 2D axial "
        "slices, FLAIR/T1/T2, E1's 5 folds, last-epoch checkpoints; test sets: 5-fold mean-probability ensembles.\n")
    add("**Deviations from the paper** (approved; details in `experiments/E7/README.md`, config `e7.deviations`): "
        "FLAIR/T1/T2 instead of FLAIR only; each training scan 8x per epoch (~2,400 phase-1 updates, roughly their "
        "MSLesSeg setup); last-epoch checkpoint instead of best validation IoU, no per-epoch validation; baseline "
        "300 epochs; 1 mm resampling also at inference, mapped back to each case's grid; 5-fold ensembles on the "
        "test sets; newer Python/PyTorch/MONAI.\n")
    add("**Statistics** as in `results/test/summary.md`: means over scans, 95% CI by patient bootstrap; ISBI's 21 "
        "scans come from 5 patients, so ISBI bootstraps resample patients and Wilcoxon tests use per-patient means "
        "(n = 5, smallest possible p = 0.0625). Paired differences A − B; Holm within each comparison and set over "
        "the per-scan metrics. `msseg_test_c07_p08` (empty consensus) is reported separately and left out.\n")

    overview, all_rec, all_cmp, all_stats = [], [], [], []
    for g, title in GROUPS:
        data, les = load_cv(cfg, tables) if g == "cv" else load_test(g, cfg, tables, index)
        for e in MODELS:
            d = data[e].copy()
            for b in names:
                d.loc[d[f"n_ref_{b}"] == 0, f"recall_{b}"] = np.nan
                d.loc[(d[f"n_ref_{b}"] == 0) & (d[f"n_pred_{b}"] == 0), [f"f1_{b}", f"precision_{b}"]] = np.nan
            data[e] = d.assign(empty_ref=((d.n_ref_lesions == 0) & (d.ref_voxels == 0)).to_numpy()).sort_values("case_id")
        add(f"## {title}\n")
        empty = sorted(data["E1"][data["E1"].empty_ref].case_id)
        if empty:
            rows = [{"scan": c, "model": LABELS[e], "false-positive lesions": int(data[e].set_index("case_id").loc[c, "fp_lesions"]),
                     "predicted volume (mL)": data[e].set_index("case_id").loc[c, "pred_volume_ml"]} for c in empty for e in MODELS]
            add(f"**Empty reference, reported separately and left out:** {', '.join(empty)}.\n")
            add(md_table(pd.DataFrame(rows), index=False))
            add("")
        data = {e: d[~d.empty_ref].reset_index(drop=True) for e, d in data.items()}
        ref_cols = [f"n_ref_{b}" for b in names] + ["n_ref_lesions"]
        for e in MODELS[1:]:
            if list(data[e].case_id) != list(data["E1"].case_id) or not np.array_equal(data[e][ref_cols], data["E1"][ref_cols]):
                raise SystemExit(f"{g} {e}: cases / reference lesions differ from E1")
        cl = np.array([subj[c] for c in data["E1"].case_id])
        add(f"{len(cl)} scans, {len(set(cl))} patients.\n")
        metric_cols = MAIN_METRICS + [f"recall_{b}" for b in names] + [f"f1_{b}" for b in names]
        st = {e: {m: ST.mean_ci(data[e][m].to_numpy(float), cl, ci_kw) for m in metric_cols} for e in MODELS}
        for e in MODELS:
            for m, r in st[e].items():
                all_stats.append({"group": g, "model": e, "metric": m, **r})

        def cell(r):
            t = f"{r['mean']:.3f} ± {r['sd']:.3f}<br>{fmt_interval(r['low'], r['high'])}"
            return t if r["n"] == len(cl) else f"{t} (n={r['n']})"
        add(md_table(pd.DataFrame({LABELS[e]: [cell(st[e][m]) for m in metric_cols] for e in MODELS},
                                  index=metric_cols).rename_axis("metric")))
        # recall / FP by bin
        rows, recs = [], []
        for e in MODELS:
            pl = les[e][les[e].case_id.isin(data[e].case_id)]
            for b in names:
                det, ref = data[e][f"detected_{b}"].to_numpy(float), data[e][f"n_ref_{b}"].to_numpy(float)
                r = bootstrap_ratio_ci(ST._agg(det, cl)[0], ST._agg(ref, cl)[0], **ci_kw)
                fpc = (~pl.true_positive & (pl.bin == b)).groupby(pl.case_id).sum().reindex(data[e].case_id).fillna(0)
                recs.append({"group": g, "model": e, "bin": b, "detected": int(det.sum()), "reference": int(ref.sum()),
                             "recall": det.sum() / ref.sum() if ref.sum() else np.nan, "low": r["low"], "high": r["high"],
                             "fp_per_scan": fpc.mean()})
        rec = pd.DataFrame(recs)
        all_rec.append(rec)
        for b in names:
            row = {"bin (mm³)": b}
            for e in MODELS:
                q = rec[(rec.model == e) & (rec.bin == b)].iloc[0]
                row[LABELS[e]] = f"R {q.detected}/{q.reference} = {q.recall:.3f} {fmt_interval(q.low, q.high)}<br>FP/scan {q.fp_per_scan:.1f}"
            rows.append(row)
        add("\nRecall (pooled) and false positives per scan by size bin:\n")
        add(md_table(pd.DataFrame(rows), index=False))
        # agreement
        rows = []
        for e in MODELS:
            for pc, rc, nm in (("n_pred_lesions", "n_ref_lesions", "lesion count"), ("pred_volume_ml", "ref_volume_ml", "volume (mL)")):
                ba = bland_altman(data[e][pc], data[e][rc])
                b_ci = ST.mean_ci((data[e][pc] - data[e][rc]).to_numpy(float), cl, ci_kw)
                rows.append({"model": LABELS[e], "quantity": nm, "bias (pred − ref)": f"{ba['bias']:.2f} {fmt_interval(b_ci['low'], b_ci['high'])}",
                             "95% limits of agreement": f"[{ba['loa_low']:.2f}, {ba['loa_high']:.2f}]", "MAE": f"{ba['mae']:.2f}"})
        add("\nLesion count and volume agreement (Bland–Altman; bias CI by patient bootstrap):\n")
        add(md_table(pd.DataFrame(rows), index=False))
        if g in ("cv", "msseg_id"):
            fig_ba(data, title, figs / f"bland_altman_{g}.png")
            add(f"\n![bland_altman_{g}](figures/bland_altman_{g}.png)\n")
        # paired
        for a, b in COMPARISONS:
            rows = []
            for m in metric_cols:
                r = ST.paired(data[a][m], data[b][m], cl, ci_kw)
                rows.append({"metric": m, "n scans": r["n_scans"], "n patients": r["n_patients"], f"mean {a}": r["mean_a"],
                             f"mean {b}": r["mean_b"], "mean diff": r["diff"], "95% CI": fmt_interval(r["low"], r["high"]),
                             "Wilcoxon p": r["p"] if r["n_scans"] else np.nan})
            t = pd.DataFrame(rows)
            t["p (Holm)"] = ST.holm(t["Wilcoxon p"])
            for bn in names:
                da, db, dr = (ST._agg(data[x][f"{c}_{bn}"].to_numpy(float), cl)[0] for x, c in ((a, "detected"), (b, "detected"), (a, "n_ref")))
                d = paired_bootstrap_ratio_diff_ci(da, db, dr, **ci_kw)
                ra = rec[(rec.model == a) & (rec.bin == bn)].iloc[0].recall
                rb = rec[(rec.model == b) & (rec.bin == bn)].iloc[0].recall
                t = pd.concat([t, pd.DataFrame([{"metric": f"pooled recall {bn}", "n scans": len(cl), "n patients": len(set(cl)),
                                                 f"mean {a}": ra, f"mean {b}": rb, "mean diff": d["estimate"],
                                                 "95% CI": fmt_interval(d["low"], d["high"]), "Wilcoxon p": np.nan, "p (Holm)": np.nan}])],
                              ignore_index=True)
            add(f"\n**{LABELS[a]} − {LABELS[b]}**\n")
            add(md_table(t, index=False))
            all_cmp.append(t.assign(group=g, comparison=f"{a} - {b}"))
        if g == "msseg_ge":
            tab = data["E1"][["case_id", "n_ref_lesions"]].copy()
            for e in MODELS:
                for m in ("dice", "lesion_f1", "fp_lesions"):
                    tab[f"{m} {e}"] = data[e][m].to_numpy()
            add("\nPer scan, GE Discovery 3T:\n")
            add(md_table(tab, index=False))
            tab.to_csv(tables / "ge_per_case.csv", index=False)
        add("")
        for m, nm in (("lesion_f1", "lesion F1"), ("dice", "Dice"), ("fp_lesions", "FP / scan")):
            overview.append({"set": SHORT[g], "metric": nm, **{e: st[e][m]["mean"] for e in MODELS}})
        overview.append({"set": SHORT[g], "metric": f"pooled recall {names[0]} mm³",
                         **{e: rec[(rec.model == e) & (rec.bin == names[0])].iloc[0].recall for e in MODELS}})

    rec_all = pd.concat(all_rec)
    rec_all.to_csv(tables / "recall_fp_by_size_bin.csv", index=False)
    pd.concat(all_cmp).to_csv(tables / "paired_comparisons.csv", index=False)
    pd.DataFrame(all_stats).to_csv(tables / "metrics.csv", index=False)
    fig_recall(rec_all, names, figs / "recall_by_size_bin.png")
    ov = pd.DataFrame(overview).rename(columns=LABELS)
    head = next(i for i, x in enumerate(md) if x.startswith("## "))
    md[head:head] = ["## Overview\n", "Means over scans (empty-reference scan excluded); CIs and tests below.\n",
                     md_table(ov, index=False), "", "![recall_by_size_bin](figures/recall_by_size_bin.png)\n"]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
