"""E9 report (EXPLORATORY, added after the results freeze): data scaling with MSSEG + MSLesSeg training.

    source scripts/env.sh
    python scripts/e9_summary.py            # -> results/E9/summary.md (+ tables/, figures/)

Each E9 model (one model per method, trained on the combined set) is scored on the frozen test sets exactly
as scripts/score_test.py (same rule, bins, metrics; ISBI resampled and tested by patient; msseg_test_c07_p08
separate) and paired with its frozen counterpart, a 5-fold ensemble of 12-patient models: E9 nnU-Net vs E1,
E9 E2c vs E2c, E9 HarDNet vs E7 HarDNet (branch e7-ms-sam-less). Holm within each comparison and set.
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

from msbench.config import REPO_ROOT, load_config, paths
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
PAIRS = [("E9_nnunet", "E1", "nnU-Net 3d_fullres"), ("E9_E2c", "E2c", "MedSAM E2c (brain box + tiles)"),
         ("E9_hardnet", "E7_hardnet", "HarDNet Segmentor (E7 baseline)")]
LABELS = {"E1": "E1 nnU-Net (frozen)", "E2c": "E2c MedSAM (frozen)", "E7_hardnet": "E7 HarDNet (E7 branch)",
          "E9_nnunet": "E9 nnU-Net", "E9_E2c": "E9 MedSAM E2c", "E9_hardnet": "E9 HarDNet"}
MODELS = [m for p in PAIRS for m in p[:2]]
TRAIN_COLORS = {"frozen": "#2a78d6", "e9": "#eb6834"}    # encodes the training set in the figure


def load(model: str, key: str, cfg, tables: Path, index) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = ST.SETS[key]
    if model in ("E1", "E2c"):
        return (pd.read_csv(RESULTS / "test" / "tables" / f"cases_{model}_{key}.csv"),
                pd.read_csv(RESULTS / "test" / "lesions" / f"pred_lesions_{model}_{key}.csv"))
    if model == "E7_hardnet":
        return (pd.read_csv(RESULTS / "E7" / "tables" / f"test_cases_E7_hardnet_{key}.csv"),
                pd.read_csv(RESULTS / "E7" / "lesions" / f"test_pred_lesions_E7_hardnet_{key}.csv"))
    fc, fl = tables / f"test_cases_{model}_{key}.csv", tables.parent / "lesions" / f"test_pred_lesions_{model}_{key}.csv"
    if not (fc.exists() and fl.exists()):
        ids = list(ST.set_cases(s, index).case_id)
        fl.parent.mkdir(parents=True, exist_ok=True)
        evaluate_predictions(ST.pred_dir(model, s), ids, model, reference=s["ref"], cfg=cfg).to_csv(fc, index=False)
        lesion_tables(ST.pred_dir(model, s), ids, reference=s["ref"], cfg=cfg)[1].to_csv(fl, index=False)
    return pd.read_csv(fc), pd.read_csv(fl)


def fig_recall(rec: pd.DataFrame, names: list[str], out: Path) -> None:
    import matplotlib.pyplot as plt
    from msbench.reporting import style_axes
    keys = list(ST.SETS)
    fig, axes = plt.subplots(len(PAIRS), len(keys), figsize=(4.2 * len(keys), 3.3 * len(PAIRS)), facecolor=SURFACE,
                             sharey=True)
    for i, (e9, fz, title) in enumerate(PAIRS):
        for j, key in enumerate(keys):
            ax = axes[i, j]
            style_axes(ax)
            x = np.arange(len(names))
            for o, (m, kind, lab) in enumerate(((fz, "frozen", "frozen: 15 patients, 5-fold ensemble"),
                                                (e9, "e9", "E9: 99 scans, single model"))):
                q = rec[(rec.set == key) & (rec.model == m)].set_index("bin").loc[names]
                xs = x + (o - 0.5) * 0.38
                ax.bar(xs, q.recall.fillna(0), width=0.38, color=TRAIN_COLORS[kind], edgecolor=SURFACE, linewidth=1.5,
                       label=lab, zorder=2)
                err = np.vstack([q.recall - q.low, q.high - q.recall]).clip(min=0)
                ax.errorbar(xs, q.recall, yerr=np.nan_to_num(err), fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
            ax.set_xticks(x, [f"{b}" for b in names], color=INK, fontsize=8)
            ax.set_ylim(0, 1.15)
            ax.set_title(f"{title}\n{ST.SHORT[key]}" if j == 0 else ST.SHORT[key], color=INK, fontsize=9, loc="left")
            if j == 0:
                ax.set_ylabel("Lesion recall (pooled), 95% CI", color=INK2, fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    fig.suptitle("Lesion recall by volume (mm³): frozen 15-patient models vs E9 combined-set models", color=INK,
                 fontsize=11, x=0.01, ha="left")
    save_fig(fig, out)


def main() -> int:
    cfg = load_config()
    names = bin_names([float(b) for b in cfg["metrics"]["size_bins_mm3"]])
    ci = cfg["statistics"]["confidence_interval"]
    ci_kw = {"n_boot": int(ci["n_boot"]), "level": float(ci["level"]), "seed": int(cfg["seed"])}
    index = pd.read_csv(paths(cfg)["data_processed"] / "index.csv", dtype=str)
    subj = index.set_index("case_id")["subject_id"]
    out = RESULTS / "E9"
    figs, tables = out / "figures", out / "tables"
    for d in (figs, tables):
        d.mkdir(parents=True, exist_ok=True)
    ts = json.loads((out / "training_set.json").read_text())
    runs = {}
    for name, f in (("E9_E2c", paths(cfg)["e9_runs"] / "E9_E2c" / "run.json"),):
        if f.exists():
            runs[name] = json.loads(f.read_text())
    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    md: list[str] = []
    add = md.append
    add("# E9 — data scaling: MSSEG + MSLesSeg training (exploratory, added after the results freeze; "
        "MSLesSeg labels follow a different annotation protocol)\n")
    add(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} from commit `{commit}` (branch `e9-data-scaling`) by "
        "`scripts/e9_summary.py`.\n")
    add("> **Exploratory, added after the results freeze; MSLesSeg labels follow a different annotation protocol.** "
        "Not part of the pre-specified comparison and not used to change any frozen result. MSLesSeg lesions were "
        "annotated under MSLesSeg's own protocol, not the MSSEG consensus protocol the MSSEG test sets are scored "
        "against, so part of any change can come from the label definition rather than the amount of data.\n")
    tr, ex = ts["training"], ts["excluded_mslesseg"]
    add(f"**Training set** (one model per method, no cross-validation): {tr['scans']} scans of {tr['patients']} "
        f"patients — MSSEG-2016 training {tr['by_dataset']['msseg2016']['scans']} scans / "
        f"{tr['by_dataset']['msseg2016']['patients']} patients and MSLesSeg training {tr['by_dataset']['mslesseg']['scans']} "
        f"scans / {tr['by_dataset']['mslesseg']['patients']} patients. Scanners: "
        + ", ".join(f"{k} {v}" for k, v in tr["by_scanner"].items())
        + f". Excluded MSLesSeg scans (GE or unknown scanner, per scan, so the GE test stays unseen): {ex['scans']} "
        f"scans of {ex['patients']} patients (" + ", ".join(f"{k} {v}" for k, v in ex["by_scanner"].items()) + ").\n")
    add("**Budgets — each method follows its own standard recipe.** MedSAM E2c and the HarDNet Segmentor keep the "
        "same updates per training scan as their 12-scan CV folds (E2c: 10,000 × 99/12 = 82,500 steps; HarDNet: 300 "
        "epochs with each scan 8× per epoch = 29,700 iterations). nnU-Net uses its standard 1000 epochs × 250 "
        "iterations, so it gets fewer updates per scan than E1. nnU-Net: new dataset 502, automatic planning on the "
        "combined data, default trainer, fold `all`. Same loss and collapse check for E2c; same recipe and "
        "deviations as E7 for HarDNet.\n")
    add("**Comparison.** Each E9 model (single model) is paired with its frozen counterpart — **a 5-fold ensemble of "
        "12-patient models** — on the same test scans: E9 nnU-Net vs E1, E9 E2c vs E2c, E9 HarDNet vs E7 HarDNet. "
        "The ensemble/single-model difference works against E9. Statistics as `results/test/summary.md`: means over "
        "scans, 95% CIs by patient bootstrap, paired Wilcoxon, Holm within each comparison and test set. **ISBI: 21 "
        "scans of 5 patients — resampled and tested by patient (smallest possible p = 0.0625). ISBI caveat: "
        "MSLesSeg, like ISBI, is Philips data registered to MNI space at 1 mm, so ISBI is a less external test for "
        "the E9 models than for the frozen ones.** `msseg_test_c07_p08` (empty consensus) is reported separately.\n")

    overview, recs, cmps = [], [], []
    for key, s in ST.SETS.items():
        data, les = {}, {}
        for m in MODELS:
            d, l_ = load(m, key, cfg, tables, index)
            for b in names:
                d.loc[d[f"n_ref_{b}"] == 0, f"recall_{b}"] = np.nan
                d.loc[(d[f"n_ref_{b}"] == 0) & (d[f"n_pred_{b}"] == 0), [f"f1_{b}", f"precision_{b}"]] = np.nan
            data[m] = d.assign(empty=((d.n_ref_lesions == 0) & (d.ref_voxels == 0)).to_numpy()).sort_values("case_id")
            les[m] = l_
        add(f"## {s['label']}\n")
        empty = sorted(data["E1"][data["E1"].empty].case_id)
        if empty:
            rows = [{"scan": c, "model": LABELS[m], "false-positive lesions": int(data[m].set_index("case_id").loc[c, "fp_lesions"]),
                     "predicted volume (mL)": data[m].set_index("case_id").loc[c, "pred_volume_ml"]} for c in empty for m in MODELS]
            add(f"**Empty reference, reported separately and left out:** {', '.join(empty)}.\n")
            add(md_table(pd.DataFrame(rows), index=False))
            add("")
        data = {m: d[~d.empty].reset_index(drop=True) for m, d in data.items()}
        refc = [f"n_ref_{b}" for b in names] + ["n_ref_lesions"]
        for m in MODELS[1:]:
            if list(data[m].case_id) != list(data["E1"].case_id) or not np.array_equal(data[m][refc], data["E1"][refc]):
                raise SystemExit(f"{key} {m}: cases / reference lesions differ")
        cl = np.array([subj[c] for c in data["E1"].case_id])
        add(f"{len(cl)} scans, {len(set(cl))} patients.\n")
        metric_cols = MAIN_METRICS + [f"recall_{b}" for b in names] + [f"f1_{b}" for b in names]
        st = {m: {k: ST.mean_ci(data[m][k].to_numpy(float), cl, ci_kw) for k in metric_cols} for m in MODELS}

        def cell(r):
            t = f"{r['mean']:.3f} ± {r['sd']:.3f}<br>{fmt_interval(r['low'], r['high'])}"
            return t if r["n"] == len(cl) else f"{t} (n={r['n']})"
        add(md_table(pd.DataFrame({LABELS[m]: [cell(st[m][k]) for k in metric_cols] for m in MODELS},
                                  index=metric_cols).rename_axis("metric")))
        rows = []
        for m in MODELS:
            pl = les[m][les[m].case_id.isin(data[m].case_id)]
            for b in names:
                det, ref = data[m][f"detected_{b}"].to_numpy(float), data[m][f"n_ref_{b}"].to_numpy(float)
                r = bootstrap_ratio_ci(ST._agg(det, cl)[0], ST._agg(ref, cl)[0], **ci_kw)
                fpc = (~pl.true_positive & (pl.bin == b)).groupby(pl.case_id).sum().reindex(data[m].case_id).fillna(0)
                recs.append({"set": key, "model": m, "bin": b, "detected": int(det.sum()), "reference": int(ref.sum()),
                             "recall": det.sum() / ref.sum() if ref.sum() else np.nan, "low": r["low"], "high": r["high"],
                             "fp_per_scan": fpc.mean()})
        rec = pd.DataFrame([r for r in recs if r["set"] == key])
        for b in names:
            row = {"bin (mm³)": b}
            for m in MODELS:
                q = rec[(rec.model == m) & (rec.bin == b)].iloc[0]
                row[LABELS[m]] = f"R {q.recall:.3f} {fmt_interval(q.low, q.high)}<br>FP/scan {q.fp_per_scan:.1f}"
            rows.append(row)
        add("\nRecall (pooled) and false positives per scan by size bin:\n")
        add(md_table(pd.DataFrame(rows), index=False))
        rows = []
        for m in MODELS:
            for pc, rc, nm in (("n_pred_lesions", "n_ref_lesions", "lesion count"), ("pred_volume_ml", "ref_volume_ml", "volume (mL)")):
                ba = bland_altman(data[m][pc], data[m][rc])
                b_ci = ST.mean_ci((data[m][pc] - data[m][rc]).to_numpy(float), cl, ci_kw)
                rows.append({"model": LABELS[m], "quantity": nm, "bias (pred − ref)": f"{ba['bias']:.2f} {fmt_interval(b_ci['low'], b_ci['high'])}",
                             "95% limits of agreement": f"[{ba['loa_low']:.2f}, {ba['loa_high']:.2f}]", "MAE": f"{ba['mae']:.2f}"})
        add("\nLesion count and volume agreement (Bland–Altman; bias CI by patient bootstrap):\n")
        add(md_table(pd.DataFrame(rows), index=False))
        for a, b, title in PAIRS:
            rows = []
            for k in metric_cols:
                r = ST.paired(data[a][k], data[b][k], cl, ci_kw)
                rows.append({"metric": k, "n scans": r["n_scans"], "n patients": r["n_patients"], f"mean {a}": r["mean_a"],
                             f"mean {b}": r["mean_b"], "mean diff": r["diff"], "95% CI": fmt_interval(r["low"], r["high"]),
                             "Wilcoxon p": r["p"] if r["n_scans"] else np.nan})
            t = pd.DataFrame(rows)
            t["p (Holm)"] = ST.holm(t["Wilcoxon p"])
            for bn in names:
                da, db, dr = (ST._agg(data[x][f"{c}_{bn}"].to_numpy(float), cl)[0] for x, c in ((a, "detected"), (b, "detected"), (a, "n_ref")))
                d = paired_bootstrap_ratio_diff_ci(da, db, dr, **ci_kw)
                t = pd.concat([t, pd.DataFrame([{"metric": f"pooled recall {bn}", "n scans": len(cl), "n patients": len(set(cl)),
                                                 f"mean {a}": rec[(rec.model == a) & (rec.bin == bn)].iloc[0].recall,
                                                 f"mean {b}": rec[(rec.model == b) & (rec.bin == bn)].iloc[0].recall,
                                                 "mean diff": d["estimate"], "95% CI": fmt_interval(d["low"], d["high"]),
                                                 "Wilcoxon p": np.nan, "p (Holm)": np.nan}])], ignore_index=True)
            add(f"\n**{title}: {LABELS[a]} − {LABELS[b]}** (single model vs 5-fold ensemble)\n")
            add(md_table(t, index=False))
            cmps.append(t.assign(set=key, comparison=f"{a} - {b}"))
            for k, nm in (("lesion_f1", "lesion F1"), ("dice", "Dice"), ("fp_lesions", "FP / scan")):
                overview.append({"test set": ST.SHORT[key], "method": title, "metric": nm, "frozen (15 pts, ensemble)": st[b][k]["mean"],
                                 "E9 (99 scans, single)": st[a][k]["mean"], "diff": st[a][k]["mean"] - st[b][k]["mean"],
                                 "p (Holm)": t.set_index("metric").loc[k, "p (Holm)"]})
            overview.append({"test set": ST.SHORT[key], "method": title, "metric": f"pooled recall {names[0]} mm³",
                             "frozen (15 pts, ensemble)": rec[(rec.model == b) & (rec.bin == names[0])].iloc[0].recall,
                             "E9 (99 scans, single)": rec[(rec.model == a) & (rec.bin == names[0])].iloc[0].recall,
                             "diff": t.set_index("metric").loc[f"pooled recall {names[0]}", "mean diff"], "p (Holm)": np.nan})
        if key == "msseg_ge":
            tab = data["E1"][["case_id", "n_ref_lesions"]].copy()
            for m in MODELS:
                tab[f"F1 {m}"] = data[m]["lesion_f1"].to_numpy()
            add("\nPer scan, GE Discovery 3T (lesion F1):\n")
            add(md_table(tab, index=False))
            tab.to_csv(tables / "ge_per_case_lesion_f1.csv", index=False)
        add("")
    rec_all = pd.DataFrame(recs)
    rec_all.to_csv(tables / "recall_fp_by_size_bin.csv", index=False)
    pd.concat(cmps).to_csv(tables / "paired_comparisons.csv", index=False)
    ov = pd.DataFrame(overview)
    ov.to_csv(tables / "overview.csv", index=False)
    fig_recall(rec_all, names, figs / "recall_by_size_bin.png")
    head = next(i for i, x in enumerate(md) if x.startswith("## "))
    md[head:head] = ["## Overview\n", "Means over scans (empty-reference scan excluded); diff = E9 − frozen; p (Holm) "
                     "from the paired comparisons below.\n", md_table(ov, index=False), "",
                     "![recall_by_size_bin](figures/recall_by_size_bin.png)\n"]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
