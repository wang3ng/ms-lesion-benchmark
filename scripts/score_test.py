"""Score the frozen test-stage predictions (experiments/test_stage/README.md) -> results/test/summary.md.

    source scripts/env.sh
    python scripts/score_test.py              # score (cached per model x set) + report
    python scripts/score_test.py --rescore    # recompute the per-case tables

Test sets: MSSEG 2016 in-distribution (Siemens/Philips, 30 scans), MSSEG GE Discovery 3T (unseen scanner,
8 scans, also per case), ISBI 2015 (21 scans of 5 patients) against each rater separately (E2a with the
boxes of the same rater). Same Anima-port lesion rule, size bins and metrics as the cross-validation.

Statistics. The resampling unit is the patient: an MSSEG scan is one patient; ISBI's 21 scans come from 5
patients, so ISBI bootstraps resample whole patients (all their scans) and paired Wilcoxon tests use one
value per patient (mean over that patient's scans; n = 5, smallest possible two-sided p = 0.0625). Means
are over scans in both cases. Holm adjustment within each comparison and test set over the per-scan
metrics. A scan with an empty reference (msseg_test_c07_p08: empty consensus) is reported separately by its
false-positive lesions and left out of all averages and tests. Experiments are frozen: nothing here feeds
back into any model or setting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msbench.config import REPO_ROOT, load_config, paths
from msbench.evaluation import evaluate_predictions, lesion_tables
from msbench.metrics import bin_names
from msbench.reporting import (
    GRID, INK, INK2, MAIN_METRICS, MODEL_COLORS, SURFACE, fmt_interval, md_table, new_fig, save_fig, style_axes,
)
from msbench.stats import bland_altman, bootstrap_ratio_ci, paired_bootstrap_ratio_diff_ci, paired_wilcoxon

RESULTS = Path(os.environ.get("MSBENCH_RESULTS", REPO_ROOT / "results"))
MODELS = ["E1", "E2a", "E2b", "E2c", "E8"]
LABELS = {"E1": "E1 nnU-Net", "E2a": "E2a MedSAM + GT boxes", "E2b": "E2b MedSAM + E1 boxes",
          "E2c": "E2c MedSAM + brain box/tiles", "E8": "E8 MedSAM + zoomed tiles"}
COMPARISONS = [("E2c", "E1"), ("E8", "E1"), ("E2b", "E1"), ("E8", "E2c")]
ID_SCANNERS = ["siemens_verio_3t", "siemens_aera_1p5t", "philips_ingenia_3t"]
SETS = {
    "msseg_id": {"label": "MSSEG test, in-distribution (Siemens/Philips)", "dataset": "msseg2016",
                 "scanners": ID_SCANNERS, "ref": "lesion"},
    "msseg_ge": {"label": "MSSEG test, GE Discovery 3T (unseen scanner)", "dataset": "msseg2016",
                 "scanners": ["ge_discovery_3t"], "ref": "lesion"},
    "isbi_r1": {"label": "ISBI 2015, rater 1", "dataset": "isbi2015", "scanners": None, "ref": "lesion_rater1"},
    "isbi_r2": {"label": "ISBI 2015, rater 2", "dataset": "isbi2015", "scanners": None, "ref": "lesion_rater2"},
}
SHORT = {"msseg_id": "MSSEG in-dist.", "msseg_ge": "MSSEG GE", "isbi_r1": "ISBI rater 1", "isbi_r2": "ISBI rater 2"}


def pred_dir(exp: str, s: dict) -> Path:
    d = RESULTS / exp / "test_predictions" / s["dataset"]
    return d / s["ref"] if (exp == "E2a" and s["dataset"] == "isbi2015") else d


def set_cases(s: dict, index: pd.DataFrame) -> pd.DataFrame:
    ix = index[index.dataset == s["dataset"]]
    if s["dataset"] == "msseg2016":
        ix = ix[(ix.release_split == "test") & ix.scanner.isin(s["scanners"])]
    return ix.sort_values("case_id")


def _score(job):
    exp, key, ids = job
    s = SETS[key]
    cfg = load_config()
    cases = evaluate_predictions(pred_dir(exp, s), ids, exp, reference=s["ref"], cfg=cfg)
    _, pl = lesion_tables(pred_dir(exp, s), ids, reference=s["ref"], cfg=cfg)
    return exp, key, cases, pl


# ---------------------------------------------------------------- patient-clustered statistics
def _agg(x: np.ndarray, cl: np.ndarray, keep: np.ndarray | None = None):
    """Per-patient sums and counts of the non-NaN values (keep: extra scan mask)."""
    ok = ~np.isnan(x) if keep is None else keep
    u = list(dict.fromkeys(cl))
    s = np.array([np.nansum(x[(cl == c) & ok]) for c in u])
    n = np.array([((cl == c) & ok).sum() for c in u], float)
    return s, n


def mean_ci(x, cl, ci_kw) -> dict:
    x = np.asarray(x, float)
    s, n = _agg(x, cl)
    r = bootstrap_ratio_ci(s, n, **ci_kw)
    v = x[~np.isnan(x)]
    return {"n": len(v), "mean": v.mean() if len(v) else np.nan, "sd": v.std(ddof=1) if len(v) > 1 else np.nan,
            "low": r["low"], "high": r["high"]}


def paired(a, b, cl, ci_kw) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    sa, n = _agg(a, cl, ok)
    sb, _ = _agg(b, cl, ok)
    d = paired_bootstrap_ratio_diff_ci(sa, sb, n, **ci_kw)
    have = n > 0
    w = paired_wilcoxon(sa[have] / n[have], sb[have] / n[have])
    return {"n_scans": int(ok.sum()), "n_patients": int(have.sum()), "mean_a": a[ok].mean(), "mean_b": b[ok].mean(),
            "diff": d["estimate"], "low": d["low"], "high": d["high"], "p": w["p"]}


def holm(p: pd.Series) -> pd.Series:
    p = p.astype(float)
    q = pd.Series(np.nan, index=p.index)
    ok = p.dropna().sort_values()
    m = len(ok)
    if m:
        q[ok.index] = np.minimum(1, np.maximum.accumulate([(m - i) * v for i, v in enumerate(ok.to_numpy())]))
    return q


# ---------------------------------------------------------------- figures
def fig_recall(rec: pd.DataFrame, names: list[str], out: Path) -> None:
    fig, axes = new_fig(18, 4.4, ncols=len(SETS))
    for ax, key in zip(axes, SETS):
        x = np.arange(len(names))
        w = 0.8 / len(MODELS)
        r = rec[rec.set == key]
        for j, e in enumerate(MODELS):
            q = r[r.model == e].set_index("bin").loc[names]
            xs = x + (j - (len(MODELS) - 1) / 2) * w
            ax.bar(xs, q.recall.fillna(0), width=w, color=MODEL_COLORS[e], edgecolor=SURFACE, linewidth=1.5,
                   label=LABELS[e], zorder=2)
            err = np.vstack([q.recall - q.low, q.high - q.recall]).clip(min=0)
            ax.errorbar(xs, q.recall, yerr=np.nan_to_num(err), fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        n = r[r.model == "E1"].set_index("bin").loc[names].reference
        ax.set_xticks(x, [f"{b} mm³\nn = {int(v)}" for b, v in zip(names, n)], color=INK, fontsize=8.5)
        ax.set_ylim(0, 1.12)
        ax.set_title(SHORT[key], color=INK, fontsize=10.5, loc="left")
    axes[0].set_ylabel("Lesion recall (pooled), 95% CI", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")
    save_fig(fig, out)


def fig_fp(fpb: pd.DataFrame, names: list[str], out: Path) -> None:
    fig, axes = new_fig(18, 4.0, ncols=len(SETS))
    for ax, key in zip(axes, SETS):
        x = np.arange(len(names))
        w = 0.8 / len(MODELS)
        r = fpb[fpb.set == key]
        for j, e in enumerate(MODELS):
            q = r[r.model == e].set_index("bin").loc[names]
            ax.bar(x + (j - (len(MODELS) - 1) / 2) * w, q.fp_per_scan, width=w, color=MODEL_COLORS[e],
                   edgecolor=SURFACE, linewidth=1.5, label=LABELS[e], zorder=2)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(x, [f"{b} mm³" for b in names], color=INK, fontsize=8.5)
        ax.set_title(SHORT[key], color=INK, fontsize=10.5, loc="left")
    axes[0].set_ylabel("False-positive lesions per scan (symlog)", color=INK2, fontsize=9)
    axes[-1].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper right")
    save_fig(fig, out)


def fig_ge(ge: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = new_fig(13, 4.2, ncols=2)
    ids = list(ge["E1"].case_id)
    x = np.arange(len(ids))
    for ax, (m, title) in zip(axes, [("dice", "Dice"), ("lesion_f1", "Lesion F1")]):
        for j, e in enumerate(MODELS):
            ax.scatter(x + (j - 2) * 0.1, ge[e].set_index("case_id").loc[ids, m], s=64, color=MODEL_COLORS[e],
                       edgecolor=SURFACE, linewidth=2, label=LABELS[e], zorder=3)
        ax.set_xticks(x, [c.replace("msseg_test_", "") for c in ids], color=INK, fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.set_title(f"GE Discovery 3T (unseen scanner): {title} per scan", color=INK, fontsize=10.5, loc="left")
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower left")
    save_fig(fig, out)


def fig_ba(data: dict[str, pd.DataFrame], key: str, out: Path) -> None:
    fig, axes = plt.subplots(2, len(MODELS), figsize=(3.6 * len(MODELS), 6.6), facecolor=SURFACE)
    for row, (pc, rc, name) in enumerate([("n_pred_lesions", "n_ref_lesions", "lesion count"),
                                          ("pred_volume_ml", "ref_volume_ml", "volume (mL)")]):
        lims = []
        for e in MODELS:
            c = data[e]
            lims.append(np.abs(c[pc] - c[rc]).max())
        lim = max(lims) * 1.15 or 1
        for col, e in enumerate(MODELS):
            ax = axes[row, col]
            style_axes(ax)
            c = data[e]
            ba = bland_altman(c[pc], c[rc])
            ax.scatter(ba["means"], ba["diffs"], s=36, color=MODEL_COLORS[e], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            for v, ls in ((ba["bias"], "-"), (ba["loa_low"], "--"), (ba["loa_high"], "--")):
                ax.axhline(v, color=INK2, linestyle=ls, linewidth=1, zorder=2)
            ax.axhline(0, color=GRID, linewidth=1.2, zorder=1)
            ax.set_ylim(-lim, lim)
            ax.set_title(f"{e}: {name}", color=INK, fontsize=9.5, loc="left")
            ax.text(0.98, 0.04, f"bias {ba['bias']:.1f}\nLoA [{ba['loa_low']:.1f}, {ba['loa_high']:.1f}]",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK)
            if col == 0:
                ax.set_ylabel(f"predicted − reference {name}", color=INK2, fontsize=8)
            if row == 1:
                ax.set_xlabel("mean of predicted and reference", color=INK2, fontsize=8)
    fig.suptitle(f"Bland–Altman, {SETS[key]['label']} (bias ± 1.96 SD)", color=INK, fontsize=11, x=0.01, ha="left")
    save_fig(fig, out)


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    names = bin_names([float(b) for b in cfg["metrics"]["size_bins_mm3"]])
    ci = cfg["statistics"]["confidence_interval"]
    ci_kw = {"n_boot": int(ci["n_boot"]), "level": float(ci["level"]), "seed": int(cfg["seed"])}
    rule = cfg["metrics"]["lesion_detection"]
    index = pd.read_csv(paths(cfg)["data_processed"] / "index.csv", dtype=str)
    out = RESULTS / "test"
    figs, tables, lesions = out / "figures", out / "tables", out / "lesions"
    for d in (figs, tables, lesions):
        d.mkdir(parents=True, exist_ok=True)

    # ---- score (cached)
    jobs = []
    for key, s in SETS.items():
        ids = list(set_cases(s, index).case_id)
        for e in MODELS:
            if args.rescore or not (tables / f"cases_{e}_{key}.csv").exists():
                jobs.append((e, key, ids))
    if jobs:
        with ProcessPoolExecutor(max_workers=min(len(jobs), 10)) as ex:
            for e, key, cases, pl in ex.map(_score, jobs):
                cases.to_csv(tables / f"cases_{e}_{key}.csv", index=False)
                pl.to_csv(lesions / f"pred_lesions_{e}_{key}.csv", index=False)
                print(f"scored {e} {key}: {len(cases)} scans", flush=True)

    subj = index.set_index("case_id")["subject_id"]
    md: list[str] = []
    add = md.append
    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    add("# Test results — E1 nnU-Net vs MedSAM (E2a, E2b, E2c, E8)\n")
    add(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} from commit `{commit}` by `scripts/score_test.py`. "
        "**Experiments frozen before scoring**: models, boxes, thresholds, size bins and the E8 choice were all "
        "fixed from the MSSEG training cross-validation (`results/E2/cv_summary.md`); nothing here was tuned.\n")
    add(f"Predictions: 5-fold ensembles (`experiments/test_stage/README.md`). Lesion rule: Anima-port "
        f"({rule['connectivity']}-connected, > {rule['min_lesion_mm3']} mm³, α {rule['alpha']} / β {rule['beta']} / "
        f"γ {rule['gamma']}). Size bins (mm³, from MSSEG training counts): {', '.join(names)}. **E2a places its "
        "boxes from the reference (MSSEG consensus; ISBI: the same rater it is scored against): an upper bound on "
        "tracing, not a fair comparison. E2b's boxes come from E1's predictions, so its recall is bounded by "
        "E1's.**\n")
    add("**Statistics.** Means are over scans; 95% CIs are percentile bootstraps (10,000 resamples) over "
        "**patients**. MSSEG: one scan per patient. **ISBI 2015: the 21 scans come from only 5 patients, so the "
        "bootstrap resamples whole patients with all their scans, and paired Wilcoxon tests use one value per "
        "patient (the mean over that patient's scans), n = 5. With 5 pairs the smallest possible two-sided "
        "Wilcoxon p is 0.0625, so no ISBI comparison can reach p < 0.05; read the CIs (which are also coarse with 5 "
        "patients).** Paired differences: A − B with a paired patient bootstrap CI; Holm adjustment within each "
        "comparison and test set over the per-scan metrics. Size-bin rows: recall leaves out scans without a "
        "reference lesion in the bin, F1 scans with no lesion of that size in either mask.\n")

    all_rec, all_fpb, all_cmp, all_stats, all_ba = [], [], [], [], []
    ge_cases = {}
    for key, s in SETS.items():
        data = {e: pd.read_csv(tables / f"cases_{e}_{key}.csv") for e in MODELS}
        for e in MODELS:
            d = data[e]
            no_lesion = (d.n_ref_lesions == 0) & (d.ref_voxels == 0)
            for b in names:  # same masking as the CV report
                d.loc[d[f"n_ref_{b}"] == 0, f"recall_{b}"] = np.nan
                d.loc[(d[f"n_ref_{b}"] == 0) & (d[f"n_pred_{b}"] == 0), [f"f1_{b}", f"precision_{b}"]] = np.nan
            data[e] = d.assign(empty_ref=no_lesion.to_numpy())
        empty_ids = sorted(data["E1"][data["E1"].empty_ref].case_id)
        add(f"## {s['label']}\n")
        n_all = len(data["E1"])
        if empty_ids:
            rows = []
            for c in empty_ids:
                for e in MODELS:
                    r = data[e].set_index("case_id").loc[c]
                    rows.append({"scan": c, "model": LABELS[e], "false-positive lesions": int(r.fp_lesions),
                                 "predicted volume (mL)": r.pred_volume_ml})
            add(f"**Empty reference, reported separately and left out of every average and test:** "
                f"{', '.join(empty_ids)} (consensus has no lesion).\n")
            add(md_table(pd.DataFrame(rows), index=False))
            pd.DataFrame(rows).to_csv(tables / f"empty_reference_{key}.csv", index=False)
            add("")
        data = {e: d[~d.empty_ref].sort_values("case_id").reset_index(drop=True) for e, d in data.items()}
        ref_cols = [f"n_ref_{b}" for b in names] + ["n_ref_lesions"]
        for e in MODELS[1:]:
            if not np.array_equal(data[e][ref_cols].to_numpy(), data["E1"][ref_cols].to_numpy()):
                raise SystemExit(f"{key} {e}: reference lesion counts differ from E1")
        cl = np.array([subj[c] for c in data["E1"].case_id])
        n_pat = len(set(cl))
        add(f"{len(data['E1'])} scans ({n_all - len(data['E1'])} left out), {n_pat} patients.\n")
        if key == "msseg_ge":
            ge_cases = data

        # pooled metrics
        metric_cols = MAIN_METRICS + [f"recall_{b}" for b in names] + [f"f1_{b}" for b in names]
        st = {e: {m: mean_ci(data[e][m], cl, ci_kw) for m in metric_cols} for e in MODELS}
        for e in MODELS:
            for m, r in st[e].items():
                all_stats.append({"set": key, "model": e, "metric": m, **r})

        def cell(r):
            t = f"{r['mean']:.3f} ± {r['sd']:.3f}<br>{fmt_interval(r['low'], r['high'])}"
            return t if r["n"] == len(data["E1"]) else f"{t} (n={r['n']})"
        add("### Per-scan metrics (mean ± SD, 95% CI)\n")
        add(md_table(pd.DataFrame({LABELS[e]: [cell(st[e][m]) for m in metric_cols] for e in MODELS},
                                  index=metric_cols).rename_axis("metric")))

        # recall and FP by size bin
        fpb_rows, rec_rows = [], []
        for e in MODELS:
            pl = pd.read_csv(lesions / f"pred_lesions_{e}_{key}.csv")
            pl = pl[pl.case_id.isin(data[e].case_id)]
            for b in names:
                det, ref = data[e][f"detected_{b}"].to_numpy(float), data[e][f"n_ref_{b}"].to_numpy(float)
                sd_, sr_ = _agg(det, cl)[0], _agg(ref, cl)[0]
                r = bootstrap_ratio_ci(sd_, sr_, **ci_kw)
                rec_rows.append({"set": key, "model": e, "bin": b, "detected": int(det.sum()), "reference": int(ref.sum()),
                                 "recall": det.sum() / ref.sum() if ref.sum() else np.nan, "low": r["low"], "high": r["high"]})
                fpc = (~pl.true_positive & (pl.bin == b)).groupby(pl.case_id).sum().reindex(data[e].case_id).fillna(0)
                m = mean_ci(fpc.to_numpy(float), cl, ci_kw)
                fpb_rows.append({"set": key, "model": e, "bin": b, "fp_per_scan": m["mean"], "low": m["low"], "high": m["high"]})
        rec, fpb = pd.DataFrame(rec_rows), pd.DataFrame(fpb_rows)
        all_rec.append(rec), all_fpb.append(fpb)
        add("\n### Recall and false positives by size bin\n")
        add("Recall: pooled detected / reference lesions (95% CI, patient bootstrap). FP / scan: false-positive "
            "lesions whose own volume is in the bin.\n")
        rows = []
        for b in names:
            row = {"bin (mm³)": b}
            for e in MODELS:
                r = rec[(rec.model == e) & (rec.bin == b)].iloc[0]
                f = fpb[(fpb.model == e) & (fpb.bin == b)].iloc[0]
                row[e] = (f"R {r.detected}/{r.reference} = {r.recall:.3f} {fmt_interval(r.low, r.high)}<br>"
                          f"FP/scan {f.fp_per_scan:.1f}")
            rows.append(row)
        add(md_table(pd.DataFrame(rows), index=False))

        # count / volume agreement
        rows = []
        for e in MODELS:
            for pc, rc, nm in (("n_pred_lesions", "n_ref_lesions", "lesion count"),
                               ("pred_volume_ml", "ref_volume_ml", "volume (mL)")):
                ba = bland_altman(data[e][pc], data[e][rc])
                bci = mean_ci((data[e][pc] - data[e][rc]).to_numpy(float), cl, ci_kw)
                rows.append({"model": LABELS[e], "quantity": nm, "bias (pred − ref)": f"{ba['bias']:.2f} "
                             f"{fmt_interval(bci['low'], bci['high'])}", "95% limits of agreement":
                             f"[{ba['loa_low']:.2f}, {ba['loa_high']:.2f}]", "MAE": f"{ba['mae']:.2f}"})
                all_ba.append({"set": key, "model": e, "quantity": nm, **{k: ba[k] for k in ("bias", "loa_low", "loa_high", "mae")},
                               "bias_ci_low": bci["low"], "bias_ci_high": bci["high"]})
        add("\n### Lesion count and volume agreement (Bland–Altman)\n")
        add("Bias with its 95% CI (patient bootstrap); limits of agreement = bias ± 1.96 SD over scans "
            "(descriptive" + ("; ISBI scans of one patient are not independent" if s["dataset"] == "isbi2015" else "")
            + ").\n")
        add(md_table(pd.DataFrame(rows), index=False))
        fig_ba(data, key, figs / f"bland_altman_{key}.png")
        add(f"\n![bland_altman_{key}](figures/bland_altman_{key}.png)\n")

        # paired comparisons
        add("### Paired comparisons\n")
        for a, b in COMPARISONS:
            rows = []
            for m in metric_cols:
                r = paired(data[a][m], data[b][m], cl, ci_kw)
                rows.append({"metric": m, "n scans": r["n_scans"], "n patients": r["n_patients"], f"mean {a}": r["mean_a"],
                             f"mean {b}": r["mean_b"], "mean diff": r["diff"], "95% CI": fmt_interval(r["low"], r["high"]),
                             "Wilcoxon p": r["p"]})
            t = pd.DataFrame(rows)
            t["p (Holm)"] = holm(t["Wilcoxon p"])
            for bn in names:
                det_a, _ = _agg(data[a][f"detected_{bn}"].to_numpy(float), cl)
                det_b, _ = _agg(data[b][f"detected_{bn}"].to_numpy(float), cl)
                ref_, _ = _agg(data[a][f"n_ref_{bn}"].to_numpy(float), cl)
                d = paired_bootstrap_ratio_diff_ci(det_a, det_b, ref_, **ci_kw)
                ra = rec[(rec.model == a) & (rec.bin == bn)].iloc[0].recall
                rb = rec[(rec.model == b) & (rec.bin == bn)].iloc[0].recall
                t = pd.concat([t, pd.DataFrame([{"metric": f"pooled recall {bn}", "n scans": len(cl), "n patients": n_pat,
                                                 f"mean {a}": ra, f"mean {b}": rb, "mean diff": d["estimate"],
                                                 "95% CI": fmt_interval(d["low"], d["high"]), "Wilcoxon p": np.nan,
                                                 "p (Holm)": np.nan}])], ignore_index=True)
            add(f"#### {LABELS[a]} − {LABELS[b]}\n")
            add(md_table(t, index=False))
            add("")
            all_cmp.append(t.assign(set=key, comparison=f"{a} - {b}"))

        if key == "msseg_ge":
            add("### Per-scan results, GE Discovery 3T\n")
            for m, title in (("dice", "Dice"), ("lesion_f1", "Lesion F1"), ("lesion_recall", "Lesion recall"),
                             ("fp_lesions", "False-positive lesions")):
                tab = data["E1"][["case_id", "n_ref_lesions", "ref_volume_ml"]].copy()
                for e in MODELS:
                    tab[e] = data[e][m].to_numpy()
                add(f"**{title}**\n")
                add(md_table(tab, index=False, floatfmt="{:.3f}" if m != "fp_lesions" else "{:.0f}"))
                add("")
                tab.to_csv(tables / f"ge_per_case_{m}.csv", index=False)
            fig_ge(data, figs / "ge_per_case.png")
            add("![ge_per_case](figures/ge_per_case.png)\n")

    rec_all, fpb_all = pd.concat(all_rec), pd.concat(all_fpb)
    rec_all.to_csv(tables / "recall_by_size_bin.csv", index=False)
    fpb_all.to_csv(tables / "fp_per_scan_by_size_bin.csv", index=False)
    pd.concat(all_cmp).to_csv(tables / "paired_comparisons.csv", index=False)
    pd.DataFrame(all_stats).to_csv(tables / "metrics.csv", index=False)
    pd.DataFrame(all_ba).to_csv(tables / "agreement.csv", index=False)
    fig_recall(rec_all, names, figs / "recall_by_size_bin.png")
    fig_fp(fpb_all, names, figs / "fp_per_scan_by_size_bin.png")
    # overview figures near the top
    head = md.index(next(x for x in md if x.startswith("## ")))
    stats_df = pd.DataFrame(all_stats)
    ov = []
    for key in SETS:
        for m, nm in (("lesion_f1", "lesion F1"), ("dice", "Dice"), ("fp_lesions", "FP / scan")):
            row = {"test set": SHORT[key], "metric": nm}
            for e in MODELS:
                r = stats_df[(stats_df.set == key) & (stats_df.model == e) & (stats_df.metric == m)].iloc[0]
                row[e] = f"{r['mean']:.3f}" if m != "fp_lesions" else f"{r['mean']:.1f}"
            ov.append(row)
        row = {"test set": SHORT[key], "metric": f"pooled recall {names[0]} mm³"}
        for e in MODELS:
            r = rec_all[(rec_all.set == key) & (rec_all.model == e) & (rec_all.bin == names[0])].iloc[0]
            row[e] = f"{r.recall:.3f}"
        ov.append(row)
    md[head:head] = ["## Overview\n", "Means over scans (empty-reference scan excluded); details and CIs below. "
                     "E2a is the upper bound with reference boxes.\n", md_table(pd.DataFrame(ov), index=False), "",
                     "![recall_by_size_bin](figures/recall_by_size_bin.png)\n",
                     "![fp_per_scan_by_size_bin](figures/fp_per_scan_by_size_bin.png)\n"]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
