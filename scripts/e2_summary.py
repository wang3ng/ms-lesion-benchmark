"""Compare E1 (nnU-Net), E2a (MedSAM + oracle boxes) and E2c (MedSAM + brain box/tiles) on the CV folds.

    source scripts/env.sh
    python scripts/e2_summary.py        # -> results/E2/cv_summary.md (+ figures/, tables/)

All three models are scored on the same MSSEG training patients, each by the fold
model that did not train on them (E1's 5 patient-level folds), with the same
Anima-port lesion rule (results/<exp>/fold<k>_cases.csv from scripts/evaluate.py).
Works on partial results: only patients evaluated by every model are compared.
Also applies the E8 decision rule (configs/experiment.yaml experiments.E8) to the
pooled <20 mm^3 recalls. Reads training cases only; no test case.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from msbench.config import REPO_ROOT, load_config, nnunet_fold_dir, paths
from msbench.evaluation import lesion_tables
from msbench.metrics import bin_names
from msbench.reporting import (
    INK, INK2, MAIN_METRICS, SCANNER_COLORS, SCANNER_LABELS, SURFACE, fmt_interval, job_alive, mask_empty_bins,
    md_table, new_fig, patient_stats, pooled_bin_recall, save_fig,
)
from msbench.stats import paired_bootstrap_diff_ci, paired_bootstrap_ratio_diff_ci, paired_wilcoxon

MODELS = ["E1", "E2a", "E2c"]
MODEL_LABELS = {"E1": "E1 nnU-Net", "E2a": "E2a MedSAM + oracle boxes", "E2c": "E2c MedSAM + brain box/tiles"}
MODEL_COLORS = {"E1": "#2a78d6", "E2a": "#eb6834", "E2c": "#1baf7a"}  # fixed per model, palette order
COMPARISONS = [("E2c", "E1"), ("E2a", "E2c")]


def pred_dir(exp: str, fold: int, cfg: dict) -> Path:
    if exp == "E1":
        return nnunet_fold_dir(fold, cfg) / "validation"
    return paths(cfg)["medsam_runs"] / exp / f"fold_{fold}" / "validation"


def load_model(exp: str, splits: list[dict], results_dir: Path) -> tuple[pd.DataFrame | None, list[int]]:
    frames, folds = [], []
    for k, sp in enumerate(splits):
        f = results_dir / exp / f"fold{k}_cases.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[df.reference == "lesion"]
        if set(df.case_id) != set(sp["val"]):
            print(f"warning: {exp} fold {k}: cases differ from splits; left out", file=sys.stderr)
            continue
        frames.append(df.assign(fold=k))
        folds.append(k)
    return (pd.concat(frames, ignore_index=True) if frames else None), folds


def holm(p: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values."""
    p = p.astype(float)
    ok = p.notna()
    q = pd.Series(np.nan, index=p.index)
    if ok.sum() == 0:
        return q
    order = p[ok].sort_values()
    m = len(order)
    adj = np.minimum(1, np.maximum.accumulate([(m - i) * v for i, v in enumerate(order.to_numpy())]))
    q[order.index] = adj
    return q


def e8_decision(rules: list[dict], recall: dict[str, float]) -> tuple[str, list[str]]:
    """Evaluate the config rules in order; returns (choice, explanation lines)."""
    lines = []
    for r in rules:
        if "else" in r:
            lines.append(f"no earlier rule fired → else → **{r['else']}**")
            return r["else"], lines
        expr = r["if"]
        num = re.sub(r"recall\((E2[a-c])\)", lambda m: f"{recall[m.group(1)]:.4f}", expr)
        if not re.fullmatch(r"[0-9.\s\-+<>=]+", num):
            raise ValueError(f"unexpected E8 rule: {expr}")
        lhs, op, rhs = re.split(r"\s*(>=|<=|<|>)\s*", num)
        val = eval(lhs, {"__builtins__": {}})  # arithmetic on numbers only (checked above)
        ok = {">=": val >= float(rhs), "<=": val <= float(rhs), "<": val < float(rhs), ">": val > float(rhs)}[op]
        lines.append(f"`{expr}` → {lhs.strip()} = {val:.4f} {op} {rhs} → **{'true' if ok else 'false'}**")
        if ok:
            lines[-1] += f" → **{r['then']}**"
            return r["then"], lines
    raise ValueError("E8 rules have no else branch")


# ---------------------------------------------------------------- figures
def fig_recall(rec: dict[str, pd.DataFrame], names: list[str], out: Path) -> None:
    fig, ax = new_fig(7.2, 4.2)
    x = np.arange(len(names))
    w = 0.8 / len(rec)
    for j, (exp, r) in enumerate(rec.items()):
        r = r[r.scanner == "all"].set_index("bin_mm3").loc[names]
        xs = x + (j - (len(rec) - 1) / 2) * w
        ax.bar(xs, r.recall.fillna(0), width=w, color=MODEL_COLORS[exp], edgecolor=SURFACE, linewidth=2,
               label=MODEL_LABELS[exp], zorder=2)
        err = np.vstack([r.recall - r.ci_low, r.ci_high - r.recall]).clip(min=0)
        ax.errorbar(xs, r.recall, yerr=np.nan_to_num(err), fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=3)
    n = next(iter(rec.values()))
    n = n[n.scanner == "all"].set_index("bin_mm3").loc[names].reference_lesions
    ax.set_xticks(x, [f"{b} mm³\nn = {int(v)} lesions" for b, v in zip(names, n)], color=INK, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Lesion recall (detected / reference), 95% CI", color=INK2, fontsize=9)
    ax.set_title("Lesion recall by volume, cross-validation (same patients)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    save_fig(fig, out)


def fig_missed(missed: pd.DataFrame, names: list[str], out: Path) -> None:
    exps = list(dict.fromkeys(missed.model))
    fig, axes = new_fig(4.2 * len(exps), 4.0, ncols=len(exps))
    axes = np.atleast_1d(axes)
    top = max(missed.missed.max(), 1)
    scanners = [s for s in SCANNER_COLORS if s in set(missed.scanner)]
    for ax, exp in zip(axes, exps):
        x = np.arange(len(names))
        w = 0.8 / len(scanners)
        for j, s in enumerate(scanners):
            m = missed[(missed.model == exp) & (missed.scanner == s)].set_index("bin").reindex(names)
            xs = x + (j - (len(scanners) - 1) / 2) * w
            ax.bar(xs, m.missed, width=w, color=SCANNER_COLORS[s], edgecolor=SURFACE, linewidth=1.5,
                   label=SCANNER_LABELS.get(s, s), zorder=2)
            for xi, mi, ni in zip(xs, m.missed, m.reference):
                ax.text(xi, mi + top * 0.015, f"{int(mi)}/{int(ni)}", ha="center", va="bottom", fontsize=6.5,
                        color=INK2)
        ax.set_xticks(x, [f"{b} mm³" for b in names], color=INK, fontsize=8.5)
        ax.set_ylim(0, top * 1.18)
        tot = missed[(missed.model == exp)]
        ax.set_title(f"{MODEL_LABELS[exp]}: {int(tot.missed.sum())} missed", color=INK, fontsize=9.5, loc="left")
    axes[0].set_ylabel("Missed reference lesions (labels: missed / reference)", color=INK2, fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper right")
    save_fig(fig, out)


def fig_fp_hist(fp: pd.DataFrame, exps: list[str], small_edge: float, min_mm3: float, n_scans: int, out: Path) -> None:
    fig, axes = new_fig(4.2 * len(exps), 3.8, ncols=len(exps))
    axes = np.atleast_1d(axes)
    hi = max(fp.volume_mm3.max() * 1.2 if len(fp) else 0, small_edge * 10)
    edges = np.logspace(np.log10(min_mm3), np.log10(hi), 22)
    ticks = [t for t in (3, 10, 20, 50, 100, 500, 2000, 10000) if min_mm3 <= t <= hi]
    scanners = [s for s in SCANNER_COLORS if s in set(fp.scanner)]
    ymax = 1
    for exp in exps:
        f = fp[fp.model == exp]
        if len(f):
            ymax = max(ymax, np.histogram(f.volume_mm3, bins=edges)[0].max())
    for ax, exp in zip(axes, exps):
        f = fp[fp.model == exp]
        if len(f):
            ax.hist([f.volume_mm3[f.scanner == s] for s in scanners], bins=edges, stacked=True,
                    color=[SCANNER_COLORS[s] for s in scanners], edgecolor=SURFACE, linewidth=0.8,
                    label=[SCANNER_LABELS.get(s, s) for s in scanners], zorder=2)
        ax.set_xscale("log")
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(0, ymax * 1.25)
        ax.axvline(small_edge, color=INK, linestyle="--", linewidth=1, zorder=3)
        n_small = int((f.volume_mm3 < small_edge).sum())
        ax.text(0.98, 0.97, f"{len(f)} FP ({len(f) / n_scans:.1f} / scan)\n{n_small} ({100 * n_small / max(len(f), 1):.0f}%)"
                f" < {small_edge:g} mm³", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
        ax.set_title(MODEL_LABELS[exp], color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("False-positive lesion volume (mm³, log)", color=INK2, fontsize=8.5)
    axes[0].set_ylabel("False-positive lesions", color=INK2, fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="center right")
    save_fig(fig, out)


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path(os.environ.get("MSBENCH_RESULTS", REPO_ROOT / "results")))
    ap.add_argument("--out", type=Path, help="default: <results-dir>/E2")
    args = ap.parse_args(argv)

    cfg = load_config()
    splits = json.loads((REPO_ROOT / "experiments" / "E1" / "splits_final.json").read_text())
    names = bin_names([float(b) for b in cfg["metrics"]["size_bins_mm3"]])
    ci_cfg = cfg["statistics"]["confidence_interval"]
    ci_kw = {"n_boot": int(ci_cfg["n_boot"]), "level": float(ci_cfg["level"]), "seed": int(cfg["seed"])}
    rule = cfg["metrics"]["lesion_detection"]
    out = args.out or args.results_dir / "E2"
    figs, tables = out / "figures", out / "tables"
    if figs.exists():
        shutil.rmtree(figs)
    figs.mkdir(parents=True)
    tables.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    alive = job_alive(Path(os.environ.get("MSBENCH_LOGS", paths(cfg)["logs"])) / "E2", marker="run_E2")

    data, folds = {}, {}
    for exp in MODELS:
        df, fl = load_model(exp, splits, args.results_dir)
        if df is not None:
            data[exp], folds[exp] = mask_empty_bins(df, names), fl
    md: list[str] = []
    add = md.append
    add("# E2 cross-validation comparison — E1 nnU-Net vs MedSAM (E2a oracle boxes, E2c brain box + tiles)\n")
    add(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} from commit `{commit}` by `scripts/e2_summary.py`.\n")
    status = pd.DataFrame([{"model": MODEL_LABELS[e], "folds evaluated": ", ".join(map(str, folds.get(e, []))) or "–",
                            "patients": len(data[e]) if e in data else 0} for e in MODELS])
    exps = [e for e in MODELS if e in data]
    common = sorted(set.intersection(*[set(data[e].case_id) for e in exps])) if exps else []
    if len(exps) < len(MODELS) or len(common) < 15:
        add(f"> **PARTIAL RESULT: {len(common)} patients evaluated by all of {', '.join(exps) or '–'}** "
            f"(E2 job {'running' if alive else 'not running'}). Numbers will change.\n")
    else:
        add("> **Complete: all 15 MSSEG training patients, each scored by the fold model that did not train on "
            "them, for all three models.**\n")
    ft = cfg["models"]["medsam"]["finetune"]
    add(f"Same patient-level folds for all models (E1's `splits_final.json`). Lesion rule: Anima-port "
        f"({rule['connectivity']}-connected, > {rule['min_lesion_mm3']} mm³, α {rule['alpha']} / β {rule['beta']} / "
        f"γ {rule['gamma']}); reference = MSSEG consensus. MedSAM: frozen image encoder, prompt encoder + mask "
        f"decoder fine-tuned per condition and fold ({ft['steps']} steps × {ft['batch_size']}, same box type at "
        "training and validation). **E2a uses the reference to place its boxes and is an upper bound on tracing, "
        "not a fair comparison; E2c is the fair head-to-head with E1.** No test case is used.\n")
    add(md_table(status, index=False))
    add("")
    if len(exps) < 2 or not common:
        (out / "cv_summary.md").write_text("\n".join(md) + "\n")
        print(f"wrote {out / 'cv_summary.md'} (not enough results yet)")
        return 0
    cases = {e: data[e][data[e].case_id.isin(common)].sort_values("case_id").reset_index(drop=True) for e in exps}
    ref_n = cases[exps[0]][[f"n_ref_{b}" for b in names] + ["n_ref_lesions"]].to_numpy()
    for e in exps[1:]:
        if not np.array_equal(cases[e][[f"n_ref_{b}" for b in names] + ["n_ref_lesions"]].to_numpy(), ref_n):
            raise SystemExit(f"{e}: reference lesion counts differ from {exps[0]}; re-run evaluate.py")

    # ---- pooled metrics
    metric_cols = MAIN_METRICS + [f"recall_{b}" for b in names] + [f"f1_{b}" for b in names]
    stats = {e: patient_stats(cases[e], metric_cols, ci_kw).set_index("metric") for e in exps}
    pd.concat([s.assign(model=e) for e, s in stats.items()]).to_csv(tables / "metrics.csv")
    add(f"## Pooled metrics ({len(common)} patients, one value per patient)\n")
    add("Mean ± SD, with the percentile-bootstrap 95% CI of the mean below (10,000 patient resamples). "
        "`fp_lesions` = false-positive lesions per scan. Size-bin rows: recall leaves out patients without a "
        "reference lesion in the bin; F1 leaves out patients with no lesion of that size in either mask.\n")
    def cell(st: pd.DataFrame, m: str) -> str:
        r = st.loc[m]
        txt = f"{r['mean']:.3f} ± {r['sd']:.3f}<br>{fmt_interval(r['ci_low'], r['ci_high'])}"
        return txt if r["n"] == len(common) else f"{txt} (n={int(r['n'])})"

    tbl = {MODEL_LABELS[e]: [cell(stats[e], m) for m in metric_cols] for e in exps}
    add(md_table(pd.DataFrame(tbl, index=metric_cols).rename_axis("metric")))

    # ---- recall by size bin
    rec = {e: pooled_bin_recall(cases[e], names, ci_kw) for e in exps}
    pd.concat([r.assign(model=e) for e, r in rec.items()]).to_csv(tables / "recall_by_size_bin.csv", index=False)
    add("\n## Lesion-level recall by size bin (pooled lesions)\n")
    add("Detected / reference lesions summed over patients (95% CI: cluster bootstrap over patients).\n")
    rows = []
    for b in names:
        row = {"bin (mm³)": b}
        for e in exps:
            r = rec[e][(rec[e].scanner == "all") & (rec[e].bin_mm3 == b)].iloc[0]
            row[MODEL_LABELS[e]] = f"{r.detected}/{r.reference_lesions} = {r.recall:.3f}<br>{fmt_interval(r.ci_low, r.ci_high)}"
        rows.append(row)
    add(md_table(pd.DataFrame(rows), index=False))
    add("\nPer scanner (detected / reference, recall):\n")
    rows = []
    for s in [s for s in SCANNER_COLORS if s in set(cases[exps[0]].scanner)]:
        for b in names:
            row = {"scanner": SCANNER_LABELS.get(s, s), "bin (mm³)": b}
            for e in exps:
                r = rec[e][(rec[e].scanner == s) & (rec[e].bin_mm3 == b)].iloc[0]
                row[e] = f"{r.detected}/{r.reference_lesions} ({r.recall:.2f})" if r.reference_lesions else "–"
            rows.append(row)
    add(md_table(pd.DataFrame(rows), index=False))
    add("\n![recall_by_size_bin](figures/recall_by_size_bin.png)\n")
    fig_recall(rec, names, figs / "recall_by_size_bin.png")

    # ---- lesion-level tables (missed lesions, FP volumes)
    refl, predl = [], []
    for e in exps:
        for k in sorted(cases[e].fold.unique()):
            ids = sorted(cases[e][cases[e].fold == k].case_id)
            r, p_ = lesion_tables(pred_dir(e, int(k), cfg), ids, cfg=cfg, fold=int(k))
            refl.append(r.assign(model=e)), predl.append(p_.assign(model=e))
    refl, predl = pd.concat(refl, ignore_index=True), pd.concat(predl, ignore_index=True)
    lesions_dir = out / "lesions"
    lesions_dir.mkdir(exist_ok=True)
    refl.to_csv(lesions_dir / "ref_lesions.csv", index=False)
    predl.to_csv(lesions_dir / "pred_lesions.csv", index=False)
    for e in exps:  # lesion tables must agree with the per-case CSVs
        fp_case = (~predl[predl.model == e].true_positive).groupby(predl[predl.model == e].case_id).sum()
        if not (fp_case.reindex(cases[e].case_id).fillna(0).to_numpy() == cases[e].fp_lesions.to_numpy()).all():
            raise SystemExit(f"{e}: lesion-level FP counts disagree with fold CSVs")

    # ---- false positives
    n_scans = len(common)
    fp = predl[~predl.true_positive]
    small_edge = float(cfg["metrics"]["size_bins_mm3"][0])
    add("## False positives per scan\n")
    rows = []
    for e in exps:
        f = fp[fp.model == e]
        row = {"model": MODEL_LABELS[e], "FP lesions": len(f), "FP / scan": len(f) / n_scans}
        for s in [s for s in SCANNER_COLORS if s in set(cases[e].scanner)]:
            row[SCANNER_LABELS.get(s, s)] = (f.scanner == s).sum() / max((cases[e].scanner == s).sum(), 1)
        for b in names:
            row[f"{b} mm³"] = int((f.bin == b).sum())
        row[f"share < {small_edge:g} mm³"] = f"{100 * (f.volume_mm3 < small_edge).mean():.0f}%" if len(f) else "–"
        rows.append(row)
    fpt = pd.DataFrame(rows)
    fpt.to_csv(tables / "fp_lesions.csv", index=False)
    add("Scanner columns: FP lesions per scan for that scanner. Bin columns: FP lesion counts by their own volume.\n")
    add(md_table(fpt, index=False, floatfmt="{:.1f}"))
    add("\n![fp_volume_hist](figures/fp_volume_hist.png)\n")
    fig_fp_hist(fp, exps, small_edge, float(rule["min_lesion_mm3"]), n_scans, figs / "fp_volume_hist.png")

    # ---- missed lesions
    mrows = []
    for e in exps:
        for s in [s for s in SCANNER_COLORS if s in set(refl.scanner)]:
            for b in names:
                rb = refl[(refl.model == e) & (refl.scanner == s) & (refl.bin == b)]
                mrows.append({"model": e, "scanner": s, "bin": b, "reference": len(rb), "missed": int((~rb.detected).sum()),
                              "missed_no_overlap": int(((~rb.detected) & (rb.covered_frac == 0)).sum())})
    missed = pd.DataFrame(mrows)
    missed.to_csv(tables / "missed_lesions.csv", index=False)
    add("## Missed lesions\n")
    rows = []
    for b in names:
        row = {"bin (mm³)": b}
        for e in exps:
            m = missed[(missed.model == e) & (missed.bin == b)]
            row[MODEL_LABELS[e]] = f"{m.missed.sum()} / {m.reference.sum()} (no voxel: {m.missed_no_overlap.sum()})"
        rows.append(row)
    add("Missed / reference lesions; in brackets, missed lesions with no predicted voxel at all.\n")
    add(md_table(pd.DataFrame(rows), index=False))
    add("\n![missed_lesions](figures/missed_lesions.png)\n")
    fig_missed(missed, names, figs / "missed_lesions.png")

    # ---- paired comparisons
    add("## Paired comparisons (same patients)\n")
    add("Per-patient metrics: mean difference (A − B) with paired bootstrap 95% CI (patients resampled together), "
        "median difference and two-sided Wilcoxon signed-rank p; p (Holm) adjusts within each comparison over the "
        "metrics listed. Patients missing a value in either model (empty size bins) are dropped pairwise. "
        "Pooled lesion recall per bin: difference of pooled recalls with a paired cluster bootstrap CI.\n")
    comp_rows = []
    for a, b in [c for c in COMPARISONS if c[0] in cases and c[1] in cases]:
        rows = []
        for m in metric_cols:
            xa, xb = cases[a][m].to_numpy(float), cases[b][m].to_numpy(float)
            ok = ~(np.isnan(xa) | np.isnan(xb))
            ci = paired_bootstrap_diff_ci(xa[ok], xb[ok], **ci_kw)
            w = paired_wilcoxon(xa[ok], xb[ok])
            rows.append({"metric": m, "n": int(ok.sum()), f"mean {a}": xa[ok].mean(), f"mean {b}": xb[ok].mean(),
                         "mean diff": ci["estimate"], "95% CI": fmt_interval(ci["low"], ci["high"]),
                         "median diff": w["median_diff"], "Wilcoxon p": w["p"]})
        t = pd.DataFrame(rows)
        t["p (Holm)"] = holm(t["Wilcoxon p"])
        for bname in names:
            ra = rec[a][(rec[a].scanner == "all") & (rec[a].bin_mm3 == bname)].iloc[0]
            rb = rec[b][(rec[b].scanner == "all") & (rec[b].bin_mm3 == bname)].iloc[0]
            d = paired_bootstrap_ratio_diff_ci(cases[a][f"detected_{bname}"], cases[b][f"detected_{bname}"],
                                               cases[a][f"n_ref_{bname}"], **ci_kw)
            t = pd.concat([t, pd.DataFrame([{"metric": f"pooled recall {bname}", "n": len(common),
                                             f"mean {a}": ra.recall, f"mean {b}": rb.recall, "mean diff": d["estimate"],
                                             "95% CI": fmt_interval(d["low"], d["high"]), "median diff": np.nan,
                                             "Wilcoxon p": np.nan, "p (Holm)": np.nan}])], ignore_index=True)
        add(f"### {MODEL_LABELS[a]} − {MODEL_LABELS[b]}\n")
        add(md_table(t, index=False))
        add("")
        comp_rows.append(t.assign(comparison=f"{a} - {b}"))
    if comp_rows:
        pd.concat(comp_rows).to_csv(tables / "paired_comparisons.csv", index=False)

    # ---- E8 decision rule
    add("## E8 decision rule\n")
    dr = cfg["experiments"]["E8"]["decision_rule"]
    if {"E2a", "E2c"} <= set(cases):
        r20 = {e: rec[e][(rec[e].scanner == "all") & (rec[e].bin_mm3 == names[0])].iloc[0] for e in ("E2a", "E2c")}
        recall = {e: float(r.recall) for e, r in r20.items()}
        d = paired_bootstrap_ratio_diff_ci(cases["E2a"][f"detected_{names[0]}"], cases["E2c"][f"detected_{names[0]}"],
                                           cases["E2a"][f"n_ref_{names[0]}"], **ci_kw)
        choice, lines = e8_decision(dr["rules"], recall)
        add(f"Rule fixed in `configs/experiment.yaml` before any MedSAM training (metric `{dr['metric']}`: pooled "
            f"lesion-level recall in the {names[0]} mm³ bin over the cross-validation patients; no test data).\n")
        add(f"- recall(E2a, {names[0]} mm³) = {r20['E2a'].detected}/{r20['E2a'].reference_lesions} = **{recall['E2a']:.4f}**")
        add(f"- recall(E2c, {names[0]} mm³) = {r20['E2c'].detected}/{r20['E2c'].reference_lesions} = **{recall['E2c']:.4f}**")
        add(f"- difference E2a − E2c = **{recall['E2a'] - recall['E2c']:.4f}** "
            f"(paired cluster bootstrap 95% CI {fmt_interval(d['low'], d['high'])}; the rule uses the point estimate)\n")
        for i, line in enumerate(lines, 1):
            add(f"{i}. {line}")
        partial = len(common) < 15 or len(exps) < 3
        add(f"\n**Selected fix: `{choice}`**" + (" (provisional: partial results)" if partial else "") + "\n")
        (tables / "e8_decision.json").write_text(json.dumps({"recall_lt20": recall, "diff": recall["E2a"] - recall["E2c"],
                                                             "diff_ci": [d["low"], d["high"]], "choice": choice,
                                                             "patients": len(common), "partial": partial}, indent=2))
    else:
        add("_Needs both E2a and E2c._\n")

    # ---- per patient
    add("## Per-patient results\n")
    pc = cases[exps[0]][["fold", "case_id", "scanner", "n_ref_lesions"]].copy()
    for m in ["dice", "lesion_f1", "fp_lesions"]:
        for e in exps:
            pc[f"{m} {e}"] = cases[e][m].to_numpy()
    add(md_table(pc, index=False))
    pc.to_csv(tables / "cases.csv", index=False)

    (out / "cv_summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'cv_summary.md'} ({len(common)} common patients; models {exps})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
