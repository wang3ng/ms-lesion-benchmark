"""Segmentation-quality metrics on the FROZEN predictions (added after the results freeze).

    source scripts/env.sh
    python scripts/seg_quality.py              # compute (cached) + append the sections
    python scripts/seg_quality.py --recompute

Metrics (msbench.metrics.segquality): lesion-wise Dice, per-lesion HD95 and average surface distance for
detected reference lesions (Anima-port matching), whole-mask HD95 and whole-mask IoU. Computed for E1, E2a,
E2b, E2c and E8 in cross-validation (E1's 5 folds, 15 MSSEG training patients) and on the three test sets
(MSSEG in-distribution, MSSEG GE, ISBI per rater). Same handling as before: resampling unit = patient (ISBI:
5 patients, Wilcoxon on per-patient means), empty-reference scans reported separately and left out of
averages and tests, Holm adjustment within each comparison and set. Test-set definitions, prediction
folders and the clustered statistics are imported from scripts/score_test.py; CV folders from
scripts/e2_summary.py.

Output: a section "Segmentation-quality metrics (added after the results freeze)" appended to (or replaced
in) results/E2/cv_summary.md and results/test/summary.md, between HTML comment markers; nothing above the
markers is touched. Re-run this script after regenerating either summary. Tables: results/E2/tables/segq_*,
results/test/tables/segq_*. No model, prediction, threshold or existing number is changed. The section ends with an additional
E2a vs E1 comparison on lesions detected by both models (also added after the freeze).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from msbench.config import REPO_ROOT, load_config, paths
from msbench.evaluation import detection_rule, load_mask
from msbench.metrics import bin_names, segmentation_quality
from msbench.reporting import INK, INK2, MODEL_COLORS, SURFACE, fmt_interval, md_table, new_fig, save_fig


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ST = _load("score_test")
ES = _load("e2_summary")
RESULTS = ST.RESULTS
MODELS = ST.MODELS
START, END = "<!-- segq:start -->", "<!-- segq:end -->"
TITLE = "## Segmentation-quality metrics (added after the results freeze)"


def _case(job):
    key, exp, case_id, pred_path, ref_path = job
    cfg = load_config()
    edges = [float(b) for b in cfg["metrics"]["size_bins_mm3"]]
    ref, _, spacing = load_mask(Path(ref_path))
    pred, _, _ = load_mask(Path(pred_path))
    row, rec = segmentation_quality(pred, ref, spacing, edges, **detection_rule(cfg))
    row.update(group=key, model=exp, case_id=case_id, n_ref=len(rec["detected"]), ref_voxels=int(ref.sum()))
    lr = pd.DataFrame({"group": key, "model": exp, "case_id": case_id, "lesion": np.arange(len(rec["detected"])),
                       "volume_mm3": rec["volume_mm3"], "bin": rec["bin"], "detected": rec["detected"],
                       "dice": rec["dice"], "hd95": rec["hd95"], "asd": rec["asd"]})
    return row, lr


def compute(jobs, out_cases: Path, out_lesions: Path, recompute: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if out_cases.exists() and out_lesions.exists() and not recompute:
        return pd.read_csv(out_cases), pd.read_csv(out_lesions)
    rows, les = [], []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for row, lr in ex.map(_case, jobs, chunksize=2):
            rows.append(row), les.append(lr)
    cases, lesions = pd.DataFrame(rows), pd.concat(les, ignore_index=True)
    out_cases.parent.mkdir(parents=True, exist_ok=True), out_lesions.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(out_cases, index=False)
    lesions.to_csv(out_lesions, index=False)
    return cases, lesions


# ---------------------------------------------------------------- report
def fig_bins(stats: pd.DataFrame, groups: list[tuple[str, str]], names: list[str], labels: dict, out: Path) -> None:
    fig, axes = new_fig(4.6 * len(groups) + 1, 4.0, ncols=len(groups))
    axes = np.atleast_1d(axes)
    for ax, (g, title) in zip(axes, groups):
        x = np.arange(len(names))
        w = 0.8 / len(MODELS)
        for j, e in enumerate(MODELS):
            q = stats[(stats.group == g) & (stats.model == e)].set_index("metric")
            m = [q.loc[f"lesion_dice_{b}", "mean"] for b in names]
            lo = [q.loc[f"lesion_dice_{b}", "low"] for b in names]
            hi = [q.loc[f"lesion_dice_{b}", "high"] for b in names]
            xs = x + (j - (len(MODELS) - 1) / 2) * w
            ax.bar(xs, np.nan_to_num(m), width=w, color=MODEL_COLORS[e], edgecolor=SURFACE, linewidth=1.5,
                   label=labels[e], zorder=2)
            err = np.vstack([np.subtract(m, lo), np.subtract(hi, m)]).clip(min=0)
            ax.errorbar(xs, m, yerr=np.nan_to_num(err), fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        ax.set_xticks(x, [f"{b} mm³" for b in names], color=INK, fontsize=8.5)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    axes[0].set_ylabel("Lesion-wise Dice of detected lesions\n(mean of patient means, 95% CI)", color=INK2, fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left", ncol=2)
    save_fig(fig, out)


def group_section(cases, lesions, g, cl_of, labels, comparisons, names, ci_kw, add, stats_rows, cmp_rows, iou_rows):
    c = {e: cases[(cases.group == g) & (cases.model == e)].sort_values("case_id").reset_index(drop=True) for e in MODELS}
    empty = c["E1"][c["E1"].ref_voxels == 0].case_id.tolist()
    if empty:
        rows = []
        for cid in empty:
            for e in MODELS:
                r = c[e].set_index("case_id").loc[cid]
                rows.append({"scan": cid, "model": labels[e], "whole-mask IoU": r.mask_iou,
                             "whole-mask HD95": "undefined (empty reference)"})
        add(f"Empty reference, reported separately and left out of the averages and tests: {', '.join(empty)} — "
            "lesion-wise metrics undefined (no reference lesion); whole-mask IoU per model (1 = both empty):\n")
        add(md_table(pd.DataFrame(rows), index=False))
        add("")
    c = {e: d[d.ref_voxels > 0].reset_index(drop=True) for e, d in c.items()}
    for e in MODELS[1:]:
        if list(c[e].case_id) != list(c["E1"].case_id) or not np.array_equal(c[e].n_ref, c["E1"].n_ref):
            raise SystemExit(f"{g} {e}: cases / reference lesions differ from E1")
    cl = np.array([cl_of[x] for x in c["E1"].case_id])
    metrics = ["lesion_dice"] + [f"lesion_dice_{b}" for b in names] + ["lesion_hd95", "lesion_asd", "mask_hd95"]
    st = {e: {m: ST.mean_ci(c[e][m].to_numpy(float), cl, ci_kw) for m in metrics + ["mask_iou"]} for e in MODELS}
    for e in MODELS:
        for m, r in st[e].items():
            stats_rows.append({"group": g, "model": e, "metric": m, **r})

    def cell(r, fmt="{:.3f}"):
        if np.isnan(r["mean"]):
            return "–"
        t = f"{fmt.format(r['mean'])} ± {fmt.format(r['sd'])}<br>{fmt_interval(r['low'], r['high'])}"
        return t if r["n"] == len(cl) else f"{t} (n={r['n']})"

    pretty = {"lesion_dice": "lesion-wise Dice", "lesion_hd95": "lesion HD95 (mm)", "lesion_asd": "lesion ASD (mm)",
              "mask_hd95": "whole-mask HD95 (mm) †", **{f"lesion_dice_{b}": f"lesion-wise Dice {b} mm³" for b in names}}
    tab = pd.DataFrame({labels[e]: [cell(st[e][m], "{:.3f}" if "dice" in m else "{:.2f}") for m in metrics]
                        for e in MODELS}, index=[pretty[m] for m in metrics]).rename_axis("metric")
    add(f"{len(cl)} scans, {len(set(cl))} patients. Mean ± SD over scans of the per-scan values, 95% CI (patient "
        "bootstrap); (n=…) = scans with at least one detected lesion (in the bin).\n")
    add(md_table(tab))
    nd = {e: int(c[e].n_detected.sum()) for e in MODELS}
    add(f"\nDetected reference lesions behind the lesion-wise values: "
        + ", ".join(f"{e} {nd[e]}" for e in MODELS) + f" (of {int(c['E1'].n_ref.sum())}).\n")

    # paired comparisons
    fam = ["lesion_dice"] + [f"lesion_dice_{b}" for b in names] + ["lesion_hd95"]
    L = lesions[lesions.group == g]
    for a, b in comparisons:
        rows = []
        for m in fam:
            r = ST.paired(c[a][m], c[b][m], cl, ci_kw)
            if r["n_scans"] == 0:   # no scan with a value in both models: no test
                r["p"] = np.nan
            rows.append({"metric": pretty[m], "n scans": r["n_scans"], "n patients": r["n_patients"],
                         f"mean {a}": r["mean_a"], f"mean {b}": r["mean_b"], "mean diff": r["diff"],
                         "95% CI": fmt_interval(r["low"], r["high"]), "Wilcoxon p": r["p"]})
        t = pd.DataFrame(rows)
        t["p (Holm)"] = ST.holm(t["Wilcoxon p"])
        # sensitivity: only lesions detected by BOTH models (same lesions compared); unadjusted
        la = L[L.model == a].set_index(["case_id", "lesion"])
        lb = L[L.model == b].set_index(["case_id", "lesion"]).loc[la.index]
        both = la.detected & lb.detected
        for m, nm in (("dice", "lesion-wise Dice, lesions detected by both ‡"), ("hd95", "lesion HD95, lesions detected by both ‡")):
            va = la[m].where(both).groupby(level=0).mean().reindex(c[a].case_id).to_numpy(float)
            vb = lb[m].where(both).groupby(level=0).mean().reindex(c[a].case_id).to_numpy(float)
            r = ST.paired(va, vb, cl, ci_kw)
            if r["n_scans"] == 0:
                r["p"] = np.nan
            t = pd.concat([t, pd.DataFrame([{"metric": nm, "n scans": r["n_scans"], "n patients": r["n_patients"],
                                             f"mean {a}": r["mean_a"], f"mean {b}": r["mean_b"], "mean diff": r["diff"],
                                             "95% CI": fmt_interval(r["low"], r["high"]), "Wilcoxon p": r["p"],
                                             "p (Holm)": np.nan}])], ignore_index=True)
        add(f"**{labels[a]} − {labels[b]}**\n")
        add(md_table(t, index=False))
        add("")
        cmp_rows.append(t.assign(group=g, comparison=f"{a} - {b}"))
    for e in MODELS:
        iou_rows.append({"set": g, "model": labels[e], "whole-mask IoU": cell(st[e]["mask_iou"])})


# ---------------------------------------------------------------- additional comparison (post freeze)
SHARED_TITLE = "### Additional comparison, added after the results freeze: E2a vs E1 on shared lesions"


def shared_lesion_section(lesions, groups, cl_of, labels, names, ci_kw, figs, tables, add) -> None:
    """E2a − E1 lesion-wise Dice on the reference lesions detected by BOTH models, overall and by bin."""
    add(SHARED_TITLE + "\n")
    add("> **Additional comparison, added after the results freeze** (not one of the pre-set comparisons). Only "
        "reference lesions detected by both E2a and E1 are compared, so both models are scored on the same lesions. "
        "Per scan: mean lesion-wise Dice over those lesions (overall and by the reference lesion's size bin); A − B "
        "with paired patient-bootstrap 95% CI and Wilcoxon signed-rank on per-patient values; **Holm adjustment "
        "over the four rows of each set**. Scans without a shared lesion (in the bin) are dropped pairwise. E2a "
        "places its boxes from the reference (on ISBI, from the rater it is scored against).\n")
    rows, fig_rows = [], []
    for g, title in groups:
        L = lesions[lesions.group == g]
        a = L[L.model == "E2a"].set_index(["case_id", "lesion"])
        b = L[L.model == "E1"].set_index(["case_id", "lesion"]).loc[a.index]
        both = a.detected & b.detected
        cases = sorted(set(a.index.get_level_values(0)))
        cl = np.array([cl_of[c] for c in cases])
        t = []
        for bn in ["all"] + names:
            sel = both if bn == "all" else both & (a["bin"] == bn)
            va = a.dice.where(sel).groupby(level=0).mean().reindex(cases).to_numpy(float)
            vb = b.dice.where(sel).groupby(level=0).mean().reindex(cases).to_numpy(float)
            r = ST.paired(va, vb, cl, ci_kw)
            if r["n_scans"] == 0:
                r["p"] = np.nan
            ok = ~(np.isnan(va) | np.isnan(vb))
            ma, mb = ST.mean_ci(np.where(ok, va, np.nan), cl, ci_kw), ST.mean_ci(np.where(ok, vb, np.nan), cl, ci_kw)
            t.append({"set": title, "bin (mm³)": bn, "shared lesions": int(sel.sum()), "n scans": r["n_scans"],
                      "n patients": r["n_patients"], "mean E2a": r["mean_a"], "mean E1": r["mean_b"],
                      "mean diff (E2a − E1)": r["diff"], "95% CI": fmt_interval(r["low"], r["high"]),
                      "Wilcoxon p": r["p"],
                      "lesions with E2a > E1": f"{int((a.dice[sel] > b.dice[sel]).sum())}/{int(sel.sum())}"})
            for e, m in (("E1", mb), ("E2a", ma)):
                fig_rows.append({"group": g, "title": title, "bin": bn, "model": e, "mean": m["mean"],
                                 "low": m["low"], "high": m["high"]})
        t = pd.DataFrame(t)
        t["p (Holm)"] = ST.holm(t["Wilcoxon p"])
        rows.append(t)
        if len(groups) > 1:
            add(f"**{title}**" + (" (5 patients: smallest possible Wilcoxon p = 0.0625)" if g.startswith("isbi") else "") + "\n")
        add(md_table(t.drop(columns="set"), index=False))
        add("")
    pd.concat(rows).to_csv(tables / "segq_e2a_vs_e1_shared.csv", index=False)
    fr = pd.DataFrame(fig_rows)
    fr.to_csv(tables / "segq_e2a_vs_e1_shared_means.csv", index=False)
    bins = ["all"] + names
    fig, axes = new_fig(4.2 * len(groups) + 0.8, 3.9, ncols=len(groups))
    axes = np.atleast_1d(axes)
    for ax, (g, title) in zip(axes, groups):
        x = np.arange(len(bins))
        for j, e in enumerate(["E1", "E2a"]):
            q = fr[(fr.group == g) & (fr.model == e)].set_index("bin").loc[bins]
            xs = x + (j - 0.5) * 0.38
            ax.bar(xs, q["mean"].fillna(0), width=0.38, color=MODEL_COLORS[e], edgecolor=SURFACE, linewidth=1.5,
                   label=labels[e], zorder=2)
            err = np.vstack([q["mean"] - q["low"], q["high"] - q["mean"]]).clip(min=0)
            ax.errorbar(xs, q["mean"], yerr=np.nan_to_num(err), fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        ax.set_xticks(x, ["all" if b == "all" else f"{b} mm³" for b in bins], color=INK, fontsize=8.5)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    axes[0].set_ylabel("Lesion-wise Dice, lesions detected by both\n(mean of patient means, 95% CI)", color=INK2, fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left", ncol=2)
    save_fig(fig, figs / "segq_e2a_vs_e1_shared.png")
    add("![segq_e2a_vs_e1_shared](figures/segq_e2a_vs_e1_shared.png)\n")


def build(cases, lesions, groups, cl_of, labels, comparisons, names, ci_kw, figs, tables, intro_extra: str) -> str:
    md: list[str] = []
    add = md.append
    add(TITLE + "\n")
    add("> **Added after the results freeze, on the frozen predictions.** No model, prediction, threshold or "
        "number above this section was changed. Same lesions, matching rule (Anima port), size bins, patients and "
        "statistical handling as above.\n")
    add("Definitions (`msbench.metrics.segquality`): for each **detected** reference lesion, its prediction is the "
        "union of the predicted components overlapping it. **Lesion-wise Dice** = Dice of that pair; **lesion "
        "HD95** = max of the two directed 95th-percentile surface distances (mm); **lesion ASD** = symmetric mean "
        "surface distance (mm). Each is averaged over the detected lesions of a scan (overall and by the "
        "reference lesion's size bin), then over scans. Missed lesions have no value: these metrics describe "
        "tracing quality *given* detection, so a model that detects more (often harder, smaller) lesions is "
        "compared on a different lesion set; rows marked ‡ restrict each pair of models to the lesions both "
        "detected (unadjusted p). **† Whole-mask HD95** uses the raw masks: one distant false positive can "
        "dominate it, so it mostly reflects stray false positives rather than outline quality. " + intro_extra + "\n")
    add("Paired comparisons: per-scan values, A − B, paired patient-bootstrap 95% CI, Wilcoxon signed-rank; "
        "**Holm adjustment within each comparison over lesion-wise Dice (overall and the three size bins) and "
        "lesion HD95**; scans where either model has no detected lesion (in the bin) are dropped pairwise.\n")
    stats_rows, cmp_rows, iou_rows = [], [], []
    for g, title in groups:
        if len(groups) > 1:
            add(f"### {title}\n")
        group_section(cases, lesions, g, cl_of, labels, comparisons, names, ci_kw, add, stats_rows, cmp_rows, iou_rows)
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(tables / "segq_metrics.csv", index=False)
    pd.concat(cmp_rows).to_csv(tables / "segq_paired_comparisons.csv", index=False)
    fig_bins(stats, groups, names, labels, figs / "segq_lesion_dice_by_bin.png")
    add("![segq_lesion_dice_by_bin](figures/segq_lesion_dice_by_bin.png)\n")
    add("### Appendix — whole-mask IoU\n")
    add("IoU = |A ∩ B| / |A ∪ B| on the raw masks (monotone in whole-mask Dice: IoU = Dice / (2 − Dice) per scan); "
        "mean ± SD over scans, 95% CI (patient bootstrap).\n")
    iou = pd.DataFrame(iou_rows)
    add(md_table(iou.pivot(index="model", columns="set", values="whole-mask IoU").reindex(
        [labels[e] for e in MODELS])[[g for g, _ in groups]].rename(columns=dict(groups)) if len(groups) > 1 else
        iou.drop(columns="set"), index=len(groups) > 1))
    add("")
    shared_lesion_section(lesions, groups, cl_of, labels, names, ci_kw, figs, tables, add)
    return "\n".join(md)


def write_section(path: Path, text: str) -> None:
    s = path.read_text()
    block = f"{START}\n{text}\n{END}\n"
    if START in s:
        s = s[:s.index(START)] + block + s[s.index(END) + len(END) + 1:]
    else:
        s = s.rstrip("\n") + "\n\n" + block
    path.write_text(s)


def main(argv=None) -> int:
    warnings.filterwarnings("ignore", message="Mean of empty slice")       # pairs with no scan (reported as –)
    warnings.filterwarnings("ignore", message="invalid value encountered")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    names = bin_names([float(b) for b in cfg["metrics"]["size_bins_mm3"]])
    ci = cfg["statistics"]["confidence_interval"]
    ci_kw = {"n_boot": int(ci["n_boot"]), "level": float(ci["level"]), "seed": int(cfg["seed"])}
    p = paths(cfg)
    index = pd.read_csv(p["data_processed"] / "index.csv", dtype=str)
    ix = index.set_index("case_id")
    cl_of = ix["subject_id"].to_dict()

    # ---- cross-validation
    splits = json.loads((REPO_ROOT / "experiments" / "E1" / "splits_final.json").read_text())
    jobs = [("cv", e, c, str(ES.pred_dir(e, k, cfg) / f"{c}.nii.gz"),
             str(p["data_processed"] / ix.loc[c, "path"] / "lesion.nii.gz"))
            for e in MODELS for k, sp in enumerate(splits) for c in sp["val"]]
    cv_cases, cv_les = compute(jobs, RESULTS / "E2" / "tables" / "segq_cases.csv",
                               RESULTS / "E2" / "lesions" / "segq_lesions.csv", args.recompute)
    text = build(cv_cases, cv_les, [("cv", "Cross-validation")], cl_of, ES.MODEL_LABELS, ES.COMPARISONS, names,
                 ci_kw, RESULTS / "E2" / "figures", RESULTS / "E2" / "tables",
                 "Cross-validation: 15 MSSEG training patients, each scored by the fold model that did not train on "
                 "them (E2a: oracle boxes, an upper bound).")
    write_section(RESULTS / "E2" / "cv_summary.md", text)

    # ---- test sets
    jobs = []
    for key, s in ST.SETS.items():
        for c in ST.set_cases(s, index).case_id:
            for e in MODELS:
                jobs.append((key, e, c, str(ST.pred_dir(e, s) / f"{c}.nii.gz"),
                             str(p["data_processed"] / ix.loc[c, "path"] / f"{s['ref']}.nii.gz")))
    te_cases, te_les = compute(jobs, RESULTS / "test" / "tables" / "segq_cases.csv",
                               RESULTS / "test" / "lesions" / "segq_lesions.csv", args.recompute)
    text = build(te_cases, te_les, [(k, ST.SHORT[k]) for k in ST.SETS], cl_of, ST.LABELS, ST.COMPARISONS, names,
                 ci_kw, RESULTS / "test" / "figures", RESULTS / "test" / "tables",
                 "Test sets as above; ISBI resampled and tested by patient (5 patients, smallest possible Wilcoxon "
                 "p = 0.0625); E2a on ISBI uses the boxes of the rater it is scored against.")
    write_section(RESULTS / "test" / "summary.md", text)
    print("sections written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
