"""Render Tables 1-3 of report/methods_results.md from the frozen result files (no computation of new results).

    python scripts/report_tables.py            # rewrites the blocks between <!-- table:N:start/end --> markers

Rounding as in the report text: Dice / F1 / recall 3 decimals, FP lesions per scan 1 decimal, p (Holm) as
"< 0.001", 3 decimals below 0.1, otherwise 2 decimals. E2a is marked as the upper bound; E7/E9 are not in
Tables 2-3.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "methods_results.md"
MODELS = ["E1", "E2a", "E2b", "E2c", "E8"]
HEAD = {"E1": "E1 nnU-Net", "E2a": "*E2a MedSAM, oracle boxes* †", "E2b": "E2b MedSAM, E1 boxes",
        "E2c": "E2c MedSAM, brain box + tiles", "E8": "E8 MedSAM, zoomed tiles"}
BINS = ["<20", "20-100", ">=100"]
BIN_LABEL = {"<20": "< 20 mm³", "20-100": "20–100 mm³", ">=100": "≥ 100 mm³"}
SETS = [("msseg_id", "MSSEG in-distribution (29 scans)"), ("msseg_ge", "MSSEG GE, unseen scanner (8 scans)"),
        ("isbi_r1", "ISBI 2015, rater 1 (21 scans, 5 patients)"), ("isbi_r2", "ISBI 2015, rater 2 (21 scans, 5 patients)")]


def f3(x):
    return f"{x:.3f}".replace("-", "−")


def f1(x):
    return f"{x:.1f}".replace("-", "−")


def ci(lo, hi, f=f3):
    return f"[{f(lo)}, {f(hi)}]"


def pfmt(p):
    if pd.isna(p):
        return "–"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def ub(model, text):
    """Italicise E2a cells (upper bound)."""
    return f"*{text}*" if model == "E2a" else text


def table1() -> str:
    ts = json.loads((ROOT / "results/E9/training_set.json").read_text())
    ml = ts["training"]["by_dataset"]["mslesseg"]
    rows = [
        "| Dataset | Scanner (site) | Field | Scans | Patients | Role |",
        "|---|---|---|---|---|---|",
        "| MSSEG-2016 training | Siemens Verio (Rennes) | 3T | 5 | 5 | 5-fold CV (development) |",
        "| | Siemens Aera (Lyon) | 1.5T | 5 | 5 | 5-fold CV (development) |",
        "| | Philips Ingenia (Lyon) | 3T | 5 | 5 | 5-fold CV (development) |",
        "| MSSEG-2016 test | Siemens Verio (Rennes) | 3T | 10 | 10 | in-distribution test |",
        "| | Siemens Aera (Lyon) | 1.5T | 10 | 10 | in-distribution test |",
        "| | Philips Ingenia (Lyon) | 3T | 10 | 10 | in-distribution test |",
        "| | GE Discovery (Bordeaux) | 3T | 8 | 8 | unseen-scanner test |",
        "| ISBI 2015 (labelled release) | Philips | 3T | 21 | 5 | external test, each rater scored separately |",
        f"| MSLesSeg training (E9 only, exploratory) | Philips Ingenia / Achieva dStream | – | {ml['scans']} | {ml['patients']} "
        f"| E9 training only ({ts['excluded_mslesseg']['scans']} GE/unknown-scanner scans excluded) |",
    ]
    note = ("Inputs: FLAIR, T1, T2 on the FLAIR grid. References: MSSEG consensus of seven experts; ISBI raters 1 and 2; "
            "MSLesSeg's own masks (different annotation protocol). One in-distribution test scan "
            "(msseg_test_c07_p08) has an empty consensus and is reported separately.")
    src = "<!-- src: README.md (MSSEG scanners, sites, field strengths, counts); results/test/summary.md (ISBI scans/patients); results/E9/training_set.json (MSLesSeg) -->"
    return "\n".join(rows) + "\n\n" + note + " " + src


def table2() -> str:
    m = pd.read_csv(ROOT / "results/E2/tables/metrics.csv").set_index(["model", "metric"])
    r = pd.read_csv(ROOT / "results/E2/tables/recall_by_size_bin.csv")
    r = r[r.scanner == "all"].set_index(["model", "bin_mm3"])
    lines = ["| Metric | " + " | ".join(HEAD[e] for e in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for metric, name, f in [("dice", "Dice", f3), ("lesion_precision", "Lesion precision", f3),
                            ("lesion_recall", "Lesion recall", f3), ("lesion_f1", "Lesion F1", f3),
                            ("fp_lesions", "FP lesions / scan", f1)]:
        cells = []
        for e in MODELS:
            q = m.loc[(e, metric)]
            cells.append(ub(e, f"{f(q['mean'])} ± {f(q['sd'])} {ci(q.ci_low, q.ci_high, f)}"))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for b in BINS:
        cells = []
        for e in MODELS:
            q = r.loc[(e, b)]
            cells.append(ub(e, f"{int(q.detected)}/{int(q.reference_lesions)} = {f3(q.recall)} {ci(q.ci_low, q.ci_high)}"))
        lines.append(f"| Pooled recall, {BIN_LABEL[b]} | " + " | ".join(cells) + " |")
    note = ("Mean ± SD over the 15 patients with the patient-bootstrap 95% CI; pooled recall = detected / reference "
            "lesions with a cluster-bootstrap 95% CI. † E2a places its boxes from the reference: an upper bound on "
            "tracing, not a fair comparison (italic).")
    src = "<!-- src: results/E2/tables/metrics.csv; results/E2/tables/recall_by_size_bin.csv -->"
    return "\n".join(lines) + "\n\n" + note + " " + src


def table3() -> str:
    m = pd.read_csv(ROOT / "results/test/tables/metrics.csv").set_index(["set", "model", "metric"])
    r = pd.read_csv(ROOT / "results/test/tables/recall_by_size_bin.csv").set_index(["set", "model", "bin"])
    c = pd.read_csv(ROOT / "results/test/tables/paired_comparisons.csv")
    c = c[c.metric == "lesion_f1"].set_index(["set", "comparison"])
    lines = ["| Test set | Model | Lesion F1 | Dice | FP lesions / scan | Pooled recall < 20 mm³ | Δ lesion F1 vs E1 [95% CI] | p (Holm) |",
             "|---|---|---|---|---|---|---|---|"]
    for key, label in SETS:
        for i, e in enumerate(MODELS):
            q = r.loc[(key, e, "<20")]
            if (key, f"{e} - E1") in c.index:
                d = c.loc[(key, f"{e} - E1")]
                lo, hi = (float(v) for v in re.findall(r"-?\d+\.\d+", d["95% CI"]))
                delta, p = f"{f3(d['mean diff'])} {ci(lo, hi)}", pfmt(d["p (Holm)"])
            else:
                delta, p = "–", "–"
            row = [label if i == 0 else "", HEAD[e],
                   ub(e, f3(m.loc[(key, e, 'lesion_f1'), 'mean'])), ub(e, f3(m.loc[(key, e, 'dice'), 'mean'])),
                   ub(e, f1(m.loc[(key, e, 'fp_lesions'), 'mean'])),
                   ub(e, f"{int(q.detected)}/{int(q.reference)} = {f3(q.recall)}"), delta, p]
            lines.append("| " + " | ".join(row) + " |")
    note = ("Means over scans (5-fold ensembles; empty-consensus scan excluded). Δ = model − E1, paired patient "
            "bootstrap 95% CI; p = two-sided Wilcoxon signed-rank, Holm-adjusted within each comparison and test set "
            "over the per-scan metrics. ISBI: CIs resample the 5 patients and Wilcoxon uses per-patient means (n = 5; "
            "smallest attainable p 0.0625). † E2a uses reference boxes (upper bound; italic) and was not part of the "
            "paired comparisons with E1.")
    src = "<!-- src: results/test/tables/metrics.csv; results/test/tables/recall_by_size_bin.csv; results/test/tables/paired_comparisons.csv -->"
    return "\n".join(lines) + "\n\n" + note + " " + src


def main() -> int:
    s = REPORT.read_text()
    for n, fn in ((1, table1), (2, table2), (3, table3)):
        start, end = f"<!-- table:{n}:start -->", f"<!-- table:{n}:end -->"
        block = f"{start}\n{fn()}\n{end}"
        if start in s:
            s = s[:s.index(start)] + block + s[s.index(end) + len(end):]
        else:
            cap = re.search(rf"\*\*Table {n}\..*?\n", s)
            if not cap:
                raise SystemExit(f"caption of Table {n} not found")
            s = s[:cap.end()] + "\n" + block + "\n" + s[cap.end():]
    REPORT.write_text(s)
    print(f"tables written to {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
