"""Shared helpers for the Markdown/figure reports (midpoint and E1 summaries).

Chart style follows the reference palette of the dataviz skill (light mode):
scanner colours are fixed per scanner identity, never assigned by rank.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ACCENT = "#2a78d6"
SCANNER_COLORS = {
    "siemens_verio_3t": "#2a78d6",
    "siemens_aera_1p5t": "#eb6834",
    "philips_ingenia_3t": "#1baf7a",
    "ge_discovery_3t": "#eda100",
}
SCANNER_LABELS = {
    "siemens_verio_3t": "Siemens Verio 3T",
    "siemens_aera_1p5t": "Siemens Aera 1.5T",
    "philips_ingenia_3t": "Philips Ingenia 3T",
    "ge_discovery_3t": "GE Discovery 3T",
}
MAIN_METRICS = ["dice", "lesion_precision", "lesion_recall", "lesion_f1", "fp_lesions"]


# ---------------------------------------------------------------- text
def md_table(df: pd.DataFrame, index: bool = True, floatfmt: str = "{:.3f}") -> str:
    d = df.reset_index() if index else df

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "–" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def mean_sd(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "–"
    if len(s) == 1:
        return f"{s.iloc[0]:.3f} (n=1)"
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f} (n={len(s)})"


def mask_empty_bins(cases: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Per-patient size-bin metrics suitable for averaging over patients.

    msbench.metrics scores a bin with no reference and no predicted lesion as
    precision = recall = F1 = 1 (the whole-case convention). Averaged over patients
    that would reward patients who simply have no lesion of that size, so here:
    recall_<bin> -> NaN when the patient has no reference lesion in the bin, and
    f1_<bin> / precision_<bin> -> NaN when neither mask has a lesion in the bin.
    Pooled counts (detected_<bin>, n_ref_<bin>) are untouched.
    """
    c = cases.copy()
    for b in names:
        no_ref = c[f"n_ref_{b}"] == 0
        neither = no_ref & (c[f"n_pred_{b}"] == 0)
        c.loc[no_ref, f"recall_{b}"] = np.nan
        c.loc[neither, [f"f1_{b}", f"precision_{b}"]] = np.nan
    return c


def job_alive(logdir: Path, marker: str = "run_E1") -> bool:
    """True if <logdir>/job.pid names a live process whose command line contains `marker`."""
    pidf = logdir / "job.pid"
    if not pidf.exists():
        return False
    pid = pidf.read_text().strip()
    try:
        return marker in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return False


_EXIT = re.compile(r"^=== (\S+) fold (\d+) exit (-?\d+) after (\d+)h(\d+)m")


def fold_wall_time(logfile: Path) -> dict:
    """Wall time of the last nnU-Net run recorded by scripts/nnunet_fold.sh in `logfile`.

    Returns {"wall_h": float or nan, "exit": int or None, "ended": str or None}. A resumed
    fold has several runs; their durations are summed.
    """
    out = {"wall_h": float("nan"), "exit": None, "ended": None}
    if not logfile.exists():
        return out
    total = 0.0
    for line in logfile.read_text(errors="replace").splitlines():
        m = _EXIT.match(line)
        if m:
            total += int(m.group(4)) + int(m.group(5)) / 60
            out.update(wall_h=total, exit=int(m.group(3)), ended=m.group(1))
    return out


# ---------------------------------------------------------------- figures
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(w, h, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE)
    for ax in np.atleast_1d(axes):
        style_axes(ax)
    return fig, axes


def save_fig(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_agreement_scatter(cases: pd.DataFrame, out: Path) -> None:
    """Predicted vs reference lesion count and volume, one dot per case, coloured by scanner."""
    fig, axes = new_fig(10, 4.4, ncols=2)
    panels = [("n_ref_lesions", "n_pred_lesions", "Lesion count", ""),
              ("ref_volume_ml", "pred_volume_ml", "Lesion volume", " (mL)")]
    for ax, (xc, yc, title, unit) in zip(axes, panels):
        hi = max(cases[xc].max(), cases[yc].max(), 1) * 1.08
        ax.plot([0, hi], [0, hi], color=INK2, linestyle="--", linewidth=1, label="identity", zorder=1)
        for sc in [s for s in SCANNER_COLORS if s in set(cases.scanner)]:
            c = cases[cases.scanner == sc]
            ax.scatter(c[xc], c[yc], s=64, color=SCANNER_COLORS[sc], edgecolor=SURFACE, linewidth=2,
                       label=SCANNER_LABELS.get(sc, sc), zorder=2)
        ax.set_xlim(0, hi), ax.set_ylim(0, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Reference (consensus){unit}", color=INK2, fontsize=9)
        ax.set_ylabel(f"Predicted (nnU-Net){unit}", color=INK2, fontsize=9)
        ax.set_title(f"{title}: predicted vs reference", color=INK, fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    save_fig(fig, out)
