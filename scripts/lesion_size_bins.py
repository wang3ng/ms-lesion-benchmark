"""Lesion counts per candidate volume bin on MSSEG TRAINING consensus masks.

    source scripts/env.sh
    python scripts/lesion_size_bins.py                     # edges 10 20 50 100 500 mm^3
    python scripts/lesion_size_bins.py --edges 20 100

Only release_split == train is read: bin edges must not depend on test data
(CLAUDE.md). Uses the lesion rule from configs/experiment.yaml.
The chosen edges live in configs/experiment.yaml (metrics.size_bins_mm3).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from msbench.config import load_config
from msbench.summaries import bin_counts, training_lesion_volumes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges", type=float, nargs="+", default=[10, 20, 50, 100, 500])
    args = ap.parse_args()

    cfg = load_config()
    rule = cfg["metrics"]["lesion_detection"]
    vols = training_lesion_volumes(cfg)
    counts = bin_counts(vols, args.edges).iloc[0]
    per_case = bin_counts(vols, args.edges, by="case_id")
    total = int(counts.sum())

    print(f"MSSEG training consensus masks: {vols.case_id.nunique()} cases, {total} lesions "
          f"(connectivity {rule['connectivity']}, min lesion {rule['min_lesion_mm3']} mm^3)")
    print(f"candidate edges (mm^3): {args.edges}\n")
    pct = 100 * counts / total
    summary = pd.DataFrame({"lesions": counts, "percent": pct, "cumulative_%": np.cumsum(pct),
                            "cases_with_any": (per_case > 0).sum()}).round(1)
    print(summary.to_string())
    print("\nLesions per bin by scanner:")
    print(bin_counts(vols, args.edges, by="scanner").to_string())


if __name__ == "__main__":
    main()
