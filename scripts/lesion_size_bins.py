"""Lesion counts per candidate volume bin on MSSEG TRAINING consensus masks.

    source scripts/env.sh
    python scripts/lesion_size_bins.py                     # edges 10 20 50 100 500 mm^3
    python scripts/lesion_size_bins.py --edges 20 100

Only release_split == train is read: bin edges must not depend on test data
(CLAUDE.md). Uses the shared metric definition of a lesion
(msbench.metrics: connectivity and minimum lesion size).
"""

from __future__ import annotations

import argparse

import nibabel as nib
import numpy as np
import pandas as pd

from msbench.config import load_config, paths
from msbench.metrics import DEFAULT_CONNECTIVITY, DEFAULT_MIN_LESION_MM3, bin_names, lesion_volume_histogram
from msbench.metrics.components import component_volumes_mm3, label_lesions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges", type=float, nargs="+", default=[10, 20, 50, 100, 500])
    args = ap.parse_args()

    cfg = load_config()
    processed = paths(cfg)["data_processed"]
    index = pd.read_csv(processed / "index.csv", dtype=str)
    tr = index[(index.dataset == cfg["primary_dataset"]) & (index.release_split == "train")]
    assert (tr.release_split == "train").all()

    pairs, per_case = [], []
    for case_id, rel, scanner in zip(tr.case_id, tr.path, tr.scanner):
        img = nib.load(str(processed / rel / "lesion.nii.gz"))
        mask = np.asanyarray(img.dataobj) > 0
        spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
        pairs.append((mask, spacing))
        lab, n = label_lesions(mask, spacing)
        vols = component_volumes_mm3(lab, n, spacing)
        per_case.append(pd.Series(np.bincount(np.digitize(vols, args.edges), minlength=len(args.edges) + 1),
                                  index=bin_names(args.edges), name=case_id).to_frame().T.assign(scanner=scanner))

    counts = lesion_volume_histogram(pairs, args.edges)
    pc = pd.concat(per_case)
    names = bin_names(args.edges)
    total = sum(counts.values())

    print(f"MSSEG training consensus masks: {len(tr)} cases, {total} lesions "
          f"(connectivity {DEFAULT_CONNECTIVITY}, min lesion {DEFAULT_MIN_LESION_MM3} mm^3)")
    print(f"candidate edges (mm^3): {args.edges}\n")
    summary = pd.DataFrame({
        "lesions": [counts[n] for n in names],
        "percent": [100 * counts[n] / total for n in names],
        "cumulative_%": np.cumsum([100 * counts[n] / total for n in names]),
        "cases_with_any": [(pc[n] > 0).sum() for n in names],
    }, index=names).round(1)
    print(summary.to_string())
    print("\nLesions per bin by scanner:")
    print(pc.groupby("scanner")[names].sum().to_string())


if __name__ == "__main__":
    main()
