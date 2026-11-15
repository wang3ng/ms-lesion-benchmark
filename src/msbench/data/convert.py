"""Convert raw releases into the common layout and rebuild the index.

    python -m msbench.data.convert                       # all datasets
    python -m msbench.data.convert msseg2016 --workers 8
    python -m msbench.data.convert --overwrite           # redo finished cases

Paths come from configs/experiment.yaml (paths.data_raw, paths.data_processed).
Prints per-case summary statistics only (shape, spacing, lesion voxels/count).
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from msbench.config import paths
from msbench.data import CONVERTERS, ConversionError, build_index, convert_case


def _run(spec, out_root, raw_root, overwrite):
    return convert_case(spec, out_root, raw_root, overwrite)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", metavar="dataset",
                    help=f"subset of {list(CONVERTERS)} (default: all)")
    ap.add_argument("--raw", type=Path, default=None, help="override paths.data_raw")
    ap.add_argument("--out", type=Path, default=None, help="override paths.data_processed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    args.datasets = args.datasets or list(CONVERTERS)
    unknown = set(args.datasets) - set(CONVERTERS)
    if unknown:
        ap.error(f"unknown dataset(s) {sorted(unknown)}; choose from {list(CONVERTERS)}")

    p = paths()
    raw_root = (args.raw or p["data_raw"]).resolve()
    out_root = (args.out or p["data_processed"]).resolve()

    specs = [s for ds in args.datasets for s in CONVERTERS[ds](raw_root)]
    ids = [s.case_id for s in specs]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate case IDs; check the converters")
    print(f"raw: {raw_root}\nout: {out_root}")
    for ds in args.datasets:
        print(f"  {ds}: {sum(s.dataset == ds for s in specs)} cases found")

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run, s, out_root, raw_root, args.overwrite): s for s in specs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except ConversionError as e:
                failures.append(str(e))
            print(f"\r  converted {i}/{len(specs)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    index, stats = build_index(out_root)
    show = index[index.dataset.isin(args.datasets)]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        cols = ["case_id", "release_split", "scanner", "shape", "spacing_mm",
                "lesion_voxels", "n_lesions", "lesion_volume_ml"]
        print(show[cols].to_string(index=False))
        rater = stats[stats.dataset.isin(args.datasets) & stats.reference.str.startswith("lesion_rater")]
        if len(rater):
            print("\nPer-rater lesion counts (pivot of lesion_stats.csv):")
            print(rater.pivot(index="case_id", columns="reference", values="n_lesions").to_string())
    print(f"\nindex: {out_root / 'index.csv'} ({len(index)} cases total)")
    print(f"stats: {out_root / 'lesion_stats.csv'}")
    if failures:
        print(f"\n{len(failures)} case(s) FAILED:", *failures, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
