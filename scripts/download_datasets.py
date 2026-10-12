"""Download public MS lesion archives into data/raw/.

The zip files are gitignored. Commit this script, not the archives.
URLs and expected sizes live in configs/downloads.yaml.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "downloads.yaml"
RAW = ROOT / "data" / "raw"
CHUNK = 1024 * 1024
UA = "msbench-download/0.1"


def load_manifest() -> dict:
    with MANIFEST.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["datasets"]


def list_datasets(datasets: dict) -> None:
    for key, spec in datasets.items():
        required = [f for f in spec["files"] if not f.get("optional")]
        nbytes = sum(f["bytes"] for f in required)
        print(f"{key:12}  {nbytes / 1e9:.2f} GB required  {spec['title']}")
        print(f"             {spec['page']}")


def download_file(url: str, dest: Path, expected: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == expected:
        print(f"skip  {dest.name}")
        return

    partial = dest.with_suffix(dest.suffix + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": UA}
    if 0 < existing < expected:
        headers["Range"] = f"bytes={existing}-"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        if status == 200 and existing:
            partial.write_bytes(b"")
            existing = 0
        done = existing
        with partial.open("ab") as out:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                out.write(block)
                done += len(block)
                print(
                    f"\r{dest.name}  {done / 1e6:.0f}/{expected / 1e6:.0f} MB",
                    end="",
                    flush=True,
                )
    print()
    got = partial.stat().st_size
    if got != expected:
        raise SystemExit(
            f"{dest.name} is {got} bytes, manifest expects {expected}. "
            f"Partial file left at {partial}."
        )
    partial.replace(dest)


def write_source(dataset_id: str, spec: dict, chosen: list[dict]) -> None:
    lines = [
        f"dataset: {dataset_id}",
        f"title: {spec['title']}",
        f"page: {spec['page']}",
        f"downloaded_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "files:",
    ]
    for item in chosen:
        lines.append(f"  - {item['name']}")
        lines.append(f"    url: {item['url']}")
        lines.append(f"    bytes: {item['bytes']}")
    path = RAW / dataset_id / "SOURCE.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fetch(dataset_id: str, spec: dict, include_optional: bool) -> None:
    print(f"\n{dataset_id}: {spec['title']}")
    print(spec["license"].strip())
    chosen = []
    for item in spec["files"]:
        if item.get("optional") and not include_optional:
            print(f"skip  {item['name']} (optional)")
            continue
        dest = RAW / dataset_id / item["name"]
        download_file(item["url"], dest, int(item["bytes"]))
        chosen.append(item)
    write_source(dataset_id, spec, chosen)


def main(argv: list[str] | None = None) -> int:
    datasets = load_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=sorted(datasets),
        help="dataset ids to fetch; omit to print the list",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="also download files marked optional (MSLesSeg raw volumes)",
    )
    args = parser.parse_args(argv)
    if not args.datasets:
        list_datasets(datasets)
        print("\nExample: python scripts/download_datasets.py isbi2015 mslesseg")
        return 0
    for dataset_id in args.datasets:
        fetch(dataset_id, datasets[dataset_id], args.include_optional)
    print(f"\nArchives are in {RAW}. They are gitignored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
