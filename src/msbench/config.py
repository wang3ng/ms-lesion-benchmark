"""Load configs/experiment.yaml and resolve its paths.

``python -m msbench.config --shell`` prints ``export`` lines used by
scripts/env.sh, so shell scripts and Python read the same config.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    """Absolute path; relative paths are taken from the repository root."""
    p = Path(p).expanduser()
    return (p if p.is_absolute() else REPO_ROOT / p).resolve()


def paths(cfg: dict | None = None) -> dict[str, Path]:
    cfg = cfg or load_config()
    return {k: resolve_path(v) for k, v in cfg["paths"].items()}


def nnunet_dataset_name(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    nn = cfg["models"]["nnunet"]
    return f"Dataset{int(nn['dataset_id']):03d}_{nn['dataset_name']}"


def shell_exports(cfg: dict | None = None) -> dict[str, str]:
    cfg = cfg or load_config()
    p = paths(cfg)
    nn_root = p["nnunet_root"]
    return {
        "MSBENCH_REPO": str(REPO_ROOT),
        "MSBENCH_DATA_RAW": str(p["data_raw"]),
        "MSBENCH_DATA_PROCESSED": str(p["data_processed"]),
        "MSBENCH_LOGS": str(p["logs"]),
        "MSBENCH_SEED": str(cfg["seed"]),
        "MSBENCH_NNUNET_ID": str(int(cfg["models"]["nnunet"]["dataset_id"])),
        "MSBENCH_NNUNET_DATASET": nnunet_dataset_name(cfg),
        "MSBENCH_NNUNET_CONFIG": cfg["models"]["nnunet"]["configuration"],
        "nnUNet_raw": str(nn_root / "nnUNet_raw"),
        "nnUNet_preprocessed": str(nn_root / "nnUNet_preprocessed"),
        "nnUNet_results": str(nn_root / "nnUNet_results"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shell", action="store_true", help="print export lines for bash")
    args = ap.parse_args()
    for k, v in shell_exports().items():
        print(f"export {k}={shlex.quote(v)}" if args.shell else f"{k}={v}")


if __name__ == "__main__":
    main()
