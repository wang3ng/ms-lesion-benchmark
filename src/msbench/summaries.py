"""Helpers for reports: training-set size-bin counts and nnU-Net training logs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import pandas as pd

from msbench.config import load_config, paths
from msbench.metrics import bin_names
from msbench.metrics.components import component_volumes_mm3, label_lesions


def training_lesion_volumes(cfg: dict | None = None) -> pd.DataFrame:
    """One row per reference lesion in the primary dataset's TRAINING consensus masks.

    Columns: case_id, scanner, volume_mm3, voxel_volume_mm3. Test cases are never read.
    """
    cfg = cfg or load_config()
    processed = paths(cfg)["data_processed"]
    index = pd.read_csv(processed / "index.csv", dtype=str)
    tr = index[(index.dataset == cfg["primary_dataset"]) & (index.release_split == "train")]
    rule = cfg["metrics"]["lesion_detection"]
    rows = []
    for case_id, rel, scanner in zip(tr.case_id, tr.path, tr.scanner):
        img = nib.load(str(processed / rel / "lesion.nii.gz"))
        mask = np.asanyarray(img.dataobj) > 0
        spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
        lab, n = label_lesions(mask, spacing, rule["connectivity"], rule["min_lesion_mm3"])
        for v in component_volumes_mm3(lab, n, spacing):
            rows.append((case_id, scanner, float(v), float(np.prod(spacing))))
    return pd.DataFrame(rows, columns=["case_id", "scanner", "volume_mm3", "voxel_volume_mm3"])


def bin_counts(volumes: pd.DataFrame, edges: Sequence[float], by: str | None = None) -> pd.DataFrame:
    """Lesion counts per volume bin (optionally per `by` column)."""
    names = bin_names(list(edges))
    v = volumes.assign(bin=pd.Categorical([names[i] for i in np.digitize(volumes.volume_mm3, edges)],
                                          categories=names))
    if by is None:
        return v.groupby("bin", observed=False).size().to_frame("lesions").T
    return v.groupby([by, "bin"], observed=False).size().unstack("bin")


@dataclass
class TrainingProgress:
    status: str                  # finished / incomplete / not started
    epochs_completed: int
    train_hours: float           # sum of nnU-Net "Epoch time" (training + per-epoch val)
    mean_epoch_s: float
    last_pseudo_dice: float
    best_ema_pseudo_dice: float
    last_log_time: str


_EPOCH = re.compile(r"^(\S+ \S+): Epoch (\d+)\s*$")
_ETIME = re.compile(r"^(\S+ \S+): Epoch time: ([\d.]+) s")
_PDICE = re.compile(r"Pseudo dice \[(.*)\]")
_BEST = re.compile(r"New best EMA pseudo Dice: ([\d.]+)")


def nnunet_training_progress(fold_dir: Path) -> TrainingProgress:
    """Parse all training_log_*.txt in a fold folder (resumed runs write several)."""
    fold_dir = Path(fold_dir)
    logs = sorted(fold_dir.glob("training_log_*.txt"))
    if not logs:
        return TrainingProgress("not started", 0, 0.0, float("nan"), float("nan"), float("nan"), "")
    epoch_time: dict[int, float] = {}   # last value wins if an epoch was re-run after resume
    last_pd, best, last_time, current = float("nan"), float("nan"), "", None
    for log in logs:
        for line in log.read_text(errors="replace").splitlines():
            line = line.rstrip()
            if m := _EPOCH.match(line):
                current, last_time = int(m.group(2)), m.group(1)
            elif (m := _ETIME.match(line)) and current is not None:
                epoch_time[current], last_time = float(m.group(2)), m.group(1)
            elif m := _PDICE.search(line):
                vals = [float(x) for x in re.findall(r"[\d.]+(?=\)|$)", m.group(1).replace(" ", ""))]
                if vals:
                    last_pd = float(np.mean(vals))
            elif m := _BEST.search(line):
                best = float(m.group(1))
    finished = (fold_dir / "checkpoint_final.pth").exists()
    status = "finished" if finished else "incomplete"   # caller decides if the job is alive
    secs = sum(epoch_time.values())
    n = len(epoch_time)
    return TrainingProgress(status, n, secs / 3600, secs / n if n else float("nan"),
                            last_pd, best, last_time[:19])
