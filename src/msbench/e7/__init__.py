"""E7 (exploratory, after the results freeze): adapter for MS-SAM-LESS (third_party/MS-SAM-LESS).

Their code is used unmodified; this package supplies our data (E1 folds, FLAIR/T1/T2), the approved
deviations (x8 repeats per epoch, no per-epoch validation, last-epoch checkpoint), verified weight
loading, no-op stand-ins for wandb/TensorBoard, and inference mapped back to each case's own grid.
"""

from pathlib import Path

from msbench.config import REPO_ROOT

UPSTREAM = REPO_ROOT / "third_party" / "MS-SAM-LESS"
UPSTREAM_COMMIT = "dde835c95aa2978030830515cace8479bc0a8eac"
MODALITIES = ("flair", "t1", "t2")
