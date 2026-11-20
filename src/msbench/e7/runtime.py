"""Run their main.py with our data and the approved deviations, without editing their files."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

from msbench.e7 import UPSTREAM

HARDNET85 = "hardnet85-a28faa00.pth"


def install_stubs() -> None:
    """No-op wandb and TensorBoard (neither is installed; their logging is not needed)."""
    class _Any:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return _Any()
        def __getattr__(self, name): return _Any()
    wb = types.ModuleType("wandb")
    wb.init = wb.log = wb.watch = wb.finish = lambda *a, **k: None
    wb.config = _Any()
    wb.Image = wb.Table = _Any
    wb.plot = _Any()
    sys.modules["wandb"] = wb
    tb = types.ModuleType("torch.utils.tensorboard")
    tb.SummaryWriter = _Any
    sys.modules["torch.utils.tensorboard"] = tb


def verified_hub(hardnet_path: Path):
    """torch.hub.load_state_dict_from_url replacement: only the verified HarDNet-85 file, weights_only."""
    def load(url, *a, map_location=None, **k):
        if not url.endswith(HARDNET85):
            raise RuntimeError(f"E7: refusing to download {url}")
        return torch.load(hardnet_path, map_location=map_location, weights_only=True)
    return load


def no_validation(*a, **k):
    """Stands in for their per-epoch inference_ds_monai (approved: no validation, last-epoch checkpoint)."""
    return 0.0, 0.0, 0.0, 0.0


def prepare(repeats: int, cache_rate: float, hardnet_path: Path) -> None:
    """Patch their modules before main.py imports them."""
    install_stubs()
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    import inference
    import utils.dataset_utils as du
    from msbench.e7.data import get_dataset_e7
    torch.hub.load_state_dict_from_url = verified_hub(hardnet_path)
    du.get_dataset = get_dataset_e7(repeats, cache_rate)
    inference.inference_ds_monai = no_validation
