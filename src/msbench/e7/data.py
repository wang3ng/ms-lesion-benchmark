"""E7 data: split file from E1's folds, FLAIR/T1/T2 training transforms (their pipeline), x8 repeats."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from msbench.config import REPO_ROOT, paths
from msbench.e7 import MODALITIES


def build_split(cfg: dict, out: Path) -> dict:
    """Their split-file format: fold<k>: {train, val} + test, each entry {flair, t1, t2, mask} (absolute paths).

    Folds = experiments/E1/splits_final.json. `test` lists MSSEG test + ISBI (masks: consensus / rater 1);
    it is never used for training or selection (inference uses scripts/e7_predict.py).
    """
    p = paths(cfg)
    index = pd.read_csv(p["data_processed"] / "index.csv", dtype=str).set_index("case_id")
    splits = json.loads((REPO_ROOT / "experiments" / "E1" / "splits_final.json").read_text())

    def entry(c: str, mask: str = "lesion") -> dict:
        d = p["data_processed"] / index.loc[c, "path"]
        return {**{m: str(d / f"{m}.nii.gz") for m in MODALITIES}, "mask": str(d / f"{mask}.nii.gz"), "case_id": c}

    data = {f"fold{k}": {"train": [entry(c) for c in sp["train"]], "val": [entry(c) for c in sp["val"]]}
            for k, sp in enumerate(splits)}
    test = index[((index.dataset == "msseg2016") & (index.release_split == "test")) | (index.dataset == "isbi2015")]
    data["test"] = [entry(c, "lesion" if index.loc[c, "dataset"] == "msseg2016" else "lesion_rater1")
                    for c in sorted(test.index)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    return data


def train_transforms(args: dict, img_dim: int):
    """Their get_mslesseg_transform (training branch) with three modalities stacked into "image".

    Same order and parameters; the only change is that FLAIR, T1 and T2 are each z-scored and scaled to
    [0, 1] (their NormalizeIntensity + ScaleIntensity on the single image key) and then concatenated.
    """
    from monai.transforms import (Compose, ConcatItemsd, DeleteItemsd, EnsureChannelFirstD, EnsureTyped,
                                  LoadImageD, NormalizeIntensityD, OrientationD, RandCropByPosNegLabelD,
                                  RandFlipD, RandRotate90D, ResizeWithPadOrCropd, ScaleIntensityD, SpacingD)
    from dataset.tfs import ResizeWithRatio   # theirs (third_party on sys.path)
    mods, mk = list(MODALITIES), args["mask_key"]
    keys = mods + [mk]
    return Compose([
        LoadImageD(keys),
        EnsureChannelFirstD(keys=keys),
        NormalizeIntensityD(keys=mods),
        ScaleIntensityD(keys=mods),
        SpacingD(keys, pixdim=(1., 1., 1.), mode=("bilinear",) * len(mods) + ("nearest",)),
        OrientationD(keys, axcodes=args["axcodes"]),
        ConcatItemsd(mods, name="image"),
        DeleteItemsd(mods),
        RandCropByPosNegLabelD(["image", mk], label_key=mk, spatial_size=(-1, -1, args["num_slices"]),
                               pos=args["pos_rand_crop"], neg=args["neg_rand_crop"]),
        EnsureTyped(keys=["image", mk], device=args["device"]),
        ResizeWithRatio(keys=["image", mk], image_size=img_dim, mode="trilinear"),
        ResizeWithPadOrCropd(keys=["image", mk], spatial_size=(img_dim, img_dim, -1), mode="constant"),
        RandFlipD(["image", mk], prob=0.5, spatial_axis=0),
        RandRotate90D(["image", mk], prob=0.5, spatial_axes=(0, 1)),
    ])


class RepeatDataset(torch.utils.data.Dataset):
    """Each training scan `repeats` times per epoch (a new random slice each time)."""

    def __init__(self, base, repeats: int):
        self.base, self.repeats = base, int(repeats)

    def __len__(self):
        return len(self.base) * self.repeats

    def __getitem__(self, i):
        return self.base[i % len(self.base)]


def get_dataset_e7(repeats: int, cache_rate: float):
    """Replacement for their utils.dataset_utils.get_dataset (training loader only; no validation)."""
    def get_dataset(args, transform, img_dim):
        from dataset.MSLesSeg import MSLesSeg
        ds = MSLesSeg(args["data_dir"], args["split_path"], args["mask_key"], section="train",
                      num_fold=args["n_fold"], transforms=train_transforms(args, img_dim), cache_rate=cache_rate)
        train = RepeatDataset(ds, repeats)
        dl = torch.utils.data.DataLoader(train, batch_size=int(args["Batch_size"]), shuffle=True,
                                         num_workers=int(args["nW"]), drop_last=True)
        return dl, None
    return get_dataset
