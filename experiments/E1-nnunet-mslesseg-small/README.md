# E1 — Small nnU-Net train on MSLesSeg (william)

Short real-data run to check that download → convert → 2D nnU-Net training
works on the laptop GPU before a full schedule.

## Machine

- Host: `william` (`~/projects/ms-lesion-benchmark`)
- GPU: NVIDIA GeForce RTX 3060 Laptop, 6 GB
- Env: conda `msbench` (Python 3.11, PyTorch CUDA 12.4, nnU-Net v2.8)

## Data

- Source: MSLesSeg Figshare zip (`MSLesSeg Dataset.zip`, ~1.37 GB)
- Used: first 12 train FLAIR/mask pairs after case pairing
- nnU-Net dataset: `Dataset501_MSLesSegSmall`
- Channels: FLAIR only (`_0000`)
- Labels: binary lesion (mask > 0)
- Split: fold 0 → 9 train / 3 validation

Scripts on the machine:

- `scripts/william_setup_env.sh` — Miniconda + torch + nnU-Net
- `scripts/william_smoke_train.sh` — synthetic 5-epoch smoke (earlier)
- `scripts/william_small_train.py` — this run
- `scripts/print_small_train_results.py` — parse `summary.json`

## Training

```text
nnUNetv2_plan_and_preprocess -d 501 -c 2d --verify_dataset_integrity
nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_20epochs
```

Planner chose 2D config with patch `[192, 160]` and batch size 80 (fits in
~4 GB used during training). About 90 s per epoch.

## Results

Full-volume validation Dice after 20 epochs (fold 0):

| Case | Dice | IoU |
| --- | ---: | ---: |
| mslesseg_001 | 0.824 | 0.701 |
| mslesseg_004 | 0.661 | 0.494 |
| mslesseg_009 | 0.282 | 0.164 |
| **Mean** | **0.589** | **0.453** |

Best EMA pseudo-Dice during training (online patch metric): **0.726**.
That number is not comparable to the full-volume mean above.

Artifacts on william (not in git):

- `data/nnUNet_results/Dataset501_MSLesSegSmall/nnUNetTrainer_20epochs__nnUNetPlans__2d/fold_0/`
- `checkpoint_best.pth`, `validation/summary.json`, `training_log_*.txt`
- Log: `~/small_train.log`

## Interpretation

The stack works on real MSLesSeg FLAIR with a 6 GB GPU. Mean Dice 0.59 after
20 epochs and 12 cases is a bring-up baseline, not a project result. The sparse
case (009) pulls the mean down. Next steps: more cases, longer schedule, add
T1/T2, then move `3d_fullres` / MedSAM to a ≥12 GB machine.
