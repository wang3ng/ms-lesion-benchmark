#!/usr/bin/env python3
import json
from pathlib import Path

base = (
    Path.home()
    / "projects/ms-lesion-benchmark/data/nnUNet_results"
    / "Dataset501_MSLesSegSmall"
    / "nnUNetTrainer_20epochs__nnUNetPlans__2d"
    / "fold_0"
)
paths = list(base.rglob("summary.json"))
print("summaries", paths)
p = paths[0]
d = json.loads(p.read_text())
print("top keys", list(d.keys()))

res = d.get("metric_per_case")
container = d
if res is None and isinstance(d.get("results"), dict):
    container = d["results"]
    res = container.get("metric_per_case")
if res is None:
    for k, v in d.items():
        if isinstance(v, dict) and "metric_per_case" in v:
            container = v
            res = v["metric_per_case"]
            break

for key in ("mean", "foreground_mean"):
    if key in container:
        print(key, container[key])

if res:
    for case in res:
        metrics = case.get("metrics", case)
        ref = Path(str(case.get("reference_file", "unknown"))).stem
        if "1" in metrics:
            m = metrics["1"]
            print(
                f"{ref}: Dice={m.get('Dice'):.4f} IoU={m.get('IoU'):.4f} "
                f"TP={m.get('TP')} FP={m.get('FP')} FN={m.get('FN')}"
            )

log = base / "training_log_2026_11_9_18_45_34.txt"
best = None
for line in log.read_text().splitlines():
    if "New best EMA pseudo Dice" in line:
        best = line.split(":")[-1].strip()
print("best_ema_pseudo_dice_during_train", best)
