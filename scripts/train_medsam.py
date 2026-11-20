"""Fine-tune MedSAM for one box condition and one fold, then predict that fold's validation patients.

    source scripts/env.sh
    python scripts/train_medsam.py --exp E2a --fold 0          # train (or resume), then predict
    python scripts/train_medsam.py --exp E2c --fold 0 --benchmark 60   # time 60 steps + 1 case, estimate
    python scripts/train_medsam.py --exp E2c --fold 1 --benchmark 3000 --dice-on all_boxes --no-predict
        # replay the first E2c loss up to the collapse check, in a temporary dir (nothing kept)

Folds are E1's (experiments/E1/splits_final.json): the 12 training patients of
fold k train the prompt encoder + mask decoder, the 3 validation patients are
predicted with the same box type (E2a oracle from the consensus, E2c brain box +
tiles, E2b boxes from E1 out-of-fold predictions, E8 brain box + zoomed tiles). Output:
<medsam_runs>/<exp>/fold_<k>/{checkpoint_final.pt, run.json, validation/<case>.nii.gz}.
Score with: python scripts/evaluate.py --exp <exp> --fold k --pred-dir .../validation
Settings: configs/experiment.yaml (models.medsam.finetune, .inference, boxes).

Collapse check (training data only, models.medsam.finetune.collapse_check): if it fails, training
stops at that step, nothing is predicted, <run_dir>/collapse_check.json says why and the exit code
is 3. A fold with a failed check is not resumed; delete its directory to retry.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from msbench.config import REPO_ROOT, load_config, paths
from msbench.medsam.model import load_medsam
from msbench.medsam.train import CONDITIONS, DICE_ON, BoxSampler, load_case, load_trainable, predict_case, save_prediction, train


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, choices=list(CONDITIONS))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--benchmark", type=int, metavar="STEPS", help="time STEPS steps + one validation case, no outputs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dice-on", choices=DICE_ON, help="override finetune.dice_on (only with --benchmark)")
    ap.add_argument("--no-predict", action="store_true", help="with --benchmark: skip the timing prediction")
    args = ap.parse_args(argv)
    if args.dice_on and not args.benchmark:
        ap.error("--dice-on is only allowed with --benchmark (real runs use the config)")

    cfg = load_config()
    cond = CONDITIONS[args.exp]
    split = json.loads((REPO_ROOT / "experiments" / "E1" / "splits_final.json").read_text())[args.fold]
    seed = int(cfg["seed"]) + 100 * args.fold
    oof = Path(os.environ.get("MSBENCH_RESULTS", REPO_ROOT / "results")) / "E1" / "oof_predictions"
    box_dir = oof if cond == "nnunet" else None
    zoom = cond == "zoomed"
    run_dir = paths(cfg)["medsam_runs"] / args.exp / f"fold_{args.fold}"

    log(f"{args.exp} ({cond} boxes) fold {args.fold}: {len(split['train'])} train / {len(split['val'])} val patients")
    t = time.time()
    train_cases = [load_case(c, cfg, with_ref=True, box_source_dir=box_dir, zoom=zoom) for c in split["train"]]
    log(f"loaded training cases in {time.time() - t:.0f} s; "
        f"pools: {BoxSampler(train_cases, cond, cfg, seed).counts()}")

    if args.benchmark:
        tmp = Path(tempfile.mkdtemp(prefix="medsam_bench_"))
        try:
            r = train(train_cases, cond, cfg, tmp, seed, steps=args.benchmark, log=log, num_workers=args.workers,
                      dice_on=args.dice_on)
            t_case = 0.0
            if not args.no_predict and not r["collapsed"]:
                model = load_medsam(cfg)
                load_trainable(model, torch.load(tmp / "checkpoint_final.pt", weights_only=False))
                cd = load_case(split["val"][0], cfg, with_ref=True, box_source_dir=box_dir, zoom=zoom)
                t = time.time()
                predict_case(model, cd, cond, cfg, seed)
                t_case = time.time() - t
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if r["collapse_check"] is not None:
            print(json.dumps({"collapse_check": r["collapse_check"]}, indent=2))
        steps = int(cfg["models"]["medsam"]["finetune"]["steps"])
        # first steps include worker start-up; use the timing of the second half
        per_fold = steps * r["s_per_step"] + t_case * len(split["val"])
        print(json.dumps({"exp": args.exp, "s_per_step": round(r["s_per_step"], 4),
                          "predict_s_per_case": round(t_case, 1), "steps_per_fold": steps,
                          "est_min_per_fold": round(per_fold / 60, 1), "est_h_5_folds": round(5 * per_fold / 3600, 2)}))
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / "checkpoint_final.pt"
    info = {}
    check_file = run_dir / "collapse_check.json"
    if check_file.exists() and not json.loads(check_file.read_text())["passed"]:
        log(f"collapse check failed earlier ({check_file}); not resuming. Delete {run_dir} to retry.")
        return 3
    if final.exists():
        log(f"already trained: {final}")
    else:
        info = train(train_cases, cond, cfg, run_dir, seed, log=log, num_workers=args.workers)
        if info["collapsed"]:
            log(f"COLLAPSE: training stopped at step {info['steps']}; fold not predicted. See {check_file}")
            return 3
        log(f"training done: {info['steps']} steps, {info['seconds'] / 60:.1f} min")
    del train_cases

    model = load_medsam(cfg)
    model.image_encoder = None
    load_trainable(model, torch.load(final, weights_only=False))
    model.eval()
    val_dir = run_dir / "validation"
    for c in split["val"]:
        if (val_dir / f"{c}.nii.gz").exists():
            continue
        t = time.time()
        cd = load_case(c, cfg, with_ref=(cond == "oracle"), box_source_dir=box_dir, zoom=zoom)
        save_prediction(predict_case(model, cd, cond, cfg, seed), c, cfg, val_dir)
        log(f"predicted {c} in {time.time() - t:.0f} s")

    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    run_json = run_dir / "run.json"
    prev = json.loads(run_json.read_text()) if run_json.exists() else {}
    prev.update({"exp": args.exp, "condition": cond, "fold": args.fold, "train": split["train"], "val": split["val"],
                 "seed": seed, "finetune": cfg["models"]["medsam"]["finetune"],
                 "inference": cfg["models"]["medsam"]["inference"], "boxes": cfg["boxes"], "git_commit": commit,
                 "collapse_check": json.loads(check_file.read_text()) if check_file.exists() else None,
                 **({"train_info": info} if info else {})})
    run_json.write_text(json.dumps(prev, indent=2))
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
