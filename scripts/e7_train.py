"""E7 (exploratory, after the results freeze): train MS-SAM-LESS phase 1 / phase 2 or the HarDNet baseline.

    source scripts/env.sh
    python scripts/e7_train.py --model samless --phase 1 --fold 0
    python scripts/e7_train.py --model samless --phase 2 --fold 0      # needs phase 1 of the fold
    python scripts/e7_train.py --model hardnet --fold 0
    python scripts/e7_train.py --model samless --phase 1 --fold 0 --epochs 2   # smoke test (separate name)

Runs their unmodified main.py (third_party/MS-SAM-LESS) with the published arguments and the deviations in
configs/experiment.yaml e7 (msbench.e7.runtime patches data, validation, weights and logging before their
imports). Output: <e7_runs>/results/<timestamp>_<name>/ckpt/net_last_<last epoch>.pth (+ run.json).
Skips a run whose final checkpoint exists; an unfinished run restarts from scratch.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

from msbench.config import REPO_ROOT, load_config, paths
from msbench.e7 import UPSTREAM, UPSTREAM_COMMIT
from msbench.e7.data import build_split


def run_name(model: str, phase: int | None, fold: int, smoke: bool = False) -> str:
    base = f"samless_p{phase}_fold{fold}" if model == "samless" else f"hardnet_fold{fold}"
    return base + ("_smoke" if smoke else "")


def final_ckpt(root: Path, name: str, epochs: int) -> Path | None:
    hits = sorted(root.glob(f"results/*_{name}/ckpt/net_last_{epochs - 1:05d}.pth"))
    return hits[-1] if hits else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["samless", "hardnet"])
    ap.add_argument("--phase", type=int, choices=[1, 2])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, help="override (smoke tests only; the run gets a _smoke name)")
    ap.add_argument("--dataset", choices=["cv", "e9"], default="cv",
                    help="e9: HarDNet baseline on the E9 combined set (fold0 = all scans, no validation)")
    args = ap.parse_args(argv)
    if args.model == "samless" and args.phase is None:
        ap.error("--phase is required for samless")
    if args.dataset == "e9" and (args.model != "hardnet" or args.fold != 0):
        ap.error("--dataset e9 is for --model hardnet --fold 0 only")
    cfg = load_config()
    e7, p = cfg["e7"], paths(cfg)
    root = p["e9_runs"] if args.dataset == "e9" else p["e7_runs"]
    (root / "cp").mkdir(parents=True, exist_ok=True)
    link = root / "cp" / "sam_vit_b.pth"
    if not link.exists():
        link.symlink_to(p["sam_vit_b"])
    if args.dataset == "e9":
        from msbench.e9 import hardnet_split
        split = hardnet_split(cfg, root / "splits" / "e9_split.json")
    else:
        split = root / "splits" / "e7_split.json"
        build_split(cfg, split)

    part = e7["phase1"] if args.model == "samless" and args.phase == 1 else \
        e7["phase2"] if args.model == "samless" else e7["hardnet_baseline"]
    smoke = args.epochs is not None
    epochs = args.epochs or int(cfg["e9"]["hardnet"]["epochs"] if args.dataset == "e9" else part["epochs"])
    name = run_name(args.model, args.phase, args.fold, smoke)
    if args.dataset == "e9":
        name = name.replace("_fold0", "_e9")
    done = final_ckpt(root, name, epochs)
    if done:
        print(f"already trained: {done}")
        return 0
    c = e7["common"]
    cli = ["--epoches", str(epochs), "-bs", str(c["batch_size"]), "-task", "mslesseg", "--exp_name", name,
           "--split_path", str(split), "--data_dir", "", "-lr", str(c["lr"]), "-WD", str(c["weight_decay"]),
           "--momentum", str(c["momentum"]), "--optim", c["optim"], "--acc_steps", str(c["acc_steps"]),
           "--theashold_discretize", str(c["threshold"]), "--image_key", "image", "--mask_key", "mask",
           "--out_channels", str(c["out_channels"]), "--order", str(c["hardnet_order"]), "--criterion", c["criterion"],
           "--n_fold", str(args.fold), "--seed", str(c["seed"]), "--axcodes", c["axcodes"],
           "--num_slices", str(c["num_slices"]), "--use_cuda_amp", str(c["amp"]),
           "--pos_rand_crop", str(part["pos_rand_crop"]), "--neg_rand_crop", str(part["neg_rand_crop"]),
           "--wandb_mode", "disabled", "--save_train_images", "False", "--save_val_images", "False",
           "--save_every", "1000000", "--cache_rate", str(e7["deviations"]["cache_rate"]), "--nW", "0"]
    if args.model == "samless" and args.phase == 1:
        cli += ["--use_sam", "True", "--sam_ckpt", "sam_vit_b", "--Idim", str(part["prompt_learner_input"])]
    else:
        cli += ["--use_sam", "False", "--use_standard_net", "True", "--net", "HardNetSegmentor",
                "--Idim", str(part["image_dim"]), "--net_segmentor", part["segmentor"],
                "--segmentor_finetune_backbone", str(part["finetune_backbone"])]
        if args.model == "samless":
            p1_epochs = args.epochs or int(e7["phase1"]["epochs"])
            p1 = final_ckpt(root, run_name("samless", 1, args.fold, smoke), p1_epochs)
            if p1 is None:
                raise SystemExit(f"phase 1 of fold {args.fold} is not finished")
            cli += ["--ckpt_path", str(p1)]
    from msbench.e7.runtime import prepare
    prepare(int(e7["deviations"]["repeats_per_epoch"]), float(e7["deviations"]["cache_rate"]), p["hardnet85"])
    os.chdir(root)
    sys.argv = ["main.py", *cli]
    t0 = time.time()
    print(f"E7 {name}: {epochs} epochs; main.py {' '.join(cli)}", flush=True)
    runpy.run_path(str(UPSTREAM / "main.py"), run_name="__main__")
    ck = final_ckpt(root, name, epochs)
    if ck is None:
        raise SystemExit("training ended without a final checkpoint")
    first = ck.parent / "net_last_00000.pth"
    if first.exists() and first != ck:
        first.unlink()                      # their loop also saves epoch 0; only the last epoch is used
    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    (ck.parent.parent / "run.json").write_text(json.dumps({
        "name": name, "model": args.model, "phase": args.phase, "fold": args.fold, "epochs": epochs,
        "minutes": round((time.time() - t0) / 60, 1), "checkpoint": str(ck), "cli": cli, "git_commit": commit,
        "upstream_commit": UPSTREAM_COMMIT, "e7": e7}, indent=2))
    print(f"E7 {name}: done in {(time.time() - t0) / 60:.1f} min -> {ck}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
