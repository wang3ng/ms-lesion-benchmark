"""E9 (exploratory, after the results freeze) steps called by scripts/run_E9.sh.

    python scripts/e9_steps.py export-nnunet          # nnUNet_raw/Dataset502_MSSEG_MSLesSeg
    python scripts/e9_steps.py e2c [--steps N]        # MedSAM E2c on the combined set (+ collapse check)
    python scripts/e9_steps.py predict-e2c            # E2c model -> test sets
    python scripts/e9_steps.py predict-hardnet        # HarDNet E9 model -> test sets
    python scripts/e9_steps.py counts                 # training-set counts (scans, patients, scanners)

Test predictions -> results/<exp>/test_predictions/<dataset>/<case>.nii.gz (+ manifest.csv); nothing is scored here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from msbench.config import REPO_ROOT, load_config, paths

RESULTS = Path(os.environ.get("MSBENCH_RESULTS", REPO_ROOT / "results"))


def log(m):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}", flush=True)


def test_cases(cfg) -> pd.DataFrame:
    ix = pd.read_csv(paths(cfg)["data_processed"] / "index.csv", dtype=str)
    return ix[((ix.dataset == "msseg2016") & (ix.release_split == "test")) | (ix.dataset == "isbi2015")].sort_values("case_id")


def write_manifest(root: Path, cases: pd.DataFrame, exp: str) -> None:
    pd.DataFrame([{"case_id": r.case_id, "dataset": r.dataset, "scanner": r.scanner,
                   "prediction": f"{r.dataset}/{r.case_id}.nii.gz", "models": 1} for r in cases.itertuples()]
                 ).to_csv(root / "manifest.csv", index=False)
    (root / "README.txt").write_text(f"{exp}: single model trained on the E9 combined set (exploratory). "
                                     "Scored only by scripts/e9_summary.py.\n")


def cmd_counts(cfg, args):
    from msbench.e9 import excluded_cases, training_cases
    tr, ex = training_cases(cfg), excluded_cases(cfg)
    out = {"training": {"scans": len(tr), "patients": int(tr.subject_id.nunique()),
                        "by_dataset": tr.groupby("dataset").agg(scans=("case_id", "size"), patients=("subject_id", "nunique")).to_dict("index"),
                        "by_scanner": tr.scanner.value_counts().to_dict()},
           "excluded_mslesseg": {"scans": len(ex), "patients": int(ex.subject_id.nunique()), "by_scanner": ex.scanner.value_counts().to_dict()}}
    print(json.dumps(out, indent=1))
    (RESULTS / "E9").mkdir(parents=True, exist_ok=True)
    (RESULTS / "E9" / "training_set.json").write_text(json.dumps(out, indent=1))
    return 0


def cmd_export_nnunet(cfg, args):
    from msbench.e9 import export_nnunet
    out = export_nnunet(cfg, Path(os.environ["nnUNet_raw"]))
    log(f"exported {json.loads((out / 'dataset.json').read_text())['numTraining']} cases -> {out}")
    return 0


def cmd_e2c(cfg, args):
    from msbench.e9 import medsam_steps, training_cases
    from msbench.medsam.train import load_case, train
    tr = training_cases(cfg)
    run_dir = paths(cfg)["e9_runs"] / "E9_E2c"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "checkpoint_final.pt").exists():
        log("already trained")
        return 0
    check = run_dir / "collapse_check.json"
    if check.exists() and not json.loads(check.read_text())["passed"]:
        log("collapse check failed earlier; not resuming")
        return 3
    if args.prewarm:   # read the embedding files once so random training reads hit the page cache
        t = time.time()
        for r in tr.itertuples():
            with open(paths(cfg)["medsam_cache"] / r.dataset / f"{r.case_id}.npy", "rb") as f:
                while f.read(64 << 20):
                    pass
        log(f"page cache warmed in {time.time() - t:.0f} s")
    steps = args.steps or medsam_steps(cfg, len(tr))
    seed = int(cfg["seed"])
    cases = [load_case(c, cfg, with_ref=True) for c in tr.case_id]
    log(f"E9 E2c: {len(cases)} training scans, {steps} steps (same updates per scan as the 12-scan CV folds)")
    info = train(cases, "independent", cfg, run_dir, seed, steps=steps, log=log)
    (run_dir / "run.json").write_text(json.dumps({"scans": len(cases), "steps": steps, "seed": seed,
                                                  "train_info": {k: v for k, v in info.items() if k != "collapse_check"},
                                                  "finetune": cfg["models"]["medsam"]["finetune"]}, indent=2, default=str))
    if info["collapsed"]:
        log("COLLAPSE: training stopped; see collapse_check.json")
        return 3
    log(f"done: {info['steps']} steps in {info['seconds'] / 60:.1f} min")
    return 0


def cmd_predict_e2c(cfg, args):
    from msbench.medsam.model import load_medsam
    from msbench.medsam.train import load_case, load_trainable, predict, save_prediction
    ck = paths(cfg)["e9_runs"] / "E9_E2c" / "checkpoint_final.pt"
    if not ck.exists():
        log(f"missing {ck}")
        return 1
    m = load_medsam(cfg)
    m.image_encoder = None
    load_trainable(m, torch.load(ck, weights_only=False))
    m.eval()
    root = RESULTS / "E9_E2c" / "test_predictions"
    cases = test_cases(cfg)
    for r in cases.itertuples():
        f = root / r.dataset / f"{r.case_id}.nii.gz"
        if f.exists():
            continue
        cd = load_case(r.case_id, cfg, with_ref=False)
        save_prediction(predict([m], cd, "independent", cfg, int(cfg["seed"]))["mask"], r.case_id, cfg, root / r.dataset)
    write_manifest(root, cases, "E9_E2c")
    log(f"E9_E2c: {len(cases)} test predictions")
    return 0


def cmd_predict_hardnet(cfg, args):
    import importlib.util
    from msbench.e7.predict import load_segmentor
    from msbench.medsam.train import save_prediction
    spec = importlib.util.spec_from_file_location("e7p", REPO_ROOT / "scripts" / "e7_predict.py")
    e7p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e7p)
    ep = int(cfg["e9"]["hardnet"]["epochs"])
    hits = sorted(paths(cfg)["e9_runs"].glob(f"results/*_hardnet_e9/ckpt/net_last_{ep - 1:05d}.pth"))
    if not hits:
        log("missing HarDNet E9 checkpoint")
        return 1
    m = load_segmentor(hits[-1], int(cfg["e7"]["phase2"]["image_dim"]))
    index = pd.read_csv(paths(cfg)["data_processed"] / "index.csv", dtype=str).set_index("case_id")
    root = RESULTS / "E9_hardnet" / "test_predictions"
    cases = test_cases(cfg)
    for r in cases.itertuples():
        f = root / r.dataset / f"{r.case_id}.nii.gz"
        if not f.exists():
            save_prediction(e7p.predict_case([m], r.case_id, index, cfg), r.case_id, cfg, root / r.dataset)
    write_manifest(root, cases, "E9_hardnet")
    log(f"E9_hardnet: {len(cases)} test predictions")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["counts", "export-nnunet", "e2c", "predict-e2c", "predict-hardnet"])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--no-prewarm", dest="prewarm", action="store_false")
    args = ap.parse_args(argv)
    cfg = load_config()
    return {"counts": cmd_counts, "export-nnunet": cmd_export_nnunet, "e2c": cmd_e2c,
            "predict-e2c": cmd_predict_e2c, "predict-hardnet": cmd_predict_hardnet}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
