"""Local GPU smoke test of the V1284 predictor patch (both arms) on a few train videos.

Runs the base notebook's own predictor (pristine support pack + cell-6 patch chain, WITHOUT the Trackastra external
patch, which needs the Kaggle runtime) + the V1284 patch, in-process through predict(), with the deployed
environment of intel/day21_nodesel/capture_run.py.  Saves cand_coords / cand_edges / ILP flags per video and the
V1284 sidecar, so that (a) the sidecar arm can be checked byte-for-byte against the day-21 capture and (b) the two
arms' refined coordinates can be compared.

usage: python -X utf8 smoke_run.py --arm full|sidecar --stems 44b6_341df25f,6bba_8b7818bf
"""
from __future__ import annotations

import os

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import v1284_patch  # noqa: E402

ROOT = common.ROOT
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data/train"


def build_repo(arm: str) -> Path:
    repo = common.LANE / f"smoke_{arm}" / "tracking_repo"
    text = common.reconstruct_predictor(repo, with_external=False)
    text = v1284_patch.v1284_patch_predictor(text, arm, require_external=False)
    (repo / "scripts" / "predict_unet_transformer.py").write_text(text, encoding="utf-8")
    (repo / "scripts" / "v1284_coordinate_refinement.py").write_text(common.module_source(), encoding="utf-8")
    shutil.copytree(common.SNAPSHOT_REPO / "weights", repo / "weights")
    return repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["full", "sidecar"], required=True)
    ap.add_argument("--stems", required=True)
    a = ap.parse_args()
    stems = [s for s in a.stems.split(",") if s]
    from intel.day21_nodesel import capture_run as cr
    repo = build_repo(a.arm)
    out = common.LANE / f"smoke_{a.arm}"
    kw = out / "kaggle_working"
    kw.mkdir(parents=True, exist_ok=True)
    for k, v in cr.DEPLOYED_ENV.items():
        os.environ[k] = v
    os.environ.pop("BIOHUB_GPU_SHARD", None)
    os.environ["V1284_MODE"] = "candidate"
    os.environ["V1284_HEAD"] = str(common.HEAD)
    os.environ["BIOHUB_V1284_ARM"] = a.arm
    os.environ["BIOHUB_V1284_REPORT_DIR"] = str(out / "v1284_reports")
    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "scripts"))
    import torch
    torch.set_num_threads(8)
    import tracksdata as td
    import dataspec
    import predict_unet_transformer as put
    assert Path(put.__file__).resolve() == (repo / "scripts" / "predict_unet_transformer.py").resolve()
    real_path = put.Path

    def _redirect(*args):
        p = real_path(*args)
        if len(args) == 1 and str(args[0]).replace("\\", "/") == "/kaggle/working":
            return real_path(kw)
        return p

    put.Path = _redirect
    cr.install_hooks(put, td, torch)
    splits = out / "splits"
    splits.mkdir(exist_ok=True)
    for stem in stems:
        sp = splits / f"{stem}.json"
        sp.write_text(json.dumps([{"split": 0, "train": [], "test": [stem]}]))
        cfg = put.PredictConfig(det_threshold=cr.DET_THRESHOLD, use_ilp=True, ilp_edge_weight=cr.ILP["edge"],
                                ilp_appearance_weight=cr.ILP["app"], ilp_disappearance_weight=cr.ILP["dis"],
                                ilp_division_weight=cr.ILP["div"])
        cr.CAP.reset()
        cr.CAP.active = True
        t0 = time.time()
        put.predict(data_dir=DATA, fold=0, splits_file=sp,
                    weights_path=Path("weights/unet_transformer/split_0/edge_predictor_best.pth"), cfg=cfg,
                    method=f"v1284_{a.arm}_{stem}", debug_video=None, unet_batch_size=cr.UNET_BATCH_SIZE,
                    video_slice=None, evaluate=False)
        wall = time.time() - t0
        cr.CAP.active = False
        il = cr.CAP.ilp or {}
        np.savez_compressed(out / f"{stem}.npz", cand_coords=cr.CAP.coords, cand_edges=cr.CAP.edges,
                            **{f"ilp_{k}": v for k, v in il.items() if isinstance(v, np.ndarray)})
        print("SMOKE_DONE", json.dumps(dict(stem=stem, arm=a.arm, wall_s=round(wall, 1), n_cand=int(len(cr.CAP.coords)),
                                           coords_dtype=str(cr.CAP.coords.dtype), n_edges=int(len(cr.CAP.edges)),
                                           ilp_selected=int(il["sol"].sum()) if il else None)), flush=True)


if __name__ == "__main__":
    main()
