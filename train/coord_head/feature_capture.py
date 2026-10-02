"""Capture the V1284 input features (frozen primary UNet features, 7 points x 32 ch = 224) at every detection that
lies within 7 um of a GT node, on train videos, through the deployed predictor (sidecar arm = unchanged pipeline).

Per video it writes LANE/features/<stem>.npz with, for the selected detections: t, grid (z,y,x on the stride-4
detector grid), features (float32 224), and the public V1284 refined position (full-resolution voxels).
GT is read only to choose which detections to keep (all detections within 7 um of a GT node of the same frame).

usage: python -X utf8 feature_capture.py --stems-file <file> [--limit N]
"""
from __future__ import annotations

import os

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import smoke_run  # noqa: E402

ROOT = common.ROOT
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data/train"
SCALE = np.array([1.625, 0.40625, 0.40625])
DS = np.array([1, 4, 4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems-file", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    stems = [s.strip() for s in Path(a.stems_file).read_text().splitlines() if s.strip()]
    if a.limit:
        stems = stems[: a.limit]
    from intel.day21_nodesel import capture_run as cr
    from scipy.spatial import cKDTree
    out = common.LANE / "features"
    out.mkdir(parents=True, exist_ok=True)
    repo = common.LANE / "featcap" / "tracking_repo"
    if not (repo / "scripts" / "v1284_coordinate_refinement.py").exists():
        text = common.reconstruct_predictor(repo, with_external=False)
        import v1284_patch
        text = v1284_patch.v1284_patch_predictor(text, "sidecar", require_external=False)
        (repo / "scripts" / "predict_unet_transformer.py").write_text(text, encoding="utf-8")
        (repo / "scripts" / "v1284_coordinate_refinement.py").write_text(common.module_source(), encoding="utf-8")
        import shutil
        shutil.copytree(common.SNAPSHOT_REPO / "weights", repo / "weights")
    kw = common.LANE / "featcap" / "kaggle_working"
    kw.mkdir(parents=True, exist_ok=True)
    for k, v in cr.DEPLOYED_ENV.items():
        os.environ[k] = v
    os.environ.pop("BIOHUB_GPU_SHARD", None)
    os.environ["V1284_MODE"] = "candidate"
    os.environ["V1284_HEAD"] = str(common.HEAD)
    os.environ["BIOHUB_V1284_ARM"] = "sidecar"
    os.environ["BIOHUB_V1284_REPORT_DIR"] = str(common.LANE / "featcap" / "v1284_reports")
    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "scripts"))
    import torch
    torch.set_num_threads(8)
    import predict_unet_transformer as put
    import v1284_coordinate_refinement as vref
    real_path = put.Path

    def _redirect(*args):
        p = real_path(*args)
        if len(args) == 1 and str(args[0]).replace("\\", "/") == "/kaggle/working":
            return real_path(kw)
        return p

    put.Path = _redirect
    gt_all = pd.read_parquet(ROOT / "remote_outputs/gt_export/gt_nodes.parquet", filters=[("dataset", "in", stems)])
    state = {}
    base_refine = put._v1284_refine

    def capture_refine(ds_path, t, arr, feature):
        refined = base_refine(ds_path, t, arr, feature)
        gt = state["gt"].get(int(t))
        if gt is not None and len(arr):
            det_um = arr[:, 1:].astype(np.float64) * DS * SCALE
            near = cKDTree(gt).query(det_um, distance_upper_bound=7.0)[0] <= 7.0
            if near.any():
                x = vref.sample_features(feature, arr[near]).float().cpu().numpy()
                state["rows"].append((np.full(int(near.sum()), int(t), np.int16), np.asarray(arr[near][:, 1:], np.int16),
                                      x.astype(np.float32), np.asarray(refined[near], np.float32)))
        return refined

    put._v1284_refine = capture_refine
    splits = common.LANE / "featcap" / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        dst = out / f"{stem}.npz"
        if dst.exists():
            continue
        g = gt_all[gt_all.dataset == stem]
        state["gt"] = {int(t): gf[["z", "y", "x"]].to_numpy(np.float64) * SCALE for t, gf in g.groupby("t")}
        state["rows"] = []
        sp = splits / f"{stem}.json"
        sp.write_text(json.dumps([{"split": 0, "train": [], "test": [stem]}]))
        cfg = put.PredictConfig(det_threshold=cr.DET_THRESHOLD, use_ilp=False, ilp_edge_weight=cr.ILP["edge"],
                                ilp_appearance_weight=cr.ILP["app"], ilp_disappearance_weight=cr.ILP["dis"],
                                ilp_division_weight=cr.ILP["div"])
        t0 = time.time()
        put.predict(data_dir=DATA, fold=0, splits_file=sp,
                    weights_path=Path("weights/unet_transformer/split_0/edge_predictor_best.pth"), cfg=cfg,
                    method=f"featcap_{stem}", debug_video=None, unet_batch_size=cr.UNET_BATCH_SIZE,
                    video_slice=None, evaluate=False)
        rows = state["rows"]
        np.savez_compressed(dst, t=np.concatenate([r[0] for r in rows]) if rows else np.empty(0, np.int16),
                            grid=np.concatenate([r[1] for r in rows]) if rows else np.empty((0, 3), np.int16),
                            features=np.concatenate([r[2] for r in rows]) if rows else np.empty((0, 224), np.float32),
                            v1284_refined=np.concatenate([r[3] for r in rows]) if rows else np.empty((0, 4), np.float32))
        print("FEATCAP", json.dumps(dict(stem=stem, wall_s=round(time.time() - t0, 1), rows=int(sum(len(r[0]) for r in rows)),
                                        gt_nodes=int(len(g)))), flush=True)


if __name__ == "__main__":
    main()
