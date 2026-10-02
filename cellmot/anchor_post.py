"""Local replica of the public anchor notebook's post-processing, executed from the exported notebook cells
(intel/anchor_cell0.py = BIOHUB_* env config, intel/anchor_cell2.py = config parsing, intel/anchor_cell5.py = graph surgery).

load_anchor() returns a namespace with the notebook's functions (filter_output_graph, motion_relink_edges, ...) and its
parsed globals, with DeepCenter disabled (fail-open) and TEST_DIR pointing at the local raw test videos (synthetic gap
midpoints are intensity-refined only for videos whose raw zarr exists locally; otherwise the plain midpoint is used).
"""
from __future__ import annotations
import os, sys, types, io, contextlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_NS_CACHE = {}


def load_anchor(env_overrides: dict | None = None, test_dir: str = "I:/kaggle/biohub/data/test", quiet: bool = True):
    key = (tuple(sorted((env_overrides or {}).items())), test_dir)
    if key in _NS_CACHE:
        return _NS_CACHE[key]
    if "IPython" not in sys.modules:  # cell 2 imports IPython.display.display
        ip = types.ModuleType("IPython"); disp = types.ModuleType("IPython.display"); disp.display = print
        ip.display = disp; sys.modules["IPython"] = ip; sys.modules["IPython.display"] = disp
    ns = {"__name__": "anchor_replica"}
    out = io.StringIO()
    with contextlib.redirect_stdout(out if quiet else sys.stdout):
        exec(compile((ROOT / "intel" / "anchor_cell0.py").read_text(encoding="utf-8"), "anchor_cell0", "exec"), ns)
        os.environ["BIOHUB_REQUIRE_DEEPCENTER_VETO"] = "0"; os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "0"
        os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"
        for k, v in (env_overrides or {}).items():
            os.environ[k] = str(v)
        src2 = (ROOT / "intel" / "anchor_cell2.py").read_text(encoding="utf-8"); src2 = src2[: src2.index("CONFIG_DISPLAY = {")]
        exec(compile(src2, "anchor_cell2", "exec"), ns)
        ns["TEST_DIR"] = Path(test_dir)
        src5 = (ROOT / "intel" / "anchor_cell5.py").read_text(encoding="utf-8"); src5 = src5[: src5.index("DEEPCENTER_VETO_DETECTOR =")]
        exec(compile(src5, "anchor_cell5", "exec"), ns)
    _NS_CACHE[key] = ns
    return ns


def candidate_edges_from_logits(E, gmap, col, pos_um, max_um=None):
    """E rows (t, i_local, j_local, logit_primary, logit_blend[, rev...]); gmap {(t, local): node index}.
    Returns (gi, gj, prob) with prob = softmax over sources per target (the pack's convention, dim 0)."""
    if len(E) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    t0 = E[:, 0].astype(np.int64); il = E[:, 1].astype(np.int64); jl = E[:, 2].astype(np.int64)
    gi = np.array([gmap.get((int(a), int(b)), -1) for a, b in zip(t0, il)], np.int64)
    gj = np.array([gmap.get((int(a) + 1, int(b)), -1) for a, b in zip(t0, jl)], np.int64)
    m = (gi >= 0) & (gj >= 0)
    if max_um is not None:
        d = np.full(len(gi), np.inf); d[m] = np.linalg.norm(pos_um[gi[m]] - pos_um[gj[m]], axis=1); m &= d <= max_um
    gi, gj, lg = gi[m], gj[m], E[m, col].astype(np.float64)
    if len(lg) == 0:
        return gi, gj, lg
    n = int(max(gi.max(), gj.max())) + 1
    mx = np.full(n, -np.inf); np.maximum.at(mx, gj, lg); e = np.exp(lg - mx[gj]); den = np.zeros(n); np.add.at(den, gj, e)
    return gi, gj, e / den[gj]


def greedy_candidates(gi, gj, prob, thr, max_parents=1, max_children=2):
    """The pack's non-ILP selection: candidates above thr, greedy by probability with per-node caps."""
    order = np.argsort(-prob); ch = {}; pa = {}; keep = []
    for k in order:
        if prob[k] <= thr:
            break
        i, j = int(gi[k]), int(gj[k])
        if ch.get(i, 0) >= max_children or pa.get(j, 0) >= max_parents:
            continue
        keep.append(k); ch[i] = ch.get(i, 0) + 1; pa[j] = pa.get(j, 0) + 1
    return np.array(keep, np.int64)


def run_post(ns, t, pos_vox, raw_edges, dataset=None):
    """t (N,), pos_vox (N,3) original-voxel coords, raw_edges list of (i, j, prob) over node indices.
    Returns (nodes_df, edges_df, stats) in our metric format (node ids 1-based)."""
    import pandas as pd
    nodes_by_id = {i + 1: {"node_id": i + 1, "t": int(t[i]), "z": float(pos_vox[i, 0]), "y": float(pos_vox[i, 1]), "x": float(pos_vox[i, 2])} for i in range(len(t))}
    edges_in = [{"source_id": int(i) + 1, "target_id": int(j) + 1, "edge_prob": (None if p is None else float(p))} for i, j, p in raw_edges]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        nodes_by_id, edges, stats = ns["filter_output_graph"](nodes_by_id, edges_in, dataset=dataset, deepcenter_bundle=None)
    nodes = pd.DataFrame([{"node_id": int(n["node_id"]), "t": int(n["t"]), "z": max(0.0, round(float(n["z"]))), "y": max(0.0, round(float(n["y"]))), "x": max(0.0, round(float(n["x"])))} for n in nodes_by_id.values()])
    edf = pd.DataFrame([{"source_id": int(e["source_id"]), "target_id": int(e["target_id"])} for e in edges]) if edges else pd.DataFrame(columns=["source_id", "target_id"])
    return nodes, edf, stats
