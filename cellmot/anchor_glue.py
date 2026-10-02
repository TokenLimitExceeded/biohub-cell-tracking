"""Glue between our detection/tracking data structures and the anchor notebook's post-processing stages
(cellmot.anchor_stages, auto-generated). Bundle-safe (numpy/pandas only)."""
from __future__ import annotations
import io, contextlib
import numpy as np
import pandas as pd


def _stages():
    """The anchor_stages module: cellmot.anchor_stages as a package, or the single-file kernel script itself."""
    import sys
    if "." not in __name__:  # bundled into one script: the stages' functions/constants are globals of __main__
        return sys.modules["__main__"]
    import importlib
    return importlib.import_module(__name__.rsplit(".", 1)[0] + ".anchor_stages")


def set_anchor_option(env_key: str, value):
    """Override a notebook constant by its BIOHUB_* env name (e.g. BIOHUB_GAP_CLOSE_UM=4.0)."""
    AS = _stages()
    name = env_key[len("BIOHUB_"):] if env_key.startswith("BIOHUB_") else env_key
    if not hasattr(AS, name):
        raise KeyError(f"unknown anchor constant {name}")
    cur = getattr(AS, name)
    if isinstance(cur, bool): value = str(value) not in ("0", "False", "false", "")
    elif isinstance(cur, int): value = int(float(value))
    elif isinstance(cur, float): value = float(value)
    setattr(AS, name, value)


def candidate_edges_from_logits(E, gmap, col, pos_um, max_um=None):
    """E rows (t, i_local, j_local, ...logit columns); gmap {(t, local): node index}.
    Returns (gi, gj, prob) with prob = softmax over sources per target of column `col` (the pack's convention)."""
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
    """The pack's non-ILP edge selection: candidates above thr, greedy by probability with per-node caps."""
    order = np.argsort(-prob); ch = {}; pa = {}; keep = []
    for k in order:
        if prob[k] <= thr:
            break
        i, j = int(gi[k]), int(gj[k])
        if ch.get(i, 0) >= max_children or pa.get(j, 0) >= max_parents:
            continue
        keep.append(k); ch[i] = ch.get(i, 0) + 1; pa[j] = pa.get(j, 0) + 1
    return np.array(keep, np.int64)


def to_dicts(nodes: pd.DataFrame, edges: pd.DataFrame, probs=None):
    """Our (node_id,t,z,y,x raw voxels) / (source_id,target_id) frames -> notebook dicts (edge distance_um filled)."""
    AS = _stages()
    nbi = {int(r.node_id): {"node_id": int(r.node_id), "t": int(r.t), "z": float(r.z), "y": float(r.y), "x": float(r.x)} for r in nodes.itertuples()}
    el = []
    for k, r in enumerate(edges.itertuples()):
        s, d = int(r.source_id), int(r.target_id)
        if s not in nbi or d not in nbi: continue
        e = {"source_id": s, "target_id": d, "edge_prob": (None if probs is None else probs[k])}
        e["distance_um"] = AS.edge_distance_um(nbi[s], nbi[d]); el.append(e)
    return nbi, el


def from_dicts(nbi, el):
    nodes = pd.DataFrame([{"node_id": int(n["node_id"]), "t": int(n["t"]), "z": float(n["z"]), "y": float(n["y"]), "x": float(n["x"])} for n in nbi.values()]) if nbi else pd.DataFrame(columns=["node_id", "t", "z", "y", "x"])
    edf = pd.DataFrame([{"source_id": int(e["source_id"]), "target_id": int(e["target_id"])} for e in el]) if el else pd.DataFrame(columns=["source_id", "target_id"])
    return nodes, edf


SAFE_DIV_STATS = ("safe_division_candidates", "safe_division_geometric_candidates", "safe_divisions_added", "safe_division_skipped_cap", "safe_division_mutual_nn_rejected",
                  "safe_division_divergence_rejected", "safe_division_symmetry_rejected", "deepcenter_safe_div_checked", "deepcenter_safe_div_accepted", "deepcenter_safe_div_rejected", "deepcenter_safe_div_missing")


_DC = {"bundles": {}, "loaded": False}


def _dev_key():
    try:
        import torch
        return f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_deepcenter(checkpoint=None, require=False):
    """Enable the anchor's DeepCenter veto (needs torch + the checkpoint from pilkwang/biohub-deepcenter-unet3d-center-prior-v1).
    Loads one model per CUDA device (call it inside each worker thread after torch.cuda.set_device); the stages' own loader
    puts the model on the current device. checkpoint: explicit path (else the stages search /kaggle/input/**/full_frame_center/**)."""
    import os
    AS = _stages()
    if checkpoint:
        os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(checkpoint)
    AS.USE_DEEPCENTER_VETO = True; AS.REQUIRE_DEEPCENTER_VETO = bool(require)
    key = _dev_key()
    with contextlib.redirect_stdout(io.StringIO()) as out:
        try:
            _DC["bundles"][key] = AS.load_deepcenter_veto_detector()
        except Exception as ex:
            _DC["bundles"][key] = None; print("DeepCenter load failed:", repr(ex))
    _DC["loaded"] = True
    return _DC["bundles"][key]


def _dc_bundle():
    return _DC["bundles"].get(_dev_key())


def apply_safe_divisions(nodes, edges, dataset=None):
    """Run the notebook's safe-division stage on our (division-free) graph. Returns (nodes, edges, n_added)."""
    AS = _stages(); nbi, el = to_dicts(nodes, edges); st = {k: 0 for k in SAFE_DIV_STATS}
    with contextlib.redirect_stdout(io.StringIO()):
        el = AS.add_safe_divisions_postlink(nbi, el, st, dataset=dataset, deepcenter_bundle=_dc_bundle(), frame_cache={}, deepcenter_cache={})
    n2, e2 = from_dicts(nbi, el)
    return n2, e2, int(st["safe_divisions_added"])


def apply_full_post(nodes, raw_edges_df, probs, dataset=None):
    """Run the notebook's whole filter_output_graph on nodes + raw candidate edges (with probabilities)."""
    AS = _stages(); nbi, el = to_dicts(nodes, raw_edges_df, probs)
    with contextlib.redirect_stdout(io.StringIO()):
        nbi, el, st = AS.filter_output_graph(nbi, el, dataset=dataset, deepcenter_bundle=_dc_bundle())
    n2, e2 = from_dicts(nbi, el)
    return n2, e2, st
