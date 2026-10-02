"""Confidence-based component pruning to exploit the metric's node-count term.

The competition metric uses adjusted = max(0, J * (1 - 0.1 * (T_pred - T_true) / T_true)), which is UNBOUNDED above for
T_pred < T_true: predicting fewer nodes than the true cell count multiplies the edge Jaccard upward. Edges between two
predicted nodes that both fail to match a GT node are ignored by the metric, so removing low-confidence tracks costs
little Jaccard while lowering T_pred.

prune_components ranks weakly-connected components by a confidence key and removes the weakest until `frac` of the nodes
are gone (components containing a division are kept unless keep_divisions=False).

prune_weak_components is the ABSOLUTE-threshold form measured in intel/next_budget.md (drop every component whose mean
detection score is below 0.995: +0.0035..0.0041 on 25 of 25 fold-C val videos, both embryos, both tracker configs) and
shift_z the +0.5 raw-z-voxel coordinate probe from the same report. Both are pure post-processes on the FINAL graph."""
from __future__ import annotations
import numpy as np


def _components(node_ids, src, tgt):
    idx = {int(n): k for k, n in enumerate(node_ids)}
    parent = np.arange(len(node_ids))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for s, d in zip(src, tgt):
        s = idx.get(int(s)); d = idx.get(int(d))
        if s is None or d is None: continue
        rs, rd = find(s), find(d)
        if rs != rd: parent[rs] = rd
    return np.array([find(i) for i in range(len(node_ids))])


def prune_components(nodes, edges, frac=0.1, key="score", keep_divisions=True, scores=None, edge_conf=None):
    """nodes: DataFrame(node_id, t, z, y, x[, score]); edges: DataFrame(source_id, target_id).
    key: 'score' (mean detection score), 'len' (component length), 'score_len' (mean score * log length).
    Returns (nodes, edges, n_removed)."""
    if frac <= 0 or len(nodes) == 0 or len(edges) == 0:
        return nodes, edges, 0
    nid = nodes.node_id.values.astype(np.int64)
    comp = _components(nid, edges.source_id.values, edges.target_id.values)
    sc = (nodes["score"].values if "score" in nodes.columns else (scores if scores is not None else np.ones(len(nid))))
    order = {}
    div = set()
    if keep_divisions:
        from collections import Counter
        c = Counter(int(s) for s in edges.source_id.values)
        idx = {int(n): k for k, n in enumerate(nid)}
        for s, n_out in c.items():
            if n_out >= 2 and s in idx: div.add(int(comp[idx[s]]))
    for c_ in np.unique(comp):
        m = comp == c_
        n = int(m.sum()); mean_s = float(np.mean(sc[m]))
        order[int(c_)] = {"score": mean_s, "len": float(n), "score_len": mean_s * np.log1p(n)}[key], n
    ranked = sorted(order.items(), key=lambda kv: kv[1][0])
    budget = int(round(frac * len(nid))); removed = 0; drop = set()
    for c_, (val, n) in ranked:
        if removed >= budget: break
        if keep_divisions and c_ in div: continue
        drop.add(c_); removed += n
    if not drop:
        return nodes, edges, 0
    keep_mask = ~np.isin(comp, list(drop))
    keep_ids = set(nid[keep_mask].tolist())
    nodes2 = nodes[keep_mask].copy()
    e = edges[[int(s) in keep_ids and int(d) in keep_ids for s, d in zip(edges.source_id.values, edges.target_id.values)]].copy()
    return nodes2, e, int((~keep_mask).sum())


def prune_weak_components(nodes, edges, min_score, score_map=None, max_frac=0.0):
    """Drop every weakly-connected component of the FINAL graph whose MEAN node detection score is < min_score.

    nodes: DataFrame(node_id, t, z, y, x[, score]); edges: DataFrame(source_id, target_id[, ...]).
    Isolated nodes are their own component; a node whose score is unknown (NaN, or absent from score_map when the
    stage that produced it dropped the column) counts as 1.0, i.e. it is kept. Components are closed under the edge
    relation, so no surviving edge is ever cut - only whole tracks disappear.
    max_frac > 0 caps the damage per video: components are dropped weakest-mean-score first and the cut stops once
    max_frac of the video's nodes would be removed. An absolute score threshold does NOT transfer between detectors
    (5% of one detector's detections sit below 0.995 but 13.5% of another's), so the cap is what makes the stage safe
    to ship: it reproduces the node fraction the threshold was tuned at, whatever the score distribution looks like.
    Returns (nodes, edges, n_nodes_removed, n_components_removed)."""
    if min_score is None or min_score <= 0 or len(nodes) == 0:
        return nodes, edges, 0, 0
    nid = nodes.node_id.values.astype(np.int64)
    src = edges.source_id.values if len(edges) else np.zeros(0, np.int64)
    tgt = edges.target_id.values if len(edges) else np.zeros(0, np.int64)
    if "score" in nodes.columns:
        sc = np.nan_to_num(np.asarray(nodes["score"].values, np.float64), nan=1.0)
    elif score_map:
        sc = np.array([float(score_map.get(int(n), 1.0)) for n in nid.tolist()], np.float64)
    else:
        return nodes, edges, 0, 0
    comp = _components(nid, src, tgt)
    n_c = int(comp.max()) + 1 if len(comp) else 0
    sums = np.bincount(comp, weights=sc, minlength=n_c); cnts = np.bincount(comp, minlength=n_c)
    mean = sums / np.maximum(cnts, 1)
    drop = (cnts > 0) & (mean < float(min_score))
    if max_frac and max_frac > 0 and drop.any():
        budget = float(max_frac) * len(nid)
        cand = np.where(drop)[0]
        cand = cand[np.argsort(mean[cand], kind="stable")]      # weakest components first
        keep_budget = np.cumsum(cnts[cand]) <= budget
        drop = np.zeros_like(drop); drop[cand[keep_budget]] = True
    keep_mask = ~drop[comp]
    if keep_mask.all():
        return nodes, edges, 0, 0
    keep_ids = set(nid[keep_mask].tolist())
    nodes2 = nodes[keep_mask].copy()
    if len(edges):
        m = np.array([int(s) in keep_ids and int(d) in keep_ids for s, d in zip(edges.source_id.values, edges.target_id.values)], bool)
        edges2 = edges[m].copy()
    else:
        edges2 = edges
    return nodes2, edges2, int((~keep_mask).sum()), int(drop.sum())


def shift_z(nodes, dz):
    """Add dz RAW Z-VOXELS to every node's z (the detector's z-convention probe; clamped at 0 like the submission
    writer). Coordinate-only: run it after every stage that uses coordinates. Returns a copy of nodes."""
    if not dz or len(nodes) == 0:
        return nodes
    out = nodes.copy()
    out["z"] = np.maximum(0.0, np.asarray(out["z"].values, np.float64) + float(dz))
    return out
