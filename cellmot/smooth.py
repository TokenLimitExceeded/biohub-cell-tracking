"""Line-fit coordinate smoothing of track interiors (port of the anchor notebook's linefit_smooth_output_graph).

For every node, walk up to `window` unique predecessors and `window` unique successors along consecutive-frame edges
(stopping at forks/merges/ends); with >= 3 points fit a degree-1 polynomial per axis against dt and move the node to
(1-w)*original + w*fit(dt=0). Topology is unchanged. Coordinates are whatever units the caller passes (raw voxels here).
"""
from __future__ import annotations
import numpy as np


def linefit_smooth(nodes, edges, weight=0.8, window=2):
    """nodes: DataFrame with node_id, t, z, y, x (raw voxels); edges: DataFrame with source_id, target_id.
    Returns a copy of nodes with smoothed z, y, x and the number of smoothed nodes."""
    if weight <= 0 or window <= 0 or len(edges) == 0 or len(nodes) < 3:
        return nodes, 0
    ids = nodes.node_id.values.astype(np.int64); t = nodes.t.values.astype(np.int64)
    pos = nodes[["z", "y", "x"]].values.astype(np.float64)
    idx = {int(n): k for k, n in enumerate(ids)}
    pred = {}; succ = {}
    for s, d in zip(edges.source_id.values, edges.target_id.values):
        s = int(s); d = int(d)
        if s not in idx or d not in idx or t[idx[d]] != t[idx[s]] + 1:
            continue
        succ.setdefault(s, []).append(d); pred.setdefault(d, []).append(s)
    new = pos.copy(); n_sm = 0
    for nid in ids:
        nid = int(nid); pts = [(0, idx[nid])]
        cur = nid
        for step in range(1, window + 1):
            p = pred.get(cur, [])
            if len(p) != 1: break
            cur = p[0]; pts.append((-step, idx[cur]))
        cur = nid
        for step in range(1, window + 1):
            s_ = succ.get(cur, [])
            if len(s_) != 1: break
            cur = s_[0]; pts.append((step, idx[cur]))
        if len(pts) < 3:
            continue
        dts = np.array([d for d, _ in pts], np.float64); c = pos[[k for _, k in pts]]
        # degree-1 least squares evaluated at dt=0 == intercept
        A = np.stack([dts, np.ones_like(dts)], 1)
        coef, *_ = np.linalg.lstsq(A, c, rcond=None)
        fit = coef[1]
        if not np.isfinite(fit).all():
            continue
        new[idx[nid]] = (1.0 - weight) * pos[idx[nid]] + weight * fit; n_sm += 1
    out = nodes.copy(); out["z"] = new[:, 0]; out["y"] = new[:, 1]; out["x"] = new[:, 2]
    return out, n_sm
