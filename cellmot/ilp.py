"""Global ILP linking with tracksdata (SCIP via ilpy): candidate edges weighted by learned probability.

Given detections (t, pos_um, score) and pair probabilities {(i,j): p} from cellmot.track.learned_pair_probs,
build a tracksdata graph and solve the ILP with appearance/disappearance/division costs. Returns edges as
(src_idx, tgt_idx, conf) over the input detection indices (no new nodes)."""
from __future__ import annotations
import numpy as np


def solve_ilp(t, pos_um, score, pair_prob: dict, edge_w=-1.0, appearance=0.0, disappearance=2.0, division=1.2,
              min_p=0.05, max_um=14.0, node_w=None):
    import polars as pl
    import tracksdata as td
    t = np.asarray(t); pos = np.asarray(pos_um, np.float32)
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -1.0)
    g.add_node_attr_key("score", pl.Float64, 0.0)
    ids = g.bulk_add_nodes([{"t": int(t[i]), "z": float(pos[i, 0]), "y": float(pos[i, 1]), "x": float(pos[i, 2]), "score": float(score[i])} for i in range(len(t))])
    ids = np.asarray(ids)
    g.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
    edges = []
    for (i, j), p in pair_prob.items():
        if p < min_p:
            continue
        if np.linalg.norm(pos[j] - pos[i]) > max_um:
            continue
        edges.append({"source_id": int(ids[i]), "target_id": int(ids[j]), "edge_prob": float(p)})
    if not edges:
        return []
    g.bulk_add_edges(edges)
    solver = td.solvers.ILPSolver(edge_weight=edge_w * td.EdgeAttr("edge_prob"), appearance_weight=appearance,
                                  disappearance_weight=disappearance, division_weight=division)
    solver.solve(g)
    sol_key = td.DEFAULT_ATTR_KEYS.SOLUTION
    ea = g.edge_attrs(attr_keys=[sol_key, "edge_prob"])
    ea = ea.filter(pl.col(sol_key))
    id2idx = {int(v): i for i, v in enumerate(ids)}
    out = [(id2idx[int(s)], id2idx[int(tt)], float(p)) for s, tt, p in zip(ea[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list(), ea[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list(), ea["edge_prob"].to_list())]
    return out
