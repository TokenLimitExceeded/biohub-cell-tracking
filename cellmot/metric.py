"""Fast numpy port of the official competition metric (royerlab/kaggle-cell-tracking-competition).

Graphs are plain arrays: nodes (node_id, t, z, y, x) and edges (source_id, target_id).
All logic mirrors src/tracking_cellmot/metrics.py and division_metrics.py.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ADJUSTMENT_ALPHA = 0.1
SCORE_DIVISION_WEIGHT = 0.1
DEFAULT_SCALE = (1.625, 0.40625, 0.40625)


class Graph:
    """Directed lineage graph with integer node ids."""
    def __init__(self, node_id, t, z, y, x, src, tgt):
        self.node_id = np.asarray(node_id, dtype=np.int64)
        self.t = np.asarray(t, dtype=np.int64)
        self.coords = np.stack([np.asarray(z, float), np.asarray(y, float), np.asarray(x, float)], 1) if len(self.node_id) else np.zeros((0, 3))
        self.src = np.asarray(src, dtype=np.int64)
        self.tgt = np.asarray(tgt, dtype=np.int64)
        assert len(np.unique(self.node_id)) == len(self.node_id), "duplicate node ids"
        self.idx = {int(n): i for i, n in enumerate(self.node_id)}
        self.succ = defaultdict(list); self.pred = defaultdict(list)
        for s, tt in zip(self.src.tolist(), self.tgt.tolist()):
            self.succ[s].append(tt); self.pred[tt].append(s)
        self.n_nodes = len(self.node_id); self.n_edges = len(self.src)

    @classmethod
    def from_frames(cls, nodes_df, edges_df):
        return cls(nodes_df["node_id"].values, nodes_df["t"].values, nodes_df["z"].values, nodes_df["y"].values,
                   nodes_df["x"].values, edges_df["source_id"].values if len(edges_df) else [],
                   edges_df["target_id"].values if len(edges_df) else [])

    def out_degree(self, n): return len(self.succ.get(n, ()))
    def in_degree(self, n): return len(self.pred.get(n, ()))
    def successors(self, n): return list(self.succ.get(n, ()))
    def predecessors(self, n): return list(self.pred.get(n, ()))
    def dividing_nodes(self): return [n for n in self.node_id.tolist() if self.out_degree(n) >= 2]
    def t_of(self, n): return int(self.t[self.idx[n]])


def _match_groups(pred_pts, gt_pts, max_distance):
    """Max-weight (1/(1+d)) one-to-one matching between candidate pairs within max_distance.
    Returns list of (pred_idx, gt_idx)."""
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return []
    d = cdist(pred_pts, gt_pts)
    cand = d <= max_distance
    if not cand.any():
        return []
    w = np.where(cand, 1.0 / (1.0 + d), -1.0)
    rows = np.where(cand.any(1))[0]; cols = np.where(cand.any(0))[0]
    sub = w[np.ix_(rows, cols)]
    r, c = linear_sum_assignment(sub, maximize=True)
    keep = sub[r, c] > 0
    return list(zip(rows[r[keep]].tolist(), cols[c[keep]].tolist()))


def match_nodes(pred: Graph, gt: Graph, scale=DEFAULT_SCALE, max_distance=7.0, gt_subset=None):
    """Per-timepoint optimal matching. Returns dict pred_node_id -> gt_node_id.
    gt_subset: optional array of gt node ids restricting the GT side (used for division windows)."""
    scale = np.asarray(scale, float)
    pc = pred.coords * scale; gc = gt.coords * scale
    if gt_subset is not None:
        gsel = np.array([gt.idx[int(n)] for n in gt_subset], dtype=np.int64)
    else:
        gsel = np.arange(gt.n_nodes)
    out = {}
    gt_t = gt.t[gsel]
    for tt in np.unique(gt_t):
        gi = gsel[gt_t == tt]
        pi = np.where(pred.t == tt)[0]
        if len(pi) == 0:
            continue
        for a, b in _match_groups(pc[pi], gc[gi], max_distance):
            out[int(pred.node_id[pi[a]])] = int(gt.node_id[gi[b]])
    return out


def edge_counts(pred: Graph, gt: Graph, p2g: dict):
    """Edge TP/FP/FN following _evaluate_matched_graph. Returns (tp, fp, fn, n_valid)."""
    if pred.n_edges == 0:
        return 0, 0, gt.n_edges, 0
    src, tgt = pred.src, pred.tgt
    # dedup identical (source,target) pairs keeping first occurrence
    _, first = np.unique(np.stack([src, tgt], 1), axis=0, return_index=True)
    keep = np.zeros(len(src), bool); keep[first] = True
    # keep only dt == 1
    ts = pred.t[[pred.idx[s] for s in src.tolist()]]; tt = pred.t[[pred.idx[s] for s in tgt.tolist()]]
    keep &= (tt - ts) == 1
    eids = np.where(keep)[0]
    src, tgt = src[eids], tgt[eids]
    # cap out-degree at 2 (lowest edge ids per source)
    cnt = defaultdict(int); keep2 = np.zeros(len(eids), bool)
    for k in range(len(eids)):
        s = int(src[k])
        if cnt[s] < 2:
            keep2[k] = True; cnt[s] += 1
    src, tgt = src[keep2], tgt[keep2]
    gt_edges = set(zip(gt.src.tolist(), gt.tgt.tolist()))
    ms = np.array([p2g.get(int(s), -1) for s in src], dtype=np.int64); mt = np.array([p2g.get(int(s), -1) for s in tgt], dtype=np.int64)
    matched = np.array([(a != -1 and b != -1 and (a, b) in gt_edges) for a, b in zip(ms.tolist(), mt.tolist())], bool)
    # collapse merges: several pred edges onto the same GT edge -> keep first
    seen = set()
    for k in np.where(matched)[0]:
        key = (int(ms[k]), int(mt[k]))
        if key in seen: matched[k] = False
        else: seen.add(key)
    out_valid = np.array([a != -1 and gt.out_degree(int(a)) > 0 for a in ms.tolist()], bool)
    in_valid = np.array([b != -1 and gt.in_degree(int(b)) > 0 for b in mt.tolist()], bool)
    pred_valid = out_valid | in_valid
    tp = int(matched.sum()); n_valid = int(pred_valid.sum())
    return tp, n_valid - tp, gt.n_edges - tp, n_valid


def _bipartite_max_matching(left, edges):
    match_r, match_l = {}, {}
    def augment(u, seen):
        for v in edges.get(u, ()):
            if v in seen: continue
            seen.add(v)
            if v not in match_r or augment(match_r[v], seen):
                match_l[u] = v; match_r[v] = u
                return True
        return False
    for u in left:
        augment(u, set())
    return match_l


def _gt_weak_component_ids(gt: Graph):
    comp = {}
    for seed in gt.node_id.tolist():
        if seed in comp: continue
        comp[seed] = seed; stack = [seed]
        while stack:
            cur = stack.pop()
            for nb in gt.successors(cur) + gt.predecessors(cur):
                if nb not in comp:
                    comp[nb] = seed; stack.append(nb)
    return comp


def _branch_component_evidence(pred: Graph, pred_div, child, p2g, gt_comp):
    if set(pred.predecessors(child)) != {pred_div}:
        return None, True
    if child in p2g:
        return gt_comp[p2g[child]], False
    gcs = pred.successors(child)
    if any(set(pred.predecessors(n)) != {child} for n in gcs):
        return None, True
    comps = {gt_comp[p2g[n]] for n in gcs if n in p2g}
    if len(comps) == 1:
        return next(iter(comps)), False
    return None, False


def _pred_division_fork_sets(pred: Graph, gt: Graph, p2g_full: dict):
    forks = {n for n in pred.node_id.tolist() if pred.out_degree(n) >= 2}
    evaluable = {f for f in forks if f in p2g_full and gt.out_degree(p2g_full[f]) >= 1}
    gt_comp = _gt_weak_component_ids(gt)
    cross, malformed = set(), set()
    for f in forks:
        ev = []; bad = False
        for child in pred.successors(f):
            comp, mal = _branch_component_evidence(pred, f, child, p2g_full, gt_comp)
            if mal:
                malformed.add(f); bad = True; break
            if comp is not None: ev.append(comp)
        if not bad and len(set(ev)) >= 2:
            cross.add(f)
    return evaluable, cross, malformed


def _is_strongly_connected_division(pred: Graph, pred_div, parent_ids, daughter_ids):
    if {pred_div, *pred.predecessors(pred_div)}.isdisjoint(parent_ids):
        return False
    lineages = [{c, *pred.successors(c)} for c in pred.successors(pred_div)]
    edges = {gi: {pi for pi, ids in enumerate(lineages) if not m.isdisjoint(ids)} for gi, m in enumerate(daughter_ids)}
    return len(_bipartite_max_matching(list(edges), edges)) >= 2


def division_counts(pred: Graph, gt: Graph, p2g_full: dict, scale=DEFAULT_SCALE, max_distance=7.0):
    """Port of score_divisions/evaluate_divisions. Returns (tp, fp, fn)."""
    gt_divs = gt.dividing_nodes()
    pred_forks = {n for n in pred.node_id.tolist() if pred.out_degree(n) >= 2}
    evaluable, cross, malformed = _pred_division_fork_sets(pred, gt, p2g_full)
    invalid = cross | malformed
    candidates = {}; considered = set()
    for div in gt_divs:
        children = gt.successors(div)
        parents = gt.predecessors(div)
        grandchildren = [g for c in children for g in gt.successors(c)]
        window = [*parents, div, *children, *grandchildren]
        p2g = match_nodes(pred, gt, scale, max_distance, gt_subset=np.array(window))
        if not p2g or len(children) < 2:
            candidates[div] = set(); continue
        gt_parent_ids = {div, *parents}
        parent_ids = {p for p, g in p2g.items() if g in gt_parent_ids}
        daughter_ids = [{p for p, g in p2g.items() if g in {c, *gt.successors(c)}} for c in children]
        if not parent_ids or sum(bool(s) for s in daughter_ids) < 2:
            candidates[div] = set(); continue
        local = parent_ids | {s for p in parent_ids for s in pred.successors(p)}
        local_forks = local & pred_forks
        considered |= local_forks
        candidates[div] = {f for f in local_forks - invalid if _is_strongly_connected_division(pred, f, parent_ids, daughter_ids)}
    pairing = _bipartite_max_matching(list(candidates), candidates)
    tp = len(pairing); fn = len(candidates) - tp
    tp_forks = set(pairing.values())
    fp_forks = (considered | evaluable | invalid) - tp_forks
    return tp, len(fp_forks), fn


def evaluate(pred: Graph, gt: Graph, scale=DEFAULT_SCALE, max_distance=7.0, with_divisions=True):
    """Returns dict with edge_tp/fp/fn, division_tp/fp/fn, num_pred_nodes, node_recall."""
    if pred.n_nodes == 0 or pred.n_edges == 0:
        return dict(edge_tp=0, edge_fp=0, edge_fn=gt.n_edges, division_tp=0, division_fp=0,
                    division_fn=len(gt.dividing_nodes()), num_pred_nodes=pred.n_nodes, node_recall=0.0)
    p2g = match_nodes(pred, gt, scale, max_distance)
    tp, fp, fn, _ = edge_counts(pred, gt, p2g)
    if with_divisions:
        dtp, dfp, dfn = division_counts(pred, gt, p2g, scale, max_distance)
    else:
        dtp = dfp = dfn = 0
    recall = len(set(p2g.values())) / max(gt.n_nodes, 1)
    return dict(edge_tp=tp, edge_fp=fp, edge_fn=fn, division_tp=dtp, division_fp=dfp, division_fn=dfn,
                num_pred_nodes=pred.n_nodes, node_recall=recall)


def per_sample_metrics(er: dict, n_total: float):
    ratio = (er["num_pred_nodes"] - n_total) / n_total if n_total and n_total > 0 else float("nan")
    denom = er["edge_tp"] + er["edge_fp"] + er["edge_fn"]
    ej = er["edge_tp"] / denom if denom > 0 else float("nan")
    adj = max(0.0, ej * (1 - ADJUSTMENT_ALPHA * ratio)) if (ej == ej and ratio == ratio) else float("nan")
    return {**er, "total_node_ratio": ratio, "edge_jaccard": ej, "adj_edge_jaccard": adj}


def summarise(rows):
    valid = [r for r in rows if r["edge_tp"] == r["edge_tp"]]
    if not valid:
        return {"n": 0, "score": float("nan")}
    tot = lambda k: sum(r[k] for r in valid)
    adj_rows = [r for r in valid if r["adj_edge_jaccard"] == r["adj_edge_jaccard"]]
    w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in adj_rows]
    adj = sum(wi * r["adj_edge_jaccard"] for wi, r in zip(w, adj_rows)) / sum(w) if sum(w) > 0 else float("nan")
    dtot = tot("division_tp") + tot("division_fp") + tot("division_fn")
    dj = tot("division_tp") / dtot if dtot > 0 else float("nan")
    e_den = tot("edge_tp") + tot("edge_fp") + tot("edge_fn")
    ej = tot("edge_tp") / e_den if e_den > 0 else float("nan")
    score = adj + SCORE_DIVISION_WEIGHT * dj if dtot > 0 else adj
    return {"n": len(valid), "edge_jaccard": ej, "adj_edge_jaccard": adj, "division_jaccard": dj,
            "division_tp": tot("division_tp"), "division_fp": tot("division_fp"), "division_fn": tot("division_fn"),
            "node_recall": sum(r["node_recall"] for r in valid) / len(valid), "score": score,
            "edge_tp": tot("edge_tp"), "edge_fp": tot("edge_fp"), "edge_fn": tot("edge_fn"),
            "num_pred_nodes": tot("num_pred_nodes")}
