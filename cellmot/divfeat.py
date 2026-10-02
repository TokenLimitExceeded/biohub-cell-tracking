"""Division candidates + features for a learned division classifier.

After linking (one child per parent), a candidate is (p, c1, q): parent p at t with its linked child c1 at t+1 and a
second node q at t+1 (typically an orphan) within `parent_um` of p and `sister_um` of c1.
Labels use GT: positive iff p matches a GT dividing node (within 7um) and {c1, q} match its two GT children.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

DIV_FEATURE_NAMES = [
    "pd1", "pd2", "sister", "sym", "cos_daughters", "pd_min", "pd_max",
    "q_orphan", "q_prev_near", "q_prev_dist", "c1_prev_near",
    "score_p", "score_c1", "score_q", "score_min",
    "p_vel", "p_vel_change", "dens_p_10", "dens_t1_10",
    "gd_next", "diverge", "c1_has_child", "q_has_child",
    "p_edge_c1", "p_edge_q", "t_frac", "z_p", "border_p",
]


def division_candidates(t, pos, score, parent, children, parent_um=13.5, sister_um=20.5, pair_prob=None,
                        allow_linked_q=False):
    """t, pos (um), score: node arrays. parent: array of parent index (-1 none). children: dict node->list.
    Returns list of tuples (p, c1, q) and a feature matrix."""
    t = np.asarray(t); pos = np.asarray(pos, np.float32); score = np.asarray(score, np.float32)
    N = len(t); T = int(t.max()) + 1
    by_t = {tt: np.where(t == tt)[0] for tt in range(T)}
    trees = {tt: cKDTree(pos[ix]) for tt, ix in by_t.items() if len(ix)}
    cands = []; feats = []
    for tt in range(T - 1):
        si = by_t[tt]; ti = by_t.get(tt + 1, np.array([], np.int64))
        if len(si) == 0 or len(ti) == 0 or (tt + 1) not in trees:
            continue
        tree_next = trees[tt + 1]; tree_prev = trees.get(tt - 1); tree_cur = trees[tt]
        for p in si:
            ch = children.get(int(p), [])
            if len(ch) != 1:
                continue
            c1 = int(ch[0])
            pd1 = float(np.linalg.norm(pos[c1] - pos[p]))
            for k in tree_next.query_ball_point(pos[p], r=parent_um):
                q = int(ti[k])
                if q == c1:
                    continue
                if parent[q] >= 0 and not allow_linked_q:
                    continue
                sister = float(np.linalg.norm(pos[q] - pos[c1]))
                if sister > sister_um:
                    continue
                pd2 = float(np.linalg.norm(pos[q] - pos[p]))
                v1 = pos[c1] - pos[p]; v2 = pos[q] - pos[p]
                cosd = float((v1 * v2).sum() / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
                sym = abs(pd1 - pd2) / max(0.5 * (pd1 + pd2), 1e-6)
                # q's history: nearest node at t-1 (if q has a plausible own parent it is less likely a daughter)
                if tree_prev is not None and len(by_t[tt - 1]):
                    dq, _ = tree_prev.query(pos[q]); dc, _ = tree_prev.query(pos[c1])
                else:
                    dq, dc = 20.0, 20.0
                q_prev_near = float(dq <= 6.0); c1_prev_near = float(dc <= 6.0)
                # parent motion
                pp = parent[p]
                if pp >= 0:
                    vp = pos[p] - pos[pp]; p_vel = float(np.linalg.norm(vp))
                    mid = 0.5 * (pos[c1] + pos[q]) - pos[p]
                    p_vel_change = float(np.linalg.norm(mid - vp))
                else:
                    p_vel = -1.0; p_vel_change = -1.0
                dens_p = len(tree_cur.query_ball_point(pos[p], r=10.0)) - 1
                dens_t1 = len(tree_next.query_ball_point(pos[c1], r=10.0)) - 1
                # divergence at t+2
                c1c = children.get(c1, []); qc = children.get(q, [])
                if len(c1c) == 1 and len(qc) == 1:
                    gd = float(np.linalg.norm(pos[c1c[0]] - pos[qc[0]])); diverge = gd - sister
                else:
                    gd = -1.0; diverge = 0.0
                pe1 = pair_prob.get((int(p), c1), -1.0) if pair_prob else -1.0
                pe2 = pair_prob.get((int(p), q), -1.0) if pair_prob else -1.0
                sz, sy, sx = 64 * 1.625, 256 * 0.40625, 256 * 0.40625
                border = min(pos[p][0], sz - pos[p][0], pos[p][1], sy - pos[p][1], pos[p][2], sx - pos[p][2])
                feats.append([pd1, pd2, sister, sym, cosd, min(pd1, pd2), max(pd1, pd2),
                              float(parent[q] < 0), q_prev_near, float(dq), c1_prev_near,
                              score[p], score[c1], score[q], min(score[p], score[c1], score[q]),
                              p_vel, p_vel_change, dens_p, dens_t1,
                              gd, diverge, float(len(c1c) > 0), float(len(qc) > 0),
                              pe1, pe2, tt / max(T - 1, 1), pos[p][0], border])
                cands.append((int(p), c1, q))
    return cands, (np.array(feats, np.float32) if feats else np.zeros((0, len(DIV_FEATURE_NAMES)), np.float32))


def label_divisions(cands, node2gt, gt_children: dict):
    """gt_children: {gt_idx: [child gt idx, ...]}. Positive iff gt(p) has 2 children matching {gt(c1), gt(q)};
    negative iff gt(p) matched and has >= 1 child (annotated continuation) but not a matching division; -1 unknown."""
    y = np.full(len(cands), -1, np.int8)
    for k, (p, c1, q) in enumerate(cands):
        gp = node2gt[p]
        if gp < 0:
            continue
        ch = gt_children.get(gp, [])
        if len(ch) >= 2 and node2gt[c1] in ch and node2gt[q] in ch and node2gt[c1] != node2gt[q]:
            y[k] = 1
        elif len(ch) >= 1:
            y[k] = 0
    return y
