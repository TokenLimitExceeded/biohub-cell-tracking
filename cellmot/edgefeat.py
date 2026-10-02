"""Candidate edge generation + geometric/context features for a learned association scorer (GBDT).

Nodes are per-dataset arrays: t (N,), pos_um (N,3), score (N,). Candidates: (i, j) with t_j = t_i + 1 and
|pos_j - pos_i| <= max_um. Labels come from GT via node matching (7 um Hungarian per frame).
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

FEATURE_NAMES = [
    "dist", "dz", "dy", "dx", "adz", "adxy",
    "score_i", "score_j", "score_diff",
    "rank_ij", "rank_ji", "gap_i", "gap_j",          # NN ranks and distance margins (2nd best - this)
    "dens_i_6", "dens_i_12", "dens_j_6", "dens_j_12",  # neighbourhood counts
    "n_cand_i", "n_cand_j",
    "prev_dist", "vel_cons", "vel_cos",              # motion consistency with i's predecessor (if any)
    "t_frac", "border_i", "border_j", "z_i", "z_j",
]


def candidates(t, pos, score, max_um=12.0, k=8):
    """Return arrays (i, j) of candidate pairs between consecutive frames (each i keeps up to k nearest j)."""
    t = np.asarray(t); pos = np.asarray(pos, np.float32)
    T = int(t.max()) + 1
    by_t = {tt: np.where(t == tt)[0] for tt in range(T)}
    I, J = [], []
    for tt in range(T - 1):
        si, ti = by_t.get(tt, []), by_t.get(tt + 1, [])
        if len(si) == 0 or len(ti) == 0:
            continue
        tree = cKDTree(pos[ti])
        d, idx = tree.query(pos[si], k=min(k, len(ti)), distance_upper_bound=max_um)
        d = np.atleast_2d(d); idx = np.atleast_2d(idx)
        for a in range(len(si)):
            for b in range(d.shape[1]):
                if np.isfinite(d[a, b]):
                    I.append(int(si[a])); J.append(int(ti[idx[a, b]]))
    return np.array(I, np.int64), np.array(J, np.int64)


def features(t, pos, score, I, J, prev_of=None, vol_shape_um=(64 * 1.625, 256 * 0.40625, 256 * 0.40625)):
    """Feature matrix (len(I), len(FEATURE_NAMES)). prev_of: optional array of predecessor index per node (-1 if none),
    e.g. from a first geometric tracking pass, used for motion-consistency features."""
    t = np.asarray(t); pos = np.asarray(pos, np.float32); score = np.asarray(score, np.float32)
    N = len(t); T = int(t.max()) + 1
    d = pos[J] - pos[I]; dist = np.linalg.norm(d, axis=1)
    # per-frame trees for density and ranks
    by_t = {tt: np.where(t == tt)[0] for tt in range(T)}
    trees = {tt: cKDTree(pos[ix]) for tt, ix in by_t.items() if len(ix)}
    dens6 = np.zeros(N, np.float32); dens12 = np.zeros(N, np.float32)
    for tt, ix in by_t.items():
        if len(ix) == 0:
            continue
        tr = trees[tt]
        dens6[ix] = np.array([len(x) - 1 for x in tr.query_ball_point(pos[ix], r=6.0)])
        dens12[ix] = np.array([len(x) - 1 for x in tr.query_ball_point(pos[ix], r=12.0)])
    # ranks: rank of j among i's neighbours at t+1, and of i among j's neighbours at t; margin to 2nd best
    rank_ij = np.zeros(len(I), np.float32); rank_ji = np.zeros(len(I), np.float32)
    gap_i = np.zeros(len(I), np.float32); gap_j = np.zeros(len(I), np.float32)
    n_cand_i = np.zeros(len(I), np.float32); n_cand_j = np.zeros(len(I), np.float32)
    # group by frame pair for efficiency
    order = np.argsort(t[I], kind="stable")
    for tt in np.unique(t[I]):
        sel = np.where(t[I] == tt)[0]
        if tt + 1 not in trees or tt not in trees:
            continue
        tr_next, tr_prev = trees[tt + 1], trees[tt]
        ix_next, ix_prev = by_t[tt + 1], by_t[tt]
        ui, inv_i = np.unique(I[sel], return_inverse=True)
        dn, idn = tr_next.query(pos[ui], k=min(3, len(ix_next)))
        dn = np.atleast_2d(dn); idn = np.atleast_2d(idn)
        uj, inv_j = np.unique(J[sel], return_inverse=True)
        dp, idp = tr_prev.query(pos[uj], k=min(3, len(ix_prev)))
        dp = np.atleast_2d(dp); idp = np.atleast_2d(idp)
        for kk, e in enumerate(sel):
            a = inv_i[kk]; b = inv_j[kk]
            # rank of j among i's nearest at t+1 (0 = nearest)
            nb = ix_next[idn[a]]
            r = np.where(nb == J[e])[0]; rank_ij[e] = r[0] if len(r) else 3
            gap_i[e] = (dn[a, 1] if dn.shape[1] > 1 else 20.0) - dist[e]
            nb2 = ix_prev[idp[b]]
            r2 = np.where(nb2 == I[e])[0]; rank_ji[e] = r2[0] if len(r2) else 3
            gap_j[e] = (dp[b, 1] if dp.shape[1] > 1 else 20.0) - dist[e]
        # candidate counts
        ci = np.bincount(inv_i); cj = np.bincount(inv_j)
        n_cand_i[sel] = ci[inv_i]; n_cand_j[sel] = cj[inv_j]
    # motion consistency
    prev_dist = np.full(len(I), -1.0, np.float32); vel_cons = np.full(len(I), -1.0, np.float32); vel_cos = np.zeros(len(I), np.float32)
    if prev_of is not None:
        p = np.asarray(prev_of)
        has = p[I] >= 0
        v_prev = pos[I[has]] - pos[p[I[has]]]
        prev_dist[has] = np.linalg.norm(v_prev, axis=1)
        vel_cons[has] = np.linalg.norm(d[has] - v_prev, axis=1)
        nv = np.linalg.norm(v_prev, axis=1) * dist[has] + 1e-6
        vel_cos[has] = (d[has] * v_prev).sum(1) / nv
    sz, sy, sx = vol_shape_um
    def border(p_):
        return np.minimum.reduce([p_[:, 0], sz - p_[:, 0], p_[:, 1], sy - p_[:, 1], p_[:, 2], sx - p_[:, 2]])
    F = np.stack([
        dist, d[:, 0], d[:, 1], d[:, 2], np.abs(d[:, 0]), np.linalg.norm(d[:, 1:], axis=1),
        score[I], score[J], score[J] - score[I],
        rank_ij, rank_ji, gap_i, gap_j,
        dens6[I], dens12[I], dens6[J], dens12[J],
        n_cand_i, n_cand_j,
        prev_dist, vel_cons, vel_cos,
        t[I] / max(T - 1, 1), border(pos[I]), border(pos[J]), pos[I][:, 0], pos[J][:, 0],
    ], 1).astype(np.float32)
    return F


def match_to_gt(t, pos, gt_t, gt_pos, max_um=7.0):
    """Per-frame Hungarian matching of detections to GT nodes (physical um). Returns array node->gt index or -1."""
    t = np.asarray(t); gt_t = np.asarray(gt_t)
    out = -np.ones(len(t), np.int64)
    for tt in np.unique(gt_t):
        gi = np.where(gt_t == tt)[0]; pi = np.where(t == tt)[0]
        if len(pi) == 0:
            continue
        D = cdist(pos[pi], gt_pos[gi])
        w = np.where(D <= max_um, 1.0 / (1.0 + D), -1.0)
        r, c = linear_sum_assignment(w, maximize=True)
        for a, b in zip(r, c):
            if w[a, b] > 0:
                out[pi[a]] = gi[b]
    return out


def label_candidates(I, J, node2gt, gt_edges: set, gt_out_deg: dict):
    """Labels: 1 if (gt(i), gt(j)) is a GT edge; 0 if gt(i) is matched and has a GT successor (so the true target is
    known and it is not j) or gt(j) is matched and has a GT predecessor; -1 unknown (ignore)."""
    y = np.full(len(I), -1, np.int8)
    for e, (i, j) in enumerate(zip(I, J)):
        gi, gj = node2gt[i], node2gt[j]
        if gi >= 0 and gj >= 0 and (gi, gj) in gt_edges:
            y[e] = 1
        elif gi >= 0 and gt_out_deg.get(gi, 0) > 0:
            y[e] = 0
    return y
