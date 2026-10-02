"""Image-evidence division classifier (vectorised port of tools/explore_division/divcands.py + the plug-in stage).

Candidate fork = (P at t, c1 at t+1, q at t+1): P has exactly one child c1 with |P-c1| <= parent_um, q is any node at
t+1 within parent_um of P and sister_um of c1 (q may be an orphan or already linked from another node X).
Features: geometry, public edge probabilities (geo + per-target softmax), track context and pool4 patch statistics
(isotropic 1.625 um, (1,4,4) mean-pooled + quantile-normalised frames, i.e. exactly what the kernel's
normalize(pool_frame(f), q_lo, q_hi) produces) at P(t), P's predecessor (t-1), P(t+1), c1, q, the sister midpoint at
t+1 and t, X(t); plus contrast ratios. Optional gates sym <= sym_max, pd2 <= pd2_max.
Then: LightGBM probability -> greedy fork insertion (one fork per parent, one use per daughter, anchor-style global cap).
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter

GRID_UM = 1.625
PATCH_KEYS = ("max", "mean", "std", "ctr", "core", "ring", "bg", "cr")
PATCH_PREFIXES = ("P0", "Pm", "Pp", "C1", "Q", "M1", "M0", "X0")

# ---- LAG WINDOW (judge item R3) ------------------------------------------------------------------
# The eight prefixes above sample the pooled volume only at t-1, t and t+1.  A competitor measured
# (paired controls, bootstrap CIs) that the volume signal of a dividing nucleus starts about two
# frames BEFORE the split and peaks about three frames AFTER it, so the window is too narrow at both
# ends.  These five extra anchors widen it to [t-2, t+3] without touching any existing column:
#   Pm2 = the parent TRACK two steps back, at t-2   (same fallback convention as Pm)
#   Cn  = c1's position, at t+2      Qn = q's position, at t+2
#   M2  = the daughter midpoint, at t+2             M3 = the same midpoint, at t+3
# Caveat carried from the source: our pooled volume is (1,4,4)-mean-pooled, i.e. 1.625 um isotropic
# and 4x coarser in y/x than the measurement that motivated this, so expect attenuation.
LAG_PREFIXES = ("Pm2", "Cn", "Qn", "M2", "M3")
LAG_RATIOS = ("r_Pm2", "r_Pm2core", "r_Pm2cr", "r_M2", "r_M3", "r_dn2", "r_cr_p2")
LAG_COLUMNS = [f"{p}_{k}" for p in LAG_PREFIXES for k in PATCH_KEYS] + list(LAG_RATIOS)


def lag_ratio_columns(o):
    """Add the seven lag ratios to a dict/DataFrame-like `o` that already holds the patch columns."""
    dmin = np.minimum(o["C1_max"], o["Q_max"])
    o["r_Pm2"] = o["P0_max"] / (o["Pm2_max"] + 1e-3)        # intensity change over the two frames into the split
    o["r_Pm2core"] = o["P0_core"] / (o["Pm2_core"] + 1e-3)
    o["r_Pm2cr"] = o["P0_cr"] / (o["Pm2_cr"] + 1e-3)        # compaction: core/ring rise as the nucleus shrinks
    o["r_M2"] = o["M2_ctr"] / (dmin + 1e-3)                 # does the gap between the daughters persist at +2 ...
    o["r_M3"] = o["M3_ctr"] / (dmin + 1e-3)                 # ... and at +3, where the source says the signal peaks
    o["r_dn2"] = (o["Cn_mean"] + o["Qn_mean"]) / (o["C1_mean"] + o["Q_mean"] + 1e-3)
    o["r_cr_p2"] = 0.5 * (o["Cn_cr"] + o["Qn_cr"]) / (o["P0_cr"] + 1e-3)
    return o
ID_COLS = ("dataset", "t", "P", "c1", "q", "X", "y", "gt_P")


def _lookup(prob: dict, N: int):
    """{(i, j): p} -> vectorised lookup f(i_arr, j_arr) -> p (0 where absent)."""
    if not prob:
        return lambda i, j: np.zeros(len(i), np.float32)
    ks = np.fromiter((k[0] * N + k[1] for k in prob.keys()), np.int64, len(prob))
    vs = np.fromiter(prob.values(), np.float64, len(prob))
    o = np.argsort(ks); ks = ks[o]; vs = vs[o]

    def f(i, j):
        i = np.asarray(i, np.int64); j = np.asarray(j, np.int64)
        key = i * N + j; pos = np.searchsorted(ks, key); pos = np.minimum(pos, len(ks) - 1)
        hit = (ks[pos] == key) & (i >= 0) & (j >= 0)
        return np.where(hit, vs[pos], 0.0).astype(np.float32)
    return f


def _frame_patch_stats(F, cz, cy, cx):
    """Patch statistics on one frame F (Z,Y,X) float32 at integer centres (clipped). Boxes are clipped at the borders,
    exactly like the explorer's per-candidate slicing: core r=2 (max/mean/std), r=1 (core mean), ring = r4 minus r2, bg r=7."""
    Z, Y, X = F.shape
    F64 = F.astype(np.float64)
    S = np.zeros((Z + 1, Y + 1, X + 1), np.float64); S[1:, 1:, 1:] = F64; S = S.cumsum(0).cumsum(1).cumsum(2)
    S2 = np.zeros((Z + 1, Y + 1, X + 1), np.float64); S2[1:, 1:, 1:] = F64 * F64; S2 = S2.cumsum(0).cumsum(1).cumsum(2)

    def box(r, ii):
        z0 = np.maximum(cz - r, 0); z1 = np.minimum(cz + r + 1, Z); y0 = np.maximum(cy - r, 0); y1 = np.minimum(cy + r + 1, Y)
        x0 = np.maximum(cx - r, 0); x1 = np.minimum(cx + r + 1, X)
        s = (ii[z1, y1, x1] - ii[z0, y1, x1] - ii[z1, y0, x1] - ii[z1, y1, x0] + ii[z0, y0, x1] + ii[z0, y1, x0] + ii[z1, y0, x0] - ii[z0, y0, x0])
        n = (z1 - z0) * (y1 - y0) * (x1 - x0)
        return s, n
    s2, n2 = box(2, S); q2, _ = box(2, S2); s1, n1 = box(1, S); s4, n4 = box(4, S); s7, n7 = box(7, S)
    mean = s2 / n2; var = np.maximum(q2 / n2 - mean * mean, 0.0)
    ring = (s4 - s2) / np.maximum(n4 - n2, 1)
    core = s1 / n1
    Fmax = maximum_filter(F, size=5, mode="nearest")
    out = {"max": Fmax[cz, cy, cx], "mean": mean, "std": np.sqrt(var), "ctr": F[cz, cy, cx], "core": core, "ring": ring,
           "bg": s7 / n7, "cr": core / (ring + 1e-3)}
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def patch_stats(V, frames, pos_um):
    """V (T,Z,Y,X) pool4 volume (float16/32, array or memmap); frames (n,) int; pos_um (n,3). Returns {key: (n,) float32},
    NaN where the frame is outside the volume."""
    n = len(frames); T, Z, Y, X = V.shape
    out = {k: np.full(n, np.nan, np.float32) for k in PATCH_KEYS}
    if n == 0:
        return out
    c = np.rint(np.asarray(pos_um, np.float32) / GRID_UM).astype(np.int64)
    c = np.clip(c, 0, np.array([Z - 1, Y - 1, X - 1]))
    frames = np.asarray(frames, np.int64); valid = (frames >= 0) & (frames < T)
    order = np.argsort(frames[valid], kind="stable"); vi = np.where(valid)[0][order]; fr = frames[vi]
    bounds = np.searchsorted(fr, np.arange(T + 1))
    for tt in range(T):
        a, b = bounds[tt], bounds[tt + 1]
        if b <= a:
            continue
        sel = vi[a:b]; F = np.asarray(V[tt], np.float32)
        st = _frame_patch_stats(F, c[sel, 0], c[sel, 1], c[sel, 2])
        for k in PATCH_KEYS:
            out[k][sel] = st[k]
    return out


def build_candidates(t, pos_um, score, edges, keep, pp, pt, V, parent_um=11.0, sister_um=16.5, sym_max=None, pd2_max=None, dens_um=10.0, lag=False):
    """Tracker output (t, pos_um, score, edges [(i, j, conf)], keep) + pair probabilities pp (geo) / pt (per-target
    softmax) + pool4 volume V -> pandas DataFrame of candidate forks with node ids (1-based, as to_graph_frames) and
    features. Vectorised; ~1-3 s per video."""
    import pandas as pd
    t = np.asarray(t); pos = np.asarray(pos_um, np.float32); score = np.asarray(score, np.float32); keep = np.asarray(keep, bool)
    N = len(t); idx = np.where(keep)[0]
    cols = ["t", "P", "c1", "q", "X", "pd1", "pd2", "sister", "sym", "cosd", "q_orphan", "dX", "X_len", "X_nchild", "X_has_parent", "P_len", "c1_nchild", "q_nchild",
            "sP", "sc1", "sq", "p_Pc1", "p_Pq", "p_Xq", "pt_Pq", "pt_Xq", "pt_best_q", "dens_P", "dens_n", "tfrac", "zP", "gd", "diverge"]
    cols += [f"{p}_{k}" for p in PATCH_PREFIXES for k in PATCH_KEYS] + ["r_daughters", "r_mass", "r_gap", "r_Pprev", "r_Pnext", "r_Pbg"]
    prefixes = PATCH_PREFIXES + (LAG_PREFIXES if lag else ())
    if lag: cols += LAG_COLUMNS
    if len(idx) == 0:
        return pd.DataFrame(columns=cols)
    nid = np.full(N, -1, np.int64); nid[idx] = 1 + np.arange(len(idx))
    E = np.array([(i, j) for i, j, c in edges if keep[i] and keep[j]], np.int64).reshape(-1, 2)
    parent = -np.ones(N, np.int64); parent[E[:, 1]] = E[:, 0]
    nch = np.bincount(E[:, 0], minlength=N) if len(E) else np.zeros(N, np.int64)
    child1 = -np.ones(N, np.int64); child1[E[:, 0]] = E[:, 1]  # the unique child where nch == 1
    Tn = int(t[idx].max()) + 1
    by_t = [idx[t[idx] == k] for k in range(Tn)]
    tlen = np.zeros(N, np.int64)
    for k in range(Tn):
        nk = by_t[k]
        if len(nk):
            par = parent[nk]; tlen[nk] = np.where(par >= 0, tlen[np.maximum(par, 0)] + 1, 1)
    # per-target best incoming tgt-prob among kept sources
    best_src = np.zeros(N, np.float32)
    if pt:
        ki = np.fromiter((k[0] for k in pt.keys()), np.int64, len(pt)); kj = np.fromiter((k[1] for k in pt.keys()), np.int64, len(pt)); kv = np.fromiter(pt.values(), np.float64, len(pt))
        m = keep[ki] & keep[kj]; np.maximum.at(best_src, kj[m], kv[m].astype(np.float32))
    fpp = _lookup(pp or {}, N); fpt = _lookup(pt or {}, N)
    # ---- enumerate (P, c1, q)
    Pl, Cl, Ql, Dp, Dn = [], [], [], [], []
    for k in range(Tn - 1):
        nk, nn = by_t[k], by_t[k + 1]
        if len(nk) == 0 or len(nn) == 0:
            continue
        Pk = nk[nch[nk] == 1]
        if len(Pk) == 0:
            continue
        c1k = child1[Pk]; ok = (t[c1k] == k + 1) & (np.linalg.norm(pos[c1k] - pos[Pk], axis=1) <= parent_um)
        Pk, c1k = Pk[ok], c1k[ok]
        if len(Pk) == 0:
            continue
        tree_next = cKDTree(pos[nn]); tree_cur = cKDTree(pos[nk])
        lists = tree_next.query_ball_point(pos[Pk], r=parent_um)
        lens = np.fromiter((len(l) for l in lists), np.int64, len(lists))
        if lens.sum() == 0:
            continue
        qk = nn[np.concatenate([np.asarray(l, np.int64) for l in lists])]
        Pr = np.repeat(Pk, lens); Cr = np.repeat(c1k, lens)
        dP = tree_cur.query_ball_point(pos[Pk], r=dens_um, return_length=True) - 1
        dN = tree_next.query_ball_point(pos[c1k], r=dens_um, return_length=True) - 1
        Pl.append(Pr); Cl.append(Cr); Ql.append(qk); Dp.append(np.repeat(dP, lens)); Dn.append(np.repeat(dN, lens))
    if not Pl:
        return pd.DataFrame(columns=cols)
    P = np.concatenate(Pl); c1 = np.concatenate(Cl); q = np.concatenate(Ql); dens_P = np.concatenate(Dp); dens_n = np.concatenate(Dn)
    m = q != c1
    P, c1, q, dens_P, dens_n = P[m], c1[m], q[m], dens_P[m], dens_n[m]
    sister = np.linalg.norm(pos[q] - pos[c1], axis=1); m = sister <= sister_um
    P, c1, q, dens_P, dens_n, sister = P[m], c1[m], q[m], dens_P[m], dens_n[m], sister[m]
    pd1 = np.linalg.norm(pos[c1] - pos[P], axis=1); pd2 = np.linalg.norm(pos[q] - pos[P], axis=1)
    sym = np.abs(pd1 - pd2) / np.maximum(0.5 * (pd1 + pd2), 1e-6)
    m = np.ones(len(P), bool)
    if sym_max is not None: m &= sym <= sym_max
    if pd2_max is not None: m &= pd2 <= pd2_max
    P, c1, q, dens_P, dens_n, sister, pd1, pd2, sym = P[m], c1[m], q[m], dens_P[m], dens_n[m], sister[m], pd1[m], pd2[m], sym[m]
    n = len(P)
    if n == 0:
        return pd.DataFrame(columns=cols)
    k = t[P]
    v1 = pos[c1] - pos[P]; v2 = pos[q] - pos[P]
    cosd = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6)
    X = parent[q]; hasX = X >= 0; Xs = np.maximum(X, 0)
    dX = np.where(hasX, np.linalg.norm(pos[q] - pos[Xs], axis=1), -1.0)
    c1c = child1[c1]; qc = child1[q]; both = (nch[c1] == 1) & (nch[q] == 1)
    gd = np.where(both, np.linalg.norm(pos[np.maximum(c1c, 0)] - pos[np.maximum(qc, 0)], axis=1), -1.0)
    diverge = np.where(both, gd - sister, -99.0)
    pP = parent[P]; pp_prev = np.where((pP >= 0)[:, None], pos[np.maximum(pP, 0)], pos[P])
    mid = 0.5 * (pos[c1] + pos[q])
    # patch statistics: one gather for all prefixes
    fr = [k, k - 1, k + 1, k + 1, k + 1, k + 1, k, np.where(hasX, k, -1)]
    pt_ = [pos[P], pp_prev, pos[P], pos[c1], pos[q], mid, mid, pos[Xs]]
    if lag:
        pP2 = np.where(pP >= 0, parent[np.maximum(pP, 0)], -1)
        anchor2 = np.where((pP2 >= 0)[:, None], pos[np.maximum(pP2, 0)], pp_prev)   # Pm's fallback chain, one step deeper
        fr += [k - 2, k + 2, k + 2, k + 2, k + 3]
        pt_ += [anchor2, pos[c1], pos[q], mid, mid]
    fr = np.concatenate(fr); pt_ = np.concatenate(pt_)
    st = patch_stats(V, fr, pt_)
    o = {"t": k, "P": nid[P], "c1": nid[c1], "q": nid[q], "X": np.where(hasX, nid[Xs], -1), "pd1": pd1, "pd2": pd2, "sister": sister, "sym": sym, "cosd": cosd,
         "q_orphan": (~hasX).astype(np.float32), "dX": dX, "X_len": np.where(hasX, tlen[Xs], 0), "X_nchild": np.where(hasX, nch[Xs], 0),
         "X_has_parent": np.where(hasX, (parent[Xs] >= 0).astype(np.float32), -1.0), "P_len": tlen[P], "c1_nchild": nch[c1], "q_nchild": nch[q],
         "sP": score[P], "sc1": score[c1], "sq": score[q], "p_Pc1": fpp(P, c1), "p_Pq": fpp(P, q), "p_Xq": np.where(hasX, fpp(Xs, q), 0.0),
         "pt_Pq": fpt(P, q), "pt_Xq": np.where(hasX, fpt(Xs, q), 0.0), "pt_best_q": best_src[q], "dens_P": dens_P, "dens_n": dens_n,
         "tfrac": k / max(Tn - 1, 1), "zP": pos[P][:, 0], "gd": gd, "diverge": diverge}
    for a_, pre in enumerate(prefixes):
        sl = slice(a_ * n, (a_ + 1) * n)
        for key in PATCH_KEYS:
            o[f"{pre}_{key}"] = st[key][sl]
    dmin = np.minimum(o["C1_max"], o["Q_max"]); dmax = np.maximum(o["C1_max"], o["Q_max"])
    o["r_daughters"] = dmin / (dmax + 1e-3); o["r_mass"] = (o["C1_mean"] + o["Q_mean"]) / (o["P0_mean"] + 1e-3)
    o["r_gap"] = o["M1_ctr"] / (dmin + 1e-3); o["r_Pprev"] = o["P0_max"] / (o["Pm_max"] + 1e-3)  # NaN propagates when Pm is off-volume
    o["r_Pnext"] = o["Pp_max"] / (o["P0_max"] + 1e-3); o["r_Pbg"] = o["P0_max"] / (o["P0_bg"] + 1e-3)
    if lag: lag_ratio_columns(o)
    df = pd.DataFrame(o)
    for c_ in ("t", "P", "c1", "q", "X", "X_len", "X_nchild", "P_len", "c1_nchild", "q_nchild", "dens_P", "dens_n"):
        df[c_] = df[c_].astype(np.int64)
    return df[cols]


def label_candidates(df, nodes, edges, gt_nodes, gt_edges, tol=1, return_detail=False):
    """GT labels for candidate forks, using the SCORER'S OWN +/-1 frame tolerance.

    The official scorer (royerlab/kaggle-cell-tracking-competition,
    `tracking_cellmot.division_metrics.score_divisions`) pairs a GT division D with a predicted fork F when

      (i)  PARENT SIDE  - F itself, or F's predecessor, is matched to D or to D's GT parent
           (`_is_strongly_connected_division`: `{F} | preds(F)` must meet the nodes matched to `{D} | preds_gt(D)`,
            and F must lie in `parent_ids | successors(parent_ids)`), and
      (ii) DAUGHTER SIDE - the two branches of F, each taken as `{child} | successors(child)`, carry matched
           evidence from two DISTINCT GT daughter lineages, each taken as `{C} | successors_gt(C)`.

    That admits a fork ONE FRAME LATE (F's predecessor matches D; the branches must then reach the two GT
    GRANDCHILDREN) and ONE FRAME EARLY (F matches D's GT parent; the branches must reach the two GT DAUGHTERS),
    and nothing at +/-2.  Frozen by the organisers' own test
    `tests/test_division_sandbox_examples.py::delayed_local_division` (fork one frame late -> div_tp = 1) and
    reproduced here, against both our metric port and the organisers' package, on a -2..+2 sweep; the tolerance is
    SYMMETRIC and does NOT depend on which daughter the prediction continues through.

    Labels
      y = 1   the fork would be PAIRED with a GT division.  At most ONE candidate per GT division is labelled 1
              (exact frame first, then late, then early) because the scorer pairs one fork per division by maximum
              bipartite matching and scores a second fork for the same division as a FALSE POSITIVE - verified:
              exact fork + late fork on one GT division gives div_tp 1 / div_fp 1.  The losing duplicates are set
              to -1 (excluded) rather than 0: their image evidence is indistinguishable from a winner's, so
              teaching the model to reject them would only add label noise.
      y = 0   a fork the scorer would count as a FALSE POSITIVE: P is matched to an annotated GT node that
              continues (>= 1 child), or the fork was CONSIDERED for a GT division (parent side satisfied) but
              failed the daughter-evidence test.
      y = -1  unknown / excluded.

    `tol=0` restores the old exact-frame-only rule (P matched to a GT divider, c1 and q matched to its two
    daughters), which is what every model before 2026-09-14 was trained on.

    Adds columns y, gt_P.  With `return_detail=True` also returns a per-row detail frame for changed rows.
    """
    from .metric import Graph, match_nodes
    gt = Graph.from_frames(gt_nodes, gt_edges); pred = Graph.from_frames(nodes, edges)
    p2g = match_nodes(pred, gt)
    gt_children = {g: set(gt.successors(g)) for g in gt.node_id.tolist()}
    gP = df.P.map(p2g).fillna(-1).astype(np.int64).values; g1 = df.c1.map(p2g).fillna(-1).astype(np.int64).values; g2 = df.q.map(p2g).fillna(-1).astype(np.int64).values
    y = -np.ones(len(df), np.int64)
    if tol <= 0:
        for i in np.where(gP >= 0)[0]:
            ch = gt_children.get(int(gP[i]), set())
            if len(ch) >= 2 and g1[i] in ch and g2[i] in ch and g1[i] != g2[i]: y[i] = 1
            elif len(ch) >= 1: y[i] = 0
        df = df.copy(); df["y"] = y; df["gt_P"] = gP
        return (df, None) if return_detail else df

    # ---- parent side: which GT dividers can a pred node standing at `g` serve as parent-side evidence for?
    dividers = {g for g, ch in gt_children.items() if len(ch) >= 2}
    div_by_parentside = {}
    for g in gt.node_id.tolist():
        s = set()
        if g in dividers: s.add(g)                       # matched to the divider itself  -> dt 0
        s |= {c for c in gt_children.get(g, ()) if c in dividers}  # matched to the divider's GT parent -> dt -1
        if s: div_by_parentside[g] = s
    # lineage of each GT daughter: {C} | successors_gt(C)
    lineage = {D: [({c} | gt_children.get(c, set())) for c in sorted(gt_children[D])] for D in dividers}

    Pv = df.P.values; c1v = df.c1.values; qv = df.q.values
    par_P = np.array([(pred.predecessors(int(p)) or [-1])[0] for p in Pv], np.int64)
    gPar = np.array([p2g.get(int(x), -1) for x in par_P], np.int64)

    def evidence(node):
        """GT nodes reachable as matched evidence from one predicted branch: {node} | successors(node)."""
        out = set()
        gn_ = p2g.get(int(node), -1)
        if gn_ >= 0: out.add(gn_)
        for s in pred.successors(int(node)):
            gs = p2g.get(int(s), -1)
            if gs >= 0: out.add(gs)
        return out

    # rows that have any parent-side evidence at all (tiny subset of a multi-million-row table)
    cand_rows = np.where((np.isin(gP, list(div_by_parentside)) & (gP >= 0)) | (np.isin(gPar, list(div_by_parentside)) & (gPar >= 0)))[0] \
        if len(div_by_parentside) else np.zeros(0, np.int64)
    qualified = []   # (rank, D, row_index, dt)
    considered = set()
    for i in cand_rows:
        Ds = set()
        if gP[i] >= 0: Ds |= div_by_parentside.get(int(gP[i]), set())
        if gPar[i] >= 0: Ds |= div_by_parentside.get(int(gPar[i]), set())
        if not Ds: continue
        e1 = evidence(c1v[i]); e2 = evidence(qv[i])
        for D in Ds:
            considered.add(int(i))
            L = lineage[D]
            m1 = {d for d, s in enumerate(L) if e1 & s}; m2 = {d for d, s in enumerate(L) if e2 & s}
            if not m1 or not m2 or len(m1 | m2) < 2: continue     # no size-2 bipartite matching
            dt = int(df.t.values[i]) - gt.t_of(int(D))
            rank = 0 if int(gP[i]) == int(D) else (1 if dt == 0 else (2 if dt > 0 else 3))
            qualified.append((rank, int(D), int(i), dt))

    # one fork per GT division, one division per fork (mirrors the scorer's bipartite pairing)
    qualified.sort(key=lambda r: (r[0], r[1], r[2]))
    used_D, used_row, winners = set(), set(), {}
    for rank, D, i, dt in qualified:
        if D in used_D or i in used_row: continue
        used_D.add(D); used_row.add(i); winners[i] = (D, dt, rank)

    for i in np.where(gP >= 0)[0]:
        if len(gt_children.get(int(gP[i]), set())) >= 1: y[i] = 0
    for i in considered:
        if y[i] < 0: y[i] = 0                     # considered-and-rejected forks are real FPs for the scorer
    for rank, D, i, dt in qualified:
        if i not in winners: y[i] = -1            # ambiguous duplicate: drop from training
    for i in winners: y[i] = 1

    df = df.copy(); df["y"] = y; df["gt_P"] = gP
    if not return_detail:
        return df
    import pandas as pd
    det = pd.DataFrame([{"row": i, "P": int(Pv[i]), "c1": int(c1v[i]), "q": int(qv[i]), "t": int(df.t.values[i]),
                         "gt_div": winners[i][0], "dt": winners[i][1], "rank": winners[i][2]} for i in sorted(winners)])
    return df, det


def feature_columns(booster):
    """Feature names stored in the LightGBM model (trained with feature_name=...)."""
    return list(booster.feature_name())


def score_candidates(booster, df):
    F = feature_columns(booster)
    return booster.predict(df[F].values.astype(np.float32)) if len(df) else np.zeros(0)


def add_divisions(edges, cands, thr=0.5, relink_thr=None, cap_frac=0.00375):
    """Greedy fork insertion on a division-free edge table (source_id, target_id[, conf]). cands: DataFrame with node-id
    columns P, q, X and probability p. Accept p >= thr, one fork per P (which must still have exactly one child), one use
    per q; a q already linked from X is only accepted when relink_thr is set and p >= relink_thr (its X->q edge is cut).
    Global cap = cap_frac x |edges| (anchor SAFE_DIV_GLOBAL_FRAC_CAP). Returns (edges, n_added, n_cut)."""
    import pandas as pd, collections
    edges = edges.copy(); src = edges.source_id.values; tgt = edges.target_id.values
    nch = collections.Counter(src.tolist()); has_par = set(tgt.tolist())
    cap = max(1, int(round(len(edges) * cap_frac)))
    if len(cands) == 0:
        return edges, 0, 0
    cands = cands.sort_values("p", ascending=False)
    add, cut, used_q, used_P = [], [], set(), set()
    for r in cands.itertuples():
        if r.p < thr or len(add) >= cap: break
        if nch.get(r.P, 0) != 1 or r.P in used_P or r.q in used_q: continue
        if r.q in has_par:
            if relink_thr is None or r.p < relink_thr: continue
            cut.append(r.q)
        add.append((r.P, r.q, float(r.p))); used_P.add(r.P); used_q.add(r.q)
    if cut:
        edges = edges[~edges.target_id.isin(set(cut))]
    if add:
        extra = pd.DataFrame(add, columns=["source_id", "target_id", "conf"])
        if "conf" not in edges.columns: extra = extra[["source_id", "target_id"]]
        edges = pd.concat([edges, extra], ignore_index=True)
    return edges, len(add), len(cut)


def apply_divclf(edges_df, t, pos_um, score, edges, keep, pp, pt, V, booster, thr=0.5, relink_thr=None, cap_frac=0.00375, sym_max=1.5, pd2_max=10.0, parent_um=11.0, sister_um=16.5):
    """Kernel entry point: tracker output + probabilities + pool4 volume + booster -> (edges_df with forks, stats)."""
    df = build_candidates(t, pos_um, score, edges, keep, pp, pt, V, parent_um=parent_um, sister_um=sister_um, sym_max=sym_max, pd2_max=pd2_max)
    if len(df) == 0:
        return edges_df, {"divclf_cands": 0, "divclf_added": 0, "divclf_cut": 0}
    df["p"] = score_candidates(booster, df)
    e2, n_add, n_cut = add_divisions(edges_df, df[["P", "q", "X", "p"]], thr=thr, relink_thr=relink_thr, cap_frac=cap_frac)
    st = {"divclf_cands": int(len(df)), "divclf_added": int(n_add), "divclf_cut": int(n_cut), "divclf_p_max": float(df.p.max()),
          "divclf_cap": int(max(1, round(len(edges_df) * cap_frac)))}
    try:   # diagnostic ladder: what every other acceptance / relink threshold WOULD have done on this same graph
        st.update(accept_ladder(edges_df, df[["P", "q", "X", "p"]]))
    except Exception as _ex:
        st["divclf_ladder_error"] = repr(_ex)[:120]
    return e2, st


LADDER = (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05)


def accept_ladder(edges, cands, thrs=LADDER):
    """DIAGNOSTIC ONLY (never changes the graph): replay the greedy of add_divisions once over all candidates in
    p-descending order and record, for every threshold in thrs, (a) how many raw candidates sit at or above it,
    (b) how many forks the greedy would accept with relink OFF, (c) how many it would accept with relink at the SAME
    threshold, and (d) how many existing edges that relink variant would cut. The greedy visits candidates in a fixed
    order and its constraints depend only on what it has already accepted, so the accepted set at a lower threshold is
    a superset of the one at a higher threshold and a single pass yields the whole ladder. The global cap is NOT
    applied here; it is reported separately as divclf_cap."""
    import collections
    src = edges.source_id.values
    nch = collections.Counter(src.tolist()); has_par = set(edges.target_id.values.tolist())
    c = cands.sort_values("p", ascending=False)
    P = c.P.values; Q = c.q.values; PR = c.p.values
    out = {}
    for relink in (0, 1):
        used_q, used_P = set(), set()
        n_add = n_cut = i = 0
        acc, cuts, raw = [], [], []
        for th in thrs:
            while i < len(PR) and PR[i] >= th:
                p_ = int(P[i]); q_ = int(Q[i]); i += 1
                if nch.get(p_, 0) != 1 or p_ in used_P or q_ in used_q:
                    continue
                if q_ in has_par:
                    if not relink:
                        continue
                    n_cut += 1
                n_add += 1; used_P.add(p_); used_q.add(q_)
            acc.append(n_add); cuts.append(n_cut); raw.append(i)
        if relink:
            out["divclf_acc_relink"] = acc; out["divclf_cut_relink"] = cuts
        else:
            out["divclf_acc"] = acc; out["divclf_raw"] = raw
    out["divclf_ladder"] = [float(x) for x in thrs]
    return out
