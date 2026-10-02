"""Tracker v1: motion-aware Hungarian linking + gap closing + heuristic divisions + pruning.

Nodes: arrays t (N,), pos_um (N,3), score (N,). Works on any detections.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

RAW_SCALE = np.array([1.625, 0.40625, 0.40625], np.float32)


class TrackConfig:
    tight_um = 10.0         # first-pass gate (with death_cost the gates matter less; CV: 10)
    relaxed_um = 10.0       # second pass on leftovers
    max_link_um = 14.0
    velocity_w = 0.0        # constant-velocity prediction weight (CV: 0 > 0.5; z-jitter)
    raw_w = 0.05            # small penalty on raw displacement
    score_w = 0.0           # bonus for detection score of target (um units)
    learned_w = 0.0         # bonus (um units) multiplied by learned edge probability p in [0,1]
    learned_min_p = 0.0     # pairs with p below this are disallowed (when a scorer is used)
    learned_gate_um = 12.0  # optional 3rd pass on leftovers (only with a learned scorer): CV +0.003
    learned_gate_p = 0.3
    death_cost = 3.5        # augmented Hungarian with per-node no-link cost (um units); CV +0.003
    reserve_thr = 0.0       # >0: detections with score < reserve_thr are a reserve pool used only to extend tracks
    reserve_um = 5.0        # max distance from an unlinked source (or to an unlinked target) to a reserve node
    gap_close = True
    gap_um_per_step = 4.5   # gate per step for 1-frame gaps (total 2x) (CV: 4.5 > 5.5)
    gap_reuse_um = 3.2      # reuse an unlinked detection near the midpoint
    gap_cap_frac = 0.05
    divisions = True
    div_min_p = 0.0         # >0 with a learned pair_prob: require p(parent -> 2nd child) >= div_min_p
    div_rank_by_p = False   # rank division proposals by learned probability instead of distance
    div_parent_um = 9.0
    div_sister_um = 14.0
    div_child_um = 10.0
    div_symmetry_tau = 0.6
    div_require_divergence = True
    div_diverge_um = 1.0    # (CV: 1.0 > 2.25)
    div_frame_cap_frac = 0.0076
    div_global_cap_frac = 0.00375
    min_track_len = 6       # holdout25 sweep 2026-09-06: 3/5/6/8 = .8757/.8773/.8780/.8784 (anchor uses 6)
    keep_fork_components = True
    prune_isolated = True
    end_prune_score = 0.0    # drop track endpoints (degree<=1) whose detection score is below this (2 iterations)
    div_mode = "heuristic"   # "heuristic" | "classifier" | "none"
    div_booster = None       # LightGBM booster over cellmot.divfeat features (classifier mode)
    div_thr = 0.5            # acceptance probability for classifier mode
    div_cand_parent_um = 13.5
    div_cand_sister_um = 20.5


def _hungarian_links(src_pos, src_vel, tgt_pos, tgt_score, gate, cfg: TrackConfig, bonus=None):
    """Return list of (i, j, cost). src_vel may contain nan for unknown velocity. bonus: optional (n_src,n_tgt)
    matrix subtracted from the cost (e.g. learned edge probability * cfg.learned_w)."""
    if len(src_pos) == 0 or len(tgt_pos) == 0:
        return []
    raw = cdist(src_pos, tgt_pos)
    pred = src_pos + cfg.velocity_w * np.nan_to_num(src_vel)
    motion = cdist(pred, tgt_pos)
    cost = motion + cfg.raw_w * raw - cfg.score_w * tgt_score[None, :]
    if bonus is not None:
        cost = cost - bonus
    allowed = raw <= gate
    big = 1e6
    cost = np.where(allowed, cost, big)
    rows = np.where(allowed.any(1))[0]; cols = np.where(allowed.any(0))[0]
    if len(rows) == 0:
        return []
    sub = cost[np.ix_(rows, cols)]
    if cfg.death_cost > 0:
        ns, nt = sub.shape; D = float(cfg.death_cost)
        aug = np.full((ns + nt, ns + nt), big, np.float64)
        aug[:ns, :nt] = sub
        aug[:ns, nt:] = np.where(np.eye(ns, dtype=bool), D, big)
        aug[ns:, :nt] = np.where(np.eye(nt, dtype=bool), D, big)
        aug[ns:, nt:] = sub.T if ns == nt else 0.0
        aug[ns:, nt:] = 0.0
        r, c = linear_sum_assignment(aug)
        out = []
        for a, b in zip(r, c):
            if a < ns and b < nt and sub[a, b] < big:
                out.append((int(rows[a]), int(cols[b]), float(sub[a, b])))
        return out
    r, c = linear_sum_assignment(sub)
    out = []
    for a, b in zip(r, c):
        if sub[a, b] < big:
            out.append((int(rows[a]), int(cols[b]), float(sub[a, b])))
    return out


def track(t: np.ndarray, pos_um: np.ndarray, score: np.ndarray, cfg: TrackConfig = TrackConfig(), pair_prob: dict | None = None, disp=None, motion_w: float = 1.0, gap_motion: bool = True):
    """Returns (t, pos, score, edges list of (src, tgt, conf), keep mask, stats); node ids are indices into the
    (possibly extended) arrays. pair_prob: optional {(i, j): p} learned edge probabilities for candidate pairs.
    disp: optional (N,3) external per-node displacement prediction in um (e.g. cellmot.flow NCC prior); when given,
    cfg.velocity_w is replaced by motion_w and the prediction pos + motion_w * disp is used instead of the track
    history velocity, in the linking passes and (gap_motion) in the 1-frame gap closing. Default path unchanged."""
    t = np.asarray(t); pos = np.asarray(pos_um, np.float32); score = np.asarray(score, np.float32)
    N = len(t)
    if disp is not None:
        import copy; cfg = copy.copy(cfg); cfg.velocity_w = float(motion_w); disp = np.nan_to_num(np.asarray(disp, np.float32))
    T = int(t.max()) + 1 if N else 0
    is_reserve = (score < cfg.reserve_thr) if cfg.reserve_thr > 0 else np.zeros(N, bool)
    active = np.zeros(N, bool); active[~is_reserve] = True   # reserve nodes become active once linked
    by_t = {tt: np.where((t == tt) & ~is_reserve)[0] for tt in range(T)}
    res_by_t = {tt: np.where((t == tt) & is_reserve)[0] for tt in range(T)} if cfg.reserve_thr > 0 else {}
    n_reserve_used = 0
    parent = -np.ones(N, np.int64); children = defaultdict(list)
    edge_conf = {}
    pair_by_src = defaultdict(list)
    if pair_prob is not None:
        for (ii, jj), p_ in pair_prob.items():
            pair_by_src[ii].append(((ii, jj), p_))
    # ---- frame-to-frame linking (two passes: tight, then relaxed on leftovers)
    for tt in range(T - 1):
        si = by_t[tt]; ti = by_t.get(tt + 1, np.array([], np.int64))
        if len(si) == 0 or len(ti) == 0:
            continue
        vel = np.full((len(si), 3), np.nan, np.float32)
        if disp is not None:
            vel[:] = disp[si]
        else:
            for k, i in enumerate(si):
                if parent[i] >= 0:
                    vel[k] = pos[i] - pos[parent[i]]
        unlinked_s = np.ones(len(si), bool); unlinked_t = np.ones(len(ti), bool)
        passes = [(cfg.tight_um, cfg.learned_min_p), (cfg.relaxed_um, cfg.learned_min_p)]
        if pair_prob is not None and cfg.learned_gate_um > cfg.relaxed_um:
            passes.append((cfg.learned_gate_um, cfg.learned_gate_p))
        for gate, min_p in passes:
            gate = min(gate, cfg.max_link_um)
            ss = np.where(unlinked_s)[0]; ts_ = np.where(unlinked_t)[0]
            if len(ss) == 0 or len(ts_) == 0:
                break
            bonus = None
            if pair_prob is not None and (cfg.learned_w != 0 or min_p > 0):
                bonus = np.full((len(ss), len(ts_)), (-1e6 if min_p > 0 else 0.0), np.float32)
                col = {int(j_): b_ for b_, j_ in enumerate(ti[ts_])}
                for a_, i_ in enumerate(si[ss]):
                    for (ii, jj), p_ in pair_by_src.get(int(i_), ()):
                        b_ = col.get(jj)
                        if b_ is not None:
                            bonus[a_, b_] = cfg.learned_w * p_ if p_ >= min_p else -1e6
            links = _hungarian_links(pos[si[ss]], vel[ss], pos[ti[ts_]], score[ti[ts_]], gate, cfg, bonus)
            for a, b, c in links:
                i = int(si[ss[a]]); j = int(ti[ts_[b]])
                parent[j] = i; children[i].append(j)
                edge_conf[(i, j)] = pair_prob.get((i, j), 1.0 / (1.0 + max(c, 0.0))) if pair_prob is not None else 1.0 / (1.0 + max(c, 0.0))
                unlinked_s[ss[a]] = False; unlinked_t[ts_[b]] = False
        # ---- reserve recovery: extend unlinked sources into low-score reserve detections at t+1
        if cfg.reserve_thr > 0 and len(res_by_t.get(tt + 1, [])) and unlinked_s.any():
            ri = res_by_t[tt + 1]; ri = ri[~active[ri]]
            # eligible sources: unlinked, with a track history (parent exists) and not themselves promoted reserve nodes
            elig = np.array([k for k in np.where(unlinked_s)[0] if parent[si[k]] >= 0 and not is_reserve[si[k]]], np.int64)
            if len(ri) and len(elig):
                ss = elig
                d = cdist(pos[si[ss]], pos[ri]); cost = np.where(d <= cfg.reserve_um, d, 1e6)
                r_, c_ = linear_sum_assignment(cost)
                for a, b in zip(r_, c_):
                    if cost[a, b] >= 1e6:
                        continue
                    i = int(si[ss[a]]); j = int(ri[b])
                    parent[j] = i; children[i].append(j); edge_conf[(i, j)] = 0.5; active[j] = True; n_reserve_used += 1
                    unlinked_s[ss[a]] = False
                # promoted reserve nodes take part in the next frame pair as sources
                by_t[tt + 1] = np.where((t == tt + 1) & (active | ~is_reserve))[0]
    # ---- 1-frame gap closing: track ends at tt <-> track starts at tt+2
    new_nodes = []  # (t, pos, score)
    if cfg.gap_close and T > 2:
        n_gap = 0; cap = int(cfg.gap_cap_frac * N) + 1
        for tt in range(T - 2):
            ends = np.array([i for i in by_t[tt] if len(children[i]) == 0], np.int64)
            starts = np.array([j for j in by_t.get(tt + 2, []) if parent[j] < 0], np.int64)
            if len(ends) == 0 or len(starts) == 0:
                continue
            vel = np.full((len(ends), 3), np.nan, np.float32)
            if disp is not None:
                vel[:] = disp[ends] if gap_motion else 0.0
            else:
                for k, i in enumerate(ends):
                    if parent[i] >= 0:
                        vel[k] = pos[i] - pos[parent[i]]
            pred = pos[ends] + 2 * cfg.velocity_w * np.nan_to_num(vel)
            raw = cdist(pos[ends], pos[starts]); motion = cdist(pred, pos[starts])
            gate = 2 * cfg.gap_um_per_step
            cost = np.where(raw <= gate, motion + cfg.raw_w * raw, 1e6)
            r, c = linear_sum_assignment(cost)
            mids = by_t.get(tt + 1, np.array([], np.int64))
            free_mid = np.array([m for m in mids if parent[m] < 0 and len(children[m]) == 0], np.int64)
            tree = cKDTree(pos[free_mid]) if len(free_mid) else None
            used_mid = set(); batch_new = []
            for a, b in zip(r, c):
                if cost[a, b] >= 1e6 or n_gap >= cap:
                    continue
                i = int(ends[a]); j = int(starts[b])
                midp = 0.5 * (pos[i] + pos[j])
                m = -1
                if tree is not None:
                    d, k = tree.query(midp)
                    if d <= cfg.gap_reuse_um and int(free_mid[k]) not in used_mid:
                        m = int(free_mid[k]); used_mid.add(m)
                if m < 0:
                    m = len(t) + len(batch_new)
                    batch_new.append((tt + 1, midp, 0.5 * (score[i] + score[j])))
                    parent = np.append(parent, -1)
                parent[m] = i; children[i].append(m); edge_conf[(i, m)] = 0.5
                parent[j] = m; children[m].append(j); edge_conf[(m, j)] = 0.5
                n_gap += 1
            if batch_new:
                t = np.concatenate([t, np.array([n[0] for n in batch_new])])
                pos = np.concatenate([pos, np.stack([n[1] for n in batch_new]).astype(np.float32)])
                score = np.concatenate([score, np.array([n[2] for n in batch_new], np.float32)])
                by_t[tt + 1] = np.where(t == tt + 1)[0]
                new_nodes.extend(batch_new)
    Nn = len(t)
    # ---- divisions
    n_div = 0
    if cfg.divisions and cfg.div_mode == "classifier" and cfg.div_booster is not None:
        from .divfeat import division_candidates
        cands, F = division_candidates(t, pos, score, parent, children, cfg.div_cand_parent_um, cfg.div_cand_sister_um, pair_prob)
        if len(cands):
            pdiv = cfg.div_booster.predict(F)
            order = np.argsort(-pdiv); used_p = set()
            for k in order:
                if pdiv[k] < cfg.div_thr:
                    break
                p_, c1, q = cands[k]
                if p_ in used_p or parent[q] >= 0 or len(children[p_]) != 1:
                    continue
                parent[q] = p_; children[p_].append(q); edge_conf[(p_, q)] = float(pdiv[k]); used_p.add(p_); n_div += 1
    elif cfg.divisions and cfg.div_mode == "heuristic":
        gcap = int(cfg.div_global_cap_frac * max(len(edge_conf), 1)) + 1
        for tt in range(T - 1):
            si = by_t[tt]; ti = by_t.get(tt + 1, np.array([], np.int64))
            orphans = np.array([j for j in ti if parent[j] < 0], np.int64)
            parents1 = np.array([i for i in si if len(children[i]) == 1], np.int64)
            if len(orphans) == 0 or len(parents1) == 0:
                continue
            tree_o = cKDTree(pos[orphans]); tree_all = cKDTree(pos[ti])
            proposals = []
            for i in parents1:
                c1 = children[i][0]
                child_d = float(np.linalg.norm(pos[c1] - pos[i]))
                if child_d > cfg.div_child_um:
                    continue
                for k in tree_o.query_ball_point(pos[i], r=cfg.div_parent_um):
                    q = int(orphans[k])
                    pd_ = float(np.linalg.norm(pos[q] - pos[i])); sd = float(np.linalg.norm(pos[q] - pos[c1]))
                    if sd > cfg.div_sister_um:
                        continue
                    # mutual nearest: q's nearest node at t+1 (excluding itself) should be c1
                    dd, kk = tree_all.query(pos[q], k=2)
                    if int(ti[kk[1]]) != c1:
                        continue
                    if cfg.div_symmetry_tau > 0 and abs(child_d - pd_) / max(0.5 * (child_d + pd_), 1e-6) > cfg.div_symmetry_tau:
                        continue
                    if cfg.div_require_divergence:
                        if len(children[c1]) != 1 or len(children[q]) != 1:
                            continue
                        gd = float(np.linalg.norm(pos[children[c1][0]] - pos[children[q][0]]))
                        if gd - sd < cfg.div_diverge_um:
                            continue
                    if pair_prob is not None and (cfg.div_min_p > 0 or cfg.div_rank_by_p):
                        pq = pair_prob.get((int(i), q), 0.0)
                        if pq < cfg.div_min_p:
                            continue
                        proposals.append(((-pq if cfg.div_rank_by_p else pd_ + 0.15 * sd), int(i), q, pd_))
                    else:
                        proposals.append((pd_ + 0.15 * sd, int(i), q, pd_))
            proposals.sort()
            fcap = max(1, int(round(len(parents1) * cfg.div_frame_cap_frac)))
            used = set()
            for s, i, q, pd_ in proposals:
                if len(used) >= fcap or n_div >= gcap:
                    break
                if i in used or parent[q] >= 0 or len(children[i]) != 1:
                    continue
                parent[q] = i; children[i].append(q); edge_conf[(i, q)] = 1.0 / (1.0 + pd_)
                used.add(i); n_div += 1
    # ---- pruning: isolated nodes, short components
    edges = [(int(i), int(j), float(c)) for (i, j), c in edge_conf.items()]
    keep = np.ones(Nn, bool)
    if cfg.prune_isolated or cfg.min_track_len > 1:
        comp = -np.ones(Nn, np.int64); adj = defaultdict(list)
        for i, j, _ in edges:
            adj[i].append(j); adj[j].append(i)
        ncomp = 0
        for s in range(Nn):
            if comp[s] >= 0:
                continue
            stack = [s]; comp[s] = ncomp; members = [s]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if comp[v] < 0:
                        comp[v] = ncomp; stack.append(v); members.append(v)
            has_fork = any(len(children[m]) >= 2 for m in members)
            if len(members) < max(cfg.min_track_len, 2 if cfg.prune_isolated else 1) and not (cfg.keep_fork_components and has_fork):
                keep[members] = False
            ncomp += 1
    if cfg.reserve_thr > 0:
        keep[:N] &= active[:N] | ~is_reserve
    n_endpruned = 0
    if cfg.end_prune_score > 0:
        for _ in range(2):
            deg = np.zeros(Nn, np.int64)
            for i, j, _c in edges:
                if keep[i] and keep[j]:
                    deg[i] += 1; deg[j] += 1
            drop = keep & (deg <= 1) & (score < cfg.end_prune_score)
            if not drop.any():
                break
            keep[drop] = False; n_endpruned += int(drop.sum())
        edges = [(i, j, c) for i, j, c in edges if keep[i] and keep[j]]
    stats = {"n_nodes_in": N, "n_gap_nodes": len(new_nodes), "n_div": n_div, "n_edges": len(edges), "n_kept": int(keep.sum()), "n_endpruned": n_endpruned, "n_reserve_used": n_reserve_used}
    return t, pos, score, edges, keep, stats


def to_graph_frames(t, pos_um, score, edges, keep, id_offset=1):
    """Convert tracker output to (nodes_df, edges_df) in raw voxel coordinates with 1-based ids; edges ordered by
    descending confidence per source."""
    import pandas as pd
    idx = np.where(keep)[0]
    remap = {int(i): id_offset + k for k, i in enumerate(idx)}
    raw = pos_um[idx] / RAW_SCALE
    nodes = pd.DataFrame({"node_id": [remap[int(i)] for i in idx], "t": t[idx].astype(int), "z": raw[:, 0], "y": raw[:, 1], "x": raw[:, 2], "score": score[idx]})
    e = [(remap[i], remap[j], c) for i, j, c in edges if i in remap and j in remap]
    e.sort(key=lambda r: (r[0], -r[2]))
    edf = pd.DataFrame(e, columns=["source_id", "target_id", "conf"]) if e else pd.DataFrame(columns=["source_id", "target_id", "conf"])
    return nodes, edf


def learned_pair_probs(t, pos_um, score, booster, max_um=12.0, k=8, cfg: TrackConfig | None = None):
    """Two-pass helper: geometric pre-tracking (for motion features) -> candidate features -> booster probabilities.
    Returns {(i, j): p}."""
    from .edgefeat import candidates, features
    c0 = TrackConfig() if cfg is None else cfg
    pre = TrackConfig(); pre.tight_um, pre.relaxed_um, pre.velocity_w, pre.raw_w = c0.tight_um, c0.relaxed_um, c0.velocity_w, c0.raw_w
    pre.divisions = False; pre.gap_close = False; pre.min_track_len = 1; pre.prune_isolated = False
    tt_, pp_, s2_, edges, keep, st = track(t, pos_um, score, pre)
    prev_of = -np.ones(len(t), np.int64)
    for i, j, c in edges:
        if j < len(t): prev_of[j] = i
    I, J = candidates(t, pos_um, score, max_um, k)
    if len(I) == 0:
        return {}
    F = features(t, pos_um, score, I, J, prev_of=prev_of)
    p = booster.predict(F)
    return {(int(i), int(j)): float(pi) for i, j, pi in zip(I, J, p)}
