"""Standalone port of the public biohub-tracking-support-pack model (TemporalUNet3D + SimpleNodeTransformer with a
detection head), for running the pack weights on our strided (1,4,4) uint16 cache without tracksdata/polars/dask.

Conventions (from repo/scripts/predict_unet_transformer.py):
  * input: frame[:, ::4, ::4] (strided, no pooling), normalized (x - q001) / (q999 - q001 + 1e-6), clamp(min=0)
  * windows of W=2 frames; detection for the *second* frame of each window (frame 0 from window 0); edges (t -> t+1)
    from the same window UNet features
  * peaks: 3x3x3 (5um) max-pool local max on logits, sigmoid > thr; integer grid coords (z, y, x); original voxel = grid*(1,4,4)
  * edge logits = transformer(feat + pos-embed, coords in ORIGINAL voxel units); the pack uses softmax over sources (dim 0)
"""
from __future__ import annotations
import json, math, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

POS_EMBED_DIM = 8
GRID_UM = 1.625  # strided grid spacing in um (z: 1.625, y/x: 4*0.40625)


def _conv_block(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
                         nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))


class _TemporalAttention(nn.Module):
    def __init__(self, channels, n_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels); self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x):
        B, T, C = x.shape[:3]; spatial = x.shape[3:]; S = math.prod(spatial)
        h = x.reshape(B, T, C, S).permute(0, 3, 1, 2).reshape(B * S, T, C)
        h = self.norm(h); h, _ = self.attn(h, h, h, need_weights=False)
        h = h.reshape(B, S, T, C).permute(0, 2, 3, 1).reshape(B, T, C, *spatial)
        return x + h


class TemporalUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, layers=(32, 64, 128), skip_fullres_temporal=True):
        super().__init__()
        layers = list(layers)
        self.encoder_blocks = nn.ModuleList(); self.temporal_blocks = nn.ModuleList(); prev = in_channels
        for i, ch in enumerate(layers):
            self.encoder_blocks.append(_conv_block(prev, ch))
            self.temporal_blocks.append(nn.Identity() if (skip_fullres_temporal and i == 0) else _TemporalAttention(ch)); prev = ch
        self.pool = nn.MaxPool3d(2, 2)
        self.upsamples = nn.ModuleList(); self.decoder_blocks = nn.ModuleList()
        for i in range(len(layers) - 1, 0, -1):
            self.upsamples.append(nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
            self.decoder_blocks.append(_conv_block(layers[i] + layers[i - 1], layers[i - 1]))
        self.head = nn.Conv3d(layers[0], out_channels, 1)

    def forward(self, x):  # (B,T,C,Z,Y,X) -> (B,T,C_out,Z,Y,X)
        B, T = x.shape[:2]; x = x.reshape(B * T, *x.shape[2:]); skips = []
        for i, (block, temporal) in enumerate(zip(self.encoder_blocks, self.temporal_blocks)):
            if i > 0: x = self.pool(x)
            x = block(x)
            x = temporal(x.reshape(B, T, *x.shape[1:])).reshape(B * T, *x.shape[1:])
            if i < len(self.encoder_blocks) - 1: skips.append(x)
        for up, block, skip in zip(self.upsamples, self.decoder_blocks, skips[::-1]):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]: x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = block(torch.cat([x, skip], 1))
        x = self.head(x)
        return x.reshape(B, T, *x.shape[1:])


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=64, n_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim); self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=dropout)
        mh = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, mh), nn.GELU(), nn.Dropout(dropout), nn.Linear(mh, hidden_dim), nn.Dropout(dropout))

    def forward(self, q, kv, kv_mask=None):
        kpm = ~kv_mask if kv_mask is not None else None
        a, _ = self.cross_attn(self.norm1(q), self.norm1(kv), self.norm1(kv), key_padding_mask=kpm)
        q = q + a; q = q + self.mlp(self.norm2(q)); return q


class SimpleNodeTransformer(nn.Module):
    def __init__(self, feat_dim=33, hidden_dim=128, n_heads=4, n_blocks=4, mlp_ratio=2.0, dropout=0.3, pair_chunk_size=32):
        super().__init__()
        self.pair_chunk_size = pair_chunk_size
        self.proj = nn.Linear(feat_dim, hidden_dim); self.norm_in = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([CrossAttentionBlock(hidden_dim, n_heads, mlp_ratio, dropout) for _ in range(n_blocks)])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.pair_mlp = nn.Sequential(nn.Linear(hidden_dim * 2 + 3, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, feat_t, feat_t1, coords_t, coords_t1, mask_t=None, mask_t1=None):
        unb = feat_t.ndim == 2
        if unb:
            feat_t, feat_t1, coords_t, coords_t1 = feat_t[None], feat_t1[None], coords_t[None], coords_t1[None]
        q = self.norm_in(self.proj(feat_t)); k = self.norm_in(self.proj(feat_t1))
        ckpt = torch.is_grad_enabled()
        for b in self.blocks:
            if ckpt:
                from torch.utils.checkpoint import checkpoint as _ck
                q = _ck(lambda q_, k_, m_, _b=b: _b(q_, k_, kv_mask=m_), q, k, mask_t1, use_reentrant=False)
                k = _ck(lambda k_, q_, m_, _b=b: _b(k_, q_, kv_mask=m_), k, q, mask_t, use_reentrant=False)
            else:
                q = b(q, k, kv_mask=mask_t1); k = b(k, q, kv_mask=mask_t)
        q = self.norm_out(q); k = self.norm_out(k)
        N_t = q.shape[1]; chunk = self.pair_chunk_size or N_t; outs = []

        def _chunk(qc, kk, cc, cc1, _pm=self.pair_mlp):
            nc = qc.shape[1]; n1 = kk.shape[1]
            qe = qc.unsqueeze(2).expand(-1, -1, n1, -1); ke = kk.unsqueeze(1).expand(-1, nc, -1, -1)
            rel = (cc.unsqueeze(2) - cc1.unsqueeze(1)) / 100.0
            return _pm(torch.cat([qe, ke, rel], -1)).squeeze(-1)
        for i in range(0, N_t, chunk):
            qc = q[:, i:i + chunk]; cc = coords_t[:, i:i + chunk]
            if ckpt:
                from torch.utils.checkpoint import checkpoint as _ck
                outs.append(_ck(_chunk, qc, k, cc, coords_t1, use_reentrant=False))
            else:
                outs.append(_chunk(qc, k, cc, coords_t1))
        logits = torch.cat(outs, 1)
        return logits[0] if unb else logits


class UNetNodeTransformer(nn.Module):
    def __init__(self, unet, unet_out_channels, pos_feat_dim, hidden_dim=128, n_heads=4, n_blocks=4, dropout=0.3):
        super().__init__()
        self.unet = unet; self.unet_out_channels = unet_out_channels
        self.detect_head = nn.Conv3d(unet_out_channels, 1, 1)
        self.transformer = SimpleNodeTransformer(feat_dim=unet_out_channels + pos_feat_dim, hidden_dim=hidden_dim, n_heads=n_heads, n_blocks=n_blocks, dropout=dropout)

    def encode(self, imgs):  # (B,W,Z,Y,X) -> unet_out (B,W,C,Z,Y,X), det logits list of (B,1,Z,Y,X)
        u = self.unet(imgs.unsqueeze(2)); W = u.shape[1]
        return u, [self.detect_head(u[:, i]) for i in range(W)]

    def index_features(self, feat_map, coords):  # feat_map (C,Z,Y,X); coords (N,3) grid ints -> (N,C)
        sp = feat_map.shape[1:]
        z = coords[:, 0].long().clamp(0, sp[0] - 1); y = coords[:, 1].long().clamp(0, sp[1] - 1); x = coords[:, 2].long().clamp(0, sp[2] - 1)
        return feat_map[:, z, y, x].T

    def predict_edges(self, f_src, f_tgt, c_src, c_tgt, p_src, p_tgt):
        return self.transformer(torch.cat([f_src, p_src], -1), torch.cat([f_tgt, p_tgt], -1), c_src, c_tgt)


def pos_features(coords_tzyx: np.ndarray, image_shape, dim=POS_EMBED_DIM) -> np.ndarray:
    """coords (N,4) [t,z,y,x] (t = window-relative), image_shape (W,Z,Y,X) -> (N, 4*dim) float32."""
    cols = [coords_tzyx[:, i] / max(s, 1) for i, s in enumerate(image_shape)]
    freqs = 2 ** np.arange(dim // 2)
    out = []
    for v in cols:
        a = v[:, None] * freqs * np.pi; out += [np.sin(a), np.cos(a)]
    return np.concatenate(out, 1).astype(np.float32)


def load_pub_model(weights_path: str, device="cuda", config_path: str | None = None):
    cfg = {"unet_out_channels": 32, "unet_layers": [32, 64, 128], "downsample": [1, 4, 4], "window_size": 2, "pool_kernel_um": 5.0}
    cp = config_path or os.path.join(os.path.dirname(weights_path), "config.json")
    if os.path.exists(cp): cfg.update(json.load(open(cp)))
    unet = TemporalUNet3D(1, cfg["unet_out_channels"], cfg["unet_layers"])
    model = UNetNodeTransformer(unet, cfg["unet_out_channels"], 4 * POS_EMBED_DIM)
    sd = torch.load(weights_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
    model.to(device).eval(); model.cfg = cfg
    return model


# ----------------------------------------------------------------------------- D4 TTA on the (Y,X) plane
def _d4_apply(x, k, tr):
    if tr: x = x.transpose(-2, -1)
    return torch.rot90(x, k, dims=(-2, -1)) if k else x


def _d4_invert(y, k, tr):
    if k: y = torch.rot90(y, -k, dims=(-2, -1))
    return y.transpose(-2, -1) if tr else y


D4 = [(k, tr) for tr in (False, True) for k in range(4)]

# EDGE-FEATURE TTA diagnostics (public patch prints EDGE_TTA_ACTIVE with the mean absolute feature delta; we print the
# first few windows and keep a running mean so a silent no-op is impossible)
_EDGE_TTA_LOG = {"n": 0, "sum": 0.0, "cnt": 0}


def edge_tta_stats():
    c = _EDGE_TTA_LOG["cnt"]
    return {"windows": c, "mean_abs_feat_delta": (_EDGE_TTA_LOG["sum"] / c) if c else 0.0}


def _align(sec, prim):
    """Align secondary logits to the primary mean/std (anchor notebook dual-seed blend)."""
    pm = prim.mean(); ps = prim.float().std(unbiased=False).clamp_min(1e-4)
    sm = sec.mean(); ss = sec.float().std(unbiased=False).clamp_min(1e-4)
    return (sec - sm) * (ps / ss).clamp(0.5, 2.0) + pm


def _n_peaks(logit, thr, pool):
    x = logit[None, None]
    pooled = F.max_pool3d(x, pool, stride=1, padding=tuple(k // 2 for k in pool))
    return int(((x == pooled) & (torch.sigmoid(x) > thr)).sum())


@torch.no_grad()
def encode_window(models, imgs, tta_views, det_w, guard_thr=None, guard_retention=0.9, pool=(3, 3, 3), det_blend="align", edge_tta=0):
    """models: list of (model, weight); the first is primary. imgs (1,W,Z,Y,X) normalized.
    det_blend: "align" = extra logits aligned to the primary mean/std (anchor dual-seed blend, default); "prob" = blend in
    probability space (scale-free: needed when the seeds were trained with different negative weights, e.g. a det_neg-1.0
    dense-pseudo-label student vs the det_neg-0.01 sparse-GT models); "raw" = plain logit blend without alignment.
    edge_tta: EDGE-FEATURE TTA (public BIOHUB_EDGE_FEATURE_TTA=1). 0 = off (identity-view features only, the pre-patch
    public behaviour); 1 = average the inverse-transformed UNet feature maps over the same TTA views for EVERY model;
    2 = the public patch verbatim, i.e. only the PRIMARY model's features are averaged (their secondary encodes its
    features once, un-augmented). The augmented encodes are already computed for the detection average, so the only new
    cost is one (W,C,Z,Y,X) accumulator per model.
    Returns (list of unet features per model, each (W,C,Z,Y,X); det logits (W,Z,Y,X) TTA-averaged, blended;
    per-frame flags 'primary used instead of blend' = the anchor's retention guard when guard_thr is given)."""
    W = imgs.shape[1]
    outs = []; feats = []
    for mi, (model, _) in enumerate(models):
        acc = None; facc = None; u_id = None
        want_f = bool(edge_tta) and (edge_tta != 2 or mi == 0) and len(tta_views) > 1
        for (k, tr) in tta_views:
            xv = _d4_apply(imgs, k, tr)
            u, det = model.encode(xv)
            if (k, tr) == (0, False):
                u_id = u[0].float()
            if want_f:
                uf = _d4_invert(u[0].float(), k, tr)
                facc = uf if facc is None else facc + uf
            d = torch.stack([_d4_invert(det[f][0, 0], k, tr) for f in range(W)], 0).float()
            acc = d if acc is None else acc + d
        if want_f:
            fav = facc / len(tta_views)
            if fav.shape != u_id.shape:
                raise RuntimeError("EDGE-TTA SHAPE MISMATCH: %s vs %s" % (tuple(fav.shape), tuple(u_id.shape)))
            _delta = float((fav - u_id).abs().mean()); _scale = float(u_id.abs().mean())
            if _delta <= 1e-6 * max(_scale, 1e-12):  # public guard is `== 0.0`; relative form also catches FP-noise no-ops
                raise RuntimeError("EDGE-TTA NO-OP: averaged features (mean|delta| %.3g, mean|u| %.3g) are the single-pass "
                                   "features, so the augmented encodes contributed nothing and this arm would read as a "
                                   "false null" % (_delta, _scale))
            if _EDGE_TTA_LOG["n"] < 3:
                _EDGE_TTA_LOG["n"] += 1
                print("EDGE_TTA_ACTIVE model=", mi, "views=", len(tta_views), "mean_abs_feat_delta=", round(_delta, 6), flush=True)
            _EDGE_TTA_LOG["sum"] += _delta; _EDGE_TTA_LOG["cnt"] += 1
            feats.append(fav)
        else:
            feats.append(u_id)
        outs.append(acc / len(tta_views))
    det = outs[0]; used_primary = [False] * W
    if len(models) > 1 and det_w > 0:
        if det_blend == "prob":
            pe = torch.stack([torch.sigmoid(o) for o in outs[1:]], 0).mean(0)
            pb = ((1 - det_w) * torch.sigmoid(outs[0]) + det_w * pe).clamp(1e-6, 1 - 1e-6)
            blend = torch.log(pb) - torch.log1p(-pb)
        elif det_blend == "raw":
            blend = (1 - det_w) * outs[0] + det_w * torch.stack(list(outs[1:]), 0).mean(0)
        else:
            extra = torch.stack([torch.stack([_align(o[f], outs[0][f]) for o in outs[1:]], 0).mean(0) for f in range(W)], 0)
            blend = (1 - det_w) * outs[0] + det_w * extra
        if guard_thr is not None:
            frames = []
            for f in range(W):
                n_p = _n_peaks(outs[0][f], guard_thr, pool); n_b = _n_peaks(blend[f], guard_thr, pool)
                use_p = n_p > 0 and (n_b / n_p) < guard_retention
                used_primary[f] = bool(use_p); frames.append(outs[0][f] if use_p else blend[f])
            det = torch.stack(frames, 0)
        else:
            det = blend
    return feats, det, used_primary


def _align_cols(x, ref):
    """Per-target-column alignment of logits x (S,T) to ref's column mean/std (anchor: stats over dim=1 of (1,S,T))."""
    rc = ref.mean(0, keepdim=True); rs = ref.float().std(0, keepdim=True, unbiased=False).clamp_min(1e-4)
    xc = x.mean(0, keepdim=True); xs = x.float().std(0, keepdim=True, unbiased=False).clamp_min(1e-4)
    return (x - xc) * (rs / xs).clamp(0.5, 2.0) + rc


def anchor_edge_logits(lp, lr=None, ls=None, bi_w=0.15, sec_w=0.15, margin_max=0.35):
    """Anchor notebook's edge logits: forward primary lp (S,T); optional reverse primary lr (S,T, already transposed)
    fused by a weighted harmonic mean of per-target softmaxes; then the secondary's logits ls mixed in per target with the
    'low_margin_consensus' rule (weight sec_w * uncertainty, only where both seeds agree on the parent)."""
    F_ = lp.float()
    if lr is not None and bi_w > 0:
        ra = _align_cols(lr.float(), F_)
        pf = torch.softmax(F_, 0).clamp_min(1e-8); pr = torch.softmax(ra, 0).clamp_min(1e-8)
        h = 1.0 / ((1.0 - bi_w) / pf + bi_w / pr); h = h / h.sum(0, keepdim=True).clamp_min(1e-8)
        F_ = _align_cols(torch.log(h.clamp_min(1e-8)), F_)
    if ls is not None and sec_w > 0:
        sa = _align_cols(ls.float(), F_)
        if F_.shape[0] >= 2:
            pp = torch.softmax(F_, 0); sp = torch.softmax(sa, 0)
            pt2 = torch.topk(pp, k=2, dim=0); st2 = torch.topk(sp, k=2, dim=0)
            margin = pt2.values[0] - pt2.values[1]; same = pt2.indices[0].eq(st2.indices[0])
            unc = ((margin_max - margin) / margin_max).clamp(0.0, 1.0)
            w = torch.where(same, sec_w * unc, torch.zeros_like(unc)).view(1, -1)
            F_ = (1.0 - w) * F_ + w * sa
    return F_


def peaks_from_logits(logit, thr, pool=(3, 3, 3)):
    """logit (Z,Y,X) -> (N,3) int grid coords and probs; local max over pool kernel (pack: _detect_cells_pooled)."""
    x = logit[None, None]
    pooled = F.max_pool3d(x, pool, stride=1, padding=tuple(k // 2 for k in pool))
    prob = torch.sigmoid(x)
    m = (x == pooled) & (prob > thr)
    idx = torch.nonzero(m[0, 0])
    return idx.cpu().numpy().astype(np.int16), prob[0, 0][idx[:, 0], idx[:, 1], idx[:, 2]].float().cpu().numpy()


def pair_probs_from_logits(E, gmap, col, n_nodes, mode="geo", min_p=0.005, rev_w=0.0, pos=None, max_um=None):
    """E rows (t, i_local, j_local, logit_primary, logit_blend); gmap {(t, local): global index} of the kept nodes.
    Returns {(gi, gj): p}. mode: 'tgt' = softmax over sources per target (pack convention), 'src' = softmax over targets
    per source, 'geo' = sqrt(tgt*src), 'sig' = sigmoid(logit). Softmax is over the stored (<= edge_max_um) pairs."""
    if len(E) == 0: return {}
    t0 = E[:, 0].astype(np.int64); il = E[:, 1].astype(np.int64); jl = E[:, 2].astype(np.int64)
    gi = np.array([gmap.get((int(a), int(b)), -1) for a, b in zip(t0, il)], np.int64)
    gj = np.array([gmap.get((int(a) + 1, int(b)), -1) for a, b in zip(t0, jl)], np.int64)
    m = (gi >= 0) & (gj >= 0)
    if pos is not None and max_um is not None:  # restrict the softmax normalisation to pairs within max_um
        d = np.full(len(gi), np.inf); d[m] = np.linalg.norm(pos[gi[m]] - pos[gj[m]], axis=1); m = m & (d <= max_um)
    gi, gj, lg = gi[m], gj[m], E[m, col].astype(np.float64)
    if len(lg) == 0: return {}
    if rev_w > 0 and E.shape[1] >= 7:
        # harmonic mutual support (anchor notebook): per-target softmax of forward and reverse logits, harmonic mean,
        # renormalised per target, back to log-space with the forward centre/scale
        lr = E[m, col + 2].astype(np.float64)
        def _sm_t(v):
            mx = np.full(n_nodes, -np.inf); np.maximum.at(mx, gj, v); e = np.exp(v - mx[gj]); den = np.zeros(n_nodes); np.add.at(den, gj, e); return e / den[gj]
        pf = np.clip(_sm_t(lg), 1e-8, None); pr = np.clip(_sm_t(lr), 1e-8, None)
        ph = 1.0 / ((1 - rev_w) / pf + rev_w / pr)
        den = np.zeros(n_nodes); np.add.at(den, gj, ph); ph = ph / np.clip(den[gj], 1e-8, None)
        lh = np.log(np.clip(ph, 1e-8, None))
        lg = (lh - lh.mean()) * (lg.std() / max(lh.std(), 1e-4)) + lg.mean()
    if mode == "sig":
        p = 1.0 / (1.0 + np.exp(-lg))
    else:
        def _sm(group):
            mx = np.full(n_nodes, -np.inf); np.maximum.at(mx, group, lg); e = np.exp(lg - mx[group]); den = np.zeros(n_nodes); np.add.at(den, group, e); return e / den[group]
        if mode == "tgt": p = _sm(gj)
        elif mode == "src": p = _sm(gi)
        else: p = np.sqrt(_sm(gj) * _sm(gi))
    return {(int(i), int(j)): float(v) for i, j, v in zip(gi, gj, p) if v > min_p}


@torch.no_grad()
def run_video(models, vol_u16, q_lo, q_hi, device="cuda", tta=8, thr=0.5, thr_hi=0.965, det_w=0.8, edge_w=0.15, edge_max_um=20.0, window=2, edge_sets=("all", "hi"), reverse=False, factor=4, anchor_edges=False, guard=False, bi_w=0.15, sec_w=0.15, margin_max=0.35, det_blend="align", edge_tta=0):
    """Full pack-style inference on one strided video.
    edge_tta: 0 = off; 1 = TTA-average the UNet feature maps fed to the edge transformer for every model; 2 = primary
    model only (the public BIOHUB_EDGE_FEATURE_TTA=1 patch verbatim). Requires tta > 1.
    Returns dict with nodes (t, grid zyx int, prob, local index within frame) and sparse edge logits between consecutive
    frames for the 'all' node set (prob>=thr) and the 'hi' node set (prob>=thr_hi): rows (t, i_local, j_local, logit_primary, logit_blend)."""
    T = vol_u16.shape[0]; W = window
    views = D4[:tta] if tta > 1 else [(0, False)]
    if edge_tta and len(views) < 2:
        raise RuntimeError("EDGE-TTA NO-OP: edge_tta=%r needs tta > 1 (got tta=%r), otherwise the averaged features are "
                           "the identity view and the arm would read as a false null" % (edge_tta, tta))
    per_frame = {}; guard_frames = {}
    edges = {"all": [], "hi": []}; seen = set()
    win_starts = list(range(0, T - W + 1, max(W - 1, 1)))
    if win_starts[-1] + W < T: win_starts.append(T - W)
    shp = (W,) + tuple(vol_u16.shape[1:])
    scale = torch.tensor([1.0, float(factor), float(factor)], device=device)  # grid -> original voxel units
    um = torch.tensor([1.625, 0.40625 * factor, 0.40625 * factor], device=device)  # grid -> microns
    pool = tuple(max(1, (k if k % 2 == 1 else k + 1)) for k in (max(1, round(5.0 / float(u))) for u in um.tolist()))
    for ws in win_starts:
        fi = list(range(ws, ws + W))
        imgs = torch.stack([torch.from_numpy(vol_u16[t].astype(np.float32)) for t in fi])
        imgs = ((imgs - q_lo) / (q_hi - q_lo + 1e-6)).clamp(0.0)[None].to(device)
        feats, det, used_p = encode_window(models, imgs, views, det_w, guard_thr=(thr_hi if guard else None), pool=pool, det_blend=det_blend, edge_tta=edge_tta)
        for f, t in enumerate(fi):
            if t not in per_frame:
                per_frame[t] = peaks_from_logits(det[f], thr, pool=pool); guard_frames[t] = used_p[f]
        for f in range(W - 1):
            ts, tt = fi[f], fi[f + 1]
            if (ts, tt) in seen: continue
            seen.add((ts, tt))
            cs, ps = per_frame[ts]; ct, pt = per_frame[tt]
            for key, sel_s, sel_t in (("all", ps >= thr, pt >= thr), ("hi", ps >= thr_hi, pt >= thr_hi)):
                if key not in edge_sets: continue
                is_, it_ = np.where(sel_s)[0], np.where(sel_t)[0]
                if len(is_) == 0 or len(it_) == 0: continue
                c_s = torch.from_numpy(cs[is_].astype(np.float32)).to(device); c_t = torch.from_numpy(ct[it_].astype(np.float32)).to(device)
                cs_rel = np.concatenate([np.full((len(is_), 1), f, np.float32), cs[is_].astype(np.float32)], 1)
                ct_rel = np.concatenate([np.full((len(it_), 1), f + 1, np.float32), ct[it_].astype(np.float32)], 1)
                p_s = torch.from_numpy(pos_features(cs_rel, shp)).to(device); p_t = torch.from_numpy(pos_features(ct_rel, shp)).to(device)
                logits = []; logits_r = []
                if reverse:  # window-relative time swapped: target frame plays "t", source frame plays "t+1"
                    cs_rev = cs_rel.copy(); cs_rev[:, 0] = f + 1; ct_rev = ct_rel.copy(); ct_rev[:, 0] = f
                    p_s_r = torch.from_numpy(pos_features(cs_rev, shp)).to(device); p_t_r = torch.from_numpy(pos_features(ct_rev, shp)).to(device)
                for mi, (model, _) in enumerate(models):
                    fs = model.index_features(feats[mi][f], c_s); ft = model.index_features(feats[mi][f + 1], c_t)
                    logits.append(model.predict_edges(fs, ft, c_s * scale, c_t * scale, p_s, p_t).float())
                    if reverse:
                        logits_r.append(model.predict_edges(ft, fs, c_t * scale, c_s * scale, p_t_r, p_s_r).float().T)
                lp = logits[0]
                lb = lp if len(logits) == 1 or edge_w <= 0 else (1 - edge_w) * lp + edge_w * torch.stack(logits[1:], 0).mean(0)
                d_um = torch.cdist(c_s * um, c_t * um)
                ii, jj = torch.nonzero(d_um <= edge_max_um, as_tuple=True)
                cols = [np.full(len(ii), ts, np.float32), is_[ii.cpu().numpy()], it_[jj.cpu().numpy()], lp[ii, jj].cpu().numpy(), lb[ii, jj].cpu().numpy()]
                if reverse:
                    lpr = logits_r[0]; lbr = lpr if len(logits_r) == 1 or edge_w <= 0 else (1 - edge_w) * lpr + edge_w * logits_r[1]
                    cols += [lpr[ii, jj].cpu().numpy(), lbr[ii, jj].cpu().numpy()]
                if anchor_edges:
                    fs0 = models[0][0].index_features(feats[0][f], c_s); ft0 = models[0][0].index_features(feats[0][f + 1], c_t)
                    lr_native = models[0][0].predict_edges(ft0, fs0, c_t * scale, c_s * scale, p_t, p_s).float().T if bi_w > 0 else None
                    ls = logits[1] if len(logits) > 1 else None
                    la = anchor_edge_logits(lp, lr_native, ls, bi_w=bi_w, sec_w=sec_w, margin_max=margin_max)
                    cols.append(la[ii, jj].cpu().numpy())
                edges[key].append(np.stack(cols, 1))
    out_t, out_c, out_p, out_local = [], [], [], []
    for t in range(T):
        c, p = per_frame[t]
        out_t.append(np.full(len(c), t, np.int16)); out_c.append(c); out_p.append(p); out_local.append(np.arange(len(c), dtype=np.int32))
    res = {"t": np.concatenate(out_t), "grid": np.concatenate(out_c), "prob": np.concatenate(out_p), "local": np.concatenate(out_local)}
    ecols = ["t", "i", "j", "primary", "blend"] + (["rev_primary", "rev_blend"] if reverse else []) + (["anchor"] if anchor_edges else [])
    for key in ("all", "hi"):
        res["edges_" + key] = np.concatenate(edges[key], 0).astype(np.float32) if edges[key] else np.zeros((0, len(ecols)), np.float32)
    res["edge_cols"] = np.array(ecols); res["guard_primary_frames"] = np.array([t for t in sorted(guard_frames) if guard_frames[t]], np.int64)
    return res
