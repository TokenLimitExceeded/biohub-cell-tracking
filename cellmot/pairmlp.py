"""Pairwise edge scorer with appearance: MLP over [geometric/context features, feat_i, feat_j, |feat_i-feat_j|, cos].

Trained on labelled candidate pairs (cellmot.edgefeat) from detections that carry UNet features (tools/extract_feats.py).
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn
from .edgefeat import candidates, features, FEATURE_NAMES


def pair_inputs(t, pos, score, feat, I, J, prev_of=None):
    """Returns float32 matrix (len(I), D): geometric features + appearance blocks."""
    G = features(t, pos, score, I, J, prev_of=prev_of)
    fi = feat[I].astype(np.float32); fj = feat[J].astype(np.float32)
    ni = np.linalg.norm(fi, axis=1, keepdims=True) + 1e-6; nj = np.linalg.norm(fj, axis=1, keepdims=True) + 1e-6
    cos = (fi * fj).sum(1, keepdims=True) / (ni * nj)
    return np.concatenate([G, fi, fj, np.abs(fi - fj), cos], 1).astype(np.float32)


class PairMLP(nn.Module):
    def __init__(self, d_in, hidden=256, depth=3, dropout=0.1):
        super().__init__()
        layers = []; d = d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]; d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(d_in)); self.register_buffer("sd", torch.ones(d_in))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd).squeeze(-1)


class PairScorer:
    """Booster-like wrapper: .predict(F) -> probabilities; built from a PairMLP checkpoint."""
    def __init__(self, path, device="cuda"):
        ck = torch.load(path, map_location=device)
        self.model = PairMLP(ck["d_in"], ck["hidden"], ck["depth"]).to(device); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.device = device; self.cfg = ck.get("cfg", {})

    @torch.no_grad()
    def predict(self, F):
        out = []
        for i in range(0, len(F), 65536):
            x = torch.from_numpy(np.asarray(F[i:i + 65536], np.float32)).to(self.device)
            out.append(torch.sigmoid(self.model(x)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0,), np.float32)


def learned_pair_probs_mlp(t, pos_um, score, feat, scorer: PairScorer, max_um=12.0, k=8, cfg=None):
    """Two-pass helper mirroring cellmot.track.learned_pair_probs but with appearance features."""
    from .track import track, TrackConfig
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
    F = pair_inputs(t, pos_um, score, feat, I, J, prev_of=prev_of)
    p = scorer.predict(F)
    return {(int(i), int(j)): float(pi) for i, j, pi in zip(I, J, p)}
