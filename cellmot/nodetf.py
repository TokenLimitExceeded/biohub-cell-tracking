"""Node transformer edge scorer (cross-attention between consecutive frames), trained on sparse GT edges.

Inputs per node: UNet feature (32) + normalized coords (3) + detection score (1). For each frame pair (t, t+1) we
compute logits (N_t, N_t1); training loss = cross-entropy over sources for each GT-linked target (one parent per child)
plus cross-entropy over targets for each GT-linked source restricted to non-dividing sources (one child per parent),
and a BCE on candidate pairs far from GT. Only rows/cols with GT supervision contribute (sparse GT).
"""
from __future__ import annotations
import math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

SCALE_UM = np.array([104.0, 104.0, 104.0], np.float32)  # coordinate normalization (volume ~104 um)


class CrossBlock(nn.Module):
    def __init__(self, d=128, heads=4, mlp=2.0, drop=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d); self.n3 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=drop)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=drop)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp)), nn.GELU(), nn.Dropout(drop), nn.Linear(int(d * mlp), d), nn.Dropout(drop))

    def forward(self, q, kv):
        h = self.n1(q); q = q + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.n2(q); k = self.n2(kv); q = q + self.cross_attn(h, k, k, need_weights=False)[0]
        q = q + self.mlp(self.n3(q))
        return q


def sinusoidal(coords, d=32):
    """coords (N,3) in [0,1] -> (N, 3*d) sinusoidal encodings."""
    N = coords.shape[0]; half = d // 2
    freqs = torch.exp(torch.arange(half, device=coords.device) * (-math.log(1000.0) / half))
    out = []
    for i in range(3):
        a = coords[:, i:i + 1] * 100.0 * freqs[None]
        out += [torch.sin(a), torch.cos(a)]
    return torch.cat(out, -1)


class NodeTransformer(nn.Module):
    def __init__(self, feat_dim=32, d=128, heads=4, blocks=4, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(feat_dim + 1 + 3 + 96, d)
        self.blocks = nn.ModuleList([CrossBlock(d, heads, 2.0, drop) for _ in range(blocks)])
        self.norm = nn.LayerNorm(d)
        self.pair = nn.Sequential(nn.Linear(2 * d + 4, d), nn.GELU(), nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    def embed(self, feat, score, coords):
        return self.proj(torch.cat([feat, score[:, None], coords, sinusoidal(coords)], -1))

    def forward(self, feat_t, score_t, coords_t, feat_t1, score_t1, coords_t1, chunk=256):
        """coords in um. Returns logits (N_t, N_t1)."""
        ct = coords_t / torch.from_numpy(SCALE_UM).to(coords_t.device); ct1 = coords_t1 / torch.from_numpy(SCALE_UM).to(coords_t.device)
        q = self.embed(feat_t, score_t, ct)[None]; k = self.embed(feat_t1, score_t1, ct1)[None]
        for b in self.blocks:
            q2 = b(q, k); k = b(k, q); q = q2
        q = self.norm(q)[0]; k = self.norm(k)[0]
        out = []
        for i in range(0, q.shape[0], chunk):
            qc = q[i:i + chunk]
            rel = (coords_t[i:i + chunk, None, :] - coords_t1[None, :, :]) / 10.0  # (c, N1, 3) in 10um units
            dist = rel.norm(dim=-1, keepdim=True)
            x = torch.cat([qc[:, None, :].expand(-1, k.shape[0], -1), k[None].expand(qc.shape[0], -1, -1), rel, dist], -1)
            out.append(self.pair(x).squeeze(-1))
        return torch.cat(out, 0)


def pair_loss(logits, gt_src_of_tgt, gt_tgt_of_src, single_child, max_um_mask=None):
    """logits (N_t, N_t1). gt_src_of_tgt: LongTensor (N_t1,) index of GT parent detection or -1.
    gt_tgt_of_src: LongTensor (N_t,) index of a GT child detection or -1 (for sources with exactly one child)."""
    loss = 0.0; n = 0
    m = gt_src_of_tgt >= 0
    if m.any():
        loss = loss + F.cross_entropy(logits[:, m].t(), gt_src_of_tgt[m]); n += 1
    m2 = (gt_tgt_of_src >= 0) & single_child
    if m2.any():
        loss = loss + F.cross_entropy(logits[m2], gt_tgt_of_src[m2]); n += 1
    return loss / max(n, 1)


@torch.no_grad()
def frame_pair_probs(model, feat, score, pos_um, t, tt, device="cuda", max_um=14.0):
    """Returns {(i,j): p} for consecutive frames tt->tt+1 using per-target softmax over sources (one parent per child),
    symmetrized with per-source softmax via geometric mean; pairs beyond max_um dropped."""
    si = np.where(t == tt)[0]; ti = np.where(t == tt + 1)[0]
    if len(si) == 0 or len(ti) == 0:
        return {}
    f0 = torch.from_numpy(feat[si].astype(np.float32)).to(device); f1 = torch.from_numpy(feat[ti].astype(np.float32)).to(device)
    s0 = torch.from_numpy(score[si]).to(device); s1 = torch.from_numpy(score[ti]).to(device)
    c0 = torch.from_numpy(pos_um[si]).to(device); c1 = torch.from_numpy(pos_um[ti]).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(f0, s0, c0, f1, s1, c1).float()
    d = torch.cdist(c0, c1)
    logits = logits.masked_fill(d > max_um, -1e4)
    p_tgt = torch.softmax(logits, 0)   # over sources for each target
    p_src = torch.softmax(logits, 1)   # over targets for each source
    p = torch.sqrt(p_tgt * p_src)
    ii, jj = torch.nonzero(d <= max_um, as_tuple=True)
    vals = p[ii, jj].cpu().numpy(); ii = ii.cpu().numpy(); jj = jj.cpu().numpy()
    return {(int(si[a]), int(ti[b])): float(v) for a, b, v in zip(ii, jj, vals) if v > 0.01}
