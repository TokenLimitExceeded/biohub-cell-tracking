"""Cell-center heatmap detector: 3D UNet, Gaussian targets, masked focal loss, peak extraction."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

POOL = (1, 4, 4)            # (z, y, x) pooling of the raw (64,256,256) frame -> (64,64,64), isotropic 1.625 um
POOLED_SCALE = 1.625        # um per pooled voxel (all axes)
RAW_SCALE = np.array([1.625, 0.40625, 0.40625])


# ----------------------------------------------------------------------------- preprocessing
def pool_frame(vol: np.ndarray, pool=POOL) -> np.ndarray:
    """Mean-pool a (Z,Y,X) frame by integer factors."""
    pz, py, px = pool
    Z, Y, X = vol.shape
    v = vol[: Z // pz * pz, : Y // py * py, : X // px * px].astype(np.float32)
    v = v.reshape(Z // pz, pz, Y // py, py, X // px, px).mean(axis=(1, 3, 5))
    return v


def normalize(vol: np.ndarray, q_lo: float, q_hi: float, clip_max: float = 4.0) -> np.ndarray:
    v = (vol.astype(np.float32) - q_lo) / (q_hi - q_lo + 1e-6)
    return np.clip(v, 0.0, clip_max)


def raw_to_pooled(coords: np.ndarray, pool=POOL) -> np.ndarray:
    """(N,3) raw voxel (z,y,x) -> pooled continuous coords (center of pooled voxel j is at raw j*p + (p-1)/2)."""
    p = np.asarray(pool, np.float32)
    return (np.asarray(coords, np.float32) - (p - 1) / 2.0) / p


def pooled_to_raw(coords: np.ndarray, pool=POOL) -> np.ndarray:
    p = np.asarray(pool, np.float32)
    return np.asarray(coords, np.float32) * p + (p - 1) / 2.0


# ----------------------------------------------------------------------------- targets
def gaussian_heatmap(shape, centers: np.ndarray, sigma: float = 1.0, radius: int = 4) -> np.ndarray:
    """Max-composited isotropic Gaussians (peak 1) at continuous centers (z,y,x) in a (Z,Y,X) grid."""
    hm = np.zeros(shape, np.float32)
    if len(centers) == 0:
        return hm
    Z, Y, X = shape
    r = radius
    for c in centers:
        cz, cy, cx = float(c[0]), float(c[1]), float(c[2])
        z0, z1 = max(0, int(math.floor(cz)) - r), min(Z, int(math.floor(cz)) + r + 2)
        y0, y1 = max(0, int(math.floor(cy)) - r), min(Y, int(math.floor(cy)) + r + 2)
        x0, x1 = max(0, int(math.floor(cx)) - r), min(X, int(math.floor(cx)) + r + 2)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue
        zz, yy, xx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
        d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        g = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
        hm[z0:z1, y0:y1, x0:x1] = np.maximum(hm[z0:z1, y0:y1, x0:x1], g)
    # exact peak voxel (nearest grid point) gets target 1 so the focal loss has explicit positives
    for c in centers:
        iz, iy, ix = int(round(float(c[0]))), int(round(float(c[1]))), int(round(float(c[2])))
        if 0 <= iz < Z and 0 <= iy < Y and 0 <= ix < X:
            hm[iz, iy, ix] = 1.0
    return hm


def supervision_mask(shape, centers: np.ndarray, image: np.ndarray, radius: int = 4, dark_thr: float = 0.1,
                     dark_weight: float = 0.3, blob_mask: np.ndarray | None = None, core_radius: int = 2,
                     ring_weight: float = 1.0, nonblob_weight: float = 0.0) -> np.ndarray:
    """Loss weights for sparse GT: 1 inside `core_radius` of any annotated center, `ring_weight` out to `radius`,
    `dark_weight` on dark voxels (known negatives), 0 elsewhere (unknown: may hold unannotated cells).
    blob_mask (bool, same shape): voxels that look like cells (e.g. DoG blobs) are set to 0 unless within the core
    of an annotated center, so unannotated neighbours are never punished."""
    w = np.zeros(shape, np.float32)
    if dark_thr is not None and dark_weight > 0:
        w[image < dark_thr] = dark_weight
    Z, Y, X = shape
    core = np.zeros(shape, bool)
    for c in centers:
        cz, cy, cx = int(round(float(c[0]))), int(round(float(c[1]))), int(round(float(c[2])))
        z0, z1 = max(0, cz - radius), min(Z, cz + radius + 1)
        y0, y1 = max(0, cy - radius), min(Y, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(X, cx + radius + 1)
        w[z0:z1, y0:y1, x0:x1] = np.maximum(w[z0:z1, y0:y1, x0:x1], ring_weight)
    for c in centers:
        cz, cy, cx = int(round(float(c[0]))), int(round(float(c[1]))), int(round(float(c[2])))
        z0, z1 = max(0, cz - core_radius), min(Z, cz + core_radius + 1)
        y0, y1 = max(0, cy - core_radius), min(Y, cy + core_radius + 1)
        x0, x1 = max(0, cx - core_radius), min(X, cx + core_radius + 1)
        w[z0:z1, y0:y1, x0:x1] = 1.0; core[z0:z1, y0:y1, x0:x1] = True
    if blob_mask is not None:
        if nonblob_weight > 0:
            w[(w == 0) & ~blob_mask] = nonblob_weight
        w[blob_mask & ~core] = 0.0
    return w


@torch.no_grad()
def dog_response(vol: torch.Tensor, sigmas=(1.0, 1.4, 2.0), ratio=1.6) -> torch.Tensor:
    """Multi-scale DoG (scale-space max) on GPU. vol: (T,Z,Y,X) float tensor -> (T,Z,Y,X)."""
    def blur(x, s):
        r = int(3 * s + 0.5); k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
        g = torch.exp(-k ** 2 / (2 * s * s)); g = g / g.sum()
        x = x[:, None]
        x = F.conv3d(F.pad(x, (0, 0, 0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1, 1))
        x = F.conv3d(F.pad(x, (0, 0, r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1, 1))
        x = F.conv3d(F.pad(x, (r, r, 0, 0, 0, 0), mode="replicate"), g.view(1, 1, 1, 1, -1))
        return x[:, 0]
    best = None
    for s in sigmas:
        d = blur(vol, s) - blur(vol, s * ratio)
        best = d if best is None else torch.maximum(best, d)
    return best.clamp(min=0)


# ----------------------------------------------------------------------------- input modes
def local_contrast(x: torch.Tensor, k=(5, 9, 9), eps: float = 0.05) -> torch.Tensor:
    """Local contrast normalization per channel: (x - local mean) / (local std + eps). x: (B,C,Z,Y,X)."""
    pad = tuple(kk // 2 for kk in k)
    mean = F.avg_pool3d(x, k, stride=1, padding=pad, count_include_pad=False)
    var = F.avg_pool3d((x - mean) ** 2, k, stride=1, padding=pad, count_include_pad=False)
    return (x - mean) / (var.sqrt() + eps)


def make_input(x: torch.Tensor, mode: str = "raw") -> torch.Tensor:
    """x: (B, frames, Z, Y, X) normalized intensities. mode 'raw' -> x; 'lcn' -> cat(x, local_contrast(x))."""
    if mode == "raw":
        return x
    if mode == "lcn":
        return torch.cat([x, local_contrast(x).clamp(-5, 5)], 1)
    raise ValueError(mode)


def input_channels(ctx: int, mode: str = "raw") -> int:
    return (2 * ctx + 1) * (2 if mode == "lcn" else 1)


# ----------------------------------------------------------------------------- model
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, norm="bn"):
        super().__init__()
        def N(c):
            return nn.BatchNorm3d(c) if norm == "bn" else nn.GroupNorm(min(8, c), c)
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), N(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), N(cout), nn.ReLU(inplace=True),
        )
        self.skip = nn.Conv3d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class UNet3D(nn.Module):
    """Residual 3D UNet. in: (B, C_in, Z, Y, X) -> out: (B, C_out, Z, Y, X) logits."""
    def __init__(self, in_channels=3, out_channels=1, channels=(32, 64, 128, 256), norm="bn"):
        super().__init__()
        self.enc = nn.ModuleList()
        prev = in_channels
        for c in channels:
            self.enc.append(ConvBlock(prev, c, norm)); prev = c
        self.pool = nn.MaxPool3d(2)
        self.up = nn.ModuleList(); self.dec = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(channels[i], channels[i - 1], 2, stride=2))
            self.dec.append(ConvBlock(channels[i - 1] * 2, channels[i - 1], norm))
        self.head = nn.Conv3d(channels[0], out_channels, 1)
        nn.init.constant_(self.head.bias, -4.0)  # start with low peak probability

    def forward(self, x, return_features: bool = False):
        skips = []
        for i, blk in enumerate(self.enc):
            if i > 0:
                x = self.pool(x)
            x = blk(x)
            if i < len(self.enc) - 1:
                skips.append(x)
        for up, dec, skip in zip(self.up, self.dec, skips[::-1]):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = dec(torch.cat([x, skip], 1))
        out = self.head(x)
        return (out, x) if return_features else out


# ----------------------------------------------------------------------------- loss
def focal_heatmap_loss(logits, target, weight, alpha=2.0, beta=4.0):
    """CenterNet-style penalty-reduced focal loss on Gaussian heatmaps, masked by `weight`.
    logits/target/weight: (B,1,Z,Y,X). Normalized by number of peak voxels (target==1)."""
    p = torch.sigmoid(logits).clamp(1e-5, 1 - 1e-5)
    pos = (target >= 0.999).float()
    neg_w = (1 - target).pow(beta)
    pos_loss = -(torch.log(p) * (1 - p).pow(alpha) * pos)
    neg_loss = -(torch.log(1 - p) * p.pow(alpha) * neg_w * (1 - pos))
    loss = (pos_loss + neg_loss) * weight
    npos = (pos * weight).sum().clamp(min=1.0)
    return loss.sum() / npos


# ----------------------------------------------------------------------------- peaks
@torch.no_grad()
def extract_peaks(prob: torch.Tensor, threshold: float = 0.3, kernel: int = 3, refine: bool = True):
    """prob: (Z,Y,X) tensor of probabilities. Returns (coords (N,3) float in grid units, scores (N,)).
    Peaks = local maxima over a kernel^3 window above threshold; sub-voxel refinement by 3^3 center of mass."""
    p = prob[None, None]
    pooled = F.max_pool3d(p, kernel, stride=1, padding=kernel // 2)
    is_peak = (p == pooled) & (p >= threshold)
    idx = torch.nonzero(is_peak[0, 0])  # (N,3)
    if len(idx) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)
    scores = prob[idx[:, 0], idx[:, 1], idx[:, 2]]
    coords = idx.float()
    if refine:
        Z, Y, X = prob.shape
        pp = F.pad(p, (1, 1, 1, 1, 1, 1))[0, 0]
        off = torch.stack(torch.meshgrid(torch.arange(-1, 2), torch.arange(-1, 2), torch.arange(-1, 2), indexing="ij"), -1).reshape(-1, 3).to(prob.device).float()
        zz = (idx[:, 0:1] + 1 + off[:, 0].long()[None]); yy = (idx[:, 1:2] + 1 + off[:, 1].long()[None]); xx = (idx[:, 2:3] + 1 + off[:, 2].long()[None])
        vals = pp[zz, yy, xx]  # (N,27)
        wsum = vals.sum(1, keepdim=True).clamp(min=1e-6)
        coords = coords + (vals[:, :, None] * off[None]).sum(1) / wsum
        coords[:, 0].clamp_(0, Z - 1); coords[:, 1].clamp_(0, Y - 1); coords[:, 2].clamp_(0, X - 1)
    return coords.cpu().numpy().astype(np.float32), scores.float().cpu().numpy().astype(np.float32)


def physical_nms(coords_um: np.ndarray, scores: np.ndarray, radius_um: float):
    """Greedy NMS on physical coordinates. Returns kept indices."""
    from scipy.spatial import cKDTree
    if len(coords_um) == 0:
        return np.zeros((0,), np.int64)
    order = np.argsort(-scores)
    tree = cKDTree(coords_um)
    killed = np.zeros(len(coords_um), bool); keep = []
    for i in order:
        if killed[i]:
            continue
        keep.append(int(i))
        killed[tree.query_ball_point(coords_um[i], r=radius_um)] = True
    return np.array(keep, np.int64)


def tta_views(x: torch.Tensor):
    """8 planar (y,x) D4 views of x (B,C,Z,Y,X). Yields (view, inverse_fn)."""
    yield x, (lambda o: o)
    yield x.flip(-1), (lambda o: o.flip(-1))
    yield x.flip(-2), (lambda o: o.flip(-2))
    yield x.flip(-2, -1), (lambda o: o.flip(-2, -1))
    xt = x.transpose(-2, -1)
    yield xt, (lambda o: o.transpose(-2, -1))
    yield xt.flip(-1), (lambda o: o.flip(-1).transpose(-2, -1))
    yield xt.flip(-2), (lambda o: o.flip(-2).transpose(-2, -1))
    yield xt.flip(-2, -1), (lambda o: o.flip(-2, -1).transpose(-2, -1))


@torch.no_grad()
def sample_features(feat: torch.Tensor, coords: np.ndarray) -> np.ndarray:
    """feat: (C,Z,Y,X) tensor; coords: (N,3) continuous pooled-grid coords (z,y,x). Trilinear sampling -> (N,C) float16."""
    if len(coords) == 0:
        return np.zeros((0, feat.shape[0]), np.float16)
    C, Z, Y, X = feat.shape
    g = torch.from_numpy(np.asarray(coords, np.float32)).to(feat.device)
    # normalize to [-1,1] for grid_sample (align_corners=True): x_norm = 2*x/(X-1) - 1 ; grid order is (x, y, z)
    gz = 2 * g[:, 0] / max(Z - 1, 1) - 1; gy = 2 * g[:, 1] / max(Y - 1, 1) - 1; gx = 2 * g[:, 2] / max(X - 1, 1) - 1
    grid = torch.stack([gx, gy, gz], -1).view(1, -1, 1, 1, 3)
    out = F.grid_sample(feat[None].float(), grid, mode="bilinear", align_corners=True)  # (1,C,N,1,1)
    return out[0, :, :, 0, 0].t().cpu().numpy().astype(np.float16)
