"""Classical per-detection displacement prior: local normalised cross-correlation (NCC) between frames t and t+1 on
the stride-2 grid (voxel 1.625 x 0.8125 x 0.8125 um; frame[:, ::2, ::2] of the raw uint16 volume, see
tools/build_cache_local.py --mode stride --factor 2). Ported from tools/explore_motion/flow.py (explorer "motion").

For every detection at frame t a (pz,py,px) patch of log1p intensities around it is correlated (grouped conv3d)
against a search window (+-sz, +-sy, +-sx voxels) in frame t+1; the NCC peak with parabolic sub-voxel refinement
gives the displacement in um.  `smooth_neighbours` replaces each estimate by the median of the estimates of the
detections within r um at the same frame.  Model-free; runs on CPU (seconds per video at 4 threads) or CUDA.

Public API: estimate(vol, t, pos_um, patch, search, device) -> (disp, ncc_peak, ncc_zero)
            smooth_neighbours(t, pos_um, disp, r, k_min) -> disp_s
            adaptive_disp(t, disp_s, thr_um) -> disp_s where the per-frame median |disp_s| > thr_um, else 0
            compute_disp(vol, t, pos_um, kind, ...) -> the displacement array for kind in {raw, smooth, adaptive}
"""
from __future__ import annotations
import numpy as np

GRID = np.array([1.625, 0.8125, 0.8125], np.float32)   # um per stride-2 voxel (z, y, x)
DEFAULT_PATCH = (5, 9, 9)
DEFAULT_SEARCH = (2, 6, 6)


def estimate(vol, t, pos_um, patch=DEFAULT_PATCH, search=DEFAULT_SEARCH, device="cpu"):
    """vol: (T,Z,Y,X) uint16 array or memmap on the stride-2 grid. t (N,) frame index, pos_um (N,3) in um.
    Returns disp (N,3) um (nan where no next frame / no estimate), ncc_peak (N,), ncc_zero (N,)."""
    import torch, torch.nn.functional as F
    t = np.asarray(t); pos_um = np.asarray(pos_um, np.float32)
    N = len(t); disp = np.full((N, 3), np.nan, np.float32); ncc_p = np.full(N, np.nan, np.float32); ncc_0 = np.full(N, np.nan, np.float32)
    if N == 0:
        return disp, ncc_p, ncc_0
    T = vol.shape[0]; pz, py, px = patch; sz, sy, sx = search
    hz, hy, hx = pz // 2, py // 2, px // 2
    pad = (hz + sz, hy + sy, hx + sx)
    ones = torch.ones(1, 1, pz, py, px, device=device)
    grid = np.rint(pos_um / GRID).astype(np.int64)
    grid[:, 0] = np.clip(grid[:, 0], 0, vol.shape[1] - 1); grid[:, 1] = np.clip(grid[:, 1], 0, vol.shape[2] - 1); grid[:, 2] = np.clip(grid[:, 2], 0, vol.shape[3] - 1)
    prev = None
    with torch.no_grad():
        for tt in range(T - 1):
            idx = np.where(t == tt)[0]
            if len(idx) == 0:
                prev = None
                continue
            f0 = torch.from_numpy(np.asarray(vol[tt], np.float32)).to(device) if prev is None else prev
            f1 = torch.from_numpy(np.asarray(vol[tt + 1], np.float32)).to(device); prev = f1
            # log-ish compression stabilises NCC on bright/dim nuclei
            a0 = torch.log1p(f0)[None, None]; a1 = torch.log1p(f1)[None, None]
            a0p = F.pad(a0, (pad[2], pad[2], pad[1], pad[1], pad[0], pad[0]), mode="replicate")
            a1p = F.pad(a1, (pad[2], pad[2], pad[1], pad[1], pad[0], pad[0]), mode="replicate")
            g = torch.from_numpy(grid[idx]).to(device)
            # gather patches / windows via unfold-free indexing
            zz = g[:, 0, None] + torch.arange(-hz, hz + 1, device=device)[None] + pad[0]
            yy = g[:, 1, None] + torch.arange(-hy, hy + 1, device=device)[None] + pad[1]
            xx = g[:, 2, None] + torch.arange(-hx, hx + 1, device=device)[None] + pad[2]
            P = a0p[0, 0][zz[:, :, None, None], yy[:, None, :, None], xx[:, None, None, :]]  # (n,pz,py,px)
            wz = g[:, 0, None] + torch.arange(-hz - sz, hz + sz + 1, device=device)[None] + pad[0]
            wy = g[:, 1, None] + torch.arange(-hy - sy, hy + sy + 1, device=device)[None] + pad[1]
            wx = g[:, 2, None] + torch.arange(-hx - sx, hx + sx + 1, device=device)[None] + pad[2]
            W = a1p[0, 0][wz[:, :, None, None], wy[:, None, :, None], wx[:, None, None, :]]  # (n,Wz,Wy,Wx)
            n = len(idx)
            P = P - P.mean(dim=(1, 2, 3), keepdim=True); pn = P.flatten(1).norm(dim=1).clamp_min(1e-6)
            Wsum = F.conv3d(W[None], ones.expand(n, 1, pz, py, px).contiguous(), groups=n)[0]      # local sums
            Wsq = F.conv3d((W * W)[None], ones.expand(n, 1, pz, py, px).contiguous(), groups=n)[0]
            num = F.conv3d(W[None], P[:, None].contiguous(), groups=n)[0]                          # (n,2sz+1,2sy+1,2sx+1)
            nvox = pz * py * px
            wstd = (Wsq - Wsum * Wsum / nvox).clamp_min(1e-6).sqrt()
            ncc = num / (wstd * pn[:, None, None, None])
            flat = ncc.flatten(1); pk = flat.argmax(1)
            oz = pk // ((2 * sy + 1) * (2 * sx + 1)); rem = pk % ((2 * sy + 1) * (2 * sx + 1)); oy = rem // (2 * sx + 1); ox = rem % (2 * sx + 1)
            ar = torch.arange(n, device=device)

            # parabolic sub-voxel refinement per axis (only where the peak is interior)
            def refine(o, axis, size):
                d = torch.zeros(n, device=device)
                inner = (o > 0) & (o < size - 1)
                if inner.any():
                    ii = ar[inner]

                    def at(off):
                        z_, y_, x_ = oz[ii], oy[ii], ox[ii]
                        if axis == 0: z_ = z_ + off
                        elif axis == 1: y_ = y_ + off
                        else: x_ = x_ + off
                        return ncc[ii, z_, y_, x_]
                    c0, cm, cp = at(0), at(-1), at(1)
                    den = (cm - 2 * c0 + cp)
                    d[inner] = torch.where(den < -1e-6, 0.5 * (cm - cp) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
                return d
            dz = (oz - sz).float() + refine(oz, 0, 2 * sz + 1); dy = (oy - sy).float() + refine(oy, 1, 2 * sy + 1); dx = (ox - sx).float() + refine(ox, 2, 2 * sx + 1)
            disp[idx] = (torch.stack([dz, dy, dx], 1).cpu().numpy() * GRID)
            ncc_p[idx] = flat.max(1).values.cpu().numpy(); ncc_0[idx] = ncc[ar, sz, sy, sx].cpu().numpy()
    return disp, ncc_p, ncc_0


def smooth_neighbours(t, pos_um, disp, r=15.0, k_min=3):
    """Median of the displacement estimates of detections within r um at the same frame (incl. self)."""
    from scipy.spatial import cKDTree
    t = np.asarray(t); pos_um = np.asarray(pos_um, np.float32)
    out = disp.copy()
    for tt in np.unique(t):
        idx = np.where(t == tt)[0]; ok = idx[np.isfinite(disp[idx, 0])]
        if len(ok) < k_min: continue
        tr = cKDTree(pos_um[ok])
        for i in ok:
            nb = tr.query_ball_point(pos_um[i], r=r)
            if len(nb) >= k_min:
                out[i] = np.median(disp[ok[nb]], axis=0)
    return out


def adaptive_disp(t, disp_s, thr_um=1.5):
    """Keep the (smoothed) prior only at frames whose median |disp_s| exceeds thr_um; zero velocity elsewhere."""
    t = np.asarray(t); out = np.zeros_like(disp_s); mag = np.linalg.norm(np.nan_to_num(disp_s), axis=1)
    for tt in np.unique(t):
        idx = np.where(t == tt)[0]
        if len(idx) and np.median(mag[idx]) > thr_um:
            out[idx] = disp_s[idx]
    return out


def compute_disp(vol, t, pos_um, kind="smooth", thr_um=1.5, patch=DEFAULT_PATCH, search=DEFAULT_SEARCH, device="cpu", r=15.0, k_min=3, return_all=False):
    """Displacement prior for the tracker. kind: 'raw' (per-detection NCC), 'smooth' (neighbour median within r um),
    'adaptive' (smooth where the per-frame median |disp_s| > thr_um, else 0). Returns (N,3) um with nan -> 0
    (or a dict with all intermediates when return_all)."""
    disp, ncc_p, ncc_0 = estimate(vol, t, pos_um, patch, search, device)
    disp_s = smooth_neighbours(t, pos_um, disp, r=r, k_min=k_min) if kind != "raw" or return_all else None
    if kind == "raw": d = disp
    elif kind == "smooth": d = disp_s
    elif kind == "adaptive": d = adaptive_disp(t, disp_s, thr_um)
    else: raise ValueError(f"unknown motion kind {kind}")
    d = np.nan_to_num(d)
    if return_all:
        return {"disp": d, "raw": disp, "smooth": disp_s, "adaptive": adaptive_disp(t, disp_s, thr_um), "ncc_peak": ncc_p, "ncc_zero": ncc_0}
    return d
