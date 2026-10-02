"""Sparse supervised 3D backward motion, inspired by Linajea and MPM.

This is a new, small Biohub adaptation, not either paper's full reproduction.
MPM's 2D spatial + time vector is extended to three spatial + one time
components. Linajea's known-parent neighbourhood mask prevents unknown sparse
annotations from becoming negatives. Image inference has no ground-truth I/O.

Public candidate coordinates are original voxels (t,z,y,x). Motion vectors
are physical (z,y,x) micrometres from a target to its parent at t-1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

RAW_SPACING_UM = (1.625, .40625, .40625)
RAW_TO_STRIDE4 = np.array([1., 4., 4.], np.float32)


@dataclass(frozen=True)
class MotionConfig:
    channels: int = 8
    max_motion_um: float = 12.
    temporal_scale_um: float = 6.
    sigma_um: float = 2.5
    radius_um: float = 4.
    grid_spacing_um: tuple = (1.625, 1.625, 1.625)

    def __post_init__(self):
        if self.channels < 4 or self.channels % 4:
            raise ValueError("channels must be a positive multiple of 4")
        if min(self.max_motion_um, self.temporal_scale_um, self.sigma_um, self.radius_um) <= 0:
            raise ValueError("Motion scales must be positive")
        if len(self.grid_spacing_um) != 3 or min(self.grid_spacing_um) <= 0:
            raise ValueError("Expected three positive spatial scales")


def sparse_parent_targets(shape, child_grid, parent_grid, config=MotionConfig()):
    """Draw targets around sparse children; NaN parent means unknown, not zero.

    Each voxel belongs to its closest annotated child (largest MPM magnitude).
    A missing parent owns an *ignored* region, so an adjacent known child cannot
    accidentally supervise it. Inputs are crop-relative grid coordinates. Both
    daughters may point to one mother. Parents can lie outside the output crop.
    """
    shape = tuple(int(x) for x in shape)
    child = np.asarray(child_grid, np.float32).reshape(-1, 3)
    parent = np.asarray(parent_grid, np.float32).reshape(-1, 3)
    if len(shape) != 3 or min(shape) <= 0 or child.shape != parent.shape:
        raise ValueError("Invalid shape or child/parent arrays")
    if not np.isfinite(child).all():
        raise ValueError("Child coordinates must be finite")
    spacing = np.asarray(config.grid_spacing_um, np.float32)
    nearest = np.full(shape, np.inf, np.float32)
    owner_valid = np.zeros(shape, bool)
    motion = np.zeros((3,) + shape, np.float32)
    amplitude = np.zeros(shape, np.float32)
    extent = np.ceil(config.radius_um / spacing).astype(int)
    # Local boxes avoid an N_cells x full-volume distance allocation.
    for c, p in zip(child, parent):
        lo = np.maximum(np.floor(c).astype(int) - extent, 0)
        hi = np.minimum(np.ceil(c).astype(int) + extent + 1, shape)
        if np.any(hi <= lo):
            continue
        ss = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        points = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij")).astype(np.float32)
        d2 = np.sum(((points - c[:, None, None, None]) * spacing[:, None, None, None]) ** 2, axis=0)
        take = (d2 <= config.radius_um ** 2) & (d2 < nearest[ss])
        if not take.any():
            continue
        nearest[ss][take] = d2[take]
        valid = bool(np.isfinite(p).all())
        owner_valid[ss][take] = valid
        amplitude[ss][take] = np.exp(-d2[take] / config.sigma_um ** 2) if valid else 0.
        local = (p[:, None, None, None] - points) * spacing[:, None, None, None] if valid else np.zeros_like(points)
        motion[(slice(None),) + ss][:, take] = local[:, take]
    # Temporal component uses a positive backward-time convention. Division by
    # this component recovers backward displacement, without a sign ambiguity.
    field = np.concatenate([motion, np.full((1,) + shape, config.temporal_scale_um, np.float32)])
    field *= amplitude[None] / np.maximum(np.linalg.norm(field, axis=0, keepdims=True), 1e-8)
    return dict(motion_um=motion, field=field, amplitude=amplitude, mask=owner_valid)


def _block(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU(),
                         nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU())


class SparseBackwardMotionNet(nn.Module):
    """Small two-frame 3D U-Net, independent of the production detector."""
    def __init__(self, config=MotionConfig()):
        super().__init__()
        self.config = config
        c = config.channels
        self.encoder = _block(2, c)
        self.middle = _block(c, 2 * c)
        self.decoder = _block(3 * c, c)
        self.head = nn.Conv3d(c, 4, 1)
        # Begin at zero motion and a neutral position magnitude.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, images):
        if images.ndim != 5 or images.shape[1] != 2:
            raise ValueError("images must be B,2,Z,Y,X, ordered previous/current")
        skip = self.encoder(images)
        deep = self.middle(F.avg_pool3d(skip, 2))
        deep = F.interpolate(deep, size=skip.shape[2:], mode="trilinear", align_corners=False)
        raw = self.head(self.decoder(torch.cat([skip, deep], dim=1)))
        motion = self.config.max_motion_um * torch.tanh(raw[:, :3])
        magnitude = torch.sigmoid(raw[:, 3:4])
        temporal = torch.full_like(magnitude, self.config.temporal_scale_um)
        direction = F.normalize(torch.cat([motion, temporal], dim=1), dim=1)
        return dict(motion_um=motion, field=direction * magnitude, amplitude=magnitude)


def sparse_motion_loss(prediction, target, *, motion_weight=1.):
    """MPM field+magnitude MSE plus Linajea-style masked vector regression.

    Unknown voxels contribute exactly zero gradient. This does not treat sparse
    unlabelled cells as negative detections, unlike unmasked full-image MPM MSE.
    Direct vector regression is scaled by temporal_scale (6 um in this pilot).
    """
    mask = target["mask"].to(prediction["field"].dtype)
    denom = mask.sum().clamp_min(1.)
    field_error = (prediction["field"] - target["field"]).square().sum(1)
    magnitude_error = (prediction["amplitude"][:, 0] - target["amplitude"]).square()
    vector_error = F.smooth_l1_loss(prediction["motion_um"] / 6., target["motion_um"] / 6., reduction="none").sum(1)
    # Weight central targets more strongly; unknown background remains absent.
    vector_weights = mask * target["amplitude"]
    terms = dict(field=(field_error * mask).sum() / denom,
                 magnitude=(magnitude_error * mask).sum() / denom,
                 vector=(vector_error * vector_weights).sum() / vector_weights.sum().clamp_min(1.))
    terms["total"] = terms["field"] + terms["magnitude"] + motion_weight * terms["vector"]
    return terms


def normalize_images(images, q_lo, q_hi):
    if not np.isfinite([q_lo, q_hi]).all() or q_hi <= q_lo:
        raise ValueError("Invalid image quantiles")
    return np.clip((np.asarray(images, np.float32) - q_lo) / (q_hi - q_lo), 0., 3.)


@torch.no_grad()
def predict_motion(model, volume, coords, *, q_lo, q_hi, device="cpu"):
    """Return motion_um[N,3] and uncalibrated position magnitude[N].

    coords are original-voxel tzyx, volume is the stride4 image. Frame zero has
    no previous image and returns NaN motion/zero support. No GT is accepted.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 4 or not np.isfinite(coords).all():
        raise ValueError("coords must be finite N,4 original-voxel tzyx")
    if len(volume.shape) != 4 or not np.equal(coords[:, 0], np.rint(coords[:, 0])).all():
        raise ValueError("Invalid volume or frame indices")
    grid = coords[:, 1:] / RAW_TO_STRIDE4
    if ((grid < 0) | (grid > np.asarray(volume.shape[1:]) - 1)).any():
        raise ValueError("Node outside image")
    if len(coords) and (coords[:, 0].min() < 0 or coords[:, 0].max() >= len(volume)):
        raise ValueError("Frame outside image sequence")
    motion = np.full((len(coords), 3), np.nan, np.float32)
    support = np.zeros(len(coords), np.float32)
    model = model.to(device).eval()
    for t in np.unique(coords[:, 0]).astype(int):
        if t == 0:
            continue
        ids = np.flatnonzero(coords[:, 0] == t)
        images = normalize_images(volume[t-1:t+1], q_lo, q_hi)
        out = model(torch.from_numpy(images)[None].to(device))
        # grid_sample expects normalized x,y,z and preserves subvoxel nodes.
        sampling = grid[ids] / np.maximum(np.asarray(volume.shape[1:]) - 1, 1) * 2 - 1
        sampling = torch.from_numpy(sampling[:, ::-1].copy()).float().to(device)[None, None, None]
        motion[ids] = F.grid_sample(out["motion_um"], sampling, align_corners=True)[0, :, 0, 0].T.cpu().numpy()
        support[ids] = F.grid_sample(out["amplitude"], sampling, align_corners=True)[0, 0, 0, 0].cpu().numpy()
    return motion, support


def apply_motion_to_candidates(coords, candidate_edges, motion, *, strength=.25,
                               sigma_um=4., voxel_size_um=RAW_SPACING_UM,
                               support=None, max_logit_change=2.):
    """Blend image motion into the existing candidate score; preserve graph.

    Gaussian *likelihood ratio* compares the motion residual with zero-motion
    distance: logit(p') = logit(p) + strength * (d0^2-d_motion^2)/(2*sigma^2).
    Thus zero motion, strength zero, unsupported motion, and frame zero are
    exact no-ops. No renormalization over a pruned candidate list is performed.
    support, when supplied, is a [0,1] attenuation, not a calibrated probability.
    Endpoints, distance column, row order, and original input remain unchanged.
    """
    coords = np.asarray(coords)
    edges = np.asarray(candidate_edges)
    motion = np.asarray(motion, np.float64)
    spacing = np.asarray(voxel_size_um, np.float64)
    if coords.ndim != 2 or coords.shape[1] != 4 or not np.isfinite(coords).all():
        raise ValueError("coords must be finite N,4 tzyx")
    if edges.ndim != 2 or edges.shape[1] != 4 or not np.issubdtype(edges.dtype, np.floating):
        raise ValueError("candidate_edges must be floating E,4 source,target,prob,distance")
    if motion.shape != (len(coords), 3) or spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Motion/spacing shape or scale mismatch")
    if not np.isfinite([strength, sigma_um, max_logit_change]).all() or strength < 0 or sigma_um <= 0 or max_logit_change <= 0:
        raise ValueError("Invalid fusion parameters")
    if not np.isfinite(edges).all() or not np.equal(edges[:, :2], np.rint(edges[:, :2])).all():
        raise ValueError("Nonfinite edges or noninteger endpoints")
    if ((edges[:, 2] < 0) | (edges[:, 2] > 1)).any() or (edges[:, 3] < 0).any():
        raise ValueError("Invalid edge probability or distance")
    out = edges.copy()
    if not len(edges):
        return out
    ij = edges[:, :2].astype(np.int64)
    if ij.min() < 0 or ij.max() >= len(coords):
        raise ValueError("Endpoint outside coords")
    src, dst = ij.T
    if not np.all(coords[dst, 0] - coords[src, 0] == 1):
        raise ValueError("Backward motion currently supports adjacent frames only")
    trust = np.ones(len(coords), np.float64) if support is None else np.asarray(support, np.float64)
    if trust.shape != (len(coords),) or not np.isfinite(trust).all() or ((trust < 0) | (trust > 1)).any():
        raise ValueError("support must be finite N values in [0,1]")
    backward = (coords[src, 1:] - coords[dst, 1:]) * spacing
    valid = np.isfinite(motion[dst]).all(1) & (trust[dst] > 0)
    if strength == 0 or not valid.any():
        return out
    raw2 = np.sum(backward[valid] ** 2, axis=1)
    residual2 = np.sum((backward[valid] - motion[dst[valid]]) ** 2, axis=1)
    delta = strength * trust[dst[valid]] * np.clip((raw2 - residual2) / (2 * sigma_um ** 2), -max_logit_change, max_logit_change)
    rows = np.flatnonzero(valid)[delta != 0]
    delta = delta[delta != 0]
    p = edges[rows, 2].astype(np.float64)
    # Endpoint probabilities remain exact 0/1; finite likelihood cannot change them.
    interior = (p > 0) & (p < 1)
    logit = np.log(p[interior]) - np.log1p(-p[interior]) + delta[interior]
    out[rows[interior], 2] = 1 / (1 + np.exp(-np.clip(logit, -700, 700)))
    return out


def load_motion_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SparseBackwardMotionNet(MotionConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, {k: v for k, v in checkpoint.items() if k != "state_dict"}
