"""Official Trackastra CTC association scores on fixed Biohub 3D candidates.

This is a detection-seeded 3D segmentation adaptation, not a reproduction of
the official CTC benchmark pipeline. No ground truth or tracking solver is read.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time

import numpy as np

UPSTREAM_COMMIT = "aa57a95160002e0fc70b915ab74178b39c99fd6a"
CTC_MODEL_SHA256 = "24c290ce74289ee6dac952a3355c9fb7a0a477fda3012656a2c6d80af892981f"
RAW_TO_GRID = np.array([1., 4., 4.], dtype=np.float32)


@dataclass(frozen=True)
class TrackastraConfig:
    radius_grid: float = 3.5
    foreground_fraction: float = .5
    gaussian_sigma_grid: float = .7
    local_background_size: int = 7

    def __post_init__(self):
        if self.radius_grid <= 0 or not 0 <= self.foreground_fraction <= 1:
            raise ValueError("Invalid segmentation radius or foreground fraction")
        if self.gaussian_sigma_grid <= 0 or self.local_background_size < 1:
            raise ValueError("Invalid smoothing/background configuration")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_candidates(coords, pairs):
    coords, pairs = np.asarray(coords), np.asarray(pairs)
    if coords.ndim != 2 or coords.shape[1] != 4 or not np.isfinite(coords).all():
        raise ValueError("coords must be finite (N,4) original-voxel t,z,y,x")
    if (coords < 0).any() or not np.equal(coords[:, 0], np.rint(coords[:, 0])).all():
        raise ValueError("Coordinates must be nonnegative, frames must be integers")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (E,2)")
    if not np.isfinite(pairs).all() or not np.equal(pairs, np.rint(pairs)).all():
        raise ValueError("Candidate IDs must be integers")
    if len(pairs):
        pairs = pairs.astype(np.int64)
        if pairs.min() < 0 or pairs.max() >= len(coords):
            raise ValueError("Candidate ID out of range")
        if len(np.unique(pairs, axis=0)) != len(pairs):
            raise ValueError("Duplicate candidate pairs")
        if not np.all(coords[pairs[:, 1], 0] - coords[pairs[:, 0], 0] == 1):
            raise ValueError("Expected forward adjacent-frame edges")


def require_3d_config(model_config, train_config):
    if model_config.get("coord_dim") != 3:
        raise ValueError("3D Biohub requires coord_dim=3; general_2d is incompatible")
    if model_config.get("feat_dim") != 12 or train_config.get("features") != "wrfeat":
        raise ValueError("Expected official 12-dimensional 3D wrfeat features")
    if model_config.get("window", 0) < 2:
        raise ValueError("Temporal window must contain at least two frames")


def load_ctc_model(folder, device="cpu"):
    import torch
    import yaml
    from trackastra.model import TrackingTransformer

    folder = Path(folder)
    model_cfg = yaml.safe_load((folder / "config.yaml").read_text())
    train_cfg = yaml.safe_load((folder / "train_config.yaml").read_text())
    require_3d_config(model_cfg, train_cfg)
    if sha256(folder / "model.pt") != CTC_MODEL_SHA256:
        raise ValueError("CTC checkpoint SHA256 does not match pinned official weights")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    model = TrackingTransformer.from_folder(folder, map_location="cpu").eval().to(device)
    return model, {"config": model_cfg, "train_config": train_cfg,
                   "model_sha256": CTC_MODEL_SHA256}


def extract_features(volume, coords, config=TrackastraConfig(), progress=None):
    """Measure all 12 official 3D wrfeat features on detection-seeded masks.

    The stride-4 grid has isotropic 1.625 um spacing; raw coordinate division is
    [1,4,4]. Original detection centers, IDs and candidates remain fixed. Mask
    centroids are not used to move detections. Upstream global percentile image
    normalization and feature ordering are retained.
    """
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    from trackastra.data.wrfeat import WRFeatures
    from trackastra.utils import normalize

    coords = np.asarray(coords)
    validate_candidates(coords, np.empty((0, 2), dtype=np.int64))
    if len(volume.shape) != 4 or volume.shape[0] < 2:
        raise ValueError("Expected at least 2 frames of a (T,Z,Y,X) volume")
    if len(coords) and coords[:, 0].max() >= volume.shape[0]:
        raise ValueError("Detection frame outside volume")
    grid = coords[:, 1:].astype(np.float32) / RAW_TO_GRID
    rounded = np.rint(grid).astype(np.int64)
    if not np.allclose(grid, rounded, atol=1e-6):
        raise ValueError("Detections must lie on stride-4 grid")
    if (rounded < 0).any() or (rounded >= np.array(volume.shape[1:])).any():
        raise ValueError("Detection outside volume")
    started = time.perf_counter()
    normalized = normalize(np.asarray(volume))
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite image intensities")
    feats = np.empty((len(coords), 12), dtype=np.float32)
    sizes = np.empty(len(coords), dtype=np.int32)
    for t in range(volume.shape[0]):
        ids = np.flatnonzero(coords[:, 0] == t)
        if not len(ids):
            continue
        points = rounded[ids]
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError(f"Duplicate detection seeds in frame {t}")
        raw = np.asarray(volume[t], dtype=np.float32)
        markers = np.zeros(raw.shape, dtype=np.int32)
        markers[tuple(points.T)] = np.arange(1, len(ids) + 1)
        smooth = ndi.gaussian_filter(raw, config.gaussian_sigma_grid)
        signal = np.maximum(smooth - ndi.minimum_filter(smooth, size=config.local_background_size), 0.)
        distance, nearest = ndi.distance_transform_edt(markers == 0, return_indices=True)
        seed_signal = signal[tuple(nearest)]
        mask = ((distance <= config.radius_grid) & (seed_signal > 0)
                & (signal >= config.foreground_fraction * seed_signal))
        mask[tuple(points.T)] = True
        labels = watershed(-smooth, markers=markers, mask=mask)
        feature = WRFeatures.from_mask_img(labels[None], normalized[t:t + 1], t_start=t)
        if not np.array_equal(feature.labels, np.arange(1, len(ids) + 1)):
            raise RuntimeError("Seed-to-feature mapping changed")
        if feature.features_stacked.shape != (len(ids), 12):
            raise RuntimeError("Unexpected upstream feature layout")
        feats[ids] = feature.features_stacked
        sizes[ids] = np.bincount(labels.ravel(), minlength=len(ids) + 1)[1:]
        if progress and (t % 20 == 0 or t == volume.shape[0] - 1):
            progress(f"features frame={t} nodes={len(ids)} seconds={time.perf_counter()-started:.2f}")
    if not np.isfinite(feats).all():
        raise RuntimeError("Nonfinite instance features")
    return feats, {"seconds": time.perf_counter() - started, "config": asdict(config),
                   "normalization": "official global 1/99.8 percentile, no clipping",
                   "feature_order": ["equivalent_diameter_area", "intensity_mean", "inertia_tensor(9)", "border_dist"],
                   "one_voxel_regions": int((sizes == 1).sum()),
                   "region_voxel_quantiles": np.quantile(sizes, [0, .1, .5, .9, 1]).tolist() if len(sizes) else []}


def predict_scores(model, coords, pairs, features, *, total_frames=None, progress=None):
    """Average official normalized 3D scores across every covering 4-frame window.

    All detected nodes are context. Only requested candidates are materialized;
    there is no score threshold, missing-score fallback or 2D projection. Output
    arrays have exactly the same row order as pairs.
    """
    import torch
    from trackastra.model.predict import predict

    coords, pairs = np.asarray(coords), np.asarray(pairs)
    validate_candidates(coords, pairs)
    pairs = pairs.astype(np.int64)
    features = np.asarray(features, dtype=np.float32)
    if features.shape != (len(coords), 12) or not np.isfinite(features).all():
        raise ValueError("features must be finite (N,12)")
    if model.config["coord_dim"] != 3 or model.config["feat_dim"] != 12:
        raise ValueError("Expected 3D model")
    total_frames = int(total_frames if total_frames is not None else coords[:, 0].max() + 1)
    if total_frames < 2 or (len(coords) and coords[:, 0].max() >= total_frames):
        raise ValueError("Invalid frame count")
    window = min(int(model.config["window"]), total_frames)
    grid = coords[:, 1:].astype(np.float32) / RAW_TO_GRID
    sums = np.zeros(len(pairs), dtype=np.float64)
    counts = np.zeros(len(pairs), dtype=np.int32)
    local = np.full(len(coords), -1, dtype=np.int64)
    started = time.perf_counter()
    max_tokens = 0
    model.eval()
    for start in range(total_frames - window + 1):
        ids = np.flatnonzero((coords[:, 0] >= start) & (coords[:, 0] < start + window))
        local.fill(-1)
        local[ids] = np.arange(len(ids))
        wanted = np.flatnonzero((local[pairs] >= 0).all(axis=1))
        if not len(wanted):
            continue
        max_tokens = max(max_tokens, len(ids))
        batch = {"coords": torch.from_numpy(grid[ids]),
                 "timepoints": torch.from_numpy(coords[ids, 0].astype(np.int64)),
                 "features": torch.from_numpy(features[ids]),
                 "labels": torch.from_numpy(ids + 1), "t1": torch.tensor(start)}
        # Upstream predict applies the pretrained quiet-softmax normalization.
        with torch.inference_mode():
            association = predict([batch], model)[0]
        lpairs = local[pairs[wanted]]
        values = association[lpairs[:, 0], lpairs[:, 1]]
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise RuntimeError("Invalid official Trackastra probabilities")
        sums[wanted] += values
        counts[wanted] += 1
        if progress and (start % 10 == 0 or start == total_frames - window):
            progress(f"scores window={start} tokens={len(ids)} edges={len(wanted)} seconds={time.perf_counter()-started:.2f}")
    if len(pairs) and (counts == 0).any():
        raise RuntimeError("Some candidate edges were never scored")
    scores = (sums / np.maximum(counts, 1)).astype(np.float32)
    return scores, counts, {"seconds": time.perf_counter() - started,
                            "window": window, "max_tokens": max_tokens,
                            "all_candidates_scored": bool((counts > 0).all()),
                            "aggregation": "mean over every covering window, no threshold",
                            "windows_per_edge": np.unique(counts, return_counts=True)[0].tolist()}
