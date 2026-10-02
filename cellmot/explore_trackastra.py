"""Trackastra 3D association on an expanded, detection-index-preserving graph.

This module reads no labels. Scores are native quiet-softmax model outputs,
never blends. Grid distances in returned edges follow Biohub's build_graph API;
multiply them by 1.625 for micrometres.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import time
import numpy as np
from cellmot.external_trackastra import (RAW_TO_GRID, extract_features,
    load_ctc_model, predict_scores, validate_candidates)

GRID_UM = 1.625
_MODELS = {}


@dataclass(frozen=True)
class Config:
    nearest_k: int = 8
    geometry_radius_um: float = 16.25
    threshold: float = .2
    maximum_distance_um: float = 16.25


def candidate_pool(coords, pool_edges, config=Config()):
    """Union all supplied pre-pruning pairs and bidirectional spatial kNN.

    Supplied probabilities only identify available pairs. They do not influence
    Trackastra features or probabilities. Geometric candidates recover pairs
    that the previous model assigned <= .01.
    """
    from scipy.spatial import cKDTree
    coords = np.asarray(coords)
    supplied = np.asarray(pool_edges)
    if supplied.ndim != 2 or supplied.shape[1] < 2:
        raise ValueError("pool_edges must have at least two columns")
    original = supplied[:, :2].astype(np.int64)
    validate_candidates(coords, original)
    if config.nearest_k < 1 or config.geometry_radius_um <= 0:
        raise ValueError("Invalid geometric candidate configuration")
    physical = coords[:, 1:].astype(np.float64) / RAW_TO_GRID * GRID_UM
    added = []
    for t in range(int(coords[:, 0].max())) if len(coords) else []:
        aa = np.flatnonzero(coords[:, 0] == t)
        bb = np.flatnonzero(coords[:, 0] == t + 1)
        if not len(aa) or not len(bb):
            continue
        for query, reference, reverse in ((aa, bb, False), (bb, aa, True)):
            dist, idx = cKDTree(physical[reference]).query(
                physical[query], k=min(config.nearest_k, len(reference)),
                distance_upper_bound=config.geometry_radius_um)
            if dist.ndim == 1:
                dist, idx = dist[:, None], idx[:, None]
            i, j = np.nonzero(np.isfinite(dist))
            src, dst = query[i], reference[idx[i, j]]
            added.append(np.column_stack((dst, src) if reverse else (src, dst)))
    geometric = np.concatenate(added) if added else np.empty((0, 2), np.int64)
    pairs = np.unique(np.concatenate((original, geometric)), axis=0)
    validate_candidates(coords, pairs)
    distances_um = np.linalg.norm(physical[pairs[:, 0]] - physical[pairs[:, 1]], axis=1)
    n = len(coords)
    keys = pairs[:, 0] * n + pairs[:, 1]
    original_keys = original[:, 0] * n + original[:, 1]
    flags = np.isin(keys, original_keys).astype(np.uint8)
    if len(geometric):
        flags |= np.isin(keys, geometric[:, 0] * n + geometric[:, 1]).astype(np.uint8) * 2
    return pairs, distances_um, flags


def select_edges(pairs, probability, distances_um, config=Config()):
    """Threshold native probabilities; return Biohub grid-distance edge rows."""
    if not 0 <= config.threshold <= 1 or config.maximum_distance_um <= 0:
        raise ValueError("Invalid selection configuration")
    keep = (np.asarray(probability) >= config.threshold) & (distances_um <= config.maximum_distance_um)
    return np.column_stack((pairs[keep], probability[keep], distances_um[keep] / GRID_UM))


def associate(coords, raw_volume, pool_edges, model_folder, device="cuda", config=Config()):
    """Deployment API: original-voxel coords and raw TZYX zarr/numpy volume.

    raw_volume is full-resolution; XY is materialized at stride four. The caller
    provides pinned Trackastra and dependency imports. CTC weights are verified
    on first use and cached per folder/device. All returned IDs index coords.
    """
    if isinstance(config, dict):
        config = Config(**config)
    started = time.perf_counter()
    coords = np.asarray(coords)
    volume = np.asarray(raw_volume[:, :, ::4, ::4], dtype=np.float32)
    pairs, distances, flags = candidate_pool(coords, pool_edges, config)
    key = (str(model_folder), device)
    if key not in _MODELS:
        _MODELS[key] = load_ctc_model(model_folder, device)
    model, mi = _MODELS[key]
    features, fi = extract_features(volume, coords)
    probability, counts, si = predict_scores(model, coords, pairs, features, total_frames=len(volume))
    edges = select_edges(pairs, probability, distances, config)
    return edges, dict(config=asdict(config), score_kind="native Trackastra quiet-softmax mean",
        distance_units="stride4 grid; 1 unit = 1.625 um", checkpoint_sha256=mi["model_sha256"],
        input_candidates=len(pool_edges), expanded_candidates=len(pairs), output_candidates=len(edges),
        geometry_only_candidates=int((flags == 2).sum()), features=fi, inference=si,
        seconds=time.perf_counter()-started, no_gt_in_inference=True)
