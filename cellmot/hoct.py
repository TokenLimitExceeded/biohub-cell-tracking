"""HOCT association scores on existing Biohub detections and candidate links.

Uses the official pretrained model and pinned upstream feature transforms. No
ground truth or tracking solver is imported by this module. Inference is offline.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time

import numpy as np

UPSTREAM_COMMIT = "8709ee9d3c4d7aae1f022b259d48dc6584237b02"
MODEL_SHA256 = "5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33"
RAW_TO_GRID = np.array([1., 4., 4.])


@dataclass(frozen=True)
class HoctConfig:
    window: int = 5
    radius: float = 3.5
    foreground_fraction: float = .5
    spacing: float = 1.625
    max_edges: int = 4096
    halo_um: float = 8.
    precision: str = "float32"

    def __post_init__(self):
        if self.window < 2 or self.radius <= 0 or self.spacing <= 0:
            raise ValueError("Invalid HOCT window or spatial scale")
        if not 0 <= self.foreground_fraction <= 1 or self.max_edges < 1 or self.halo_um < 0:
            raise ValueError("Invalid foreground fraction or tile budget")
        if self.precision not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported inference precision")


def validate_candidates(coords, pairs):
    coords = np.asarray(coords)
    pairs = np.asarray(pairs)
    if coords.ndim != 2 or coords.shape[1] != 4 or not np.isfinite(coords).all():
        raise ValueError("coords must be finite (N, 4) t,z,y,x")
    if not np.equal(coords[:, 0], np.rint(coords[:, 0])).all():
        raise ValueError("Frame numbers must be integers")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (E, 2)")
    if not np.isfinite(pairs).all() or not np.equal(pairs, pairs.astype(np.int64)).all():
        raise ValueError("Candidate endpoints must be integers")
    if len(pairs):
        if pairs.min() < 0 or pairs.max() >= len(coords):
            raise ValueError("Candidate endpoint out of range")
        pairs = pairs.astype(np.int64)
        if len(np.unique(pairs, axis=0)) != len(pairs):
            raise ValueError("Duplicate candidate edges")
        if not np.all(coords[pairs[:, 1], 0] - coords[pairs[:, 0], 0] == 1):
            raise ValueError("This integration requires adjacent-frame edges")


def extract_features(volume, coords, config=HoctConfig(), progress=None):
    """One watershed region per original peak; shape/intensity are measured.

    volume is a stride-4 XY view of raw images (T,Z,Y/4,X/4). Position features
    retain detection centers, rather than relocating nodes to mask centroids.
    """
    from scipy import ndimage as ndi
    from skimage.measure import regionprops_table
    from skimage.segmentation import watershed
    from hoct.features.constants import REGIONPROPS
    from hoct.features.features import normalize_image, _border_dist_nd

    coords = np.asarray(coords)
    validate_candidates(coords, np.empty((0, 2), np.int64))
    if len(volume.shape) != 4:
        raise ValueError("Expected T,Z,Y,X image volume")
    grid = coords[:, 1:] / RAW_TO_GRID
    rounded = np.rint(grid).astype(np.int64)
    if not np.allclose(grid, rounded, atol=1e-6):
        raise ValueError("Detection points must lie on the stride-4 image grid")
    if (rounded < 0).any() or (rounded >= np.asarray(volume.shape[1:])).any():
        raise ValueError("Detection outside image volume")
    if len(coords) and (coords[:, 0].min() < 0 or coords[:, 0].max() >= volume.shape[0]):
        raise ValueError("Detection frame outside image volume")
    features = {p: np.empty((len(coords), 3, 3) if p == "inertia_tensor" else len(coords), np.float32)
                for p in REGIONPROPS}
    features["border_dist"][:] = _border_dist_nd(grid, volume.shape[1:])
    sizes = np.empty(len(coords), np.int32)
    t0 = time.perf_counter()
    for t in np.unique(coords[:, 0]).astype(int):
        ids = np.flatnonzero(coords[:, 0] == t)
        points = rounded[ids]
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError(f"Duplicate detection seeds at frame {t}")
        raw = np.asarray(volume[t], np.float32)
        if not np.isfinite(raw).all():
            raise ValueError(f"Nonfinite image at frame {t}")
        markers = np.zeros(raw.shape, np.int32)
        markers[tuple(points.T)] = np.arange(1, len(ids) + 1)
        smooth = ndi.gaussian_filter(raw, .7)
        signal = np.maximum(smooth - ndi.minimum_filter(smooth, size=7), 0.)
        distance, nearest = ndi.distance_transform_edt(markers == 0, return_indices=True)
        seed_signal = signal[tuple(nearest)]
        mask = (distance <= config.radius) & (signal >= config.foreground_fraction * seed_signal) & (seed_signal > 0)
        mask[tuple(points.T)] = True
        labels = watershed(-smooth, markers=markers, mask=mask)
        props = regionprops_table(labels, intensity_image=normalize_image(raw),
                                 properties=["label", "area", *[p for p in REGIONPROPS if p != "border_dist"]])
        if not np.array_equal(props["label"], np.arange(1, len(ids) + 1)):
            raise RuntimeError("Watershed lost a detection seed")
        sizes[ids] = props["area"]
        for p in REGIONPROPS:
            if p == "border_dist":
                continue
            if p == "inertia_tensor":
                for a in range(3):
                    for b in range(3):
                        features[p][ids, a, b] = props[f"{p}-{a}-{b}"]
            else:
                features[p][ids] = props[p]
        if progress and (t % 20 == 0 or t == int(coords[:, 0].max())):
            progress(f"features frame={t} nodes={len(ids)} elapsed={time.perf_counter()-t0:.1f}s")
    if any(not np.isfinite(v).all() for v in features.values()):
        raise RuntimeError("Nonfinite HOCT region features")
    info = dict(nodes=len(coords), one_voxel_regions=int((sizes == 1).sum()),
                region_voxel_quantiles=np.quantile(sizes, [0, .1, .5, .9, 1]).tolist() if len(sizes) else [],
                seconds=time.perf_counter()-t0, config=asdict(config))
    return features, info


def feature_graph(coords, pairs, features):
    import polars as pl
    import tracksdata as td
    from hoct.features.constants import REGIONPROPS
    validate_candidates(coords, pairs)
    graph = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        graph.add_node_attr_key(k, pl.Float32, 0.)
    for k in REGIONPROPS:
        tensor = k == "inertia_tensor"
        graph.add_node_attr_key(k, pl.Array(pl.Float32, (3, 3)) if tensor else pl.Float32,
                                np.zeros((3, 3), np.float32) if tensor else 0.)
    grid = np.asarray(coords[:, 1:], float) / RAW_TO_GRID
    rows = [dict(t=int(c[0]), z=float(p[0]), y=float(p[1]), x=float(p[2]),
                 **{k: features[k][i].tolist() for k in REGIONPROPS})
            for i, (c, p) in enumerate(zip(coords, grid))]
    ids = np.asarray(graph.bulk_add_nodes(rows))
    if not np.array_equal(ids, np.arange(len(coords))):
        raise RuntimeError("Upstream changed node ID assignment")
    graph.add_edge_attr_key("delta_t", pl.Float32, 1.)
    graph.bulk_add_edges([dict(source_id=int(s), target_id=int(t), delta_t=1.) for s, t in pairs])
    ea = graph.edge_attrs(attr_keys=[])
    if not np.array_equal(ea.select("source_id", "target_id").to_numpy(), pairs):
        raise RuntimeError("Upstream changed candidate edge order")
    return graph


def _tiles(batch, max_edges, halo_um):
    """Partition output edges, retaining a spatial context halo in each model call.

    Every input edge owns exactly one output per temporal window; context edges
    do not receive extra votes merely because they occur in overlapping tiles.
    """
    ep = batch["edge_pos"].numpy()
    np_ = batch["node_pos"].numpy()
    pairs = batch["edge_batch_id"].numpy()
    stack = [np.arange(len(pairs))]
    while stack:
        core = stack.pop()
        if len(core) == len(pairs):
            nodes = np.arange(len(np_))
            edges = core
        else:
            lo, hi = ep[core].min(axis=0)-halo_um, ep[core].max(axis=0)+halo_um
            inside = ((np_ >= lo) & (np_ <= hi)).all(axis=1)
            inside[pairs[core].ravel()] = True
            nodes = np.flatnonzero(inside)
            edges = np.flatnonzero(inside[pairs].all(axis=1))
        if len(edges) > max_edges:
            if len(core) == 1:
                raise RuntimeError("HOCT context exceeds max_edges even for one core edge; increase budget")
            axis = np.ptp(ep[core], axis=0).argmax()
            ordered = core[np.argsort(ep[core, axis], kind="stable")]
            half = len(ordered)//2
            stack.extend([ordered[half:], ordered[:half]])
            continue
        yield nodes, edges, np.searchsorted(edges, np.sort(core))


def parental_probabilities(edge_ids, edge_exp, node_ids, orphan_exp, pairs, n_nodes):
    """Official median-of-exp aggregation and parental softmax for dt=1."""
    import pandas as pd
    ee = pd.Series(edge_exp, index=edge_ids).groupby(level=0).median()
    oe = pd.Series(orphan_exp, index=node_ids).groupby(level=0).median()
    if not np.array_equal(ee.index.to_numpy(), np.arange(len(pairs))):
        raise RuntimeError("Incomplete HOCT candidate edge coverage")
    target = np.asarray(pairs, np.int64)[:, 1]
    if not np.isin(target, oe.index).all():
        raise RuntimeError("Missing orphan evidence for candidate targets")
    scores = ee.to_numpy(dtype=np.float64)
    orphan = np.zeros(n_nodes, np.float64)
    orphan[oe.index] = oe.values
    denom = np.bincount(target, weights=scores, minlength=n_nodes) + orphan
    p = scores / denom[target]
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise RuntimeError("Invalid HOCT normalized probabilities")
    return p.astype(np.float32)


def load_offline_model(checkpoint, device="cuda"):
    import torch
    path = Path(checkpoint)
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("HOCT general_v1 checkpoint SHA256 mismatch")
    return torch.jit.load(str(path), map_location=device).eval()


def predict_scores(model, coords, pairs, features, config=HoctConfig(), progress=None):
    import torch
    from hoct._api import _create_dataset
    validate_candidates(coords, pairs)
    if not len(pairs):
        return np.empty(0, np.float32), dict(edges=0, calls=0, seconds=0.)
    graph = feature_graph(coords, pairs, features)
    ds = _create_dataset(graph, window_size=config.window, scale=(1., config.spacing, config.spacing, config.spacing))
    device = next(model.parameters()).device
    # Match upstream model_predict: avoid costly JIT recompilation as graph sizes vary.
    torch._C._jit_set_bailout_depth(0)
    if config.precision == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("Device does not support bfloat16; use float32 or float16")
    if device.type != "cuda" and config.precision != "float32":
        raise ValueError("Reduced precision is only enabled on CUDA")
    ctx = nullcontext() if config.precision == "float32" else torch.autocast("cuda", dtype=getattr(torch, config.precision))
    edge_ids, edge_exp, node_ids, orphan_exp = [], [], [], []
    calls = 0
    t0 = time.perf_counter()
    with torch.inference_mode(), ctx:
        for w in range(len(ds)):
            batch = ds[w]
            if batch is None or not len(batch["edge_id"]):
                continue
            for nodes, edges, core_idx in _tiles(batch, config.max_edges, config.halo_um):
                mapping = np.full(len(batch["node_id"]), -1, np.int64)
                mapping[nodes] = np.arange(len(nodes))
                indices = torch.from_numpy(mapping[batch["edge_batch_id"][edges].numpy()]).unsqueeze(0).to(device)
                nf = batch["node_feats"][nodes].unsqueeze(0).to(device)
                pos = batch["node_pos"][nodes].unsqueeze(0).to(device)
                ep = batch["edge_pos"][edges].unsqueeze(0).to(device)
                pred, _, _, orphan = model(nf, pos, ep, indices,
                    torch.ones(nf.shape[:2], dtype=torch.bool, device=device),
                    torch.ones(indices.shape[:2], dtype=torch.bool, device=device))
                pred = pred[0, core_idx].float().clamp(max=20).exp().cpu().numpy().ravel()
                orphan = orphan[0].float().clamp(max=20).exp().cpu().numpy().ravel()
                if not np.isfinite(pred).all() or not np.isfinite(orphan).all():
                    raise RuntimeError("Nonfinite HOCT model output")
                original_edges = edges[core_idx]
                edge_ids.append(batch["edge_id"][original_edges].numpy())
                edge_exp.append(pred)
                # Store orphan estimates only for nodes incident to output edges.
                core_nodes = np.unique(batch["edge_batch_id"][original_edges].numpy())
                node_ids.append(batch["node_id"][core_nodes].numpy())
                orphan_exp.append(orphan[mapping[core_nodes]])
                calls += 1
            if progress and (w % 20 == 0 or w == len(ds)-1):
                progress(f"HOCT window={w+1}/{len(ds)} calls={calls} elapsed={time.perf_counter()-t0:.1f}s")
    p = parental_probabilities(np.concatenate(edge_ids), np.concatenate(edge_exp), np.concatenate(node_ids),
                              np.concatenate(orphan_exp), pairs, len(coords))
    return p, dict(edges=len(p), calls=calls, windows=len(ds), seconds=time.perf_counter()-t0,
                   model_sha256=MODEL_SHA256, upstream_commit=UPSTREAM_COMMIT, config=asdict(config))


def blend_candidates(candidate_edges, context_pairs, hoct_prob, alpha):
    """Replace scores only: preserve every solver edge, its order and distance."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    ce = np.asarray(candidate_edges, np.float64).copy()
    if ce.ndim != 2 or ce.shape[1] != 4:
        raise ValueError("candidate_edges must be (E,4) source,target,prob,distance")
    hp = np.asarray(hoct_prob)
    if len(hp) != len(context_pairs) or not np.isfinite(hp).all() or (hp < 0).any() or (hp > 1).any():
        raise ValueError("Invalid HOCT probability array")
    if len(np.unique(context_pairs, axis=0)) != len(context_pairs):
        raise ValueError("Duplicate context pairs")
    lookup = {(int(s), int(t)): i for i, (s, t) in enumerate(context_pairs)}
    try:
        inds = np.array([lookup[(int(s), int(t))] for s, t in ce[:, :2]], np.int64)
    except KeyError as e:
        raise ValueError(f"Missing HOCT score for solver edge {e}") from e
    if alpha:
        ce[:, 2] = (1-alpha)*ce[:, 2] + alpha*hp[inds]
    return ce
