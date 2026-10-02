"""Fixed-node CELLECT image-embedding / link-scoring adapter.

This is an adaptation, not the complete CELLECT detector/tracker. It evaluates
the official pretrained U-Net and inter-frame MLP at existing detection centers.
No labels, metric, solver, or network downloads are used during inference.
Upstream implementation remains separate under its GPL-2.0 license.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time

import numpy as np

UPSTREAM_COMMIT = "3586070926f7f1fd5d8df37456861d22bdc63236"


@dataclass(frozen=True)
class CellectConfig:
    patch_xy: int = 256
    patch_z: int = 32
    overlap_xy: int = 8
    overlap_z: int = 4
    z_to_xy: float = 4.0
    enhance: float = 1.0
    device: str = "cuda"

    def __post_init__(self):
        if self.patch_xy < 32 or self.patch_xy % 16 or self.patch_z < 2:
            raise ValueError("U-Net requires XY multiple of 16 and nontrivial 3D depth")
        if not (0 <= self.overlap_xy < self.patch_xy and 0 <= self.overlap_z < self.patch_z):
            raise ValueError("Invalid patch overlap")
        if self.z_to_xy <= 0 or self.enhance <= 0:
            raise ValueError("Scale and enhancement must be positive")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def load_models(root, config=CellectConfig()):
    """Strictly load a pinned, locally verified state_dict with weights_only."""
    import torch
    root = Path(root)
    run = root / "remote_outputs/external_methods_20260921/cellect"
    manifest = json.loads((run / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest["commit"] != UPSTREAM_COMMIT:
        raise ValueError("Unexpected CELLECT source revision")
    for entry in manifest["files"]:
        if sha256(root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Source/checkpoint SHA mismatch: {entry['path']}")
    src = root / "tools/external_methods/cellect_upstream/unetext3Dn_con7.py"
    spec = importlib.util.spec_from_file_location("_biohub_cellect_upstream", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    unet, matcher = module.UNet3D(2, 6), module.EXNet(64, 8)
    for model, prefix in ((unet, "U-ext"), (matcher, "EX")):
        model.load_state_dict(torch.load(run / "weights" / f"{prefix}+-x3rd-149.0-4.6540.pth",
                                        map_location="cpu", weights_only=True), strict=True)
        model.eval().to(config.device)
    return unet, matcher, manifest


def _origins(length, patch, overlap):
    if length <= patch:
        return [0]
    return sorted(set([*range(0, length - patch + 1, patch - overlap), length - patch]))


def validate_inputs(coords, candidate_edges, volume_shape):
    coords = np.asarray(coords)
    edges = np.asarray(candidate_edges)
    if coords.ndim != 2 or coords.shape[1] != 4 or not np.isfinite(coords).all():
        raise ValueError("coords must be finite (N,4) raw t,z,y,x voxels")
    if not np.equal(coords, np.rint(coords)).all():
        raise ValueError("Fixed-node sampling requires integer voxel centers")
    if len(volume_shape) != 4 or (coords < 0).any() or (coords >= np.asarray(volume_shape)).any():
        raise ValueError("Detection center outside raw T,Z,Y,X image")
    if edges.ndim != 2 or edges.shape[1] != 4 or not np.isfinite(edges).all():
        raise ValueError("candidate_edges must be finite (E,4) source,target,p,dist")
    pairs = edges[:, :2].astype(np.int64)
    if not np.equal(edges[:, :2], pairs).all():
        raise ValueError("Candidate IDs must be integers")
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= len(coords)):
        raise ValueError("Candidate endpoint outside node array")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("Duplicate candidate edges")
    if len(pairs) and not np.all(coords[pairs[:, 1], 0] - coords[pairs[:, 0], 0] == 1):
        raise ValueError("CELLECT adapter accepts adjacent-frame edges only")
    if (edges[:, 2] < 0).any() or (edges[:, 2] > 1).any():
        raise ValueError("Baseline scores must be in [0,1]")


def extract_node_features(unet, volume, coords, config=CellectConfig(), progress=None):
    """Evaluate native anisotropic 3D patches, sampling exactly the frozen nodes.

    Raw (Z,Y,X) becomes the upstream (H,W,Z) order. Like upstream, each frame is
    represented with itself and its next frame as two channels. The final frame
    repeats itself, an explicit boundary extension. Overlaps choose the patch
    whose center is closest to the node; no image-to-label operation occurs.
    """
    import torch
    coords = np.asarray(coords)
    validate_inputs(coords, np.empty((0, 4)), volume.shape)
    n = len(coords)
    features = np.empty((n, 64), np.float32)
    sizes = np.empty(n, np.float32)
    division = np.empty((n, 2), np.float32)
    foreground = np.empty(n, np.float32)
    patch_shape = np.asarray([config.patch_z, config.patch_xy, config.patch_xy])
    spatial_shape = np.asarray(volume.shape[1:])
    starts = list(itertools.product(*[_origins(int(d), int(p), int(o)) for d, p, o in
                       zip(spatial_shape, patch_shape, [config.overlap_z, config.overlap_xy, config.overlap_xy])]))
    input_hash = hashlib.sha256()
    read_seconds = gpu_seconds = 0.0
    frame_cache = {}
    hashed = set()
    patch_count = 0
    t0 = time.perf_counter()

    def read(t):
        nonlocal read_seconds
        if t not in frame_cache:
            begin = time.perf_counter()
            raw = np.asarray(volume[t])
            if raw.shape != tuple(spatial_shape) or not np.isfinite(raw).all() or raw.min() < 0:
                raise ValueError(f"Invalid raw volume frame {t}")
            if t not in hashed:
                input_hash.update(np.asarray([t, *raw.shape], np.int64).tobytes())
                input_hash.update(str(raw.dtype).encode("ascii"))
                input_hash.update(raw.tobytes(order="C"))
                hashed.add(t)
            frame_cache[t] = raw
            read_seconds += time.perf_counter() - begin
        return frame_cache[t]

    with torch.inference_mode():
        for t in sorted(np.unique(coords[:, 0]).astype(int)):
            ids = np.flatnonzero(coords[:, 0] == t)
            points = coords[ids, 1:].astype(np.int64)
            raw = [read(t), read(min(t + 1, volume.shape[0] - 1))]
            # The upstream transform uses the minimum positive value over the
            # entire two-frame tensor, adds 1900 and then applies log1p.
            positive_min = min(float(x[x > 0].min()) if np.any(x > 0) else float("inf") for x in raw)
            if not np.isfinite(positive_min):
                raise ValueError(f"All-zero image pair at {t}")
            distances = []
            for start in starts:
                start = np.asarray(start)
                valid = ((points >= start) & (points < np.minimum(start + patch_shape, spatial_shape))).all(1)
                d = np.sum(((points - (start + (patch_shape - 1) / 2)) / patch_shape) ** 2, axis=1)
                distances.append(np.where(valid, d, np.inf))
            choice = np.argmin(distances, axis=0)
            if not np.isfinite(np.min(distances, axis=0)).all():
                raise RuntimeError("Patch assignment omitted a detection")
            for tile, start_tuple in enumerate(starts):
                selected = np.flatnonzero(choice == tile)
                if not len(selected):
                    continue
                start = np.asarray(start_tuple)
                stop = np.minimum(start + patch_shape, spatial_shape)
                sl = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
                batch = np.full((2, *patch_shape), positive_min, dtype=np.float32)
                dst = tuple(slice(0, int(b - a)) for a, b in zip(start, stop))
                for channel in range(2):
                    batch[(channel, *dst)] = raw[channel][sl]
                np.maximum(batch, positive_min, out=batch)
                batch = np.log1p(batch * config.enhance + 1900)
                x = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 2, 3, 1))).unsqueeze(0).to(config.device)
                if config.device.startswith("cuda"):
                    torch.cuda.synchronize()
                gpu_start = time.perf_counter()
                seg, center, embed, div, size = unet(x)
                local = points[selected] - start
                z, y, xx = (torch.as_tensor(local[:, j], device=config.device) for j in range(3))
                out_ids = ids[selected]
                features[out_ids] = embed[0, :, y, xx, z].T.float().cpu().numpy()
                sizes[out_ids] = size[0, 0, y, xx, z].float().cpu().numpy()
                division[out_ids] = torch.sigmoid(div[0, :, y, xx, z]).T.float().cpu().numpy()
                foreground[out_ids] = torch.softmax(seg[0, :, y, xx, z], dim=0)[1:].sum(0).float().cpu().numpy()
                if config.device.startswith("cuda"):
                    torch.cuda.synchronize()
                gpu_seconds += time.perf_counter() - gpu_start
                patch_count += 1
                del x, seg, center, embed, div, size
            frame_cache = {k: v for k, v in frame_cache.items() if k > t}
            if progress and (t % 10 == 0 or t == int(coords[:, 0].max())):
                progress(f"frame={t} nodes={len(ids)} patches={patch_count} elapsed={time.perf_counter()-t0:.1f}s")
    if any(not np.isfinite(x).all() for x in (features, sizes, division, foreground)):
        raise RuntimeError("Nonfinite CELLECT features")
    if (np.linalg.norm(features, axis=1) == 0).any():
        raise RuntimeError("Zero feature vector cannot enter upstream cosine computations")
    info = dict(seconds=time.perf_counter() - t0, gpu_seconds=gpu_seconds, image_read_hash_seconds=read_seconds,
                patches=patch_count, input_frame_stream_sha256=input_hash.hexdigest(),
                input_frames=[int(t) for t in sorted(hashed)], raw_volume_shape=[int(d) for d in volume.shape],
                feature_dim=64, config=asdict(config), precision="float32",
                boundary_nodes=int(((coords[:, 1:] < 3) | (coords[:, 1:] >= spatial_shape - 3)).any(1).sum()),
                terminal_frame="repeat final frame as its own next-frame channel")
    return dict(embedding=features, size=sizes, division=division, foreground=foreground), info


def score_candidates(matcher, coords, candidate_edges, node_features, config=CellectConfig()):
    """Return aligned probabilities and a mask of genuine CELLECT-scored edges.

    The official MLP sees the five spatially nearest target nodes. Its first five
    logits use independent sigmoid link outputs, its sixth uses sigmoid no-match,
    and its final two use softmax division classification. Out-of-neighborhood
    frozen candidates retain B30 probabilities and are marked scored_mask=False.
    This fallback is explicit; these outputs are not calibrated Biohub posteriors.
    """
    import torch
    from scipy.spatial.distance import cdist
    coords = np.asarray(coords)
    edges = np.asarray(candidate_edges)
    pairs = edges[:, :2].astype(np.int64)
    probabilities = edges[:, 2].astype(np.float32).copy()
    scored = np.zeros(len(edges), bool)
    no_match = np.full(len(coords), np.nan, np.float32)
    split_probability = np.full(len(coords), np.nan, np.float32)
    positions = coords[:, [2, 3, 1]].astype(np.float32)
    positions[:, 2] *= config.z_to_xy
    f = torch.as_tensor(node_features["embedding"], device=config.device)
    ss = torch.as_tensor(node_features["size"], device=config.device)
    div = torch.as_tensor(node_features["division"], device=config.device)
    pos = torch.as_tensor(positions, device=config.device)
    nearest_pairs, nearest_probs = [], []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for t in sorted(np.unique(coords[:, 0]).astype(int)):
            sources = np.flatnonzero(coords[:, 0] == t)
            targets = np.flatnonzero(coords[:, 0] == t + 1)
            if not len(sources) or not len(targets):
                continue
            # Stable ordering gives deterministic handling of exact distance ties.
            neighbor = np.argsort(cdist(positions[sources], positions[targets]), axis=1, kind="stable")[:, :5]
            neighbor = np.pad(neighbor, ((0, 0), (0, max(0, 5 - neighbor.shape[1]))), mode="edge")
            q_np = targets[neighbor]
            q = torch.as_tensor(q_np, device=config.device)
            src = torch.as_tensor(sources, device=config.device)
            target_size, source_size = ss[q], ss[src, None]
            delta = torch.abs(pos[q] - pos[src, None])
            distance = torch.sqrt((delta * delta).sum(-1))
            size_features = torch.stack([target_size - source_size,
                target_size / (source_size + 1), distance, target_size - distance,
                (target_size > distance).float(), target_size / (distance + .0001)], dim=-1)
            source_div = div[src, None].expand(-1, 5, -1)
            target_div = div[q]
            interaction = torch.stack([(source_div[:, :, 0] > .5) * (target_div[:, :, 1] > .5),
                                        source_div[:, :, 0] * target_div[:, :, 1]], dim=-1)
            division_features = torch.cat([source_div, target_div, interaction], dim=-1)
            logits = matcher(f[src, None], f[q], delta, size_features, division_features)
            link = torch.sigmoid(logits[:, :5]).cpu().numpy()
            no_match[sources] = torch.sigmoid(logits[:, 5]).cpu().numpy()
            split_probability[sources] = torch.softmax(logits[:, 6:], dim=-1)[:, 1].cpu().numpy()
            lookup = {(int(s), int(target)): float(p) for s, row, probs in zip(sources, q_np, link)
                      for target, p in zip(row, probs)}
            edge_ids = np.flatnonzero(coords[pairs[:, 0], 0] == t)
            for edge_id in edge_ids:
                pair = tuple(pairs[edge_id])
                if pair in lookup:
                    probabilities[edge_id] = lookup[pair]
                    scored[edge_id] = True
            nearest_pairs.extend(lookup)
            nearest_probs.extend(lookup.values())
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Nonfinite CELLECT link probabilities")
    return dict(prob=probabilities, scored_mask=scored, no_match=no_match, split_probability=split_probability,
                nearest_pairs=np.asarray(nearest_pairs, np.int64).reshape(-1, 2),
                nearest_prob=np.asarray(nearest_probs, np.float32)), dict(
                    seconds=time.perf_counter() - t0, scored_edges=int(scored.sum()),
                    total_edges=len(scored), coverage=float(scored.mean()) if len(scored) else 1.,
                    fallback="Preserve original B30 probability outside official nearest-five neighborhood",
                    score_semantics="sigmoid(EX logits 0:5) for nearest-five links; sigmoid(logit5) no-match; softmax(logits6:8)[1] division; no Biohub calibration")
