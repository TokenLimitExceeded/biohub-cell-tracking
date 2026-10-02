"""CELLECT-owned candidate graphs and additive division costs, without GT.

Inference reuses the audited pretrained adapter. This graph construction does
not depend on B30's association threshold or candidate edges unless union is
explicitly requested. Costs are exploratory interventions, not calibration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
import numpy as np

from cellmot.external_cellect import CellectConfig, extract_node_features, load_models, score_candidates


@dataclass(frozen=True)
class NativeCellectConfig:
    gating: str = "official_inspired"
    min_link: float = .1
    use_division_hint: bool = False
    division_base: float = 1.2
    division_strength: float = .4
    division_logodds_clip: float = 2.
    union_baseline: bool = False

    def __post_init__(self):
        if self.gating not in ("all", "official_inspired"):
            raise ValueError("Unknown CELLECT candidate gate")
        if not 0 <= self.min_link <= 1 or self.division_strength < 0 or self.division_logodds_clip <= 0:
            raise ValueError("Invalid link threshold or division cost parameters")
        if self.division_base < self.division_strength * self.division_logodds_clip:
            raise ValueError("Division cost must remain nonnegative")


def candidates_from_features(coords, nearest_pairs, nearest_prob, no_match, split_probability, sizes,
                             config=NativeCellectConfig(), baseline_edges=None):
    """Build N x 4 (source,target,affinity,grid_distance) edges from native KNN.

    The optional gate follows upstream's p>0.1 first-link acceptance, source
    no-match comparison and size-derived geometric radius. The ILP chooses up to
    two children globally; this is not upstream's greedy full tracker.
    """
    coords = np.asarray(coords)
    pairs = np.asarray(nearest_pairs, np.int64)
    p = np.asarray(nearest_prob, np.float64)
    if coords.ndim != 2 or coords.shape[1] != 4 or not np.isfinite(coords).all():
        raise ValueError("Expected raw t,z,y,x centers")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or p.shape != (len(pairs),):
        raise ValueError("Invalid native edge arrays")
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= len(coords)):
        raise ValueError("Native candidate endpoint out of bounds")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid native link affinity")
    if len(pairs) and not np.all(coords[pairs[:, 1], 0] - coords[pairs[:, 0], 0] == 1):
        raise ValueError("Native candidates must join adjacent frames")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("Duplicate native candidate")
    size = np.asarray(sizes, float)
    nomatch = np.asarray(no_match, float)
    split = np.asarray(split_probability, float)
    if any(x.shape != (len(coords),) for x in (size, nomatch, split)):
        raise ValueError("Node feature size mismatch")
    grid = coords[:, 1:].astype(np.float64) / [1., 4., 4.]
    distance = np.linalg.norm(grid[pairs[:, 1]] - grid[pairs[:, 0]], axis=1)
    keep = np.ones(len(pairs), bool)
    if config.gating == "official_inspired":
        maximum = np.zeros(len(coords))
        np.maximum.at(maximum, pairs[:, 0], p)
        source = pairs[:, 0]
        # Native model geometry is expressed in raw XY-equivalent voxels.
        # grid_distance*4 is the matching coordinate metric [y,x,z*4].
        keep = ((p > config.min_link) & (maximum[source] > nomatch[source]) &
                (distance * 4 < np.maximum(size[source], 5.) * 6))
    edges = np.column_stack([pairs[keep], p[keep], distance[keep]])
    scored = np.ones(len(edges), bool)
    fallback = 0
    if config.union_baseline:
        if baseline_edges is None:
            raise ValueError("Union mode explicitly requires baseline edges")
        existing = {tuple(pair) for pair in pairs[keep]}
        extra = np.asarray([row for row in np.asarray(baseline_edges) if tuple(row[:2].astype(int)) not in existing], float).reshape(-1, 4)
        fallback = len(extra)
        edges = np.concatenate([edges, extra])
        scored = np.r_[scored, np.zeros(len(extra), bool)]
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, scored = edges[order], scored[order]
    div_cost = np.full(len(coords), config.division_base, np.float64)
    valid = np.isfinite(split)
    if config.use_division_hint:
        q = np.clip(split[valid], 1e-6, 1 - 1e-6)
        term = np.clip(np.log((1 - q) / q), -config.division_logodds_clip, config.division_logodds_clip)
        div_cost[valid] += config.division_strength * term
    info = dict(config=asdict(config), native_edges=len(pairs), accepted_native_edges=int(keep.sum()),
                output_edges=len(edges), baseline_fallback_edges=fallback,
                distance_units="stride4 grid voxels; multiply by 1.625 for micrometers",
                gating_units="raw XY-equivalent voxels: grid distance multiplied by 4",
                affinity_semantics="native independent sigmoid link outputs, not calibrated Biohub posteriors",
                division_cost_semantics="base + strength * clipped log((1-p_div)/p_div), applied to the parent division variable only",
                division_hint_nodes=int(valid.sum()) if config.use_division_hint else 0,
                division_cost_quantiles=np.quantile(div_cost, [0, .1, .5, .9, 1]).tolist())
    return edges, div_cost, scored, info


_MODELS = {}


def associate(coords, raw_volume, project_root, device="cuda", config=NativeCellectConfig(), baseline_edges=None):
    """Portable runtime: returns candidate_edges, per-node division_cost, info.

    project_root retains the original CELLECT manifest/source/weight layout.
    coords are detector centers; no pre-pruned association graph is necessary.
    raw_volume is indexable T,Z,Y,X and must be original XY resolution.
    """
    begin = time.perf_counter()
    cfg = CellectConfig(device=device)
    key = str(project_root), device
    if key not in _MODELS:
        _MODELS[key] = load_models(project_root, cfg)[:2]
    unet, matcher = _MODELS[key]
    features, finfo = extract_node_features(unet, raw_volume, coords, cfg)
    scores, sinfo = score_candidates(matcher, coords, np.empty((0, 4)), features, cfg)
    edges, div_cost, scored, ginfo = candidates_from_features(
        coords, scores["nearest_pairs"], scores["nearest_prob"], scores["no_match"],
        scores["split_probability"], features["size"], config, baseline_edges)
    return edges, div_cost, dict(features=finfo, scorer=sinfo, graph=ginfo,
                                scored_mask=scored, seconds=time.perf_counter() - begin)


def attach_division_cost(graph, division_cost, attr="cellect_division_cost"):
    """Register per-parent division costs and return a tracksdata expression."""
    import polars as pl
    import tracksdata as td
    values = np.asarray(division_cost, np.float64)
    node_ids = graph.node_attrs(attr_keys=["node_id"])["node_id"].to_numpy()
    if not np.array_equal(node_ids, np.arange(len(values))):
        raise ValueError("Graph node IDs must preserve CELLECT candidate indices")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite division costs")
    if attr not in graph.node_attr_keys():
        graph.add_node_attr_key(attr, pl.Float64, 1.2)
    graph.update_node_attrs(node_ids=node_ids.tolist(), attrs={attr: values.tolist()})
    return td.NodeAttr(attr)
