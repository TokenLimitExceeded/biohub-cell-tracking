"""Offline deployment of independent external association candidates."""
from pathlib import Path
import json
import os
import time
import numpy as np

DIVISION_COSTS = {}


def rebuild(coords, original_edges, pool_edges, raw_volume, dataset, device='cuda', q_lo=None, q_hi=None):
    start = time.monotonic()
    root = Path(os.environ['BIOHUB_EXTERNAL_ASSET'])
    recipe = json.loads(os.environ['BIOHUB_EXTERNAL_RECIPE'])
    original = np.asarray(original_edges, np.float64).reshape(-1, 4)
    pool = np.asarray(pool_edges, np.float64).reshape(-1, 4)
    kind = recipe['method']
    costs = None
    if kind == 'trackastra':
        from cellmot.explore_trackastra import associate
        edges, report = associate(coords, raw_volume, pool, root/'models/ctc',
                                  device=str(device), config=recipe['config'])
    elif kind == 'trackastra_guarded':
        from cellmot.explore_trackastra_guarded import associate
        edges, report = associate(coords, raw_volume, pool, root/'models/ctc',
                                  device=str(device), config=recipe['config'], original_edges=original)
    elif kind == 'cellect':
        from cellmot.explore_cellect import associate, NativeCellectConfig
        edges, costs, report = associate(coords, raw_volume, root, device=str(device),
                              config=NativeCellectConfig(**recipe['config']), baseline_edges=original)
        report.pop('scored_mask', None)
    elif kind == 'consensus':
        from cellmot.explore_consensus import associate
        edges, report = associate(coords, raw_volume, pool, original, root,
                                   device=str(device), config=recipe['config'])
    elif kind == 'ht50':
        from cellmot.explore_ht import associate
        edges, report = associate(coords, raw_volume, pool, original, root, device=str(device))
    elif kind == 'trackastra_motion':
        from cellmot.explore_trackastra import associate
        from cellmot.explore_motion_safe import build_safe_candidates, reweight_safe_candidates
        all_te, ti = associate(coords, raw_volume, pool, root/'models/ctc', device=str(device),
                               config=dict(threshold=0., maximum_distance_um=1e6))
        grid = np.asarray(raw_volume[:, :, ::4, ::4], np.float32)
        safe, mi = build_safe_candidates(coords, original, pool, grid,
                     checkpoint=Path(__file__).parent.parent/'motion_checkpoint.pt',
                     q_lo=q_lo, q_hi=q_hi, device=str(device), apply_motion=False)
        n=len(coords)
        keys=all_te[:,0].astype(np.int64)*n+all_te[:,1].astype(np.int64)
        order=np.argsort(keys);keys=keys[order]
        wanted=safe[:,0].astype(np.int64)*n+safe[:,1].astype(np.int64)
        loc=np.searchsorted(keys,wanted)
        if np.any(loc>=len(keys)) or not np.array_equal(keys[loc],wanted):
            raise ValueError('Trackastra pool does not cover safe candidate edges')
        safe[:,2]=all_te[order[loc],2]
        edges=reweight_safe_candidates(coords,safe,mi.pop('motion_um'),mi.pop('motion_gate'))
        report=dict(trackastra=ti,motion=mi,quantiles=dict(q_lo=q_lo,q_hi=q_hi))
    elif kind == 'hoct':
        from cellmot.hoct import HoctConfig, load_offline_model, extract_features, predict_scores
        cfg = HoctConfig()
        volume = np.asarray(raw_volume[:, :, ::4, ::4], np.float32)
        model = load_offline_model(os.environ['BIOHUB_HOCT_CHECKPOINT'], device=str(device))
        features, fi = extract_features(volume, coords, cfg)
        pairs = pool[:, :2].astype(np.int64)
        probability, si = predict_scores(model, coords, pairs, features, cfg)
        mask = probability > recipe['config']['threshold']
        edges = pool[mask].copy()
        edges[:, 2] = probability[mask]
        report = dict(features=fi, inference=si)
        del model
    else:
        raise ValueError('Unknown external recipe: '+kind)
    edges = np.asarray(edges, np.float64).reshape(-1, 4)
    if not np.isfinite(edges).all():
        raise ValueError('Nonfinite rebuilt candidates')
    if costs is not None:
        DIVISION_COSTS[dataset] = costs
    report.update(dataset=dataset, status='complete', recipe=recipe,
                  original_candidate_edges=len(original), pre_pruning_pool_edges=len(pool),
                  output_candidate_edges=len(edges), seconds=time.monotonic()-start,
                  ground_truth_used_for_inference=False)
    pairs0 = {tuple(p) for p in original[:, :2].astype(np.int64)}
    pairs1 = {tuple(p) for p in edges[:, :2].astype(np.int64)}
    report.update(new_candidate_pairs=len(pairs1-pairs0), removed_candidate_pairs=len(pairs0-pairs1))
    out = Path(os.environ['BIOHUB_EXTERNAL_REPORT_DIR'])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/(dataset+'.npz'), coords=coords, original_edges=original,
                        pool_edges=pool, edges=edges,
                        division_cost=np.asarray(costs) if costs is not None else np.empty(0))
    (out/(dataset+'.json')).write_text(json.dumps(report, indent=2))
    print('EXTERNAL_VIDEO '+json.dumps(report), flush=True)
    return [(int(s), int(t), float(p), float(d)) for s,t,p,d in edges]


def division_weight(graph, dataset, default):
    if dataset not in DIVISION_COSTS:
        return default
    from cellmot.explore_cellect import attach_division_cost
    return attach_division_cost(graph, DIVISION_COSTS.pop(dataset))
