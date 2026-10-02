"""Conservative Trackastra re-association with native score margins.

Original B30 pairs are retained with Trackastra probabilities. Additional pairs
can compete only when physically short, confident, and clearly stronger than
the existing parent in the same Trackastra score system.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import time
import numpy as np
from cellmot.explore_trackastra import Config, GRID_UM, candidate_pool, _MODELS
from cellmot.external_trackastra import extract_features, load_ctc_model, predict_scores

@dataclass(frozen=True)
class GuardedConfig:
    addition_threshold: float = .5
    addition_radius_um: float = 5.5
    replacement_margin: float = .2
    max_children: int = 2

def select_guarded(coords, original_edges, pairs, probability, distance_um, config=GuardedConfig()):
    coords, old, pairs = np.asarray(coords), np.asarray(original_edges), np.asarray(pairs)
    if isinstance(config, dict): config = GuardedConfig(**config)
    n = len(coords)
    keys = pairs[:,0] * n + pairs[:,1]
    order = np.argsort(keys)
    keys = keys[order]
    original_keys = old[:,0].astype(np.int64) * n + old[:,1].astype(np.int64)
    loc = np.searchsorted(keys, original_keys)
    if (loc >= len(keys)).any() or not np.array_equal(keys[loc],original_keys):
        raise ValueError("Expanded score pool must include all original B30 pairs")
    old_prob = probability[order[loc]]
    rescored = old.copy()
    rescored[:,2] = old_prob
    # Score margins compare values from the same model, with no mixture.
    strongest = np.zeros(n)
    np.maximum.at(strongest,old[:,1].astype(np.int64),old_prob)
    parents = np.bincount(old[:,1].astype(np.int64),minlength=n)
    children = np.bincount(old[:,0].astype(np.int64),minlength=n)
    new = ~np.isin(pairs[:,0]*n+pairs[:,1],original_keys)
    eligible = np.flatnonzero(new & (probability >= config.addition_threshold) & (distance_um <= config.addition_radius_um)
        & ((parents[pairs[:,1]] == 0) | (probability >= strongest[pairs[:,1]] + config.replacement_margin)))
    chosen=[]
    selected_target=set()
    for row in eligible[np.argsort(-probability[eligible],kind="stable")]:
        src,dst=map(int,pairs[row])
        if children[src] >= config.max_children or dst in selected_target:
            continue
        chosen.append(row)
        children[src] += 1
        selected_target.add(dst)
    chosen=np.asarray(chosen,dtype=np.int64)
    extra=np.column_stack((pairs[chosen],probability[chosen],distance_um[chosen]/GRID_UM))
    result=np.concatenate((rescored,extra))
    return result,dict(config=asdict(config),score_kind="native Trackastra quiet-softmax mean; no mixture",
        original_candidates=len(old),eligible_additions=len(eligible),added_candidates=len(extra),
        competitive_added_candidates=int((parents[pairs[chosen,1]]>0).sum()),
        formerly_parentless_additions=int((parents[pairs[chosen,1]]==0).sum()),
        output_candidates=len(result),distance_units="stride4 grid; 1 unit = 1.625 um")

def associate(coords,raw_volume,pool_edges,model_folder,device="cuda",config=GuardedConfig(),*,original_edges):
    """Deployment entry point; original_edges is predictor's actual pruned graph."""
    started=time.perf_counter()
    coords=np.asarray(coords)
    volume=np.asarray(raw_volume[:,:,::4,::4],dtype=np.float32)
    pairs,distances,flags=candidate_pool(coords,pool_edges,Config())
    key=(str(model_folder),device)
    if key not in _MODELS: _MODELS[key]=load_ctc_model(model_folder,device)
    model,mi=_MODELS[key]
    feats,fi=extract_features(volume,coords)
    prob,counts,si=predict_scores(model,coords,pairs,feats,total_frames=len(volume))
    edges,info=select_guarded(coords,original_edges,pairs,prob,distances,config)
    info.update(checkpoint_sha256=mi["model_sha256"],features=fi,inference=si,
        expanded_candidates=len(pairs),no_gt_in_inference=True,seconds=time.perf_counter()-started)
    return edges,info
