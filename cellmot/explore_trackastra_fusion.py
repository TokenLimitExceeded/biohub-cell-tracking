"""Explicit uncalibrated affinity fusion after strict native candidate admission."""
from __future__ import annotations
import numpy as np
from cellmot.explore_trackastra_guarded import associate as native_associate, GuardedConfig

def fuse(edges, pool_edges, number_of_nodes, track_weight=.5):
    if not 0 <= track_weight <= 1:
        raise ValueError("Invalid Trackastra weight")
    edges=np.asarray(edges,dtype=np.float64).copy()
    pool=np.asarray(pool_edges)
    keys=pool[:,0].astype(np.int64)*number_of_nodes+pool[:,1].astype(np.int64)
    order=np.argsort(keys)
    keys=keys[order]
    query=edges[:,0].astype(np.int64)*number_of_nodes+edges[:,1].astype(np.int64)
    pos=np.searchsorted(keys,query)
    found=pos<len(keys)
    found[found] &= keys[pos[found]]==query[found]
    base=np.zeros(len(edges))
    base[found]=pool[order[pos[found]],2]
    edges[:,2]=track_weight*edges[:,2]+(1-track_weight)*base
    return edges,dict(track_weight=track_weight,base_weight=1-track_weight,
        absent_base_scores_set_to_zero=int((~found).sum()),
        score_kind="uncalibrated convex affinity fusion; not native or calibrated probability")

def associate(coords,raw_volume,pool_edges,model_folder,device="cuda",config=None,*,original_edges):
    config=dict(config or {})
    weight=config.pop("track_weight",.5)
    if not config:
        config=dict(addition_threshold=.9,addition_radius_um=3.25,replacement_margin=.3,max_children=2)
    edges,info=native_associate(coords,raw_volume,pool_edges,model_folder,device,config,original_edges=original_edges)
    result,fi=fuse(edges,pool_edges,len(coords),weight)
    info["native_admission_score_kind"]=info.pop("score_kind")
    info.update(fi)
    return result,info
