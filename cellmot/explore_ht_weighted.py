"""Bounded HOCT/Trackastra affinity weights on the frozen strict-T graph.

The coefficient is not a probability calibration. Candidate admission always
uses the same native Trackastra gates as explore_ht v1; weights cannot change it.
"""
from __future__ import annotations
import os
import numpy as np
from cellmot.explore_trackastra_guarded import select_guarded, GuardedConfig

ALLOWED_TRACK_WEIGHTS = (.25, .5, .75)

def combine(coords, original_edges, track_pairs, track_prob, track_distance_um,
            hoct_pairs, hoct_prob, track_weight=.5):
    track_weight=float(track_weight)
    if track_weight not in ALLOWED_TRACK_WEIGHTS:
        raise ValueError("This bounded experiment permits only T weights .25, .5, .75")
    edges,info=select_guarded(coords,original_edges,track_pairs,track_prob,track_distance_um,
        GuardedConfig(addition_threshold=.9,addition_radius_um=3.25,replacement_margin=.3,max_children=2))
    n=len(coords)
    hp=np.asarray(hoct_pairs,np.int64)
    scores=np.asarray(hoct_prob)
    if hp.ndim!=2 or hp.shape[1]!=2 or scores.shape!=(len(hp),):
        raise ValueError("HOCT pairs/probability shape mismatch")
    if not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
        raise ValueError("Invalid native HOCT scores")
    keys=hp[:,0]*n+hp[:,1]
    if len(np.unique(keys))!=len(keys):
        raise ValueError("Duplicate HOCT pair")
    order=np.argsort(keys)
    keys=keys[order]
    wanted=edges[:,0].astype(np.int64)*n+edges[:,1].astype(np.int64)
    loc=np.searchsorted(keys,wanted)
    valid=loc<len(keys)
    valid[valid] &= keys[loc[valid]]==wanted[valid]
    if not valid[:len(original_edges)].all():
        raise ValueError("HOCT must cover every original candidate")
    edges[valid,2]=track_weight*edges[valid,2]+(1.-track_weight)*scores[order[loc[valid]]]
    info.update(score_kind="weighted HOCT/Trackastra affinity; not calibrated probability",
        track_weight=track_weight,hoct_weight=1.-track_weight,mixed_edges=int(valid.sum()),
        trackastra_only_geometry_edges=int((~valid).sum()),
        candidate_admission="fixed T>=.9, radius<=3.25um, parent margin>=.3, source capacity2",
        missing_hoct_policy="Retain native Trackastra score, exactly as HT v1")
    return edges,info

def associate(coords,raw,pool,original,root,device="cuda",config=None):
    """Same arguments as HT v1, with config={'track_weight': .25 or .75}."""
    from cellmot.explore_trackastra import associate as ta
    from cellmot.hoct import HoctConfig,load_offline_model,extract_features,predict_scores
    config=dict(config or {})
    track_weight=float(config.pop("track_weight",.5))
    if config or track_weight not in ALLOWED_TRACK_WEIGHTS:
        raise ValueError("Only the prespecified Trackastra mixture coefficient may change")
    te,ti=ta(coords,raw,pool,root/"models/ctc",device=device,
        config=dict(threshold=0.,maximum_distance_um=1e6))
    volume=np.asarray(raw[:,:,::4,::4],np.float32)
    cfg=HoctConfig()
    model=load_offline_model(os.environ["BIOHUB_HOCT_CHECKPOINT"],device=device)
    features,fi=extract_features(volume,coords,cfg)
    pairs=pool[:,:2].astype(np.int64)
    prob,pi=predict_scores(model,coords,pairs,features,cfg)
    edges,info=combine(coords,original,te[:,:2].astype(np.int64),te[:,2],te[:,3]*1.625,
        pairs,prob,track_weight=track_weight)
    return edges,dict(trackastra=ti,hoct=dict(features=fi,inference=pi),graph=info)
