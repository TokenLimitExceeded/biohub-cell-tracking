"""Prespecified equal-affinity HOCT/Trackastra mixture on strict T additions."""
from pathlib import Path
import os
import numpy as np
from cellmot.explore_trackastra_guarded import select_guarded, GuardedConfig


def combine(coords,original_edges,track_pairs,track_prob,track_distance_um,hoct_pairs,hoct_prob):
    edges,info=select_guarded(coords,original_edges,track_pairs,track_prob,track_distance_um,
        GuardedConfig(addition_threshold=.9,addition_radius_um=3.25,replacement_margin=.3,max_children=2))
    n=len(coords);hp=np.asarray(hoct_pairs,np.int64)
    keys=hp[:,0]*n+hp[:,1];order=np.argsort(keys);keys=keys[order]
    wanted=edges[:,0].astype(np.int64)*n+edges[:,1].astype(np.int64)
    loc=np.searchsorted(keys,wanted);valid=loc<len(keys)
    valid[valid]&=keys[loc[valid]]==wanted[valid]
    assert valid[:len(original_edges)].all()
    edges[valid,2]=.5*edges[valid,2]+.5*np.asarray(hoct_prob)[order[loc[valid]]]
    info.update(score_kind='equal-weight HOCT/Trackastra affinity; not calibrated probability',
                mixture='0.5 Trackastra + 0.5 HOCT affinity; not calibrated probability',
                mixed_edges=int(valid.sum()),trackastra_only_geometry_edges=int((~valid).sum()))
    return edges,info


def associate(coords,raw,pool,original,root,device='cuda'):
    from cellmot.explore_trackastra import associate as ta
    from cellmot.hoct import HoctConfig,load_offline_model,extract_features,predict_scores
    te,ti=ta(coords,raw,pool,root/'models/ctc',device=device,config=dict(threshold=0.,maximum_distance_um=1e6))
    volume=np.asarray(raw[:,:,::4,::4],np.float32)
    cfg=HoctConfig();model=load_offline_model(os.environ['BIOHUB_HOCT_CHECKPOINT'],device=device)
    features,fi=extract_features(volume,coords,cfg)
    pairs=pool[:,:2].astype(np.int64)
    prob,pi=predict_scores(model,coords,pairs,features,cfg)
    edges,info=combine(coords,original,te[:,:2].astype(np.int64),te[:,2],te[:,3]*1.625,pairs,prob)
    return edges,dict(trackastra=ti,hoct=dict(features=fi,inference=pi),graph=info)
