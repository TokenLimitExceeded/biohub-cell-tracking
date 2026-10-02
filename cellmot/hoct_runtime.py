"""Optional production adapter; requires local HOCT package and checkpoint."""
from __future__ import annotations
from dataclasses import asdict
import numpy as np
from cellmot.hoct import HoctConfig,extract_features,predict_scores,blend_candidates,load_offline_model

_MODELS={}


class Stride4Volume:
    """Read one strided frame at a time from a raw Zarr volume."""
    def __init__(self,raw,frames=None):
        self.raw=raw
        self.shape=(raw.shape[0] if frames is None else frames,raw.shape[1],
                    (raw.shape[2]+3)//4,(raw.shape[3]+3)//4)

    def __getitem__(self,t):
        return np.asarray(self.raw[t])[:,::4,::4]


def rescore_video(coords,edges,context_pairs,volume,checkpoint,alpha,device="cuda",config=HoctConfig()):
    """Apply HOCT immediately before the existing graph builder/ILP.

    context_pairs must contain the original association model's p>0.01 pairs,
    collected before its solver candidate threshold. This function preserves
    solver edges exactly and never downloads weights or reads annotations.
    """
    if not 0<=alpha<=1:
        raise ValueError("HOCT alpha must be in [0,1]")
    if alpha==0 or not len(edges):
        return edges,dict(status="disabled" if alpha==0 else "empty_candidates",alpha=alpha)
    key=(str(checkpoint),str(device))
    if key not in _MODELS:
        _MODELS[key]=load_offline_model(checkpoint,device=str(device))
    pairs=np.asarray(context_pairs,np.int64).reshape(-1,2)
    features,fi=extract_features(volume,np.asarray(coords),config)
    probabilities,pi=predict_scores(_MODELS[key],np.asarray(coords),pairs,features,config)
    original=np.asarray(edges,np.float64).reshape(-1,4)
    changed=blend_candidates(original,pairs,probabilities,alpha)
    report=dict(status="complete",alpha=alpha,features=fi,inference=pi,config=asdict(config),
                changed_scores=int(np.count_nonzero(changed[:,2]!=original[:,2])),
                candidate_edges=len(edges),context_edges=len(pairs))
    return [(int(s),int(t),float(p),float(d)) for s,t,p,d in changed],report
