"""Trackastra re-association supported by independent CELLECT image features."""
import numpy as np


def combine(coords, original_edges, track_pairs, track_prob, cell_pairs, cell_prob,
            minimum_track=.4, minimum_cellect=.8, margin=.15, radius_um=5.5):
    n=len(coords)
    orig=np.asarray(original_edges,np.float64).reshape(-1,4)
    tp=np.asarray(track_pairs,np.int64); cp=np.asarray(cell_pairs,np.int64)
    tk=tp[:,0]*n+tp[:,1]; ck=cp[:,0]*n+cp[:,1]
    order=np.argsort(tk); tk=tk[order]; tv=np.asarray(track_prob)[order]
    keys=orig[:,0].astype(np.int64)*n+orig[:,1].astype(np.int64)
    positions=np.searchsorted(tk,keys)
    if len(keys) and (np.any(positions>=len(tk)) or not np.array_equal(tk[positions],keys)):
        raise ValueError('Original edges must be scored by Trackastra')
    out=orig.copy();out[:,2]=tv[positions]
    old_best=np.zeros(n)
    np.maximum.at(old_best,orig[:,1].astype(np.int64),out[:,2])
    # Only candidate construction uses CELLECT; edge costs remain native Trackastra.
    loc=np.searchsorted(tk,ck)
    valid=loc<len(tk)
    valid[valid]&=tk[loc[valid]]==ck[valid]
    selected=np.flatnonzero(valid & (np.asarray(cell_prob)>=minimum_cellect) & ~np.isin(ck,keys))
    pair=cp[selected]; p=tv[loc[selected]]
    grid=np.asarray(coords)[:,1:]/[1.,4.,4.]
    dist=np.linalg.norm(grid[pair[:,0]]-grid[pair[:,1]],axis=1)
    keep=(p>=minimum_track)&(p>=old_best[pair[:,1]]+margin)&(dist*1.625<=radius_um)
    extra=np.column_stack([pair[keep],p[keep],dist[keep]])
    out=np.concatenate([out,extra])
    out=out[np.lexsort((out[:,1],out[:,0]))]
    return out,dict(original_edges=len(orig),added_edges=len(extra),
                    existing_parent_targets=int(np.sum(old_best[extra[:,1].astype(np.int64)]>0)) if len(extra) else 0,
                    config=dict(minimum_track=minimum_track,minimum_cellect=minimum_cellect,margin=margin,radius_um=radius_um),
                    edge_cost_semantics='native Trackastra affinity; CELLECT controls new candidate admission')


def associate(coords,raw_volume,pool_edges,original_edges,root,device='cuda',config=None):
    from cellmot.explore_trackastra import associate as ta
    from cellmot.explore_cellect import associate as ca, NativeCellectConfig
    te,ti=ta(coords,raw_volume,pool_edges,root/'models/ctc',device=device,
             config=dict(threshold=0.,maximum_distance_um=1e6))
    ce,_,ci=ca(coords,raw_volume,root,device=device,config=NativeCellectConfig(gating='all'))
    ci.pop('scored_mask',None)
    edges,info=combine(coords,original_edges,te[:,:2],te[:,2],ce[:,:2],ce[:,2],**(config or {}))
    return edges,dict(trackastra=ti,cellect=ci,graph=info)
