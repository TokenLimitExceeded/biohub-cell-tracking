"""Frozen sparse-motion/geometry safe additions for portable Biohub inference.

Coordinates are original voxels t,z,y,x. Pool rows are (source,target,B30 p).
Output distance remains Euclidean stride4-grid distance, as in production B30.
No GT access. This copies the predeclared motion_native candidate rules exactly.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from .external_motion import RAW_SPACING_UM, apply_motion_to_candidates, load_motion_checkpoint, predict_motion

def _node_patches(volume,coords):
    grid=np.rint(coords[:,1:]/[1,4,4]).astype(int)
    offsets=np.stack(np.meshgrid(*[np.arange(-2,3)]*3,indexing='ij')).reshape(3,-1).T
    patches=np.empty((len(coords),125),np.float32)
    for t in np.unique(coords[:,0]).astype(int):
        ids=np.flatnonzero(coords[:,0]==t)
        frame=np.pad(np.log1p(np.asarray(volume[t],np.float32)),2,mode='edge')
        loc=grid[ids,None,:]+offsets[None]+2
        a=frame[loc[:,:,0],loc[:,:,1],loc[:,:,2]]
        a-=a.mean(1,keepdims=True);a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-6)
        patches[ids]=a
    return patches


def _build_graphs(cap,motion,volume):
    c=cap['cand_coords'];original=cap['cand_edges'];n=len(c)
    pos=c[:,1:]*RAW_SPACING_UM
    ps,pt,pp=cap['probs_src'],cap['probs_tgt'],cap['probs_p']
    order=np.argsort(pt,kind='stable');ps,pt,pp=ps[order],pt[order],pp[order]
    starts=np.r_[0,np.flatnonzero(np.diff(pt))+1,len(pt)]
    original_pairs=set(map(tuple,original[:,:2].astype(int)))
    orig_by_t={}
    for s,t,_,_ in original:orig_by_t.setdefault(int(t),set()).add(int(s))
    extras=[]
    for a,b in zip(starts[:-1],starts[1:]):
        target=int(pt[a]);ss=ps[a:b];p=pp[a:b]
        d=np.linalg.norm(pos[ss]-pos[target],axis=1);ok=d<=12.
        ss,p,d=ss[ok],p[ok],d[ok]
        if not len(ss):continue
        proposed=np.argsort(-p,kind='stable')[:3].tolist()+[int(np.argmin(d))]
        if np.isfinite(motion[target]).all():proposed.append(int(np.argmin(np.linalg.norm(pos[ss]-pos[target]-motion[target],axis=1))))
        chosen=set(orig_by_t.get(target,set()))
        for j in proposed:
            source=int(ss[j])
            if source in chosen or len(chosen)>=6:continue
            chosen.add(source)
            dist=float(np.linalg.norm((c[source,1:].astype(np.float32)-c[target,1:].astype(np.float32))/[1,4,4]))
            extras.append((source,target,float(p[j]),dist))
    extras=np.asarray(sorted(extras,key=lambda e:(e[1],e[0])),np.float64).reshape(-1,4)
    full=np.vstack([original,extras])
    if len(np.unique(full[:,:2],axis=0))!=len(full):raise AssertionError('duplicate proposals')
    assert np.bincount(full[:,1].astype(int),minlength=n).max()<=6
    patches=_node_patches(volume,c)
    ij=full[:,:2].astype(int);ncc=np.sum(patches[ij[:,0]]*patches[ij[:,1]],axis=1)
    raw=np.linalg.norm(pos[ij[:,0]]-pos[ij[:,1]],axis=1)
    residual=np.linalg.norm(pos[ij[:,0]]-pos[ij[:,1]]-motion[ij[:,1]],axis=1)
    gate=np.zeros(n,np.float32);best_ncc=np.full(n,-1,np.int64);ncc_margin=np.zeros(n)
    target_order=np.argsort(ij[:,1],kind='stable')
    target_starts=np.r_[0,np.flatnonzero(np.diff(ij[target_order,1]))+1,len(ij)]
    for a,b in zip(target_starts[:-1],target_starts[1:]):
        rr=target_order[a:b];t=int(ij[rr[0],1])
        ranks=rr[np.argsort(-ncc[rr],kind='stable')]
        best_ncc[t]=ranks[0];ncc_margin[t]=float(ncc[ranks[0]]-ncc[ranks[1]]) if len(rr)>1 else 2.
        if len(rr)<2 or not np.isfinite(motion[t]).all():continue
        mr=rr[np.argsort(residual[rr],kind='stable')]
        mag=np.linalg.norm(motion[t])
        if .5<=mag<=4. and residual[mr[0]]<=3. and residual[mr[1]]-residual[mr[0]]>=1. and mr[0]==ranks[0] and ncc[mr[0]]>=.6 and ncc_margin[t]>=.1:
            gate[t]=1.
    # Independent conservative addition: source and target were both unlinked,
    # are mutual spatial nearest neighbours, and image patch agreement is clear.
    outdeg=np.bincount(original[:,0].astype(int),minlength=n);indeg=np.bincount(original[:,1].astype(int),minlength=n)
    nearest_next=np.full(n,-1,np.int64);nearest_prev=np.full(n,-1,np.int64)
    for t in np.unique(c[:,0]).astype(int):
        s=np.flatnonzero(c[:,0]==t-1);d=np.flatnonzero(c[:,0]==t)
        if not len(s) or not len(d):continue
        nearest_next[s]=d[cKDTree(pos[d]).query(pos[s])[1]]
        nearest_prev[d]=s[cKDTree(pos[s]).query(pos[d])[1]]
    kept=[]
    for row in range(len(original),len(full)):
        s,t=ij[row]
        if indeg[t]==0 and outdeg[s]==0 and nearest_next[s]==t and nearest_prev[t]==s and raw[row]<=5.5 and full[row,2]>=.05 and best_ncc[t]==row and ncc[row]>=.6 and ncc_margin[t]>=.1:
            kept.append(row)
    safe=np.vstack([original,full[kept]])
    stats=dict(original_edges=len(original),expanded_edges=len(full),expanded_new_edges=len(extras),safe_new_edges=len(kept),
               motion_gated_targets=int(gate.sum()),nodes=len(c),candidate_counts={str(int(k)):int(v) for k,v in zip(*np.unique(np.bincount(ij[:,1],minlength=n)[c[:,0]>0],return_counts=True))},
               no_gt_in_candidate_generation=True,gate_semantics='NCC and geometric consistency; learned position magnitude not treated as calibrated confidence')
    return full,safe,gate,ncc,stats


def build_safe_candidates(coords, original_edges, pool_edges, volume, *,
                          motion_um=None, checkpoint=None, q_lo=None, q_hi=None,
                          image_grid="stride4", device="cpu", apply_motion=False):
    """Return (safe_edges, info); info contains motion_um and binary motion_gate.

    volume: indexable (T,Z,Y,X) stride4 cache or raw Zarr/ndarray (image_grid="raw").
    q_lo/q_hi: frozen preprocessing quantiles, required if motion is inferred.
    apply_motion=False permits replacing all edge probabilities with another
    scorer before calling reweight_safe_candidates. Original edges stay first.
    """
    started=time.perf_counter()
    coords=np.asarray(coords);original_edges=np.asarray(original_edges)
    pool_edges=np.asarray(pool_edges)
    if coords.ndim!=2 or coords.shape[1]!=4 or original_edges.ndim!=2 or original_edges.shape[1]!=4:
        raise ValueError("Expected coords[N,4] and original_edges[E,4]")
    if pool_edges.ndim!=2 or pool_edges.shape[1]<3:
        raise ValueError("Expected pool_edges[P,>=3] with native B30 probability")
    if image_grid=="raw":
        volume=np.stack([np.asarray(volume[t])[:,::4,::4] for t in range(volume.shape[0])])
    elif image_grid!="stride4":raise ValueError("image_grid must be raw or stride4")
    grid=np.rint(coords[:,1:]/[1,4,4]).astype(int)
    if not ((grid>=0).all() and (grid<np.asarray(volume.shape[1:])).all()):
        raise ValueError("Detection coordinates outside image")
    if not np.isfinite(coords).all() or not np.isfinite(original_edges).all() or not np.isfinite(pool_edges).all():
        raise ValueError("Nonfinite candidate input")
    for edges in (original_edges,pool_edges):
        pairs=edges[:,:2].astype(np.int64)
        if not np.array_equal(pairs,edges[:,:2]) or (pairs<0).any() or (pairs>=len(coords)).any():
            raise ValueError("Invalid candidate endpoints")
        if not np.all(coords[pairs[:,1],0]-coords[pairs[:,0],0]==1):
            raise ValueError("Candidates must link consecutive frames")
    model_seconds=0.
    if motion_um is None:
        if checkpoint is None or q_lo is None or q_hi is None:
            raise ValueError("Checkpoint and frozen q_lo/q_hi required for motion inference")
        model,_=load_motion_checkpoint(Path(checkpoint),device)
        t0=time.perf_counter()
        motion_um,_=predict_motion(model,volume,coords,q_lo=q_lo,q_hi=q_hi,device=device)
        model_seconds=time.perf_counter()-t0
    motion_um=np.asarray(motion_um)
    if motion_um.shape!=(len(coords),3):raise ValueError("motion_um must be [N,3] backward z,y,x um")
    pool_edges=pool_edges[pool_edges[:,2]>.01]
    cap=dict(cand_coords=coords,cand_edges=original_edges,
             probs_src=pool_edges[:,0].astype(np.int64),probs_tgt=pool_edges[:,1].astype(np.int64),probs_p=pool_edges[:,2])
    _,safe,gate,_,stats=_build_graphs(cap,motion_um,volume)
    if apply_motion:safe=reweight_safe_candidates(coords,safe,motion_um,gate)
    return safe,dict(stats=stats,motion_um=motion_um,motion_gate=gate,
        model_seconds=model_seconds,seconds=time.perf_counter()-started,
        motion_applied=bool(apply_motion),no_gt_in_inference=True,
        probability_semantics="native B30 probabilities unless caller subsequently replaces them",
        distance_units="Euclidean stride4 grid; each grid axis 1.625 um")


def reweight_safe_candidates(coords, edges, motion_um, motion_gate):
    """Apply the frozen strength=.25, sigma=3 um binary support gate."""
    return apply_motion_to_candidates(coords,edges,motion_um,strength=.25,sigma_um=3.,support=motion_gate)
