"""Frozen-feature coordinate regression at first-seen fused detector centers."""
import os
from pathlib import Path
import numpy as np
import torch

SPACING = np.array([1.625, 1.625, 1.625], dtype=np.float32)
OFFSETS = ((0,0,0), (-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1))
_CACHE = None


def make_head():
    head = torch.nn.Sequential(torch.nn.Linear(224, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    return head


def bounded(head, x):
    delta = head(x)
    return 2.0 * delta / (1.0 + torch.linalg.vector_norm(delta, dim=-1, keepdim=True))


def sample_features(feature, arr):
    xyz = torch.as_tensor(arr[:, 1:], device=feature.device, dtype=torch.long)
    blocks = []
    for offset in OFFSETS:
        loc = xyz + torch.tensor(offset, device=feature.device)
        for axis, size in enumerate(feature.shape[-3:]):
            loc[:, axis].clamp_(0, size-1)
        blocks.append(feature[0, :, loc[:,0], loc[:,1], loc[:,2]].T)
    # Directional differences plus the central representation.
    return torch.cat([blocks[0]] + [b - blocks[0] for b in blocks[1:]], dim=1)


def index_features(self, maps, coords, mask):
    """Trilinear lookup; integer coordinates reproduce native gather exactly."""
    out = torch.zeros((*coords.shape[:2], maps.shape[1]), device=maps.device, dtype=maps.dtype)
    for batch in range(len(maps)):
        n = int(mask[batch].sum())
        if not n:
            continue
        q = coords[batch, :n].clone()
        for axis, size in enumerate(maps.shape[-3:]):
            q[:,axis].clamp_(0, size-1)
        low = q.floor().long()
        frac = q-low
        for z in (0,1):
            for y in (0,1):
                for x in (0,1):
                    shift = torch.tensor([z,y,x], device=maps.device)
                    loc = low+shift
                    for axis, size in enumerate(maps.shape[-3:]):
                        loc[:,axis].clamp_(0,size-1)
                    weight = torch.where(shift.bool(), frac, 1-frac).prod(dim=1)
                    out[batch,:n] += maps[batch,:,loc[:,0],loc[:,1],loc[:,2]].T * weight[:,None]
    return out


def refine(ds_path, t, arr, feature):
    global _CACHE
    mode = os.environ['V1284_MODE']
    if not len(arr):
        return arr
    if mode == 'zero':
        return arr.astype(np.float32)
    x = sample_features(feature, arr).float()
    if mode == 'capture':
        folder = Path(os.environ['V1284_CAPTURE']) / ds_path.stem
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/f'{int(t):04d}.npz', coords=arr, features=x.cpu().numpy())
        return arr
    if _CACHE is None:
        saved = torch.load(os.environ['V1284_HEAD'], map_location='cpu', weights_only=True)
        head = make_head().to(feature.device)
        head.load_state_dict(saved['state_dict']); head.eval()
        _CACHE = (head, saved['mean'].to(feature.device), saved['scale'].to(feature.device))
    head, mean, scale = _CACHE
    shift = bounded(head, (x-mean)/scale).cpu().numpy() / SPACING
    result = arr.astype(np.float32).copy()
    result[:,1:] += shift
    result[:,1:] = np.clip(result[:,1:], 0, np.asarray(feature.shape[-3:])-1)
    if not np.isfinite(result).all() or np.max(np.linalg.norm((result[:,1:]-arr[:,1:])*SPACING,axis=1)) > 2.00001:
        raise RuntimeError('invalid V1284 displacement')
    return result
