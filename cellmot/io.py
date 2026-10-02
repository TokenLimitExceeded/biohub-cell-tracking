"""Readers for the competition zarr v3 image volumes and geff ground-truth graphs."""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd

SCALE = (1.625, 0.40625, 0.40625)  # z, y, x in um/voxel


def read_geff(path: str):
    """Return (nodes_df[node_id,t,z,y,x], edges_df[source_id,target_id], meta_dict)."""
    import zarr
    g = zarr.open_group(path, mode="r")
    ids = np.asarray(g["nodes/ids"][:])
    props = {k: np.asarray(g[f"nodes/props/{k}/values"][:]) for k in ("t", "z", "y", "x")}
    e = np.asarray(g["edges/ids"][:]).reshape(-1, 2)
    nodes = pd.DataFrame({"node_id": ids.astype(np.int64), **{k: v for k, v in props.items()}})
    edges = pd.DataFrame({"source_id": e[:, 0].astype(np.int64), "target_id": e[:, 1].astype(np.int64)})
    attrs = json.loads(json.dumps(dict(g.attrs), default=str))
    meta = (attrs.get("geff", {}) or {})
    extra = meta.get("extra", {}) or {}
    return nodes, edges, {"estimated_number_of_nodes": extra.get("estimated_number_of_nodes"), "attrs": attrs}


class ZarrVolume:
    """Lazy (T,Z,Y,X) uint16 volume backed by a zarr v3 array at path/0."""
    def __init__(self, path: str):
        import zarr
        self.path = path
        self.group = zarr.open_group(path, mode="r")
        self.arr = self.group["0"]
        self.shape = tuple(self.arr.shape)
        self.attrs = json.loads(json.dumps(dict(self.group.attrs), default=str))
        ms = self.attrs.get("multiscales")
        self.scale = tuple(ms[0]["datasets"][0]["coordinateTransformations"][0]["scale"][-3:]) if ms else SCALE
        self.quantiles = (self.attrs.get("image_statistics", {}) or {}).get("quantiles", {})

    def frame(self, t: int) -> np.ndarray:
        return np.asarray(self.arr[t])

    def frames(self, t0: int, t1: int) -> np.ndarray:
        return np.asarray(self.arr[t0:t1])

    def __len__(self):
        return self.shape[0]


def list_datasets(root: str, split: str = "train", require_geff: bool = True):
    d = os.path.join(root, split)
    names = sorted(n[:-5] for n in os.listdir(d) if n.endswith(".zarr"))
    if require_geff:
        names = [n for n in names if os.path.isdir(os.path.join(d, n + ".geff"))]
    return names


def quantile_normalize(vol: np.ndarray, q_lo: float = 0.001, q_hi: float = 0.999, gamma: float = 1.0,
                       clip_max: float = 4.0, sub: int = 50) -> np.ndarray:
    v = vol.astype(np.float32)
    q1, q2 = np.quantile(v.ravel()[::sub], [q_lo, q_hi])
    v = (v - q1) / (q2 - q1 + 1e-6)
    v = np.clip(v, 0.0, None)
    if gamma != 1.0:
        v = v ** gamma
    return np.clip(v, 0.0, clip_max)
