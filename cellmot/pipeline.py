"""Shared inference pipeline pieces: detection (model or DoG) and tracking on one dataset."""
from __future__ import annotations
import glob, os, json
import numpy as np, pandas as pd, torch
from .detect import UNet3D, extract_peaks, physical_nms, pooled_to_raw, input_channels, POOLED_SCALE
from .eval_detect import predict_heatmaps
from .track import track, to_graph_frames, TrackConfig, RAW_SCALE


def load_models(weights, device="cuda"):
    """weights: dir with *.pt or list of paths. Returns (models, ctx, thr_default)."""
    paths = sorted(glob.glob(os.path.join(weights, "*.pt"))) if isinstance(weights, str) and os.path.isdir(weights) else list(weights if isinstance(weights, (list, tuple)) else [weights])
    if not paths:
        raise FileNotFoundError(f"no checkpoints found for {weights}")
    models = []; ctx = 1; thr = None
    for p in paths:
        ck = torch.load(p, map_location=device)
        cfg = ck["config"]; ch = tuple(int(c) for c in cfg["channels"].split(","))
        mode = cfg.get("input_mode", "raw")
        m = UNet3D(in_channels=input_channels(cfg["ctx"], mode), out_channels=1, channels=ch, norm=cfg.get("norm", "bn")).to(device)
        m.load_state_dict(ck["model"]); m.eval(); m.input_mode = mode; m.ctx = cfg["ctx"]; models.append(m); ctx = cfg["ctx"]
        if thr is None:
            thr = ck.get("thr_sel", None)
    return models, ctx, thr


def dog_heatmap(vol):
    from scipy.ndimage import gaussian_filter
    best = None
    for s in (1.0, 1.4, 2.0):
        d = gaussian_filter(vol, s) - gaussian_filter(vol, s * 1.6)
        best = d if best is None else np.maximum(best, d)
    return np.clip(best, 0, None)


def heatmaps(vol_f16, models=None, ctx=1, tta=False, device="cuda"):
    """Returns prob (T,Z,Y,X) float32 tensor (CPU): model ensemble average, or DoG response if no models."""
    T = vol_f16.shape[0]
    if models:
        prob = None
        for m in models:
            p = predict_heatmaps(m, vol_f16, device=device, tta=tta, ctx=getattr(m, "ctx", ctx), input_mode=getattr(m, "input_mode", "raw")).float()
            prob = p if prob is None else prob + p
        return prob / len(models)
    return torch.from_numpy(np.stack([dog_heatmap(vol_f16[t].astype(np.float32)) for t in range(T)]))


def peaks(prob, thr, nms_um=3.0, device="cuda", max_per_frame=6000):
    """prob (T,Z,Y,X) -> (t, pos_um (N,3), score (N,)). Coordinates in physical um (raw voxel * scale)."""
    ts, ps, ss = [], [], []
    for t in range(prob.shape[0]):
        c, s = extract_peaks(prob[t].to(device), threshold=thr)
        if len(c) and nms_um > 0:
            keep = physical_nms(c * POOLED_SCALE, s, nms_um); c, s = c[keep], s[keep]
        if len(c) > max_per_frame:
            o = np.argsort(-s)[:max_per_frame]; c, s = c[o], s[o]
        ts.append(np.full(len(c), t)); ps.append(pooled_to_raw(c) * RAW_SCALE); ss.append(s)
    return np.concatenate(ts), np.concatenate(ps).astype(np.float32), np.concatenate(ss).astype(np.float32)


def track_dataset(t, pos, score, tcfg: TrackConfig):
    tt, pp, s2, edges, keep, stats = track(t, pos, score, tcfg)
    nodes, edf = to_graph_frames(tt, pp, s2, edges, keep)
    return nodes, edf, stats


def make_track_config(overrides=None):
    tcfg = TrackConfig()
    if overrides:
        if isinstance(overrides, str):
            overrides = json.loads(overrides)
        for k, v in overrides.items():
            if not hasattr(tcfg, k):
                raise KeyError(f"unknown TrackConfig field {k}")
            setattr(tcfg, k, v)
    return tcfg
