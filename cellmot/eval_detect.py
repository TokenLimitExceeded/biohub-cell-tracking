"""Detector evaluation: recall@7um on GT nodes, node-count ratio vs estimated_number_of_nodes."""
from __future__ import annotations
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from .detect import extract_peaks, physical_nms, pooled_to_raw, tta_views, make_input, POOLED_SCALE, RAW_SCALE


@torch.no_grad()
def predict_heatmaps(model, vol_f16: np.ndarray, device="cuda", batch=16, tta=False, ctx=1, input_mode="raw"):
    """vol_f16: (T,Z,Y,X) pooled normalized. Returns prob (T,Z,Y,X) float16 tensor on CPU."""
    model.eval()
    T = vol_f16.shape[0]
    v = torch.from_numpy(vol_f16.astype(np.float32))
    out = torch.empty((T,) + tuple(vol_f16.shape[1:]), dtype=torch.float16)
    for t0 in range(0, T, batch):
        ts = list(range(t0, min(T, t0 + batch)))
        x = torch.stack([torch.stack([v[min(max(t + d, 0), T - 1)] for d in range(-ctx, ctx + 1)], 0) for t in ts], 0).to(device)
        x = make_input(x, input_mode)
        with torch.autocast("cuda", dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)):
            if tta:
                acc = None
                for xv, inv in tta_views(x):
                    o = inv(model(xv).float())
                    acc = o if acc is None else acc + o
                logits = acc / 8.0
            else:
                logits = model(x).float()
        out[t0:t0 + len(ts)] = torch.sigmoid(logits[:, 0]).half().cpu()
    return out


def peaks_from_prob(prob_t: torch.Tensor, threshold: float, nms_um: float = 3.0):
    """Returns pooled-grid coords (N,3) float, scores (N,), raw coords (N,3)."""
    c, s = extract_peaks(prob_t, threshold=threshold)
    if len(c) and nms_um > 0:
        keep = physical_nms(c * POOLED_SCALE, s, nms_um)
        c, s = c[keep], s[keep]
    return c, s, pooled_to_raw(c)


def recall_stats(gt_nodes, pred_raw_by_t: dict, max_um=7.0):
    """gt_nodes: DataFrame(t,z,y,x) raw voxels; pred_raw_by_t: {t: (N,3) raw coords}. Hungarian per frame."""
    hit = 0; n = 0; dists = []
    for t, g in gt_nodes.groupby("t"):
        gp = g[["z", "y", "x"]].values * RAW_SCALE
        n += len(gp)
        p = pred_raw_by_t.get(int(t))
        if p is None or len(p) == 0:
            continue
        d = cdist(gp, p * RAW_SCALE)
        r, c = linear_sum_assignment(d)
        m = d[r, c] <= max_um
        hit += int(m.sum()); dists.extend(d[r, c][m].tolist())
    return {"recall": hit / max(n, 1), "n_gt": n, "hit": hit, "mean_dist": float(np.mean(dists)) if dists else float("nan"),
            "p90_dist": float(np.percentile(dists, 90)) if dists else float("nan")}


def evaluate_detector(model, cache_dir, names, gt_nodes_all, meta_est: dict, thresholds=(0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5),
                      device="cuda", tta=False, nms_um=3.0, ctx=1, verbose=True, input_mode="raw"):
    """Returns dict threshold -> summary(recall, ratio, mean_dist) averaged over datasets, plus per-dataset rows."""
    import os
    rows = []
    for name in names:
        vol = np.load(os.path.join(cache_dir, name + ".npy"))
        prob = predict_heatmaps(model, vol, device=device, tta=tta, ctx=ctx, input_mode=input_mode)
        gt = gt_nodes_all[gt_nodes_all.dataset == name]
        est = meta_est.get(name, np.nan)
        for thr in thresholds:
            by_t = {}; total = 0
            for t in range(prob.shape[0]):
                _, _, raw = peaks_from_prob(prob[t].float().to(device), thr, nms_um)
                by_t[t] = raw; total += len(raw)
            st = recall_stats(gt, by_t)
            rows.append({"dataset": name, "thr": thr, "n_pred": total, "est": est, "ratio": total / est if est == est and est else np.nan, **st})
        if verbose:
            r = [x for x in rows if x["dataset"] == name]
            print(f"  {name}: " + " | ".join(f"thr{x['thr']:.2f} R={x['recall']:.3f} ratio={x['ratio']:.2f} d={x['mean_dist']:.2f}" for x in r), flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    summ = df.groupby("thr").agg(recall=("hit", "sum"), n_gt=("n_gt", "sum"), ratio=("ratio", "mean"), mean_dist=("mean_dist", "mean"), n_pred=("n_pred", "sum"))
    summ["recall"] = summ["recall"] / summ["n_gt"]
    return summ, df
