"""Can a head on the frozen public UNet features beat the public V1284 head?  Held-out-by-video and by-embryo study.

Data: LANE/features/<stem>.npz from feature_capture.py (detections within 7 um of a GT node; 224 V1284 features; the
public head's refined position).  Each GT node is matched to the detections of its frame with the metric's matching
(max-weight 1/(1+d) within 7 um, on the unrefined grid positions); the target is GT - detection in um.

Models (all predict a bounded 3D offset in um, like V1284):
  zero     no refinement
  v1284    the public head (anvithpothula/biohub-v1284-head-s075), as captured in the deployed predictor
  mlp      V1284 architecture (224 -> 32 -> SiLU -> 3, 2d/(1+|d|) bound), trained on the training folds
  mlp64    wider (224 -> 64 -> 3) variant
Folds: 5-fold video folds (embryo-stratified) and the two cross-embryo splits.
Reports mean matched 3D distance, change vs zero, share of nodes brought closer, mean dz; writes the out-of-fold
offsets of EVERY captured detection (matched or not) to LANE/head_oof/<stem>.npz for the metric replay, and a final
head trained on all videos in the V1284 file format.

usage: python head_study.py [--epochs 60]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
LANE = ROOT / "v1284_20260926"
SCALE = np.array([1.625, 0.40625, 0.40625])
DS = np.array([1, 4, 4])


def match(gt_um, det_um, max_d=7.0):
    tree = cKDTree(det_um)
    pairs = tree.query_ball_point(gt_um, max_d)
    cols = sorted({j for p in pairs for j in p})
    if not cols:
        return np.empty(0, int), np.empty(0, int)
    cidx = {j: k for k, j in enumerate(cols)}
    w = np.zeros((len(gt_um), len(cols)))
    for i, p in enumerate(pairs):
        for j in p:
            w[i, cidx[j]] = 1.0 / (1.0 + np.linalg.norm(gt_um[i] - det_um[j]))
    r, c = linear_sum_assignment(w, maximize=True)
    keep = w[r, c] > 0
    return r[keep], np.asarray(cols)[c[keep]]


def load(stems=None):
    """All captured rows (A*) plus the matched training pairs (index into A, target)."""
    files = sorted(f for f in (LANE / "features").glob("*.npz") if stems is None or f.stem in set(stems))
    stems = [f.stem for f in files]
    gt = pd.read_parquet(ROOT / "remote_outputs/gt_export/gt_nodes.parquet", filters=[("dataset", "in", stems)])
    AX, AV, AS, AT, AG = [], [], [], [], []
    PI, PY = [], []
    base = 0
    for f in files:
        d = np.load(f)
        n = len(d["t"])
        if not n:
            continue
        det_um = d["grid"].astype(np.float64) * DS * SCALE
        ref_um = d["v1284_refined"][:, 1:].astype(np.float64) * DS * SCALE   # refine() returns stride-4 grid units
        AX.append(d["features"])
        AV.append((ref_um - det_um).astype(np.float32))
        AS += [f.stem] * n
        AT.append(d["t"].astype(np.int64))
        AG.append(d["grid"].astype(np.int64))
        g = gt[gt.dataset == f.stem]
        for t, gf in g.groupby("t"):
            idx = np.flatnonzero(d["t"] == t)
            if not len(idx):
                continue
            gt_um = gf[["z", "y", "x"]].to_numpy(np.float64) * SCALE
            r, c = match(gt_um, det_um[idx])
            PI.append(base + idx[c])
            PY.append(gt_um[r] - det_um[idx[c]])
        base += n
    return dict(X=np.concatenate(AX).astype(np.float32), V=np.concatenate(AV), S=np.asarray(AS),
                T=np.concatenate(AT), G=np.concatenate(AG),
                pi=np.concatenate(PI).astype(np.int64), py=np.concatenate(PY).astype(np.float32))


def bounded(delta):
    return 2.0 * delta / (1.0 + torch.linalg.vector_norm(delta, dim=-1, keepdim=True))


def fit_mlp(x, y, hidden=32, epochs=60, seed=0, wd=1e-4, lr=3e-3):
    torch.manual_seed(seed)
    mean = x.mean(0)
    scale = x.std(0) + 1e-6
    xt = torch.from_numpy((x - mean) / scale)
    yt = torch.from_numpy(y)
    head = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 3))
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = len(xt)
    bs = 1024
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((n + bs - 1) // bs))
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            pred = bounded(head(xt[b]))
            loss = torch.nn.functional.smooth_l1_loss(pred, yt[b], beta=0.5)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    head.eval()
    return head, mean, scale


def predict(model, x):
    head, mean, scale = model
    with torch.no_grad():
        return bounded(head(torch.from_numpy((x - mean) / scale))).numpy()


def summary(y, off):
    d0 = np.linalg.norm(y, axis=1)
    d1 = np.linalg.norm(y - off, axis=1)
    return dict(n=int(len(y)), d0=float(d0.mean()), d1=float(d1.mean()), change=float(d1.mean() / d0.mean() - 1),
                closer=float((d1 < d0).mean()), dz_before=float(-y[:, 0].mean()), dz_after=float((off - y)[:, 0].mean()))


def fmt(row):
    return " | ".join(f"{m}: d={r['d1']:.3f} ({r['change']:+.1%}) closer={r['closer']:.0%} dz={r['dz_after']:+.2f}"
                      for m, r in row.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--models", default="mlp,mlp64")
    a = ap.parse_args()
    D = load()
    X, V, S, pi, py = D["X"], D["V"], D["S"], D["pi"], D["py"]
    pS = S[pi]
    pemb = np.array([s[:4] for s in pS])
    aemb = np.array([s[:4] for s in S])
    vids = np.unique(S)
    print(f"captured rows {len(X)}, matched pairs {len(pi)} from {len(vids)} videos "
          f"({(pemb == '44b6').sum()} 44b6 / {(pemb == '6bba').sum()} 6bba pairs)", flush=True)
    rng = np.random.default_rng(0)
    fold_of = {}
    for e in ("44b6", "6bba"):
        vs = sorted(v for v in vids if v.startswith(e))
        rng.shuffle(vs)
        for k, v in enumerate(vs):
            fold_of[v] = k % 5
    pfold = np.array([fold_of[s] for s in pS])
    afold = np.array([fold_of[s] for s in S])
    models = {"mlp": 32, "mlp64": 64}
    models = {m: models[m] for m in a.models.split(",")}
    oof_all = {m: np.zeros((len(X), 3), np.float32) for m in models}
    results = {}
    for k in range(5):
        tr, te = pfold != k, pfold == k
        row = dict(zero=summary(py[te], np.zeros_like(py[te])), v1284=summary(py[te], V[pi[te]]))
        for m, hidden in models.items():
            model = fit_mlp(X[pi[tr]], py[tr], hidden=hidden, epochs=a.epochs)
            row[m] = summary(py[te], predict(model, X[pi[te]]))
            oof_all[m][afold == k] = predict(model, X[afold == k])
        results[f"video_fold{k}"] = row
        print(f"video_fold{k}", fmt(row), flush=True)
    for name, trm, tem in (("train6bba_test44b6", pemb == "6bba", pemb == "44b6"),
                           ("train44b6_test6bba", pemb == "44b6", pemb == "6bba")):
        row = dict(zero=summary(py[tem], np.zeros_like(py[tem])), v1284=summary(py[tem], V[pi[tem]]))
        for m, hidden in models.items():
            model = fit_mlp(X[pi[trm]], py[trm], hidden=hidden, epochs=a.epochs)
            row[m] = summary(py[tem], predict(model, X[pi[tem]]))
        results[name] = row
        print(name, fmt(row), flush=True)
    for e in ("44b6", "6bba", None):
        sel = np.ones(len(py), bool) if e is None else pemb == e
        row = dict(zero=summary(py[sel], np.zeros_like(py[sel])), v1284=summary(py[sel], V[pi[sel]]))
        for m in models:
            row[m] = summary(py[sel], oof_all[m][pi[sel]])
        results[f"oof_{e or 'all'}"] = row
        print(f"OOF {e or 'all'}", fmt(row), flush=True)
    first = next(iter(models))
    per_video = []
    for v in vids:
        sel = pS == v
        per_video.append(dict(video=v, n=int(sel.sum()), zero=float(np.linalg.norm(py[sel], axis=1).mean()),
                              v1284=float(np.linalg.norm(py[sel] - V[pi[sel]], axis=1).mean()),
                              own=float(np.linalg.norm(py[sel] - oof_all[first][pi[sel]], axis=1).mean())))
    pv = pd.DataFrame(per_video)
    print(f"videos where OOF {first} beats v1284: {int((pv.own < pv.v1284).sum())} of {len(pv)} | "
          f"v1284 beats zero: {int((pv.v1284 < pv.zero).sum())} | {first} beats zero: {int((pv.own < pv.zero).sum())}")
    pv.to_csv(LANE / "head_study_per_video.csv", index=False)
    (LANE / "head_study.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    out = LANE / "head_oof"
    out.mkdir(exist_ok=True)
    for v in vids:
        sel = S == v
        np.savez_compressed(out / f"{v}.npz", t=D["T"][sel], grid=D["G"][sel],
                            **{f"oof_{m}": oof_all[m][sel] for m in models}, v1284=V[sel])
    model = fit_mlp(X[pi], py, hidden=32, epochs=a.epochs)
    head, mean, scale = model
    torch.save(dict(state_dict=head.state_dict(), mean=torch.from_numpy(mean.astype(np.float32)),
                    scale=torch.from_numpy(scale.astype(np.float32))), LANE / "own_head_all.pt")
    print("saved OOF offsets for", len(vids), "videos and", LANE / "own_head_all.pt")


if __name__ == "__main__":
    main()
