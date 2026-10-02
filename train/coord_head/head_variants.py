"""Deployable variants of our coordinate head, evaluated out-of-fold on the same video folds as head_study.py.

  halfz    trained on targets whose z component is halved (hedge against the embryo-dependent z bias: 6bba videos sit
           ~0.9 um low, 44b6 ~0.2 um); same architecture and file format as V1284, so it drops into the same module
  balanced full targets, 44b6 pairs up-weighted to the same total weight as 6bba

Writes oof_<variant> into LANE/head_oof/<stem>.npz (keeps the existing keys) and LANE/own_head_<variant>.pt trained on
all videos that head_oof covers.

usage: python head_variants.py [--variants halfz,balanced] [--epochs 60]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_study as hs  # noqa: E402


def fit_weighted(x, y, w, epochs, seed=0, lr=3e-3, wd=1e-4):
    if w is None:
        return hs.fit_mlp(x, y, epochs=epochs, seed=seed)
    torch.manual_seed(seed)
    mean, scale = x.mean(0), x.std(0) + 1e-6
    xt, yt, wt = torch.from_numpy((x - mean) / scale), torch.from_numpy(y), torch.from_numpy(w.astype(np.float32))
    head = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n, bs = len(xt), 1024
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((n + bs - 1) // bs))
    for _ in range(epochs):
        perm = torch.multinomial(wt / wt.sum(), n, replacement=True)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = torch.nn.functional.smooth_l1_loss(hs.bounded(head(xt[b])), yt[b], beta=0.5)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    head.eval()
    return head, mean, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="halfz,balanced")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    stems = sorted(p.stem for p in (hs.LANE / "head_oof").glob("*.npz"))
    D = hs.load(stems)
    X, S, pi, py, V = D["X"], D["S"], D["pi"], D["py"], D["V"]
    pS = S[pi]
    vids = np.unique(S)
    rng = np.random.default_rng(0)
    fold_of = {}
    for e in ("44b6", "6bba"):
        vs = sorted(v for v in vids if v.startswith(e))
        rng.shuffle(vs)
        for k, v in enumerate(vs):
            fold_of[v] = k % 5
    pfold = np.array([fold_of[s] for s in pS])
    afold = np.array([fold_of[s] for s in S])
    pemb = np.array([s[:4] for s in pS])
    for variant in a.variants.split(","):
        target = py.copy()
        weight = None
        if variant == "halfz":
            target[:, 0] *= 0.5
        elif variant == "balanced":
            weight = np.where(pemb == "44b6", (pemb == "6bba").sum() / max((pemb == "44b6").sum(), 1), 1.0)
        else:
            raise ValueError(variant)
        oof = np.zeros((len(X), 3), np.float32)
        for k in range(5):
            tr = pfold != k
            model = fit_weighted(X[pi[tr]], target[tr], None if weight is None else weight[tr], a.epochs)
            oof[afold == k] = hs.predict(model, X[afold == k])
        off = oof[pi]
        for e in ("all", "44b6", "6bba"):
            m = np.ones(len(py), bool) if e == "all" else pemb == e
            r = hs.summary(py[m], off[m])
            print(f"{variant:9s} {e:4s} d={r['d1']:.3f} ({r['change']:+.1%}) closer={r['closer']:.0%} dz={r['dz_after']:+.2f}",
                  flush=True)
        for v in vids:
            f = hs.LANE / "head_oof" / f"{v}.npz"
            old = dict(np.load(f))
            old[f"oof_{variant}"] = oof[S == v]
            np.savez_compressed(f, **old)
        head, mean, scale = fit_weighted(X[pi], target, weight, a.epochs)
        torch.save(dict(state_dict=head.state_dict(), mean=torch.from_numpy(mean.astype(np.float32)),
                        scale=torch.from_numpy(scale.astype(np.float32))), hs.LANE / f"own_head_{variant}.pt")
        print("saved", hs.LANE / f"own_head_{variant}.pt", flush=True)


if __name__ == "__main__":
    main()
