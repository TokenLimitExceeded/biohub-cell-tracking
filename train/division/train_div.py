"""Train a LightGBM division classifier on candidate triples (divcands.py output), report AP on val, plug into the
val graphs as the division stage and score with the official-port metric (baseline = anchor safe-div, same run).
Reports per-embryo (44b6 / 6bba) splits and the number of videos that improve over the safe-div baseline.

usage: python tools/explore_division/train_div.py --train tools/explore_division/cands_train_ilp.parquet \
         --val tools/explore_division/cands_val_ilp.parquet --graph-dir tools/explore_division/graphs_val_ilp --tag ilp_all
gate-2 (cross-embryo): --train-embryo 6bba --val-embryo 44b6 [--val-extra tools/explore_division/cands_train_ilp.parquet:tools/explore_division/graphs_train_ilp]
"""
import argparse, os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from cellmot.metric import Graph, evaluate, per_sample_metrics, summarise
from cellmot.submission import graph_rows
from cellmot.divclf import add_divisions, ID_COLS
from sklearn.metrics import average_precision_score, roc_auc_score
import lightgbm as lgb

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True, help="comma-separated candidate parquets"); ap.add_argument("--val", required=True); ap.add_argument("--graph-dir", required=True)
ap.add_argument("--val-extra", default=None, help="extra test videos 'parquet:graph_dir' (e.g. leaked-det 44b6 train videos for the cross-embryo check)")
ap.add_argument("--train-embryo", default="all", choices=["all", "6bba", "44b6"]); ap.add_argument("--val-embryo", default="all", choices=["all", "6bba", "44b6"])
ap.add_argument("--gt-dir", default="remote_outputs/gt_export"); ap.add_argument("--unknown-w", type=float, default=0.0, help="weight of unknown-label candidates used as negatives (0 = known labels only)")
ap.add_argument("--feats", default="all", choices=["all", "geo", "img", "geo_prob", "noprob"]); ap.add_argument("--thrs", default="0.3,0.4,0.5,0.6,0.7")
ap.add_argument("--relink-thrs", default="", help="comma-separated relink thresholds to try in addition to 'off'"); ap.add_argument("--gates", default="none,1.5:10", help="inference gates 'sym_max:pd2_max' (none = off), comma-separated")
ap.add_argument("--drop-feats", default="", help="comma-separated feature-name PREFIXES to drop from F (day-10: trim the lag block)")
ap.add_argument("--train-gate", default=None, help="'sym_max:pd2_max' applied to the training candidates")
ap.add_argument("--smooth", type=float, default=0.8); ap.add_argument("--round", type=int, default=1, help="1 = round coordinates as in an int submission (0 = float coordinates)")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-seeds", type=int, default=1, help="average the probabilities of this many seeds (seed, seed+1, ...)")
ap.add_argument("--rounds", type=int, default=400); ap.add_argument("--leaves", type=int, default=7); ap.add_argument("--lr", type=float, default=0.03)
ap.add_argument("--save-model", default=None); ap.add_argument("--tag", default="divclf"); ap.add_argument("--out-dir", default="tools/explore_division")
a = ap.parse_args()
tr = pd.concat([pd.read_parquet(p) for p in a.train.split(",")], ignore_index=True); va = pd.read_parquet(a.val)
graph_dirs = {n: a.graph_dir for n in va.dataset.unique()}
if a.val_extra:
    pq, gd = a.val_extra.split(":"); ex = pd.read_parquet(pq)
    if a.val_embryo != "all": ex = ex[ex.dataset.str.startswith(a.val_embryo)]
    ex = ex[~ex.dataset.isin(set(va.dataset))]
    graph_dirs.update({n: gd for n in ex.dataset.unique()}); va = pd.concat([va, ex], ignore_index=True)
    print(f"val-extra: {ex.dataset.nunique()} videos, {int((ex.y == 1).sum())} positives ({pq})")
if a.val_embryo != "all": va = va[va.dataset.str.startswith(a.val_embryo)].reset_index(drop=True)
if a.train_embryo != "all": tr = tr[tr.dataset.str.startswith(a.train_embryo)].reset_index(drop=True)
tr = tr[~tr.dataset.isin(set(va.dataset))].reset_index(drop=True)  # never train on a test video
ID = set(ID_COLS)
img_keys = tuple(k for k in tr.columns if k.split("_")[0] in ("P0", "Pm", "Pp", "C1", "Q", "M1", "M0", "X0") or k.startswith("r_"))
prob_keys = ("p_Pc1", "p_Pq", "p_Xq", "pt_Pq", "pt_Xq", "pt_best_q", "sP", "sc1", "sq")
if a.feats == "all": F = [c for c in tr.columns if c not in ID]
elif a.feats == "img": F = list(img_keys) + ["pd1", "pd2", "sister", "sym", "q_orphan"]
elif a.feats == "geo": F = [c for c in tr.columns if c not in ID and c not in img_keys and c not in prob_keys]
elif a.feats == "noprob": F = [c for c in tr.columns if c not in ID and c not in prob_keys]
else: F = [c for c in tr.columns if c not in ID and c not in img_keys]


def gate_mask(d, g):
    if g in (None, "", "none"): return np.ones(len(d), bool)
    s, p = (float(x) for x in g.split(":")); return (d.sym.values <= s) & (d.pd2.values <= p)


if a.drop_feats:
    _pre = tuple(x for x in a.drop_feats.split(",") if x)
    F = [c for c in F if not c.startswith(_pre)]
if a.train_gate: tr = tr[gate_mask(tr, a.train_gate)].reset_index(drop=True)
emb = lambda d: d.dataset.str[:4]
print(f"train cands {len(tr)} pos {(tr.y == 1).sum()} neg {(tr.y == 0).sum()} unknown {(tr.y == -1).sum()} videos {tr.dataset.nunique()} (embryo {a.train_embryo}: pos by embryo {tr[tr.y == 1].pipe(emb).value_counts().to_dict()}) "
      f"| val cands {len(va)} pos {(va.y == 1).sum()} neg {(va.y == 0).sum()} videos {va.dataset.nunique()} (pos by embryo {va[va.y == 1].pipe(emb).value_counts().to_dict()}) | feats {len(F)} ({a.feats})")
print("val positives with q already linked (stolen):", int(((va.y == 1) & (va.X >= 0)).sum()), "orphan:", int(((va.y == 1) & (va.X < 0)).sum()),
      "| train positives stolen/orphan:", int(((tr.y == 1) & (tr.X >= 0)).sum()), int(((tr.y == 1) & (tr.X < 0)).sum()))

m = tr.y >= 0
if a.unknown_w > 0:
    Xtr = tr[F].values.astype(np.float32); ytr = (tr.y == 1).astype(int).values; w = np.where(tr.y >= 0, 1.0, a.unknown_w)
else:
    Xtr = tr.loc[m, F].values.astype(np.float32); ytr = (tr.y[m] == 1).astype(int).values; w = np.ones(len(ytr))
pos_w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1)) ** 0.5
Xva = va[F].values.astype(np.float32); pv = np.zeros(len(va)); boosters = []
for s_ in range(a.n_seeds):
    params = dict(objective="binary", learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=15, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=5.0, scale_pos_weight=pos_w, verbose=-1, seed=a.seed + s_, num_threads=2)
    bst = lgb.train(params, lgb.Dataset(Xtr, ytr, weight=w, feature_name=F), num_boost_round=a.rounds); boosters.append(bst)
    pv += bst.predict(Xva) / a.n_seeds
if a.save_model:
    boosters[0].save_model(a.save_model)
    json.dump({"features": F, "thr": 0.5, "n_features": len(F), "train_pos": int(ytr.sum()), "train_neg": int((ytr == 0).sum()), "train_videos": int(tr.dataset.nunique()), "seed": a.seed,
               "params": {k: v for k, v in params.items() if k != "seed"}, "rounds": a.rounds, "candidates": {"parent_um": 11.0, "sister_um": 16.5}, "tag": a.tag},
              open(os.path.splitext(a.save_model)[0] + ".json", "w"), indent=1)
    print("saved", a.save_model)
va["p"] = pv
mv = va.y >= 0
yv = (va.y[mv] == 1).astype(int).values
if yv.sum() and (yv == 0).sum():
    print(f"VAL known-labelled: AP={average_precision_score(yv, va.p[mv]):.3f} AUC={roc_auc_score(yv, va.p[mv]):.3f} (n={mv.sum()}, pos={yv.sum()})")
    for e_ in ("6bba", "44b6"):
        me = mv & va.dataset.str.startswith(e_); ye = (va.y[me] == 1).astype(int).values
        if ye.sum() and (ye == 0).sum(): print(f"   {e_}: AP={average_precision_score(ye, va.p[me]):.3f} AUC={roc_auc_score(ye, va.p[me]):.3f} (n={me.sum()}, pos={ye.sum()})")
    for ref, sgn in [("pt_Pq", 1), ("p_Pq", 1), ("pd2", -1), ("sym", -1)]:
        print(f"   reference scorer {ref}: AP={average_precision_score(yv, sgn * va.loc[mv, ref]):.3f}")
imp = sorted(zip(boosters[0].feature_importance("gain"), F), reverse=True)[:15]
print("top features:", ", ".join(f"{n}:{g:.0f}" for g, n in imp))

# ---------------- plug-in evaluation on the val graphs
from cellmot.anchor_glue import apply_safe_divisions, _stages
from cellmot.smooth import linefit_smooth
_stages().TEST_DIR = __import__("pathlib").Path("I:/kaggle/biohub/data/train")
gn = pd.read_parquet(f"{a.gt_dir}/gt_nodes.parquet"); ge = pd.read_parquet(f"{a.gt_dir}/gt_edges.parquet"); gm = pd.read_csv(f"{a.gt_dir}/gt_meta.csv"); est = dict(zip(gm.dataset, gm.estimated_number_of_nodes))
names = sorted(va.dataset.unique())
gts = {n: Graph.from_frames(gn[gn.dataset == n], ge[ge.dataset == n]) for n in names}
graphs = {n: (pd.read_parquet(os.path.join(graph_dirs[n], n + ".nodes.parquet")), pd.read_parquet(os.path.join(graph_dirs[n], n + ".edges.parquet"))) for n in names}
cands_by = {n: va[va.dataset == n] for n in names}


def score_graph(name, nodes, edges):
    nodes2, _ = linefit_smooth(nodes, edges, weight=a.smooth, window=2) if a.smooth > 0 else (nodes, None)
    if a.round:
        rows = graph_rows(name, nodes2, edges)  # rounded coordinates, as in an int submission
        nn = rows[rows.row_type == "node"][["node_id", "t", "z", "y", "x"]]; ee = rows[rows.row_type == "edge"][["source_id", "target_id"]]
    else:
        nn = nodes2[["node_id", "t", "z", "y", "x"]]; ee = edges[["source_id", "target_id"]]
    er = evaluate(Graph.from_frames(nn, ee), gts[name]); r = per_sample_metrics(er, float(est[name])); r["dataset"] = name; r["score"] = summarise([r])["score"]; return r


KEYS = ("division_tp", "division_fp", "division_fn", "division_jaccard", "adj_edge_jaccard", "edge_jaccard", "score")


def report(lab, rows, base_rows=None, extra=None):
    s = summarise(rows); line = f"{lab:34s}: " + (f"added {extra[0]:4d} cut {extra[1]:3d} " if extra else "") + f"div {s['division_tp']}/{s['division_fp']}/{s['division_fn']} divJ={s['division_jaccard']:.4f} adj={s['adj_edge_jaccard']:.4f} J={s['edge_jaccard']:.4f} score={s['score']:.4f}"
    out = dict(variant=lab, **{k: s[k] for k in KEYS}, n_videos=len(rows))
    if extra: out["added"], out["cut"] = extra
    for e_ in ("6bba", "44b6"):
        sub = [r for r in rows if r["dataset"].startswith(e_)]
        if sub:
            se = summarise(sub); line += f" | {e_}({len(sub)}) divJ={se['division_jaccard']:.3f} score={se['score']:.4f}"
            out[f"score_{e_}"] = se["score"]; out[f"divJ_{e_}"] = se["division_jaccard"]; out[f"adj_{e_}"] = se["adj_edge_jaccard"]
    if base_rows is not None:
        sb = summarise(base_rows); d = s["score"] - sb["score"]
        bm = {r["dataset"]: r["score"] for r in base_rows}; dv = np.array([r["score"] - bm[r["dataset"]] for r in rows])
        line += f" | delta {d:+.4f} up/down {(dv > 1e-9).sum()}/{(dv < -1e-9).sum()}"
        out["delta"] = d; out["n_up"] = int((dv > 1e-9).sum()); out["n_down"] = int((dv < -1e-9).sum())
        for e_ in ("6bba", "44b6"):
            sub = [r for r in rows if r["dataset"].startswith(e_)]; subb = [r for r in base_rows if r["dataset"].startswith(e_)]
            if sub: out[f"delta_{e_}"] = summarise(sub)["score"] - summarise(subb)["score"]; line += f" [{e_} {out[f'delta_{e_}']:+.4f}]"
    print(line, flush=True)
    return out, s


t0 = time.time(); base_rows, none_rows = [], []
for n in names:
    nodes, edges = graphs[n]
    none_rows.append(score_graph(n, nodes, edges))
    n2, e2, _ = apply_safe_divisions(nodes, edges, dataset=n); base_rows.append(score_graph(n, n2, e2))
res = [report("no divisions", none_rows)[0], report("BASELINE anchor safediv", base_rows)[0]]
print(f"   ({time.time() - t0:.0f}s; round={a.round} smooth={a.smooth})")
per_video = {"safediv": {r["dataset"]: r["score"] for r in base_rows}}
for g in a.gates.split(","):
    for thr in [float(x) for x in a.thrs.split(",")]:
        for rl in ([float(x) for x in a.relink_thrs.split(",") if x] + [None]):
            if rl is not None and rl < thr: continue
            rows = []; nadd = ncut = 0
            for n in names:
                nodes, edges = graphs[n]; c = cands_by[n]; c = c[gate_mask(c, g)]
                e2, na, nc = add_divisions(edges, c[["P", "q", "X", "p"]], thr=thr, relink_thr=rl); nadd += na; ncut += nc
                rows.append(score_graph(n, nodes, e2))
            lab = f"clf thr={thr:.2f} gate={g} relink={'off' if rl is None else f'{rl:.2f}'}"
            o, _ = report(lab, rows, base_rows, (nadd, ncut)); res.append(o); per_video[lab] = {r["dataset"]: r["score"] for r in rows}
os.makedirs(a.out_dir, exist_ok=True)
pd.DataFrame(res).to_csv(os.path.join(a.out_dir, f"{a.tag}_plugin.csv"), index=False)
pd.DataFrame(per_video).to_csv(os.path.join(a.out_dir, f"{a.tag}_pervideo.csv"))
print("wrote", os.path.join(a.out_dir, f"{a.tag}_plugin.csv"), f"({time.time() - t0:.0f}s)")
