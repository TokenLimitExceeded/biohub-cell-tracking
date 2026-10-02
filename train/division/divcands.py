"""Division candidates with IMAGE evidence (pool4 cache) + geometry + public edge probabilities, labelled by GT.

Per video: public dets (npz) -> [optional tracksdata ILP node selection, as in tools/cv_pub.py --ilp-select 1] ->
pair probs (geo + tgt) -> our tracker with divisions OFF -> candidate triples (cellmot.divclf.build_candidates, vectorised)
-> GT labels (1 = GT division with these two daughters, 0 = P matched to an annotated GT node that continues but not this
division, -1 = unknown). Also saves the division-free graph per video so a plug-in evaluation can add divisions without
re-tracking.

usage: python tools/explore_division/divcands.py --pub-dirs I:/kaggle/biohub/dets/pubC_s0b_val --names "$(cat configs/foldC_val.txt)" \
         --ilp-select 1 --ilp-cand 0.35 --out tools/explore_division/cands_val_ilp.parquet --graph-dir tools/explore_division/graphs_val_ilp
"""
import argparse, os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from cellmot.track import track, to_graph_frames
from cellmot.pipeline import make_track_config
from cellmot.pubmodel import pair_probs_from_logits
from cellmot.divclf import build_candidates, label_candidates

ap = argparse.ArgumentParser()
ap.add_argument("--pub-dirs", required=True, help="comma-separated detection dirs; the first one holding <name>.npz is used"); ap.add_argument("--names", required=True)
ap.add_argument("--thr", type=float, default=0.965); ap.add_argument("--thr-hi", type=float, default=0.965)
ap.add_argument("--track-json", default='{"learned_gate_um": 12.0, "learned_gate_p": 0.3, "min_track_len": 3, "div_diverge_um": 2.25, "death_cost": 2.5, "gap_um_per_step": 3.5, "div_min_p": 0.4}')
ap.add_argument("--lw", type=float, default=10.0); ap.add_argument("--minp", type=float, default=0.05); ap.add_argument("--edge-col", default="blend")
ap.add_argument("--ilp-select", type=int, default=0); ap.add_argument("--ilp-cand", type=float, default=0.35); ap.add_argument("--ilp-disap", type=float, default=2.0)
ap.add_argument("--cand-parent-um", type=float, default=11.0); ap.add_argument("--cand-sister-um", type=float, default=16.5)
ap.add_argument("--sym-max", type=float, default=None); ap.add_argument("--pd2-max", type=float, default=None)
ap.add_argument("--cache-dir", default="I:/kaggle/biohub/cache/pool4"); ap.add_argument("--gt-dir", default="remote_outputs/gt_export")
ap.add_argument("--out", required=True); ap.add_argument("--graph-dir", required=True); ap.add_argument("--only-div-videos", type=int, default=0, help=">0: keep videos with GT divisions plus this many others")
ap.add_argument("--skip-existing", type=int, default=0, help="1 = reuse <graph-dir>/<name>.cands.parquet when present (resume)")
a = ap.parse_args()
os.makedirs(a.graph_dir, exist_ok=True)
gn = pd.read_parquet(f"{a.gt_dir}/gt_nodes.parquet"); ge = pd.read_parquet(f"{a.gt_dir}/gt_edges.parquet")
pub_dirs = [d for d in a.pub_dirs.split(",") if d]
names = [n.strip() for n in a.names.replace("\n", ",").split(",") if n.strip()]


def pub_path(name):
    for d in pub_dirs:
        p = os.path.join(d, name + ".npz")
        if os.path.exists(p): return p
    return None


names = [n for n in names if pub_path(n) and os.path.exists(os.path.join(a.cache_dir, n + ".npy"))]
if a.only_div_videos:
    oc = ge.groupby(["dataset", "source_id"]).size(); dv = set(oc[oc >= 2].reset_index().dataset)
    names = [n for n in names if n in dv] + [n for n in names if n not in dv][: a.only_div_videos]
cfg = make_track_config(a.track_json); cfg.learned_w = a.lw; cfg.learned_min_p = a.minp; cfg.divisions = False
print(len(names), "videos; track", a.track_json, "ilp", a.ilp_select, a.ilp_cand, flush=True)


def load_pub(name):
    z = np.load(pub_path(name)); f = float(z["factor"]) if "factor" in z.files else 4.0
    t, grid, prob, local = z["t"].astype(np.int64), z["grid"].astype(np.float32), z["prob"].astype(np.float32), z["local"].astype(np.int64)
    idx = np.where(prob >= a.thr)[0]
    gmap = {(int(t[i]), int(local[i])): k for k, i in enumerate(idx)}
    pos = grid[idx] * np.array([1.625, 0.40625 * f, 0.40625 * f], np.float32)
    E = z["edges_hi"] if a.thr >= a.thr_hi - 1e-9 else z["edges_all"]
    cols = list(z["edge_cols"]) if "edge_cols" in z.files else ["t", "i", "j", "primary", "blend"]
    return t[idx], pos, prob[idx], gmap, E, cols


all_parts = []; t0 = time.time()
for name in names:
    cp = os.path.join(a.graph_dir, name + ".cands.parquet")
    if a.skip_existing and os.path.exists(cp):
        all_parts.append(pd.read_parquet(cp)); print(f"  {name}: reused ({len(all_parts[-1])} cands)", flush=True); continue
    t_start = time.time(); t, pos, sc, gmap, E, ecols = load_pub(name)
    col = ecols.index(a.edge_col) if a.edge_col in ecols else ecols.index("blend")
    n_pub = len(t); n_drop = 0
    if a.ilp_select:
        from cellmot.ilp import solve_ilp
        pt_ = pair_probs_from_logits(E, gmap, col, len(t), mode="tgt", min_p=a.ilp_cand)
        sol_ = solve_ilp(t, pos, sc, pt_, edge_w=-1.0, appearance=0.0, disappearance=a.ilp_disap, division=1.2, min_p=0.0, max_um=1e9) if pt_ else []
        keep_ = np.zeros(len(t), bool)
        for i_, j_, c_ in sol_: keep_[i_] = True; keep_[j_] = True
        remap_ = -np.ones(len(t), np.int64); remap_[keep_] = np.arange(int(keep_.sum())); n_drop = int((~keep_).sum())
        gmap = {k_: int(remap_[v_]) for k_, v_ in gmap.items() if keep_[v_]}
        t, pos, sc = t[keep_], pos[keep_], sc[keep_]
    t_ilp = time.time()
    pp = pair_probs_from_logits(E, gmap, col, len(t), mode="geo")
    pt = pair_probs_from_logits(E, gmap, col, len(t), mode="tgt")
    tt, p2, s2, edges, keep, st = track(t, pos, sc, cfg, pair_prob=pp)
    nodes, edf = to_graph_frames(tt, p2, s2, edges, keep)
    nodes.to_parquet(os.path.join(a.graph_dir, name + ".nodes.parquet")); edf.to_parquet(os.path.join(a.graph_dir, name + ".edges.parquet"))
    t_trk = time.time()
    V = np.load(os.path.join(a.cache_dir, name + ".npy"), mmap_mode="r")
    df = build_candidates(tt, p2, s2, edges, keep, pp, pt, V, parent_um=a.cand_parent_um, sister_um=a.cand_sister_um, sym_max=a.sym_max, pd2_max=a.pd2_max)
    t_cand = time.time()
    df = label_candidates(df, nodes, edf, gn[gn.dataset == name], ge[ge.dataset == name])
    df.insert(0, "dataset", name)
    df.to_parquet(cp); all_parts.append(df)
    print(f"  {name}: [{os.path.basename(os.path.dirname(pub_path(name)))}] pub {n_pub} ilp-dropped {n_drop} nodes {len(nodes)} edges {len(edf)} cands {len(df)} pos {int((df.y == 1).sum())} neg {int((df.y == 0).sum())} "
          f"[ilp {t_ilp - t_start:.0f}s | track {t_trk - t_ilp:.1f}s cands {t_cand - t_trk:.1f}s label {time.time() - t_cand:.1f}s] ({time.time() - t0:.0f}s)", flush=True)
df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
df.to_parquet(a.out)
print("total", len(df), "pos", int((df.y == 1).sum()), "neg", int((df.y == 0).sum()), "unknown", int((df.y == -1).sum()))
