"""Convert div_lgb_noprob.txt into a compact numpy-evaluable JSON, and prove the numpy
evaluator reproduces lightgbm.Booster.predict bit-for-bit on the val candidates.

Why: the kernel would die ~10 minutes into a 90-minute run if `import lightgbm` failed in the
pinned image.  With this, the graft cell can fall back to numpy and the arm survives.
"""
import os, sys, json
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
# Day-10: SRC/OUT/VAL are overridable so the same dumper serves the relabelled and lag models.
# With no environment set it reproduces the day-7 dump exactly.
SRC = os.environ.get("BIOHUB_NUMPY_SRC", os.path.join(ROOT, "tools/explore_division/div_lgb_noprob.txt"))
OUT = os.environ.get("BIOHUB_NUMPY_OUT", os.path.join(ROOT, "kaggle_datasets/biohub-divclf-v1/div_lgb_noprob_numpy.json"))
VAL = os.environ.get("BIOHUB_NUMPY_VAL", os.path.join(ROOT, "tools/explore_division/cands_val_ilp.parquet"))

b = lgb.Booster(model_file=SRC)
dump = b.dump_model()
F = list(b.feature_name())
assert dump["objective"].startswith("binary")

trees = []
for t in dump["tree_info"]:
    nodes, leaves = [], []
    def walk(n):
        if "leaf_value" in n:
            leaves.append(float(n["leaf_value"]))
            return -len(leaves)          # negative encodes a leaf index (-1 -> leaves[0])
        i = len(nodes)
        nodes.append(None)
        assert n["decision_type"] == "<=", n["decision_type"]
        left = walk(n["left_child"]); right = walk(n["right_child"])
        nodes[i] = [int(n["split_feature"]), float(n["threshold"]), left, right,
                    bool(n["default_left"]), str(n.get("missing_type", "None"))]
        return i
    root = walk(t["tree_structure"])
    trees.append({"root": root, "nodes": nodes, "leaves": leaves})

model = {"features": F, "trees": trees, "objective": "binary_sigmoid"}
json.dump(model, open(OUT, "w"), separators=(",", ":"))
print("wrote", OUT, os.path.getsize(OUT), "bytes |", len(trees), "trees")

MISS_ZERO = 1e-35


def predict_numpy(model, X):
    n = X.shape[0]
    raw = np.zeros(n, np.float64)
    for tr in model["trees"]:
        nodes = tr["nodes"]; leaves = np.asarray(tr["leaves"], np.float64)
        out = np.empty(n, np.float64)
        stack = [(tr["root"], np.ones(n, bool))]
        while stack:
            idx, mask = stack.pop()
            if not mask.any():
                continue
            if idx < 0:
                out[mask] = leaves[-idx - 1]
                continue
            f, thr, left, right, dleft, mtype = nodes[idx]
            v = X[:, f]
            if mtype == "NaN":
                miss = np.isnan(v)
            elif mtype == "Zero":
                miss = np.isnan(v) | (np.abs(v) <= MISS_ZERO)
            else:
                miss = np.zeros(n, bool)
            vv = np.where(np.isnan(v), 0.0, v)
            go_left = np.where(miss, dleft, vv <= thr)
            stack.append((left, mask & go_left))
            stack.append((right, mask & ~go_left))
        raw += out
    return 1.0 / (1.0 + np.exp(-raw))


va = pd.read_parquet(VAL)
sub = va.sample(n=min(200000, len(va)), random_state=0)
X = sub[F].values.astype(np.float32)
p1 = b.predict(X)
p2 = predict_numpy(model, X.astype(np.float64))
d = np.abs(p1 - p2)
print(f"rows {len(X):,}  max|diff| {d.max():.3e}  mean {d.mean():.3e}  nan cols in sample: "
      f"{int(np.isnan(X).any(0).sum())}")
assert d.max() < 1e-9, d.max()
print("NUMPY EVALUATOR MATCHES LIGHTGBM")
