"""Shared helpers for the 2026-09-26 V1284 port: rebuild the flow-t predictor text exactly as the notebook does."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE_NB = ROOT / "kaggle_kernels/biohub-flow-t-20260922/biohub-flow-t-20260922.ipynb"
PRISTINE_REPO = ROOT / "public_pack/pack/repo"
SNAPSHOT_REPO = ROOT / "day21_nodesel/snapshot/tracking_repo"
EXTERNAL_RUNTIME = ROOT / "remote_outputs/multidir_20260922/platform/biohub-flow-t-20260922/external_runtime"
LANE = ROOT / "v1284_20260926"
HEAD = ROOT / "models/v1284_head/v1284_head.pt"
HEAD_SHA256 = "625a0d9340f48193f2ec294fc2d81c5bb3c03087eab78ef0ae998a9c4c7da00c"
MODULE_FILE = HERE / "x138_module_verbatim.py"
MODULE_SHA256 = "34a2b4653f35ae73b5a7bdb46a095c812da5533c875cd1cdc4db83e4c3fd303c"
EXTERNAL_MARKER = "print('EXTERNAL_PREDICTOR_PATCHED', EX_RECIPE, flush=True)\n"


def notebook_cells(path=BASE_NB):
    return ["".join(c["source"]) for c in json.loads(Path(path).read_text(encoding="utf-8"))["cells"]]


def reconstruct_predictor(out_repo: Path, with_external: bool) -> str:
    """Materialise the pristine support-pack repo into out_repo and run the base notebook's own patch chain on it
    (cell 1 environment + cell 6 up to and including the external predictor patch).  Returns the patched text."""
    cells = notebook_cells()
    if out_repo.exists():
        shutil.rmtree(out_repo)
    for sub in ("scripts", "src"):
        shutil.copytree(PRISTINE_REPO / sub, out_repo / sub, ignore=shutil.ignore_patterns("__pycache__"))
    c6 = cells[6]
    if c6.count(EXTERNAL_MARKER) != 1:
        raise RuntimeError("external marker not unique in cell 6")
    head = c6[: c6.index(EXTERNAL_MARKER) + len(EXTERNAL_MARKER)]
    if not with_external:
        ext = "from patch_predictor import patch_source as _ex_patch\n"
        if head.count(ext) != 1:
            raise RuntimeError("external import not unique")
        head = head[: head.index(ext)]
    saved_env = dict(os.environ)
    g = {"__name__": "__v1284_reconstruct__"}
    work = out_repo.parent / (out_repo.name + "_kw")
    work.mkdir(parents=True, exist_ok=True)
    try:
        exec(compile(cells[1], "base::cell1", "exec"), g)
        ex_line = [ln for ln in cells[5].splitlines() if ln.startswith("EX_RECIPE = ")]
        exec(compile(ex_line[0], "base::cell5", "exec"), g)
        g["REPO_DIR"] = out_repo
        g["WORKING_DIR"] = work
        sys.path.insert(0, str(EXTERNAL_RUNTIME))
        try:
            exec(compile(head, "base::cell6", "exec"), g)
        finally:
            sys.path.remove(str(EXTERNAL_RUNTIME))
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
    return (out_repo / "scripts" / "predict_unet_transformer.py").read_text(encoding="utf-8")


def module_source() -> str:
    import hashlib
    text = MODULE_FILE.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != MODULE_SHA256:
        raise RuntimeError("V1284 module text changed")
    return text
