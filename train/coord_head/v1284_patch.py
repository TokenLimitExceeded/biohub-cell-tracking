"""V1284 learned coordinate refinement, ported onto the flow-t predictor (strict Trackastra + local-flow relink).

The head is anvithpothula/biohub-v1284-head-s075 (CC0): an MLP on the frozen primary UNet features at each
first-seen detection, displacement bounded to 2 um.  It is applied exactly where the public x138 notebook applies it
(right after a frame's detections are produced, before any edge is scored).  Two arms:

  full     x138 semantics: the refined float coordinates replace the detections everywhere downstream (edge features
           by trilinear lookup, candidate distances, ILP graph, post-process, output).  Trackastra's feature extractor
           only accepts stride-4 grid points, so it is handed the unrefined grid coordinates of the same nodes.
  sidecar  the predictor is unchanged; the refined coordinates are only written to a sidecar file, and the
           post-process swaps them in for detection nodes right before the output line-fit smoothing
           (see OUTPUT_HOOK_SOURCE in build_kernel.py).

Both arms write BIOHUB_V1284_REPORT_DIR/<stem>.npz with `refined` (float32 t,z,y,x, full resolution) and `grid`
(int16, the unrefined detections in the same order), and print one V1284_VIDEO line per video.
"""


def v1284_patch_predictor(source, arm, require_external=True):
    if arm not in ("full", "sidecar"):
        raise ValueError(arm)
    helper = '''import tracksdata as td
from v1284_coordinate_refinement import refine as _v1284_refine, index_features as _v1284_index


def _v1284_write(stem, refined, grid):
    import json as _v1284_json
    refined = np.asarray(refined, dtype=np.float32)
    grid = np.asarray(grid)
    if refined.shape != grid.shape or refined.ndim != 2 or refined.shape[1] != 4:
        raise RuntimeError(("V1284 sidecar shape mismatch", refined.shape, grid.shape))
    if not np.array_equal(refined[:, 0], grid[:, 0].astype(np.float32)):
        raise RuntimeError("V1284 sidecar frame mismatch")
    shift_um = (refined[:, 1:] - grid[:, 1:].astype(np.float32)) * np.array([1.625, 0.40625, 0.40625], np.float32)
    norm = np.linalg.norm(shift_um, axis=1) if len(shift_um) else np.zeros(0, np.float32)
    if len(norm) and (not np.isfinite(norm).all() or float(norm.max()) > 2.0001):
        raise RuntimeError("invalid V1284 displacement in sidecar")
    out = Path(os.environ["BIOHUB_V1284_REPORT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{stem}.npz", refined=refined, grid=grid.astype(np.int16))
    record = dict(dataset=stem, arm=os.environ.get("BIOHUB_V1284_ARM", ""), nodes=int(len(grid)),
                  mean_shift_um=float(norm.mean()) if len(norm) else 0.0,
                  median_shift_um=float(np.median(norm)) if len(norm) else 0.0,
                  mean_dz_um=float(shift_um[:, 0].mean()) if len(norm) else 0.0,
                  mean_dy_um=float(shift_um[:, 1].mean()) if len(norm) else 0.0,
                  mean_dx_um=float(shift_um[:, 2].mean()) if len(norm) else 0.0)
    print("V1284_VIDEO " + _v1284_json.dumps(record, sort_keys=True), flush=True)
'''
    det_anchor = '                coord_offset[t] = (global_node_count, global_node_count + len(arr))'
    grid_src = '''    _v1284_grid = np.concatenate(_v1284_grid_lists) if _v1284_grid_lists else np.empty((0, 4), dtype=np.int16)
    _v1284_grid = _v1284_grid.astype(np.float32)
    _v1284_grid[:, 1:] *= ds_arr
    _v1284_grid = _v1284_grid.astype(np.int16)
'''
    if arm == "full":
        det_new = ('                _v1284_grid_lists.append(np.asarray(arr).copy())\n'
                   '                arr = _v1284_refine(ds_path, t, arr, unet_out[:, f_idx])\n' + det_anchor)
        tail_new = ('    # V1284 (full): refined geometry is kept through association, the ILP graph and the output.\n'
                    + grid_src + '    _v1284_write(ds_path.stem, coords, _v1284_grid)\n')
    else:
        det_new = ('                _v1284_grid_lists.append(np.asarray(arr).copy())\n'
                   '                _v1284_ref_lists.append(np.asarray(\n'
                   '                    _v1284_refine(ds_path, t, arr, unet_out[:, f_idx]), dtype=np.float32))\n' + det_anchor)
        tail_new = ('    coords = coords.astype(np.int16)\n'
                    '    # V1284 (sidecar): the pipeline keeps the grid detections; refined coordinates go to the sidecar only.\n'
                    + grid_src +
                    '    if not np.array_equal(_v1284_grid, coords):\n'
                    '        raise RuntimeError("V1284 sidecar grid does not match the detections")\n'
                    '    _v1284_ref = np.concatenate(_v1284_ref_lists) if _v1284_ref_lists else np.empty((0, 4), dtype=np.float32)\n'
                    '    _v1284_ref = _v1284_ref.astype(np.float32)\n'
                    '    _v1284_ref[:, 1:] *= ds_arr\n'
                    '    _v1284_write(ds_path.stem, _v1284_ref, coords)\n')
    replacements = [
        ("import tracksdata as td\n", helper, True),
        ("    coord_lists: list[np.ndarray] = []\n",
         "    coord_lists: list[np.ndarray] = []\n    _v1284_grid_lists: list = []\n"
         + ("    _v1284_ref_lists: list = []\n" if arm == "sidecar" else ""), True),
        (det_anchor, det_new, True),
        ("    coords = coords.astype(np.int16)\n", tail_new, True),
    ]
    if arm == "full":
        replacements.append(("    model, window_size, downsample = load_model(weights_path, device)",
                             "    model, window_size, downsample = load_model(weights_path, device)\n"
                             "    UNetNodeTransformer._index_features = _v1284_index", True))
        replacements.append(("    all_edges = rebuild(coords, all_edges, _ex_pool, zarr_arr, ds_path.stem, device=device)",
                             "    all_edges = rebuild(_v1284_grid, all_edges, _ex_pool, zarr_arr, ds_path.stem, device=device)",
                             require_external))
    elif require_external:
        replacements.append(("    all_edges = rebuild(coords, all_edges, _ex_pool, zarr_arr, ds_path.stem, device=device)",
                             "    all_edges = rebuild(coords, all_edges, _ex_pool, zarr_arr, ds_path.stem, device=device)",
                             True))
    for old, new, required in replacements:
        n = source.count(old)
        if n != 1:
            if required or n > 1:
                raise RuntimeError(f"V1284 anchor count {n} for {old[:70]!r}")
            continue
        source = source.replace(old, new, 1)
    compile(source, "predict_unet_transformer.py", "exec")
    return source
