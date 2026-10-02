"""Fixed-candidate mixtures for external association experiments; no GT access.

Scores are model-specific affinities. Arithmetic mixing is an experimental
intervention, not a claim that the component probabilities are calibrated.
"""
from __future__ import annotations

import numpy as np


def validate_edges(coords, candidate_edges):
    coords, edges = np.asarray(coords), np.asarray(candidate_edges)
    if coords.ndim != 2 or coords.shape[1] != 4 or not np.isfinite(coords).all():
        raise ValueError("Expected finite raw-voxel coordinates (t,z,y,x)")
    if edges.ndim != 2 or edges.shape[1] != 4 or not np.isfinite(edges).all():
        raise ValueError("Expected finite (source,target,score,distance) edges")
    pairs = edges[:, :2].astype(np.int64)
    if not np.array_equal(pairs, edges[:, :2]):
        raise ValueError("Candidate IDs must be integers")
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= len(coords)):
        raise ValueError("Candidate endpoint outside coordinate array")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("Duplicate candidate edges")
    if len(pairs) and not np.all(coords[pairs[:, 1], 0] - coords[pairs[:, 0], 0] == 1):
        raise ValueError("Only adjacent-frame candidate edges are supported")
    if np.any((edges[:, 2] < 0) | (edges[:, 2] > 1)):
        raise ValueError("Association scores must be in [0,1]")
    return pairs


def mix_candidate_scores(coords, candidate_edges, components, weights):
    """Mix aligned scores without deleting or reordering solver candidates.

    components maps names to length-E arrays. `baseline` is always provided
    from the capture. A zero-weight component is not loaded or required.
    """
    validate_edges(coords, candidate_edges)
    values = {"baseline": np.asarray(candidate_edges)[:, 2], **components}
    if "baseline" in components:
        raise ValueError("The captured baseline cannot be overwritten")
    if not weights or any(not np.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Mixture weights must be finite and nonnegative")
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-12, rtol=0):
        raise ValueError("Mixture weights must sum to one")
    active = [(name, w) for name, w in weights.items() if w > 0]
    # Preserve exact baseline bytes at the disabled setting.
    if active == [("baseline", 1.0)]:
        return np.asarray(candidate_edges).copy()
    combined = np.zeros(len(candidate_edges), np.float64)
    for name, weight in active:
        if name not in values:
            raise ValueError(f"Missing component {name}")
        p = np.asarray(values[name], np.float64)
        if p.shape != (len(candidate_edges),) or not np.isfinite(p).all():
            raise ValueError(f"Invalid aligned component {name}")
        if np.any((p < 0) | (p > 1)):
            raise ValueError(f"Out-of-range component {name}")
        combined += weight * p
    output = np.asarray(candidate_edges).copy()
    output[:, 2] = combined
    if not np.array_equal(output[:, [0, 1, 3]], np.asarray(candidate_edges)[:, [0, 1, 3]]):
        raise RuntimeError("Mixture changed candidate identities or distances")
    return output


def disagreement_table(candidate_edges, components):
    """Return inspectable disagreement, not an estimated error probability."""
    names = ["baseline", *sorted(components)]
    matrix = np.column_stack([candidate_edges[:, 2], *[components[k] for k in names[1:]]])
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite disagreement input")
    return names, matrix, np.ptp(matrix, axis=1)
