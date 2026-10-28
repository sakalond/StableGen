"""Confidence and sequencing primitives for sequential texturing.

The functions in this module intentionally avoid Blender imports.  The live
shader graph and the baked pipeline use the same rules, while unit tests can
exercise the policy without opening Blender.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np




def acquisition_quality(
    visibility,
    signed_frontality,
    projected_density=1.0,
    depth_edge_confidence=1.0,
    frame_edge_confidence=1.0,
):
    """Return source-camera quality in ``[0, 1]``.

    Each factor describes the camera that originally produced the colour.
    Calling this function on a re-rendered/copying camera is therefore a policy
    error; copied pixels retain their prior source quality instead.
    """
    factors = [
        visibility,
        signed_frontality,
        projected_density,
        depth_edge_confidence,
        frame_edge_confidence,
    ]
    result = np.asarray(factors[0], dtype=np.float32)
    result = np.clip(result, 0.0, 1.0)
    for factor in factors[1:]:
        result = result * np.clip(np.asarray(factor, dtype=np.float32), 0.0, 1.0)
    return np.clip(result, 0.0, 1.0)


def best_source_quality(qualities, write_masks=None):
    """Return the best legitimately acquired source quality per sample.

    Colour fusion is intentionally absent from this policy.  A camera may
    improve confidence only where it was permitted to write; copied/protected
    context therefore cannot inherit the new camera's geometric quality.
    """
    q = np.asarray(qualities, dtype=np.float32)
    if q.ndim < 1 or q.shape[0] == 0:
        raise ValueError("qualities must contain a non-empty camera axis")
    if write_masks is None:
        writes = np.ones_like(q, dtype=bool)
    else:
        writes = np.asarray(write_masks) > 0
        if writes.shape != q.shape:
            raise ValueError("write_masks must match qualities")
    accepted = np.where(writes, np.clip(q, 0.0, 1.0), 0.0)
    return np.max(accepted, axis=0)




def trusted_repair_masks(source_quality, visible_target, trust_threshold=0.7):
    """Return explicit trusted and repair masks without inspecting RGB."""
    quality = np.asarray(source_quality, dtype=np.float32)
    visible = np.asarray(visible_target) > 0
    trusted = visible & (quality >= float(trust_threshold))
    repair = visible & ~trusted
    return trusted, repair




def camera_direction_from_matrix(matrix_rows) -> np.ndarray:
    """Extract normalized world-space camera forward direction (-local Z)."""
    matrix = np.asarray(matrix_rows, dtype=np.float64)
    direction = -matrix[:3, 2]
    length = np.linalg.norm(direction)
    if length <= 1e-12:
        return np.array((0.0, 0.0, -1.0), dtype=np.float64)
    return direction / length


def overlap_aware_order(
    directions: Sequence[Iterable[float]],
    *,
    anchor_index=0,
    reliable_new_coverage=None,
    trusted_overlap=None,
    occlusion_risk=None,
    reliable_coverage=None,
):
    """Return a bridge-view order using the approved overlap-aware score.

    Optional coverage/overlap/risk matrices allow a renderer to supply exact
    measurements.  Without them, angular proximity is used as a conservative
    overlap proxy, which avoids front-to-back jumps for character turntables.
    """
    dirs = np.asarray(directions, dtype=np.float64)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError("directions must be an N x 3 sequence")
    lengths = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(lengths, 1e-12)
    count = len(dirs)
    if not count:
        return []
    anchor = min(max(0, int(anchor_index)), count - 1)
    remaining = set(range(count))
    remaining.remove(anchor)
    order = [anchor]

    new_cov = np.ones(count) if reliable_new_coverage is None else np.asarray(reliable_new_coverage)
    overlap = None if trusted_overlap is None else np.asarray(trusted_overlap)
    risk = np.zeros(count) if occlusion_risk is None else np.asarray(occlusion_risk)
    coverage = None if reliable_coverage is None else np.asarray(reliable_coverage, dtype=bool)
    if coverage is not None and (coverage.ndim != 2 or coverage.shape[0] != count):
        raise ValueError("reliable_coverage must be camera x sample")
    covered = coverage[anchor].copy() if coverage is not None else None

    while remaining:
        previous = order[-1]
        best = None
        best_score = -float("inf")
        for candidate in sorted(remaining):
            cosine = float(np.clip(np.dot(dirs[previous], dirs[candidate]), -1.0, 1.0))
            angular_jump = math.acos(cosine) / math.pi
            if coverage is not None:
                candidate_coverage = coverage[candidate]
                new_score = float(np.mean(candidate_coverage & ~covered))
                denominator = max(int(np.count_nonzero(candidate_coverage)), 1)
                overlap_score = float(
                    np.count_nonzero(candidate_coverage & covered) / denominator
                )
            else:
                new_score = float(new_cov[candidate])
                overlap_score = (
                    float(overlap[previous, candidate])
                    if overlap is not None
                    else max(0.0, cosine)
                )
            score = new_score + overlap_score - angular_jump - float(risk[candidate])
            if score > best_score:
                best_score = score
                best = candidate
        order.append(best)
        remaining.remove(best)
        if coverage is not None:
            covered |= coverage[best]
    return order
