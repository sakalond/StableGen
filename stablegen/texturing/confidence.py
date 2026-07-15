"""Confidence, ownership, and sequencing primitives for sequential texturing.

The functions in this module intentionally avoid Blender imports.  The live
shader graph and the baked pipeline use the same rules, while unit tests can
exercise the policy without opening Blender.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _gaussian_blur(image, sigma):
    """Use OpenCV when available, with a deterministic NumPy test fallback."""
    sigma = max(0.1, float(sigma))
    try:
        import cv2
        return cv2.GaussianBlur(image, (0, 0), sigma)
    except ImportError:
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
        kernel /= kernel.sum()

        def convolve_axis(values, axis):
            pad = [(0, 0)] * values.ndim
            pad[axis] = (radius, radius)
            padded = np.pad(values, pad, mode="reflect")
            return np.apply_along_axis(
                lambda row: np.convolve(row, kernel, mode="valid"),
                axis,
                padded,
            )

        return convolve_axis(convolve_axis(image, 0), 1)


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
    error; copied pixels retain their prior owner quality instead.
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


def resolve_ownership(
    qualities,
    write_masks=None,
    *,
    trust_threshold=0.7,
    replacement_margin=0.05,
    allow_trusted_replacement=False,
    initial_priority_strength=0.0,
):
    """Resolve a single high-frequency source owner for every sample.

    Uncovered and untrusted samples remain replaceable.  Trusted samples lock
    unless quality upgrades are explicitly enabled.  Initial-view priority is
    applied only as extra replacement margin for an already-trusted early
    owner; it never changes acquisition quality or promotes grazing coverage.
    """
    q = np.asarray(qualities, dtype=np.float32)
    if q.ndim < 1:
        raise ValueError("qualities must contain a camera axis")
    if write_masks is None:
        writes = np.ones_like(q, dtype=bool)
    else:
        writes = np.asarray(write_masks) > 0
        if writes.shape != q.shape:
            raise ValueError("write_masks must match qualities")

    q = np.where(writes, np.clip(q, 0.0, 1.0), 0.0)
    sample_shape = q.shape[1:]
    owner = np.full(sample_shape, -1, dtype=np.int32)
    owner_quality = np.zeros(sample_shape, dtype=np.float32)
    threshold = float(np.clip(trust_threshold, 0.0, 1.0))
    base_margin = max(0.0, float(replacement_margin))
    priority = float(np.clip(initial_priority_strength, 0.0, 1.0))
    count = q.shape[0]

    for camera_index in range(count):
        new_q = q[camera_index]
        has_write = writes[camera_index] & (new_q > 0.0)
        uncovered = owner < 0
        owner_trusted = owner_quality >= threshold

        # Earlier reliable owners receive an additional margin.  This is
        # deliberately indexed by the current owner, not by the candidate.
        if count > 1:
            owner_rank = np.clip(owner, 0, count - 1) / float(count - 1)
        else:
            owner_rank = np.zeros(sample_shape, dtype=np.float32)
        priority_margin = priority * (1.0 - owner_rank) * 0.25
        quality_upgrade = new_q > owner_quality + base_margin + priority_margin

        replace = has_write & (
            uncovered
            | (~owner_trusted)
            | (bool(allow_trusted_replacement) & quality_upgrade)
        )
        owner[replace] = camera_index
        owner_quality[replace] = new_q[replace]

    coverage = owner >= 0
    trusted = coverage & (owner_quality >= threshold)
    return {
        "owner": owner,
        "owner_quality": owner_quality,
        "trusted": trusted,
        "coverage": coverage,
    }


def trusted_repair_masks(owner_quality, visible_target, trust_threshold=0.7):
    """Return explicit trusted and repair masks without inspecting RGB."""
    quality = np.asarray(owner_quality, dtype=np.float32)
    visible = np.asarray(visible_target) > 0
    trusted = visible & (quality >= float(trust_threshold))
    repair = visible & ~trusted
    return trusted, repair


def low_frequency_seam_blend(images, owner, transition_mask, sigma=8.0):
    """Blend only low-frequency colour inside a narrow ownership transition.

    High-frequency detail always comes from the discrete owner image.  The
    returned image changes only the low-pass component in ``transition_mask``.
    """
    stack = np.asarray(images)
    owner = np.asarray(owner, dtype=np.int32)
    transition = np.asarray(transition_mask) > 0
    if stack.ndim != 4 or stack.shape[-1] < 3:
        raise ValueError("images must be camera x height x width x channels")
    if owner.shape != stack.shape[1:3]:
        raise ValueError("owner dimensions must match images")

    yy, xx = np.indices(owner.shape)
    safe_owner = np.clip(owner, 0, stack.shape[0] - 1)
    hard = stack[safe_owner, yy, xx].astype(np.float32)
    low_stack = np.stack([
        _gaussian_blur(image.astype(np.float32), sigma)
        for image in stack
    ])
    owner_low = low_stack[safe_owner, yy, xx]
    average_low = np.mean(low_stack, axis=0)
    result = hard.copy()
    result[transition] = np.clip(
        hard[transition] - owner_low[transition] + average_low[transition],
        0,
        255 if np.issubdtype(stack.dtype, np.integer) else 1.0,
    )
    return result.astype(stack.dtype)


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
