"""Pure helpers for deterministic texturing view plans.

This module intentionally has no Blender imports so camera ordering and aligned
resolution selection can be regression-tested outside Blender.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Sequence


_NATURAL_PARTS = re.compile(r"(\d+)")


def natural_name_key(name: str) -> tuple:
    """Return a stable natural-sort key (``Camera_2`` before ``Camera_10``)."""
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in _NATURAL_PARTS.split(name or "")
    )


def order_cameras(cameras: Iterable[Any], custom_names: Sequence[str] = ()) -> list[Any]:
    """Order camera-like objects once for an entire generation run."""
    natural = sorted(list(cameras), key=lambda camera: natural_name_key(camera.name))
    by_name = {camera.name: camera for camera in natural}
    ordered: list[Any] = []
    seen: set[str] = set()
    for name in custom_names:
        if name in by_name and name not in seen:
            ordered.append(by_name[name])
            seen.add(name)
    ordered.extend(camera for camera in natural if camera.name not in seen)
    return ordered


def choose_aligned_resolution(
    width: int,
    height: int,
    target_pixels: int,
    alignment: int,
    search_radius: int = 4,
) -> tuple[int, int]:
    """Rescale to the best nearby aligned dimensions without padding."""
    width = max(1, int(width))
    height = max(1, int(height))
    target_pixels = max(1, int(target_pixels))
    alignment = max(1, int(alignment))

    scale = math.sqrt(target_pixels / float(width * height))
    ideal_w = width * scale
    ideal_h = height * scale
    target_ratio = width / float(height)

    center_w = max(1, round(ideal_w / alignment))
    center_h = max(1, round(ideal_h / alignment))
    candidates: list[tuple[float, int, int]] = []
    for wi in range(max(1, center_w - search_radius), center_w + search_radius + 1):
        for hi in range(max(1, center_h - search_radius), center_h + search_radius + 1):
            candidate_w = wi * alignment
            candidate_h = hi * alignment
            ratio_error = abs(math.log((candidate_w / candidate_h) / target_ratio))
            area_error = abs(math.log((candidate_w * candidate_h) / target_pixels))
            scale_error = (
                abs(math.log(candidate_w / ideal_w))
                + abs(math.log(candidate_h / ideal_h))
            )
            score = ratio_error * 8.0 + area_error + scale_error * 0.25
            candidates.append((score, candidate_w, candidate_h))

    _, best_w, best_h = min(candidates, key=lambda item: (item[0], item[1] * item[2]))
    return best_w, best_h


def initial_view_prior(index: int, count: int, strength: float) -> float:
    """Return a bounded global rank prior for an ordered camera view.

    At maximum strength the final camera receives 25% of the first camera's
    prior. Applying this once to all weights avoids the grouping-dependent
    behavior of recursive pairwise remapping.
    """
    count = max(1, int(count))
    index = min(max(0, int(index)), count - 1)
    strength = min(max(float(strength), 0.0), 1.0)
    rank = index / float(max(count - 1, 1))
    return 1.0 - (0.75 * strength * rank)


def should_replace_owner(
    currently_covered: bool,
    owner_confidence: float,
    new_confidence: float,
    replacement_margin: float,
) -> bool:
    """Reference rule used by the shader-node ownership implementation."""
    return (
        not currently_covered
        or float(new_confidence)
        > float(owner_confidence) + max(0.0, float(replacement_margin))
    )


@dataclass(frozen=True)
class ViewRecord:
    """Immutable camera identity used by every stage of one generation run."""

    view_id: str
    index: int
    name: str
    camera: Any
    resolution_x: int
    resolution_y: int

    def to_manifest(self) -> dict[str, Any]:
        camera_data = getattr(self.camera, "data", None)
        matrix = getattr(self.camera, "matrix_world", None)
        matrix_rows = None
        if matrix is not None:
            try:
                matrix_rows = [[float(value) for value in row] for row in matrix]
            except (TypeError, ValueError):
                matrix_rows = None
        return {
            "schema": 1,
            "view_id": self.view_id,
            "index": self.index,
            "name": self.name,
            "resolution": [self.resolution_x, self.resolution_y],
            "matrix_world": matrix_rows,
            "camera": {
                "type": getattr(camera_data, "type", None),
                "lens": getattr(camera_data, "lens", None),
                "sensor_width": getattr(camera_data, "sensor_width", None),
                "sensor_height": getattr(camera_data, "sensor_height", None),
                "shift_x": getattr(camera_data, "shift_x", None),
                "shift_y": getattr(camera_data, "shift_y", None),
            },
        }
