"""Binary context/write-mask helpers for sequential edit-model texturing."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # Blender's clean Python omits OpenCV; addon builds include it.
    class _NumpyCV2Fallback:
        """Small deterministic fallback for binary-mask operations and tests."""
        COLOR_BGR2GRAY = 0
        INTER_NEAREST = 0
        INTER_LINEAR = 1
        MORPH_ELLIPSE = 0
        MORPH_RECT = 1
        MORPH_CLOSE = 2
        CC_STAT_AREA = 4

        @staticmethod
        def cvtColor(image, _code):
            return np.rint(np.mean(image[:, :, :3], axis=2)).astype(image.dtype)

        @staticmethod
        def resize(image, size, interpolation=0):
            width, height = size
            y = np.linspace(0, image.shape[0] - 1, height)
            x = np.linspace(0, image.shape[1] - 1, width)
            if interpolation == _NumpyCV2Fallback.INTER_NEAREST:
                return image[np.rint(y).astype(int)[:, None], np.rint(x).astype(int)]
            # Bilinear accuracy is unnecessary for binary policy fallback;
            # nearest keeps masks and IDs categorical and deterministic.
            return image[np.rint(y).astype(int)[:, None], np.rint(x).astype(int)]

        @staticmethod
        def getStructuringElement(shape, size):
            width, height = size
            if shape == _NumpyCV2Fallback.MORPH_RECT:
                return np.ones((height, width), dtype=np.uint8)
            yy, xx = np.ogrid[:height, :width]
            cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
            ry, rx = max(height / 2.0, 0.5), max(width / 2.0, 0.5)
            return (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0).astype(np.uint8)

        @staticmethod
        def _morph(image, kernel, maximum):
            kh, kw = kernel.shape
            py, px = kh // 2, kw // 2
            pad_value = 0 if maximum else 255
            padded = np.pad(image, ((py, py), (px, px)), constant_values=pad_value)
            samples = []
            for y, x in np.argwhere(kernel > 0):
                samples.append(padded[y:y + image.shape[0], x:x + image.shape[1]])
            stack = np.stack(samples)
            return (np.max(stack, axis=0) if maximum else np.min(stack, axis=0)).astype(image.dtype)

        @classmethod
        def dilate(cls, image, kernel, iterations=1):
            result = image
            for _ in range(max(1, int(iterations))):
                result = cls._morph(result, kernel, True)
            return result

        @classmethod
        def erode(cls, image, kernel, iterations=1):
            result = image
            for _ in range(max(1, int(iterations))):
                result = cls._morph(result, kernel, False)
            return result

        @classmethod
        def morphologyEx(cls, image, operation, kernel):
            if operation != cls.MORPH_CLOSE:
                raise ValueError("Only binary close is supported without OpenCV")
            return cls.erode(cls.dilate(image, kernel), kernel)

        @classmethod
        def connectedComponentsWithStats(cls, image, connectivity=8):
            foreground = image > 0
            labels = np.zeros(foreground.shape, dtype=np.int32)
            component = 0
            stats = [[0, 0, image.shape[1], image.shape[0], int(np.count_nonzero(~foreground))]]
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            if connectivity == 4:
                offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
            for start_y, start_x in np.argwhere(foreground):
                if labels[start_y, start_x]:
                    continue
                component += 1
                queue = [(int(start_y), int(start_x))]
                labels[start_y, start_x] = component
                points = []
                while queue:
                    y, x = queue.pop()
                    points.append((y, x))
                    for dy, dx in offsets:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < foreground.shape[0] and 0 <= nx < foreground.shape[1]
                                and foreground[ny, nx] and labels[ny, nx] == 0):
                            labels[ny, nx] = component
                            queue.append((ny, nx))
                ys, xs = zip(*points)
                stats.append([min(xs), min(ys), max(xs) - min(xs) + 1,
                              max(ys) - min(ys) + 1, len(points)])
            centroids = np.zeros((component + 1, 2), dtype=np.float64)
            return component + 1, labels, np.asarray(stats, dtype=np.int32), centroids

    cv2 = _NumpyCV2Fallback()


def regularize_repair_mask(
    mask,
    threshold=127,
    dilation=0,
    min_island_area=0,
    min_hole_area=0,
    min_strip_width=0,
    boundary_labels=None,
    keep_largest_components=0,
):
    """Return a regularized exact binary repair mask (255 means editable).

    Morphology can be constrained independently inside object/material labels,
    preventing repair regions from bleeding across part boundaries.  Thin
    trusted strips are removed with a close operation on the repair side.
    """
    if mask is None:
        raise ValueError("mask is required")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = np.where(mask > int(threshold), 255, 0).astype(np.uint8)

    def _regularize_region(region_mask, region):
        # Zero other parts before morphology. Masking only after dilation lets
        # a hand/accessory region seed repair pixels across a dress/body part.
        result = np.where(region, region_mask, 0).astype(np.uint8)
        if min_island_area > 0:
            # Small trusted islands are holes in the repair mask.
            trusted = np.where(region & (result == 0), 255, 0).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(trusted, 8)
            for component in range(1, count):
                if stats[component, cv2.CC_STAT_AREA] < int(min_island_area):
                    result[labels == component] = 255
        if min_hole_area > 0:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                np.where(region & (result > 0), 255, 0).astype(np.uint8), 8
            )
            # Fill small repair components back to trusted.
            for component in range(1, count):
                if stats[component, cv2.CC_STAT_AREA] < int(min_hole_area):
                    result[labels == component] = 0
        if min_strip_width > 0:
            size = max(1, int(min_strip_width)) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            closed = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
            result[region] = closed[region]
        if dilation > 0:
            size = int(dilation) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            expanded = cv2.dilate(result, kernel, iterations=1)
            result[region] = expanded[region]
        result[~region] = 0
        return result

    if boundary_labels is None:
        binary = _regularize_region(binary, np.ones(binary.shape, dtype=bool))
    else:
        labels = np.asarray(boundary_labels)
        if labels.shape != binary.shape:
            labels = cv2.resize(
                labels.astype(np.float32),
                (binary.shape[1], binary.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        combined = np.zeros_like(binary)
        for label in np.unique(labels):
            if label == 0:
                continue
            region = labels == label
            combined = np.maximum(combined, _regularize_region(binary, region))
        binary = combined

    keep = max(0, int(keep_largest_components))
    if keep:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        ranked = sorted(
            range(1, count), key=lambda item: stats[item, cv2.CC_STAT_AREA], reverse=True
        )[:keep]
        binary = np.where(np.isin(labels, ranked), 255, 0).astype(np.uint8)
    return binary


def apply_binary_context_mask(context_bgr, repair_mask, fallback_rgb):
    """Replace every editable context pixel with the exact fallback colour."""
    if context_bgr is None:
        raise ValueError("context image is required")
    if context_bgr.shape[:2] != repair_mask.shape[:2]:
        repair_mask = cv2.resize(
            repair_mask,
            (context_bgr.shape[1], context_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    result = context_bgr.copy()
    fallback_bgr = np.array(
        [round(fallback_rgb[2] * 255), round(fallback_rgb[1] * 255), round(fallback_rgb[0] * 255)],
        dtype=np.uint8,
    )
    result[repair_mask > 0, :3] = fallback_bgr
    return result


def reject_fallback_from_write_mask(
    write_mask,
    generated_bgr,
    fallback_rgb,
    tolerance=8,
):
    """Disable write pixels where Qwen left the sentinel colour behind."""
    if generated_bgr.shape[:2] != write_mask.shape[:2]:
        generated_bgr = cv2.resize(
            generated_bgr,
            (write_mask.shape[1], write_mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    fallback_bgr = np.array(
        [round(fallback_rgb[2] * 255), round(fallback_rgb[1] * 255), round(fallback_rgb[0] * 255)],
        dtype=np.int16,
    )
    delta = generated_bgr[:, :, :3].astype(np.int32) - fallback_bgr.astype(np.int32)[None, None, :]
    sentinel = np.sqrt(np.sum(delta * delta, axis=2)) <= max(0, int(tolerance))
    result = write_mask.copy()
    rejected = (write_mask > 0) & sentinel
    result[rejected] = 0
    return result


def residual_fallback_percentage(write_mask, generated_bgr, fallback_rgb, tolerance=8):
    """Measure sentinel leakage only inside the explicit editable region."""
    editable = np.asarray(write_mask) > 0
    if not np.any(editable):
        return 0.0
    if generated_bgr.shape[:2] != editable.shape:
        generated_bgr = cv2.resize(
            generated_bgr,
            (editable.shape[1], editable.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    fallback_bgr = np.array(
        [round(fallback_rgb[2] * 255), round(fallback_rgb[1] * 255), round(fallback_rgb[0] * 255)],
        dtype=np.int32,
    )
    delta = generated_bgr[:, :, :3].astype(np.int32) - fallback_bgr[None, None, :]
    sentinel = np.sqrt(np.sum(delta * delta, axis=2)) <= max(0, int(tolerance))
    return float(np.count_nonzero(editable & sentinel) * 100.0 / np.count_nonzero(editable))


def feather_projection_guard(guard, width, *, return_influence=False):
    """Fade into safe pixels without admitting any rejected silhouette pixels."""
    width = max(0.0, float(width))
    if width == 0:
        return (guard.copy(), np.zeros_like(guard)) if return_influence else guard.copy()
    allowed = (guard > 0).astype(np.uint8)
    distance = cv2.distanceTransform(allowed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    ramp = np.clip(distance / width, 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    feather = np.rint(ramp * 255).astype(np.uint8)
    if not return_influence:
        return feather
    shoulder = np.clip((distance - width) / width, 0.0, 1.0)
    shoulder = shoulder * shoulder * (3.0 - 2.0 * shoulder)
    return feather, np.rint((1.0 - shoulder) * 255).astype(np.uint8)


def projection_guard_feather_influence(guard, width):
    """Keep sharpening off across the fade; restore it over one extra width."""
    width = max(0.0, float(width))
    if width == 0:
        return np.zeros_like(guard)
    distance = cv2.distanceTransform((guard > 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    ramp = np.clip((distance - width) / width, 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    return np.rint((1.0 - ramp) * 255).astype(np.uint8)


def projection_occlusion_fields(depth, width, depth_threshold, object_ids=None, surface_distance=None, *, include_legacy=False):
    """Fade only the far surface approaching a resolved depth discontinuity.

    Unlike a part/margin guard, this field must survive coverage rescue. Carry
    the nearest boundary's occluder depth so its fade cannot spill onto the
    near surface (e.g. from the neck onto the visible chin). Object boundaries
    with a resolved depth order catch thin nearby occluders below the general
    depth threshold. Flat ID seams and foreground margins do not seed it.

    With surface distance and include_legacy, also return the two legacy
    fields before the geometry fields, sharing the same boundary transforms.
    """
    if include_legacy and surface_distance is None:
        raise ValueError('Combined occlusion fields require surface distance')
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    depth = depth.astype(np.uint8)
    width = max(0.0, float(width))
    white = np.full(depth.shape, 255, dtype=np.uint8)
    empty = np.zeros_like(depth)
    def empty_fields():
        geometry = (white, empty, np.zeros(depth.shape, np.float32))
        if surface_distance is None:
            return white, empty
        return (white, empty) + geometry if include_legacy else geometry
    if width == 0:
        return empty_fields()
    threshold = max(0, int(depth_threshold))
    near = cv2.dilate(depth, np.ones((3, 3), np.uint8))
    seeds = near.astype(np.int16) - depth > threshold
    seed_threshold = np.full(depth.shape, threshold, dtype=np.int16)
    if object_ids is not None:
        if object_ids.ndim == 3:
            object_ids = cv2.cvtColor(object_ids, cv2.COLOR_BGR2GRAY)
        if object_ids.shape != depth.shape:
            object_ids = cv2.resize(object_ids, (depth.shape[1], depth.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        padded_depth = np.pad(depth, 1, mode='edge')
        padded_ids = np.pad(object_ids, 1, mode='edge')
        depth_signed = depth.astype(np.int16)
        object_near = depth.copy()
        for dy, dx in ((dy, dx) for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)):
            neighbor = padded_depth[dy:dy+depth.shape[0], dx:dx+depth.shape[1]]
            ids = padded_ids[dy:dy+depth.shape[0], dx:dx+depth.shape[1]]
            # Ignore a single quantization step and never infer ordering from
            # IDs alone. Compare against the OTHER object's depth explicitly.
            ordered = (ids != object_ids) & (neighbor.astype(np.int16) - depth_signed > 1)
            np.maximum(object_near, neighbor, out=object_near, where=ordered)
        object_seeds = object_near.astype(np.int16) - depth > 1
        near = np.where(object_seeds, object_near, near)
        seed_threshold[object_seeds] = 1
        seeds |= object_seeds
    if not np.any(seeds):
        return empty_fields()
    allowed = (~seeds).astype(np.uint8)
    # Labels identify the occluder belonging to the nearest boundary pixel.
    # Use the precise transform separately for the continuous feather width.
    _, labels = cv2.distanceTransformWithLabels(
        allowed, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    references = np.zeros(int(labels.max()) + 1, dtype=np.int16)
    references[labels[seeds]] = near[seeds]
    thresholds = np.zeros_like(references)
    thresholds[labels[seeds]] = seed_threshold[seeds]
    far_side = references[labels] - depth > thresholds[labels]
    distance = cv2.distanceTransform(allowed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    ramp = np.clip(distance / width, 0.0, 1.0)
    fade = ramp * ramp * (3.0 - 2.0 * ramp)
    shoulder = np.clip((distance - width) / width, 0.0, 1.0)
    influence = 1.0 - shoulder * shoulder * (3.0 - 2.0 * shoulder)
    if surface_distance is not None:
        # Keep the ramp continuous on BOTH sides of the image boundary.
        # The shader identifies the receiving surface by camera distance;
        # baking the foreground exemption into this image lets filtering
        # reintroduce the foreground's white value onto the far surface.
        nearest = cv2.erode(surface_distance, np.ones((3, 3), np.uint8))
        midpoint = (nearest + surface_distance) * 0.5
        reference = np.zeros(int(labels.max()) + 1, dtype=np.float32)
        reference[labels[seeds]] = midpoint[seeds]
        geometry = (np.rint(fade * 255).astype(np.uint8),
                    np.rint(influence * 255).astype(np.uint8), reference[labels])
        if not include_legacy:
            return geometry
        return (np.where(far_side, geometry[0], 255).astype(np.uint8),
                np.where(far_side, geometry[1], 0).astype(np.uint8)) + geometry
    return (np.where(far_side, np.rint(fade * 255), 255).astype(np.uint8),
            np.where(far_side, np.rint(influence * 255), 0).astype(np.uint8))


def build_depth_part_rejection_mask(
    depth,
    object_ids=None,
    material_ids=None,
    *,
    margin=3,
    depth_threshold=3,
    foreground_margin=None,
    return_diagnostics=False,
):
    """Build a per-pixel projection guard around depth/part silhouettes.

    White pixels may project; black pixels are rejected.  ID boundaries catch
    hand/dress and accessory/body transitions even when normalized depth is
    locally similar.
    """
    if depth is None:
        raise ValueError("depth is required")
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    depth = depth.astype(np.uint8)
    margin = max(0, int(margin))
    if margin == 0 and (foreground_margin is None or foreground_margin <= 0):
        guard = np.full(depth.shape, 255, dtype=np.uint8)
        diagnostics = {
            "mode": "disabled",
            "fallback_triggered": False,
            "id_guard_allowed_foreground_percent": 100.0,
            "allowed_foreground_percent": 100.0,
        }
        return (guard, diagnostics) if return_diagnostics else guard
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    depth_range = cv2.dilate(depth, kernel3) - cv2.erode(depth, kernel3)
    depth_edge = depth_range > max(0, int(depth_threshold))
    edge = depth_edge.copy()
    id_edge = np.zeros(depth.shape, dtype=bool)
    used_part_ids = False

    for ids in (object_ids, material_ids):
        if ids is None:
            continue
        if ids.ndim == 3:
            ids = cv2.cvtColor(ids, cv2.COLOR_BGR2GRAY)
        if ids.shape != depth.shape:
            ids = cv2.resize(
                ids, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        boundary = cv2.dilate(ids, kernel3) != cv2.erode(ids, kernel3)
        id_edge |= boundary
        edge |= boundary
        used_part_ids = True

    def _guard_from_edges(edges):
        edge_u8 = np.where(edges, 255, 0).astype(np.uint8)
        if margin:
            size = margin * 2 + 1
            edge_u8 = cv2.dilate(
                edge_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
            )
        return np.where(edge_u8 > 0, 0, 255).astype(np.uint8)

    def _directional_guard(include_ids):
        # Inverted normalized depth: larger values are nearer the camera.
        # Carry the opposite side's depth while expanding a rejection band,
        # then compare at the destination. Ordinary dilation crosses onto the
        # foreground (e.g. the visible chin above a neck occlusion).
        threshold = max(0, int(depth_threshold))
        fg_margin = max(0, int(foreground_margin))
        near = cv2.dilate(depth, kernel3)
        far = cv2.erode(depth, kernel3)
        far_seed = near.astype(np.int16) - depth > threshold
        near_seed = depth.astype(np.int16) - far > threshold
        ambiguous = np.zeros_like(id_edge)
        if include_ids:
            radius = max(margin, fg_margin, 1) + 1
            neighborhood = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
            id_near, id_far = cv2.dilate(depth, neighborhood), cv2.erode(depth, neighborhood)
            far_ids = id_edge & (id_near.astype(np.int16) - depth > threshold)
            near_ids = id_edge & (depth.astype(np.int16) - id_far > threshold)
            near = np.where(far_ids, np.maximum(near, id_near), near)
            far = np.where(near_ids, np.minimum(far, id_far), far)
            far_seed |= far_ids
            near_seed |= near_ids
            # Without a resolvable depth order, keep symmetric part protection.
            ambiguous = id_edge & ((id_near.astype(np.int16) - id_far) <= threshold)
        rejected = np.zeros_like(id_edge)
        if margin > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*margin+1, 2*margin+1))
            nearer_reference = cv2.dilate(np.where(far_seed, near, 0).astype(np.uint8), kernel)
            rejected |= nearer_reference.astype(np.int16) - depth > threshold
            rejected |= cv2.dilate(ambiguous.astype(np.uint8), kernel) > 0
        if fg_margin > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*fg_margin+1, 2*fg_margin+1))
            farther_reference = cv2.erode(np.where(near_seed, far, 255).astype(np.uint8), kernel)
            rejected |= depth.astype(np.int16) - farther_reference > threshold
        return np.where(rejected, 0, 255).astype(np.uint8)

    guard = _guard_from_edges(edge) if foreground_margin is None else _directional_guard(used_part_ids)

    # Categorical render passes must be noise-free, but older outputs and
    # third-party render settings can contain one-LSB dither. Treating that as
    # real object/material boundaries expands rejection across virtually the
    # whole character. If part IDs would leave less than 10% of foreground
    # writable, retain the depth-discontinuity protection and discard the
    # contaminated ID contribution instead of blacking out the texture.
    foreground = depth > 1
    diagnostics = {
        "mode": "depth_part" if used_part_ids else "depth_only",
        "fallback_triggered": False,
        "id_guard_allowed_foreground_percent": None,
        "allowed_foreground_percent": None,
    }
    if used_part_ids and np.any(foreground):
        allowed_fraction = float(np.count_nonzero((guard > 0) & foreground)) / float(
            np.count_nonzero(foreground)
        )
        diagnostics["id_guard_allowed_foreground_percent"] = allowed_fraction * 100.0
        if allowed_fraction < 0.10:
            guard = (_guard_from_edges(depth_edge) if foreground_margin is None
                     else _directional_guard(False))
            diagnostics["mode"] = "depth_only_fallback"
            diagnostics["fallback_triggered"] = True

    if np.any(foreground):
        diagnostics["allowed_foreground_percent"] = (
            float(np.count_nonzero((guard > 0) & foreground))
            * 100.0 / float(np.count_nonzero(foreground))
        )

    if return_diagnostics:
        return guard, diagnostics
    return guard
