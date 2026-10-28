"""Camera math, BVH, coverage algorithms, and resolution helpers."""

import math
import bpy, bmesh  # pylint: disable=import-error
import numpy as np
import mathutils
from mathutils import Matrix
from ..core.architectures import QWEN_21_ARCHITECTURES

def _existing_camera_directions(mesh_center):
    """Return unit-direction vectors from *mesh_center* toward each existing
    camera in the scene.  Used by 'Consider existing cameras' to treat
    pre-existing cameras as already-placed directions."""
    dirs = []
    center = np.array(mesh_center, dtype=float)
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            pos = np.array(obj.location, dtype=float)
            d = pos - center
            norm = np.linalg.norm(d)
            if norm > 1e-6:
                dirs.append(d / norm)
    return dirs


def _filter_near_existing(directions, existing_dirs, min_angle_deg=30.0):
    """Remove directions that are within *min_angle_deg* of any existing
    camera direction.  Both *directions* and *existing_dirs* should be
    lists of unit-length numpy arrays."""
    if not existing_dirs or not directions:
        return directions
    cos_thresh = math.cos(math.radians(min_angle_deg))
    existing_np = np.array(existing_dirs)          # (M, 3)
    filtered = []
    for d in directions:
        d_np = np.asarray(d, dtype=float)
        n = np.linalg.norm(d_np)
        if n < 1e-12:
            continue
        d_unit = d_np / n
        dots = existing_np @ d_unit                # (M,)
        if dots.max() < cos_thresh:
            filtered.append(d)
    return filtered

def _fibonacci_sphere_points(n):
    """Generate *n* approximately evenly-spaced unit vectors on a sphere
    using a Fibonacci spiral.  Returns list of (x, y, z) tuples."""
    points = []
    golden_ratio = (1 + math.sqrt(5)) / 2
    for i in range(n):
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        phi = 2 * math.pi * i / golden_ratio
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        points.append((x, y, z))
    return points


def _gather_target_meshes(context):
    """Return a list of mesh objects to cover.
    - If any selected objects are meshes, use those.
    - Otherwise, use ALL mesh objects in the scene.
    """
    selected = [o for o in context.selected_objects if o.type == 'MESH']
    if selected:
        return selected
    return [o for o in context.scene.objects if o.type == 'MESH']


def _get_mesh_face_data(objs):
    """Return world-space face (normals, areas, centers) as numpy arrays.
    *objs* can be a single object or a list of mesh objects."""
    if not isinstance(objs, (list, tuple)):
        objs = [objs]
    all_normals, all_areas, all_centers = [], [], []
    for obj in objs:
        mesh = obj.data
        mat = obj.matrix_world
        rot = mat.to_3x3()
        scale_det = abs(rot.determinant())
        n = len(mesh.polygons)
        normals = np.empty((n, 3))
        areas = np.empty(n)
        centers = np.empty((n, 3))
        for idx, poly in enumerate(mesh.polygons):
            wn = rot @ poly.normal
            wn.normalize()
            normals[idx] = (wn.x, wn.y, wn.z)
            wc = mat @ poly.center
            centers[idx] = (wc.x, wc.y, wc.z)
            areas[idx] = poly.area * (scale_det ** 0.5)
        all_normals.append(normals)
        all_areas.append(areas)
        all_centers.append(centers)
    return np.vstack(all_normals), np.concatenate(all_areas), np.vstack(all_centers)


def _filter_bottom_faces(normals, areas, centers, angle_rad):
    """Remove faces whose normals point more than *angle_rad* below the
    horizon (negative-Z).  Returns filtered (normals, areas, centers).

    *angle_rad* = 80° (≈1.40 rad) removes faces whose normal is > 80° below
    horizontal, i.e. nearly straight down.  The Z-component threshold is
    ``-cos(angle_rad)``."""
    threshold_z = -math.cos(angle_rad)
    mask = normals[:, 2] >= threshold_z  # keep faces *above* the threshold
    return normals[mask], areas[mask], centers[mask]


def _get_mesh_verts_world(objs):
    """Return world-space vertex positions as (N, 3) numpy array.
    *objs* can be a single object or a list of mesh objects."""
    if not isinstance(objs, (list, tuple)):
        objs = [objs]
    parts = []
    for obj in objs:
        mesh = obj.data
        mat = obj.matrix_world
        n = len(mesh.vertices)
        verts = np.empty((n, 3))
        for i, v in enumerate(mesh.vertices):
            wv = mat @ v.co
            verts[i] = (wv.x, wv.y, wv.z)
        parts.append(verts)
    return np.vstack(parts)


def _camera_basis(cam_dir_np):
    """Build an orthonormal camera basis from a centre-to-camera direction.

    Returns ``(right, up, d_unit)`` where
    * *right*   – camera local X (numpy unit vector)
    * *up*      – camera local Y (numpy unit vector)
    * *d_unit*  – normalised centre-to-camera direction (camera local Z)

    When the direction is nearly vertical the world-up fallback switches
    from Z to Y, giving a deterministic orientation that avoids
    ``to_track_quat`` gimbal degeneracy.
    """
    d = cam_dir_np / np.linalg.norm(cam_dir_np)
    forward = -d
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(d, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return right, up, d


def _rotation_from_basis(right, up, d_unit):
    """Build a Blender rotation (Euler) from camera basis vectors.

    The resulting rotation aligns the camera's local axes:
    * local  X → *right*
    * local  Y → *up*
    * local  Z → *d_unit*   (away from mesh; -Z is the look direction)
    """
    rot_mat = mathutils.Matrix((
        (right[0], up[0], d_unit[0]),
        (right[1], up[1], d_unit[1]),
        (right[2], up[2], d_unit[2]),
    ))
    return rot_mat.to_euler()


def _compute_silhouette_distance(verts_world, center_np, cam_dir_np, fov_x, fov_y, margin=0.10):
    """Compute the optimal distance along *cam_dir_np* so every mesh vertex
    fits inside the camera frame with *margin* breathing room.

    Uses a perspective-correct formula so objects with depth along the
    viewing direction are never clipped.

    Returns ``(distance, aim_offset)`` where *aim_offset* is a 3-element numpy
    vector that shifts from *center_np* to the visual center of the silhouette.
    The caller should aim the camera at ``center_np + aim_offset`` and place
    it at ``center_np + aim_offset + direction * distance``.
    """
    right, up, d = _camera_basis(cam_dir_np)
    forward = -d

    # Project every vertex onto the camera's right / up plane
    rel = verts_world - center_np
    proj_r = rel @ right
    proj_u = rel @ up

    r_min, r_max = float(proj_r.min()), float(proj_r.max())
    u_min, u_max = float(proj_u.min()), float(proj_u.max())

    # Visual centre of the silhouette (midpoint of extents)
    mid_r = (r_max + r_min) / 2.0
    mid_u = (u_max + u_min) / 2.0

    # Aim offset: shifts from mesh center to silhouette visual centre
    aim_offset = mid_r * right + mid_u * up

    eff_fov_x = fov_x * (1.0 - margin)
    eff_fov_y = fov_y * (1.0 - margin)

    tan_hx = math.tan(eff_fov_x / 2) if eff_fov_x > 0.02 else 1e-6
    tan_hy = math.tan(eff_fov_y / 2) if eff_fov_y > 0.02 else 1e-6

    # Perspective-correct distance: for each vertex the minimum camera
    # distance along the view direction is  |lateral| / tan(half_fov) - depth
    # where depth is signed distance from the aim point along the view ray.
    aim_point = center_np + aim_offset
    rel_aim = verts_world - aim_point
    pr = rel_aim @ right
    pu = rel_aim @ up
    # Depth: positive = in front of aim point (toward camera)
    pd = rel_aim @ d  # dot with camera direction (away from mesh)

    min_dist_r = np.abs(pr) / tan_hx + pd
    min_dist_u = np.abs(pu) / tan_hy + pd

    dist = max(float(min_dist_r.max()), float(min_dist_u.max()), 0.5)

    # --- Refine aim_offset using perspective angular centre ---
    # The orthographic midpoint doesn't account for perspective foreshortening.
    # Compute where the visual centre actually is from the camera position and
    # shift the aim to centre it, then recompute the distance.
    cam_pos = aim_point + d * dist
    rel_cam = verts_world - cam_pos
    depth = -(rel_cam @ d)  # positive = in front of camera
    depth = np.maximum(depth, 0.001)
    ang_r = np.arctan2(rel_cam @ right, depth)
    ang_u = np.arctan2(rel_cam @ up, depth)
    ang_mid_r = (float(ang_r.max()) + float(ang_r.min())) / 2.0
    ang_mid_u = (float(ang_u.max()) + float(ang_u.min())) / 2.0
    # Convert angular offset to world-space shift (at the computed distance)
    aim_offset = aim_offset + dist * math.tan(ang_mid_r) * right + dist * math.tan(ang_mid_u) * up

    # Recompute distance with refined aim
    aim_point = center_np + aim_offset
    rel_aim = verts_world - aim_point
    pr = rel_aim @ right
    pu = rel_aim @ up
    pd = rel_aim @ d
    min_dist_r = np.abs(pr) / tan_hx + pd
    min_dist_u = np.abs(pu) / tan_hy + pd
    dist = max(float(min_dist_r.max()), float(min_dist_u.max()), 0.5)

    return dist, aim_offset


def _kmeans_on_sphere(directions, weights, k, max_iter=50):
    """Spherical K-means: cluster unit vectors weighted by area.
    Returns (k, 3) numpy array of cluster-centre unit vectors."""
    n_pts = len(directions)
    if n_pts == 0 or k == 0:
        return np.zeros((max(k, 1), 3))
    k = min(k, n_pts)
    rng = np.random.default_rng(42)
    probs = weights / weights.sum()
    indices = rng.choice(n_pts, size=k, replace=False, p=probs)
    centers = directions[indices].copy()
    for _ in range(max_iter):
        dots = directions @ centers.T
        labels = np.argmax(dots, axis=1)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = labels == j
            if mask.any():
                ws = (directions[mask] * weights[mask, np.newaxis]).sum(axis=0)
                nrm = np.linalg.norm(ws)
                new_centers[j] = ws / nrm if nrm > 0 else centers[j]
            else:
                new_centers[j] = centers[j]
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers
    return centers


def _compute_pca_axes(verts):
    """Return the 3 principal axes of a (N, 3) vertex array (rows of a
    3x3 array, sorted by descending eigenvalue)."""
    mean = verts.mean(axis=0)
    centered = verts - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, idx].T  # rows = principal axes


def _greedy_coverage_directions(normals, areas, max_cameras=12,
                                coverage_target=0.95, n_candidates=200,
                                existing_dirs=None):
    """Greedy set-cover: iteratively pick the camera direction that adds the
    most newly-visible surface area (back-face culling only, no occlusion,
    for speed).  Returns (selected_directions, final_coverage_fraction)."""
    total_area = areas.sum()
    if total_area <= 0:
        return [], 0.0

    candidates = np.array(_fibonacci_sphere_points(n_candidates))

    # visibility[i, j] = True if face i faces toward candidate j
    # cos(75°) ≈ 0.26 – ignore near-grazing faces that won't texture well
    visibility = normals @ candidates.T > 0.26  # (n_faces, n_candidates)

    covered = np.zeros(len(areas), dtype=bool)
    # Pre-seed coverage from existing cameras
    if existing_dirs:
        for edir in existing_dirs:
            covered |= normals @ np.asarray(edir, dtype=float) > 0.26
    selected = []

    for _ in range(max_cameras):
        uncovered = ~covered
        # Vectorised: for each candidate sum the area of faces that are
        # both visible from it AND not yet covered
        new_vis = visibility & uncovered[:, np.newaxis]  # broadcast
        new_areas = (new_vis * areas[:, np.newaxis]).sum(axis=0)

        best_idx = int(np.argmax(new_areas))
        if new_areas[best_idx] < total_area * 0.005:   # < 0.5 % new coverage
            break

        selected.append(candidates[best_idx].copy())
        covered |= visibility[:, best_idx]

        coverage = float(areas[covered].sum() / total_area)
        if coverage >= coverage_target:
            break

    final_coverage = float(areas[covered].sum() / total_area) if covered.any() else 0.0
    return selected, final_coverage


# ──────────────────────────────────────────────────────────────────────
# BVH occlusion helpers
# ──────────────────────────────────────────────────────────────────────

def _build_bvh_trees(objs, depsgraph):
    """Build a list of BVHTree objects (one per mesh) for raycasting.

    Returns
    -------
    list[mathutils.bvhtree.BVHTree]
        One BVH per object, in world space.
    """
    from mathutils.bvhtree import BVHTree
    trees = []
    for obj in objs:
        bm = bmesh.new()
        bm.from_object(obj, depsgraph)
        bm.transform(obj.matrix_world)
        tree = BVHTree.FromBMesh(bm)
        bm.free()
        trees.append(tree)
    return trees


def _ray_occluded(bvh_trees, origin, direction, max_dist):
    """Return True if *any* BVH tree has a hit between *origin* and
    *origin + direction * max_dist* (exclusive of the starting face).

    *origin* should already be slightly offset along the face normal to avoid
    self-intersection.
    """
    for tree in bvh_trees:
        hit_loc, _normal, _index, _dist = tree.ray_cast(
            mathutils.Vector(origin), mathutils.Vector(direction), max_dist)
        if hit_loc is not None:
            return True
    return False


def _greedy_select_from_visibility(vis, areas, max_cameras, coverage_target,
                                   candidates, existing_dirs=None,
                                   normals=None):
    """Run greedy set-cover on a pre-computed visibility matrix.

    Parameters
    ----------
    vis : ndarray (n_faces, n_candidates), bool
    areas : ndarray (n_faces,)
    max_cameras : int
    coverage_target : float
    candidates : ndarray (n_candidates, 3)
    existing_dirs : list[ndarray] or None
        Directions of existing cameras to pre-seed coverage.
    normals : ndarray (n_faces, 3) or None
        Face normals, required when *existing_dirs* is given.

    Returns
    -------
    (selected_directions, final_coverage)
    """
    total_area = areas.sum()
    if total_area <= 0:
        return [], 0.0

    covered = np.zeros(len(areas), dtype=bool)
    # Pre-seed coverage from existing cameras
    if existing_dirs and normals is not None:
        for edir in existing_dirs:
            covered |= normals @ np.asarray(edir, dtype=float) > 0.26
    selected = []

    for _ in range(max_cameras):
        uncovered = ~covered
        new_vis = vis & uncovered[:, np.newaxis]
        new_areas = (new_vis * areas[:, np.newaxis]).sum(axis=0)

        best_idx = int(np.argmax(new_areas))
        if new_areas[best_idx] < total_area * 0.005:
            break

        selected.append(candidates[best_idx].copy())
        covered |= vis[:, best_idx]

        coverage = float(areas[covered].sum() / total_area)
        if coverage >= coverage_target:
            break

    final_cov = float(areas[covered].sum() / total_area) if covered.any() else 0.0
    return selected, final_cov


# ──────────────────────────────────────────────────────────────────────
# Occlusion generators (modal-friendly, yield progress)
# ──────────────────────────────────────────────────────────────────────

def _occ_filter_faces_generator(normals, areas, centers, bvh_trees,
                                n_candidates=200):
    """Generator: determine which faces are visible from at least one
    camera direction (exterior faces).

    Yields progress floats in [0, 1].  Final result (via
    ``StopIteration.value``) is a boolean mask ``(n_faces,)`` where True
    means the face is visible from at least one direction.

    Uses early-exit per face: as soon as one unoccluded direction is found,
    the face is marked exterior and skipped for remaining candidates.
    """
    n_faces = len(normals)
    candidates = np.array(_fibonacci_sphere_points(n_candidates))
    backface_vis = normals @ candidates.T > 0.26  # (F, C)
    exterior = np.zeros(n_faces, dtype=bool)
    epsilon = 0.001
    BATCH = 5

    for j in range(n_candidates):
        cam_dir = candidates[j]
        for i in range(n_faces):
            if exterior[i]:
                continue  # already proven exterior
            if not backface_vis[i, j]:
                continue
            origin = centers[i] + normals[i] * epsilon
            if not _ray_occluded(bvh_trees, origin, cam_dir, 1e6):
                exterior[i] = True
        if (j + 1) % BATCH == 0 or j == n_candidates - 1:
            yield (j + 1) / n_candidates
            # Early termination: all faces proven exterior
            if exterior.all():
                break

    return exterior


def _occ_vis_count_generator(normals, areas, centers, bvh_trees,
                              n_candidates=200):
    """Generator: count how many candidate directions can see each face.

    Like ``_occ_filter_faces_generator`` but returns per-face visibility
    *counts* instead of a boolean mask.  This enables continuous weighting
    rather than binary keep/remove filtering.

    Yields progress floats in [0, 1].  Final result (via
    ``StopIteration.value``) is an int array ``(n_faces,)`` with the number
    of unoccluded candidate directions per face.
    """
    n_faces = len(normals)
    candidates = np.array(_fibonacci_sphere_points(n_candidates))
    backface_vis = normals @ candidates.T > 0.26  # (F, C)
    vis_count = np.zeros(n_faces, dtype=int)
    epsilon = 0.001
    BATCH = 5

    for j in range(n_candidates):
        cam_dir = candidates[j]
        for i in range(n_faces):
            if not backface_vis[i, j]:
                continue
            origin = centers[i] + normals[i] * epsilon
            if not _ray_occluded(bvh_trees, origin, cam_dir, 1e6):
                vis_count[i] += 1
        if (j + 1) % BATCH == 0 or j == n_candidates - 1:
            yield (j + 1) / n_candidates

    return vis_count


def _occ_fom_generator(normals, areas, centers, bvh_trees,
                       max_cameras, coverage_target, n_candidates=200,
                       existing_dirs=None):
    """Generator: Full Occlusion Matrix approach.

    Yields progress floats in [0, 1].  Final result is available via
    ``StopIteration.value`` as ``(directions, coverage)``.
    """
    total_area = areas.sum()
    if total_area <= 0:
        yield 1.0
        return [], 0.0

    candidates = np.array(_fibonacci_sphere_points(n_candidates))
    backface_vis = normals @ candidates.T > 0.26
    n_faces = len(normals)
    vis = np.zeros((n_faces, n_candidates), dtype=bool)
    epsilon = 0.001
    BATCH = 5

    for j in range(n_candidates):
        cam_dir = candidates[j]
        for i in range(n_faces):
            if not backface_vis[i, j]:
                continue
            origin = centers[i] + normals[i] * epsilon
            if not _ray_occluded(bvh_trees, origin, cam_dir, 1e6):
                vis[i, j] = True
        if (j + 1) % BATCH == 0 or j == n_candidates - 1:
            yield (j + 1) / n_candidates

    return _greedy_select_from_visibility(
        vis, areas, max_cameras, coverage_target, candidates,
        existing_dirs=existing_dirs, normals=normals)


def _occ_tpr_generator(normals, areas, centers, bvh_trees,
                       max_cameras, coverage_target, n_candidates=200,
                       existing_dirs=None):
    """Generator: Two-Pass Refinement approach.

    Yields progress floats in [0, 1].  Final result is available via
    ``StopIteration.value`` as ``(directions, coverage)``.
    """
    total_area = areas.sum()
    if total_area <= 0:
        yield 1.0
        return [], 0.0

    candidates = np.array(_fibonacci_sphere_points(n_candidates))
    backface_vis = normals @ candidates.T > 0.26

    # ── Pass 1: fast greedy (back-face only, instant) ─────────────────
    covered_bf = np.zeros(len(areas), dtype=bool)
    # Pre-seed coverage from existing cameras
    if existing_dirs:
        for edir in existing_dirs:
            covered_bf |= normals @ np.asarray(edir, dtype=float) > 0.26
    selected_indices = []

    for _ in range(max_cameras):
        uncov = ~covered_bf
        new_vis = backface_vis & uncov[:, np.newaxis]
        new_areas_arr = (new_vis * areas[:, np.newaxis]).sum(axis=0)
        best = int(np.argmax(new_areas_arr))
        if new_areas_arr[best] < total_area * 0.005:
            break
        selected_indices.append(best)
        covered_bf |= backface_vis[:, best]
        if float(areas[covered_bf].sum() / total_area) >= coverage_target:
            break

    if not selected_indices:
        yield 1.0
        return [], 0.0

    yield 0.0  # pass-1 done

    # ── Phase 2a: BVH validate selected set (0 % → 30 %) ─────────────
    epsilon = 0.001
    true_covered = np.zeros(len(areas), dtype=bool)
    n_sel = len(selected_indices)
    for idx, ci in enumerate(selected_indices):
        cam_dir = candidates[ci]
        for fi in range(len(normals)):
            if not backface_vis[fi, ci]:
                continue
            origin = centers[fi] + normals[fi] * epsilon
            if not _ray_occluded(bvh_trees, origin, cam_dir, 1e6):
                true_covered[fi] = True
        yield 0.3 * (idx + 1) / n_sel

    # ── Phase 2b: patch phantom-uncovered faces (30 % → 100 %) ───────
    phantom_uncovered = covered_bf & ~true_covered

    if phantom_uncovered.any():
        remaining_budget = max_cameras - len(selected_indices)
        if remaining_budget > 0:
            used_set = set(selected_indices)
            unused_mask = np.array([i not in used_set
                                    for i in range(n_candidates)])
            unused_indices = np.where(unused_mask)[0]

            if len(unused_indices) > 0:
                phantom_indices = np.where(phantom_uncovered)[0]
                ph_normals = normals[phantom_indices]
                ph_centers = centers[phantom_indices]
                ph_areas = areas[phantom_indices]
                ph_bf = ph_normals @ candidates[unused_indices].T > 0.26

                ph_vis = np.zeros_like(ph_bf)
                n_unused = len(unused_indices)
                BATCH = 5
                for j_local, j_global in enumerate(unused_indices):
                    cam_dir = candidates[j_global]
                    for i_local in range(len(phantom_indices)):
                        if not ph_bf[i_local, j_local]:
                            continue
                        origin = (ph_centers[i_local]
                                  + ph_normals[i_local] * epsilon)
                        if not _ray_occluded(bvh_trees, origin,
                                             cam_dir, 1e6):
                            ph_vis[i_local, j_local] = True
                    if (j_local + 1) % BATCH == 0 or j_local == n_unused - 1:
                        yield 0.3 + 0.7 * (j_local + 1) / n_unused

                # Greedy on sub-matrix (instant)
                ph_covered = np.zeros(len(ph_areas), dtype=bool)
                for _ in range(remaining_budget):
                    uncov = ~ph_covered
                    nv = ph_vis & uncov[:, np.newaxis]
                    na = (nv * ph_areas[:, np.newaxis]).sum(axis=0)
                    best_local = int(np.argmax(na))
                    if na[best_local] < total_area * 0.005:
                        break
                    best_global = int(unused_indices[best_local])
                    selected_indices.append(best_global)
                    ph_covered |= ph_vis[:, best_local]
                    for i_local in range(len(phantom_indices)):
                        if ph_vis[i_local, best_local]:
                            true_covered[phantom_indices[i_local]] = True

    yield 1.0  # ensure caller sees 100 %
    selected = [candidates[ci].copy() for ci in selected_indices]
    final_cov = (float(areas[true_covered].sum() / total_area)
                 if true_covered.any() else 0.0)
    return selected, final_cov


def _sort_directions_spatially(directions, ref_direction=None):
    """Sort direction vectors by azimuth angle so cameras progress smoothly
    around the subject.  The camera whose azimuth is closest to
    *ref_direction* (typically the viewport look direction) becomes the first
    entry.  Important for sequential generation mode where each camera needs
    spatial context from the previous one."""
    if len(directions) <= 1:
        return directions
    # Azimuth = atan2(y, x) gives a smooth circular ordering
    angles = [math.atan2(float(d[1]), float(d[0])) for d in directions]
    paired = sorted(zip(angles, directions), key=lambda p: p[0])

    if ref_direction is not None:
        # Use full 3D angular distance (dot product) so elevation is
        # considered when choosing the start camera, not just azimuth.
        ref_v = mathutils.Vector(ref_direction).normalized()
        best_idx = 0
        best_dot = -2.0
        for idx, (_, d) in enumerate(paired):
            dot = mathutils.Vector(d).normalized().dot(ref_v)
            if dot > best_dot:
                best_dot = dot
                best_idx = idx
        # Rotate so closest-to-ref is first
        paired = paired[best_idx:] + paired[:best_idx]

    return [d for _, d in paired]


def _classify_camera_direction(cam_dir_np, ref_front_np):
    """Classify a camera direction into a human-readable view label.

    Uses Option-B scheme: elevation tiers first (top/bottom if >60°),
    then 4 azimuth quadrants (front/right/left/rear, 90° each),
    with 'from above'/'from below' modifiers for 30–60° elevation.

    Parameters
    ----------
    cam_dir_np : array-like, shape (3,)
        Centre-to-camera unit vector (where the camera is placed relative to
        the mesh centre).
    ref_front_np : array-like, shape (3,)
        The reference "front" direction (centre-to-camera direction that
        corresponds to the user's viewport, i.e. Camera_0's neighbourhood).

    Returns
    -------
    str
        A prompt-friendly label such as ``"front view"``,
        ``"right side view, from above"``, ``"top view"``, etc.
    """
    d = np.array(cam_dir_np, dtype=float)
    d /= max(np.linalg.norm(d), 1e-12)

    # --- Elevation (angle above / below the horizontal XY plane) ---
    elevation_rad = math.asin(np.clip(d[2], -1.0, 1.0))
    elevation_deg = math.degrees(elevation_rad)

    if elevation_deg > 60.0:
        return "top view"
    if elevation_deg < -60.0:
        return "bottom view"

    # --- Azimuth relative to ref_front (projected onto XY plane) ---
    ref = np.array(ref_front_np, dtype=float)
    # Project both onto XY
    d_h = np.array([d[0], d[1]], dtype=float)
    r_h = np.array([ref[0], ref[1]], dtype=float)
    d_h_len = np.linalg.norm(d_h)
    r_h_len = np.linalg.norm(r_h)
    if d_h_len < 1e-8 or r_h_len < 1e-8:
        # Nearly vertical – should have been caught by the elevation check,
        # but fall back just in case.
        return "top view" if d[2] >= 0 else "bottom view"
    d_h /= d_h_len
    r_h /= r_h_len

    # Signed angle: positive = counterclockwise from above = to the user's right
    sin_a = r_h[0] * d_h[1] - r_h[1] * d_h[0]
    cos_a = r_h[0] * d_h[0] + r_h[1] * d_h[1]
    azimuth_deg = math.degrees(math.atan2(sin_a, cos_a))

    # Quadrant classification (each 90°)
    abs_az = abs(azimuth_deg)
    if abs_az <= 45.0:
        base = "front view"
    elif abs_az >= 135.0:
        base = "rear view"
    elif azimuth_deg > 0:
        base = "right side view"
    else:
        base = "left side view"

    # Elevation modifier
    if elevation_deg > 30.0:
        return f"{base}, from above"
    if elevation_deg < -30.0:
        return f"{base}, from below"
    return base


def _compute_per_camera_aspect(direction_np, verts_world, center):
    """Compute the silhouette aspect ratio (width / height) from a single
    camera direction.  Returns the aspect ratio as a float."""
    right, up, _d = _camera_basis(direction_np)
    rel = verts_world - center
    proj_r = rel @ right
    proj_u = rel @ up
    w = max(float(proj_r.max() - proj_r.min()), 0.001)
    h = max(float(proj_u.max() - proj_u.min()), 0.001)
    return w / h


def _perspective_aspect(verts_world, cam_pos_np, cam_dir_np):
    """Compute the visible angular aspect ratio (width / height) as seen
    from a perspective camera at *cam_pos_np* looking along *cam_dir_np*."""
    right, up, d = _camera_basis(cam_dir_np)
    forward = -d

    rel = verts_world - cam_pos_np
    depth = rel @ forward  # positive = in front of camera
    depth = np.maximum(depth, 0.001)
    pr = rel @ right
    pu = rel @ up

    angle_r = np.arctan2(pr, depth)
    angle_u = np.arctan2(pu, depth)

    angular_w = float(angle_r.max() - angle_r.min())
    angular_h = float(angle_u.max() - angle_u.min())
    if angular_h < 0.001:
        return 1.0
    return angular_w / angular_h


def _resolution_from_aspect(aspect, total_px, align=8):
    """Compute (res_x, res_y) for a given *aspect* ratio (w/h) keeping
    approximately *total_px* pixels, snapped to *align*."""
    new_x = math.sqrt(total_px * aspect)
    new_y = total_px / new_x
    new_x = max(align, int(round(new_x / align)) * align)
    new_y = max(align, int(round(new_y / align)) * align)
    return new_x, new_y


def _get_resolution_align(context):
    """Use each Qwen encoder's image grid for camera resolution planning."""
    architecture = context.scene.model_architecture
    if architecture in QWEN_21_ARCHITECTURES:
        return 32
    if architecture in {'moss_image', 'fibo_edit'}:
        # Both use 16px latent patches (8x VAE with 2x2 patchify).
        return 16
    if (architecture == 'qwen_image_edit'
            and getattr(context.scene, 'qwen_rescale_alignment', False)):
        return 112
    return 8


def _apply_auto_aspect(directions_np, context, verts_world):
    """Adjust scene render resolution to match the mesh's average apparent
    aspect ratio across the given camera *directions_np*.  This is the
    'shared' mode — all cameras use the same resolution.
    Keeps total pixel count approximately constant and snaps to alignment step.
    Returns (new_res_x, new_res_y)."""
    center = verts_world.mean(axis=0)
    aspects = [_compute_per_camera_aspect(d, verts_world, center)
               for d in directions_np]
    avg_aspect = float(np.mean(aspects))

    render = context.scene.render
    total_px = render.resolution_x * render.resolution_y
    align = _get_resolution_align(context)
    new_x, new_y = _resolution_from_aspect(avg_aspect, total_px, align=align)
    render.resolution_x = new_x
    render.resolution_y = new_y
    return new_x, new_y


def _store_per_camera_resolution(cam_obj, res_x, res_y):
    """Store per-camera resolution as custom properties on the camera object."""
    cam_obj["sg_res_x"] = res_x
    cam_obj["sg_res_y"] = res_y


def _get_camera_resolution(cam_obj, scene):
    """Return (res_x, res_y) for a camera.  Falls back to scene render
    resolution if no per-camera resolution is stored."""
    if "sg_res_x" in cam_obj and "sg_res_y" in cam_obj:
        return int(cam_obj["sg_res_x"]), int(cam_obj["sg_res_y"])
    return scene.render.resolution_x, scene.render.resolution_y


class _SGCameraResolution:
    """Context manager: temporarily set scene render resolution to a camera's
    per-camera values (if stored), and restore the original on exit.

    Usage::

        with _SGCameraResolution(context, camera_obj):
            bpy.ops.render.render(write_still=True)
    """
    def __init__(self, context, cam_obj):
        self._render = context.scene.render
        self._cam = cam_obj
        self._scene = context.scene
        self._orig_x = self._render.resolution_x
        self._orig_y = self._render.resolution_y

    def __enter__(self):
        rx, ry = _get_camera_resolution(self._cam, self._scene)
        self._render.resolution_x = rx
        self._render.resolution_y = ry
        return self

    def __exit__(self, *exc):
        self._render.resolution_x = self._orig_x
        self._render.resolution_y = self._orig_y
        return False


def _get_fov(cam_settings, context, res_x=None, res_y=None):
    """Return (fov_x, fov_y) in radians for the given camera data block.
    If *res_x* / *res_y* are not provided, reads from scene render settings."""
    fov_x = cam_settings.angle_x
    if res_x is None:
        res_x = context.scene.render.resolution_x
    if res_y is None:
        res_y = context.scene.render.resolution_y
    if res_y > res_x:
        fov_x = 2 * math.atan(math.tan(fov_x / 2) * res_x / res_y)
    fov_y = 2 * math.atan(math.tan(fov_x / 2) * res_y / res_x)
    return fov_x, fov_y
