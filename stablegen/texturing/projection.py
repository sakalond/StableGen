import bpy
import os
from math import atan, tan, pi

from ..utils import get_last_material_index, get_file_path, get_dir_path
from .rendering import prepare_baking, bake_texture, unwrap
from ..cameras.geometry import _get_camera_resolution
from mathutils import Matrix, Vector


_SG_BUFFER_UV_NAME = "_SG_ProjectionBuffer"


def _is_projection_mesh(obj):
    """Consistent render/visibility filter for production geometry."""
    if obj.type != 'MESH' or obj.hide_get():
        return False
    if not getattr(bpy.context.scene, 'sg_rework_projection_geometry', False):
        return True
    if obj.hide_render:
        return False
    if obj.get("sg_rig_helper", False) or obj.get("sg_exclude_texturing", False):
        return False
    upper_name = obj.name.upper()
    if upper_name.startswith(("WGT-", "WGT_", "RIG_WIDGET", "CTRL_SHAPE")):
        return False
    return True


def _scene_projection_scale(context):
    corners = []
    for obj in context.view_layer.objects:
        if _is_projection_mesh(obj):
            corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not corners:
        return 1.0
    return max(
        (Vector((
            max(point.x for point in corners),
            max(point.y for point in corners),
            max(point.z for point in corners),
        )) - Vector((
            min(point.x for point in corners),
            min(point.y for point in corners),
            min(point.z for point in corners),
        ))).length,
        1e-6,
    )


def _ensure_lowpass_texture(image_path, sigma=8.0):
    """Create/update a low-pass companion used only for ownership seams."""
    if not image_path or not os.path.exists(image_path):
        return None
    root, extension = os.path.splitext(image_path)
    sigma_value = max(0.1, float(sigma))
    sigma_key = f"{sigma_value:.2f}".replace('.', '_')
    output_path = f"{root}_sg_lowpass_s{sigma_key}{extension or '.png'}"
    try:
        if (os.path.exists(output_path)
                and os.path.getmtime(output_path) >= os.path.getmtime(image_path)):
            return output_path
        import cv2
        source = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if source is None:
            return None
        result = source.copy()
        channels = result[:, :, :3] if result.ndim == 3 else result
        blurred = cv2.GaussianBlur(
            channels, (0, 0), sigmaX=sigma_value
        )
        if result.ndim == 3:
            result[:, :, :3] = blurred
        else:
            result = blurred
        cv2.imwrite(output_path, result)
        return output_path
    except Exception as error:
        print(f"[StableGen] Low-frequency seam texture failed: {error}")
        return None


def _camera_world_location(camera):
    """Return camera position in world space (including parent transforms)."""
    if getattr(bpy.context.scene, 'sg_rework_projection_geometry', False):
        return camera.matrix_world.translation.copy()
    return camera.location.copy()


def _scene_ray_epsilon(context):
    """Scale ray tolerances to the scene instead of assuming metre-sized assets."""
    if not getattr(context.scene, 'sg_rework_projection_geometry', False):
        # The legacy native graph used +0.01 ray length and +0.001 hit
        # tolerance; callers multiply this base by ten for ray length.
        return 0.001
    corners = []
    for obj in context.view_layer.objects:
        if not _is_projection_mesh(obj):
            continue
        corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not corners:
        return 1e-5
    min_corner = Vector((
        min(point.x for point in corners),
        min(point.y for point in corners),
        min(point.z for point in corners),
    ))
    max_corner = Vector((
        max(point.x for point in corners),
        max(point.y for point in corners),
        max(point.z for point in corners),
    ))
    return max((max_corner - min_corner).length * 1e-5, 1e-7)


def _copy_uv_to_attribute(obj, uv_layer_name, attr_name):
    """Copy UV data from a UV layer to a named FLOAT_VECTOR corner-domain attribute.
    Uses FLOAT_VECTOR (3-component) instead of FLOAT2 so the attribute does NOT
    appear in the UV Maps list (FLOAT2 + CORNER = UV map in Blender 3.5+).
    """
    uv_layer = obj.data.uv_layers.get(uv_layer_name)
    if not uv_layer:
        return
    n = len(obj.data.loops)
    uv_buf = [0.0] * (n * 2)
    uv_layer.data.foreach_get("uv", uv_buf)
    # Convert (u,v) pairs to (u,v,0) triples for FLOAT_VECTOR storage
    vec_buf = [0.0] * (n * 3)
    for i in range(n):
        vec_buf[i * 3] = uv_buf[i * 2]
        vec_buf[i * 3 + 1] = uv_buf[i * 2 + 1]
        # vec_buf[i * 3 + 2] already 0.0
    # Remove existing attribute if present
    existing = obj.data.attributes.get(attr_name)
    if existing:
        obj.data.attributes.remove(existing)
    attr = obj.data.attributes.new(name=attr_name, type='FLOAT_VECTOR', domain='CORNER')
    attr.data.foreach_set("vector", vec_buf)
    obj.data.update()


def _bake_visibility_weights(
    context,
    objects,
    cameras,
    mat_id,
    stop_index=1000000,
    angle_threshold=None,
    power_exponent=None,
):
    """Pre-compute per-vertex, per-camera visibility weights and store them
    as ``FLOAT`` ``POINT``-domain mesh attributes.

    Attribute naming: ``_SG_VisWeight_{cam_idx}_{mat_id}``

    The computation replicates the same logic as the runtime
    ``create_native_raycast_visibility`` + ``create_native_feather`` shader
    nodes but runs once in Python so the resulting material has no world-space
    dependencies — projected textures survive object transforms.

    Uses NumPy for bulk frustum/angle/feather computation and only falls back
    to per-vertex BVH raycasting for the subset that passes initial checks.
    """
    import bmesh
    import numpy as np
    from mathutils.bvhtree import BVHTree
    from math import pow as fpow

    scene = context.scene
    angle_threshold = (
        scene.discard_factor if angle_threshold is None else angle_threshold
    )
    power_exponent = (
        scene.weight_exponent if power_exponent is None else power_exponent
    )
    feather_active  = scene.visibility_vignette
    feather_width   = scene.visibility_vignette_width if feather_active else 0.0
    feather_gamma   = scene.visibility_vignette_softness if feather_active else 1.0
    feather_width   = max(0.0, min(feather_width, 0.49))

    # Pre-compute camera data once -----------------------------------------------
    cam_data = []
    for camera in cameras:
        cam_loc = _camera_world_location(camera)
        quat    = camera.matrix_world.to_quaternion()
        cam_dir = quat @ Vector((0, 0, -1))
        cam_up  = quat @ Vector((0, 1, 0))
        cam_right   = cam_dir.cross(cam_up).normalized()
        cam_up_vec  = cam_right.cross(cam_dir).normalized()

        cam_res_x, cam_res_y = _get_camera_resolution(camera, scene)
        fov = camera.data.angle_x
        if cam_res_y > cam_res_x:
            fov = 2 * atan(tan(fov / 2) * cam_res_x / cam_res_y)
        filmw  = tan(fov / 2)
        aspect = cam_res_x / cam_res_y
        filmh  = filmw / aspect
        camera_type = (
            camera.data.type
            if getattr(scene, 'sg_rework_projection_geometry', False)
            else 'PERSP'
        )
        ortho_half_h = max(float(camera.data.ortho_scale) * 0.5, 1e-8)
        ortho_half_w = ortho_half_h * aspect

        cam_data.append((
            np.array(cam_loc, dtype=np.float64),
            np.array(cam_dir, dtype=np.float64),
            np.array(cam_right, dtype=np.float64),
            np.array(cam_up_vec, dtype=np.float64),
            filmw, filmh,
            float(camera.data.shift_x) if getattr(scene, 'sg_rework_projection_geometry', False) else 0.0,
            float(camera.data.shift_y) if getattr(scene, 'sg_rework_projection_geometry', False) else 0.0,
            cam_loc,  # keep mathutils Vector for BVH calls
            camera_type, ortho_half_w, ortho_half_h,
        ))

    # Build scene-wide BVH for occlusion (matches ShaderNodeRaycast behaviour,
    # which checks ALL visible scene geometry, not just the current object).
    use_corrected_geometry = getattr(scene, 'sg_rework_projection_geometry', False)
    depsgraph = context.evaluated_depsgraph_get() if use_corrected_geometry else None
    scene_bm = bmesh.new()
    vert_offset = 0
    for scene_obj in context.view_layer.objects:
        if not _is_projection_mesh(scene_obj):
            continue
        tmp = bmesh.new()
        evaluated_obj = None
        if use_corrected_geometry:
            evaluated_obj = scene_obj.evaluated_get(depsgraph)
            evaluated_mesh = evaluated_obj.to_mesh()
            tmp.from_mesh(evaluated_mesh)
            tmp.transform(evaluated_obj.matrix_world)
        else:
            tmp.from_mesh(scene_obj.data)
            tmp.transform(scene_obj.matrix_world)
        for v in tmp.verts:
            scene_bm.verts.new(v.co)
        scene_bm.verts.ensure_lookup_table()
        for f in tmp.faces:
            try:
                scene_bm.faces.new(
                    [scene_bm.verts[v.index + vert_offset] for v in f.verts])
            except ValueError:
                pass  # skip degenerate / duplicate faces
        vert_offset += len(tmp.verts)
        tmp.free()
        if evaluated_obj is not None:
            evaluated_obj.to_mesh_clear()
    scene_bm.normal_update()
    scene_tree = BVHTree.FromBMesh(scene_bm)
    scene_bm.free()
    print(f"[StableGen] Scene BVH built ({vert_offset} verts across visible meshes)")

    # Per-object bake -------------------------------------------------------------
    n_cameras = len(cameras)
    for obj_idx, obj in enumerate(objects):
        bm = bmesh.new()
        evaluated_obj = None
        if use_corrected_geometry:
            evaluated_obj = obj.evaluated_get(depsgraph)
            evaluated_mesh = evaluated_obj.to_mesh()
            if len(evaluated_mesh.vertices) == len(obj.data.vertices):
                source_mesh = evaluated_mesh
                source_matrix = evaluated_obj.matrix_world
            else:
                print(
                    f"[StableGen] Warning: '{obj.name}' modifiers change vertex count; "
                    "visibility weights use the base mesh because POINT attributes "
                    "cannot be mapped safely to evaluated topology."
                )
                source_mesh = obj.data
                source_matrix = obj.matrix_world
        else:
            source_mesh = obj.data
            source_matrix = obj.matrix_world
        bm.from_mesh(source_mesh)
        bm.transform(source_matrix)
        bm.normal_update()

        n_verts = len(bm.verts)
        # Build numpy arrays (Nx3) for positions and normals
        positions_np = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
        normals_np   = np.array([v.normal[:] for v in bm.verts], dtype=np.float64)
        bm.free()
        if evaluated_obj is not None:
            evaluated_obj.to_mesh_clear()

        for cam_idx, camera in enumerate(cameras):
            print(f"[StableGen] Baking visibility: object {obj_idx+1}/{len(objects)}, "
                  f"camera {cam_idx+1}/{n_cameras} ({n_verts} verts)")

            weights = np.zeros(n_verts, dtype=np.float64)

            if cam_idx > stop_index:
                # Store zeros (camera disabled) and continue
                attr_name = f"_SG_VisWeight_{cam_idx}_{mat_id}"
                existing  = obj.data.attributes.get(attr_name)
                if existing:
                    obj.data.attributes.remove(existing)
                attr = obj.data.attributes.new(
                    name=attr_name, type='FLOAT', domain='POINT')
                attr.data.foreach_set("value", weights.tolist())
                continue

            (cam_loc_np, cam_dir_np, cam_right_np, cam_up_np, filmw, filmh,
             shift_x, shift_y, cam_loc_mu, camera_type,
             ortho_half_w, ortho_half_h) = cam_data[cam_idx]

            # ── Vectorised frustum + angle pre-filter ──────────────────
            to_vert = positions_np - cam_loc_np                     # (N, 3)
            if camera_type == 'ORTHO':
                dists = np.einsum('ij,j->i', to_vert, cam_dir_np)
                valid = dists > 1e-8
                direction = np.repeat(cam_dir_np[None, :], n_verts, axis=0)
                x_proj = np.einsum('ij,j->i', to_vert, cam_right_np)
                y_proj = np.einsum('ij,j->i', to_vert, cam_up_np)
                x_centered = x_proj / ortho_half_w - 2.0 * shift_x
                y_centered = y_proj / ortho_half_h - 2.0 * shift_y
                valid &= (np.abs(x_centered) < 1.0) & (np.abs(y_centered) < 1.0)
            elif camera_type == 'PERSP':
                dists = np.linalg.norm(to_vert, axis=1)
                valid = dists > 1e-8
                direction = np.zeros_like(to_vert)
                direction[valid] = to_vert[valid] / dists[valid, None]
                d_neg_fwd = np.einsum('ij,j->i', direction, -cam_dir_np)
                valid &= d_neg_fwd < 0
                neg_cam_z = np.where(valid, -d_neg_fwd, 1.0)
                d_right = np.einsum('ij,j->i', direction, cam_right_np)
                d_up = np.einsum('ij,j->i', direction, cam_up_np)
                x_proj = d_right / neg_cam_z
                y_proj = d_up / neg_cam_z
                x_centered = x_proj / max(filmw, 1e-12) - 2.0 * shift_x
                y_centered = y_proj / max(filmh, 1e-12) - 2.0 * shift_y
                valid &= (np.abs(x_centered) < 1.0) & (np.abs(y_centered) < 1.0)
            else:
                valid = np.zeros(n_verts, dtype=bool)
                dists = np.ones(n_verts, dtype=np.float64)
                direction = np.repeat(cam_dir_np[None, :], n_verts, axis=0)
                x_centered = np.zeros(n_verts)
                y_centered = np.zeros(n_verts)

            # Angle threshold
            raw_dot = np.einsum('ij,ij->i', direction, normals_np)
            if getattr(scene, 'sg_allow_backface_projection', False):
                dot_dn = np.abs(raw_dot)
            else:
                # ``direction`` points camera -> surface, so a visible front
                # face has a negative dot product with its outward normal.
                dot_dn = np.maximum(-raw_dot, 0.0)
            dot_dn    = np.clip(dot_dn, 0.0, 1.0)
            angle_deg = np.degrees(np.arccos(dot_dn))
            valid &= angle_deg < angle_threshold

            # ── BVH raycast only for candidates that passed filters ───
            candidate_indices = np.nonzero(valid)[0]
            occluded = np.zeros(n_verts, dtype=bool)
            ray_epsilon = _scene_ray_epsilon(context)
            for vi in candidate_indices:
                d_mu = Vector(direction[vi])
                ray_origin = cam_loc_mu
                if camera_type == 'ORTHO':
                    ray_origin = Vector(
                        positions_np[vi] - cam_dir_np * (dists[vi] + ray_epsilon * 10.0)
                    )
                hit_loc, _, _, hit_dist = scene_tree.ray_cast(
                    ray_origin, d_mu, float(dists[vi]) + ray_epsilon * (
                        20.0 if getattr(scene, 'sg_rework_projection_geometry', False) else 10.0
                    ))
                if hit_loc is not None and dists[vi] > hit_dist + ray_epsilon:
                    occluded[vi] = True

            valid &= ~occluded

            # ── Silhouette erosion ─────────────────────────────────────
            # Cast offset rays around each visible vertex.  If any offset
            # ray hits an occluder significantly closer, this vertex is
            # near a depth discontinuity (e.g. hand edge on body) and its
            # weight should be zeroed so another camera fills that area.
            silhouette_margin_px = getattr(scene, 'sg_silhouette_margin', 3)
            if silhouette_margin_px > 0:
                # Angular offset per pixel (radians)
                cam_res_x_n, cam_res_y_n = _get_camera_resolution(camera, scene)
                rad_per_px = fov / max(cam_res_x_n, 1)
                offset_angle = rad_per_px * silhouette_margin_px

                visible_indices = np.nonzero(valid)[0]
                near_silhouette = np.zeros(n_verts, dtype=bool)

                # Offset directions in camera space
                offsets_right = cam_right_np * offset_angle
                offsets_up    = cam_up_np * offset_angle

                use_8_rays = getattr(scene, 'sg_silhouette_rays', '4') == '8'
                if use_8_rays:
                    diag = offset_angle * 0.7071067811865476  # 1/sqrt(2)
                    offsets_diag_ru = cam_right_np * diag + cam_up_np * diag
                    offsets_diag_rd = cam_right_np * diag - cam_up_np * diag
                    all_offsets = (offsets_right, -offsets_right,
                                   offsets_up, -offsets_up,
                                   offsets_diag_ru, -offsets_diag_ru,
                                   offsets_diag_rd, -offsets_diag_rd)
                else:
                    all_offsets = (offsets_right, -offsets_right,
                                   offsets_up, -offsets_up)

                for vi in visible_indices:
                    d_base = direction[vi]
                    dist_v = float(dists[vi])
                    is_edge = False
                    for off in all_offsets:
                        d_off = d_base + off if camera_type == 'PERSP' else d_base
                        # Normalise (offset is small, so direction barely changes)
                        norm = float(np.linalg.norm(d_off))
                        if norm < 1e-8:
                            continue
                        d_off_mu = Vector(d_off / norm)
                        offset_origin = cam_loc_mu
                        if camera_type == 'ORTHO':
                            world_per_pixel = (2.0 * ortho_half_w) / max(cam_res_x_n, 1)
                            offset_world = off / max(offset_angle, 1e-12) * (
                                world_per_pixel * silhouette_margin_px
                            )
                            offset_origin = Vector(
                                positions_np[vi] - cam_dir_np * (dist_v + ray_epsilon * 10.0)
                                + offset_world
                            )
                        h_loc, _, _, h_dist = scene_tree.ray_cast(
                            offset_origin, d_off_mu, dist_v + ray_epsilon * (
                                20.0 if getattr(scene, 'sg_rework_projection_geometry', False) else 10.0
                            ))
                        if getattr(scene, 'sg_rework_projection_geometry', False):
                            scene_scale = _scene_projection_scale(context)
                            relative_depth = getattr(scene, 'sg_silhouette_depth', 0.05)
                            if camera_type == 'ORTHO':
                                pixel_footprint = (
                                    (2.0 * ortho_half_w) / max(cam_res_x_n, 1)
                                    * max(silhouette_margin_px, 1)
                                )
                            else:
                                pixel_footprint = (
                                    dist_v * rad_per_px * max(silhouette_margin_px, 1)
                                )
                            depth_thr = max(
                                scene_scale * relative_depth,
                                pixel_footprint * 0.5,
                            )
                        else:
                            depth_thr = getattr(scene, 'sg_silhouette_depth', 0.05)
                        if h_loc is not None and dist_v > h_dist + depth_thr:
                            is_edge = True
                            break
                    if is_edge:
                        near_silhouette[vi] = True

                valid &= ~near_silhouette

            # ── Vectorised weight + feather ────────────────────────────
            # Projected density estimates how many camera pixels describe a
            # model-scale span at this depth.  It penalizes far/stretched
            # coverage while remaining dimensionless across scene scales.
            cam_res_x_n, cam_res_y_n = _get_camera_resolution(camera, scene)
            if camera_type == 'ORTHO':
                projected_span = np.full(
                    n_verts,
                    _scene_projection_scale(context) * cam_res_x_n
                    / max(2.0 * ortho_half_w, 1e-8),
                )
            else:
                focal_pixels = cam_res_x_n / max(2.0 * filmw, 1e-12)
                projected_span = (
                    _scene_projection_scale(context) * focal_pixels
                    / np.maximum(dists, 1e-8)
                )
            density_confidence = np.clip(
                projected_span / max(0.5 * min(cam_res_x_n, cam_res_y_n), 1.0),
                0.0,
                1.0,
            )
            # Store unsharpened acquisition quality.  The material owns the
            # single authoritative SharpW exponent stage.
            w = np.maximum(dot_dn, 0.0) * density_confidence

            if feather_width > 0:
                nx   = np.abs(x_centered)
                ny   = np.abs(y_centered)
                edge = np.maximum(nx, ny)
                low  = 1.0 - feather_width
                need_feather = edge > low
                t  = np.clip((edge - low) / feather_width, 0.0, 1.0)
                ss = t * t * (3.0 - 2.0 * t)      # smoothstep
                ff = np.power(np.maximum(1.0 - ss, 0.0),
                              max(feather_gamma, 0.01))
                w  = np.where(need_feather, w * ff, w)

            weights[valid] = w[valid]

            # Store as FLOAT POINT-domain attribute on the *base* mesh
            attr_name = f"_SG_VisWeight_{cam_idx}_{mat_id}"
            existing  = obj.data.attributes.get(attr_name)
            if existing:
                obj.data.attributes.remove(existing)
            attr = obj.data.attributes.new(
                name=attr_name, type='FLOAT', domain='POINT')
            attr.data.foreach_set("value", weights.tolist())

            # Force UI update between cameras to keep Blender responsive
            if context.window:
                context.window.cursor_set('WAIT')
            for area in context.screen.areas:
                area.tag_redraw()

        obj.data.update()


def create_native_raycast_visibility(nodes, links, camera, geometry, context, i, mat_id, stop_index):
    """
    Creates a visibility weight node group using the native Raycast shader node (Blender 5.1+).
    Replicates the OSL raycast.osl shader logic:
      1. Frustum check (is the surface within the camera FOV)
      2. Occlusion check (Raycast from camera, compare Hit Distance with expected distance)
      3. Angle check (surface normal vs ray direction)
      4. Weight = pow(orthogonality, Power)
    
    Returns: (weight_output_node, subtract_node, normalize_node, length_node,
              camera_loc_node, camera_fov_node, camera_aspect_node, camera_dir_node, camera_up_node)
    """
    x_base = -400
    y_base = (-800) * i

    # --- Camera location (CombineXYZ) ---
    add_camera_loc = nodes.new("ShaderNodeCombineXYZ")
    add_camera_loc.location = (-600, 200 + 300 * i)
    camera_location = _camera_world_location(camera)
    add_camera_loc.inputs[0].default_value = camera_location.x
    add_camera_loc.inputs[1].default_value = camera_location.y
    add_camera_loc.inputs[2].default_value = camera_location.z

    # --- Direction: Geometry.Position - CameraLoc ---
    subtract = nodes.new("ShaderNodeVectorMath")
    subtract.operation = 'SUBTRACT'
    subtract.location = (x_base, -300 + y_base)
    subtract.inputs[1].default_value = camera_location

    normalize_node = nodes.new("ShaderNodeVectorMath")
    normalize_node.operation = 'NORMALIZE'
    normalize_node.location = (x_base, -500 + y_base)

    links.new(geometry.outputs["Position"], subtract.inputs[0])
    links.new(subtract.outputs["Vector"], normalize_node.inputs[0])

    # --- Length (camera-to-surface distance) ---
    length_node = nodes.new("ShaderNodeVectorMath")
    length_node.operation = 'LENGTH'
    length_node.location = (x_base, 200 * (i + 1))
    links.new(subtract.outputs["Vector"], length_node.inputs[0])

    # --- Camera FOV ---
    camera_fov_node = nodes.new("ShaderNodeValue")
    camera_fov_node.location = (-800, 200 + 300 * i)
    cam_res_x, cam_res_y = _get_camera_resolution(camera, context.scene)
    fov = camera.data.angle_x
    if cam_res_y > cam_res_x:
        fov = 2 * atan(tan(fov / 2) * cam_res_x / cam_res_y)
    camera_fov_node.outputs[0].default_value = fov

    # --- Camera aspect ratio ---
    camera_aspect_node = nodes.new("ShaderNodeValue")
    camera_aspect_node.location = (-800, 100 + 300 * i)
    camera_aspect_node.outputs[0].default_value = cam_res_x / cam_res_y

    # --- Camera direction (CombineXYZ) ---
    cam_dir_vec = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    camera_dir_node = nodes.new("ShaderNodeCombineXYZ")
    camera_dir_node.location = (-800, 0 + 300 * i)
    camera_dir_node.inputs[0].default_value = cam_dir_vec.x
    camera_dir_node.inputs[1].default_value = cam_dir_vec.y
    camera_dir_node.inputs[2].default_value = cam_dir_vec.z

    # --- Camera up (CombineXYZ) ---
    cam_up_vec = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    camera_up_node = nodes.new("ShaderNodeCombineXYZ")
    camera_up_node.location = (-800, -100 + 300 * i)
    camera_up_node.inputs[0].default_value = cam_up_vec.x
    camera_up_node.inputs[1].default_value = cam_up_vec.y
    camera_up_node.inputs[2].default_value = cam_up_vec.z

    # ========================================
    # FRUSTUM CHECK (using vector math nodes)
    # ========================================
    norm_forward = nodes.new("ShaderNodeVectorMath")
    norm_forward.operation = 'NORMALIZE'
    norm_forward.location = (-700, y_base - 100)
    links.new(camera_dir_node.outputs["Vector"], norm_forward.inputs[0])

    cross_fwd_up = nodes.new("ShaderNodeVectorMath")
    cross_fwd_up.operation = 'CROSS_PRODUCT'
    cross_fwd_up.location = (-700, y_base - 200)
    links.new(norm_forward.outputs["Vector"], cross_fwd_up.inputs[0])
    links.new(camera_up_node.outputs["Vector"], cross_fwd_up.inputs[1])

    norm_right = nodes.new("ShaderNodeVectorMath")
    norm_right.operation = 'NORMALIZE'
    norm_right.location = (-600, y_base - 200)
    links.new(cross_fwd_up.outputs["Vector"], norm_right.inputs[0])

    cross_right_fwd = nodes.new("ShaderNodeVectorMath")
    cross_right_fwd.operation = 'CROSS_PRODUCT'
    cross_right_fwd.location = (-700, y_base - 300)
    links.new(norm_right.outputs["Vector"], cross_right_fwd.inputs[0])
    links.new(norm_forward.outputs["Vector"], cross_right_fwd.inputs[1])

    norm_up_vec = nodes.new("ShaderNodeVectorMath")
    norm_up_vec.operation = 'NORMALIZE'
    norm_up_vec.location = (-600, y_base - 300)
    links.new(cross_right_fwd.outputs["Vector"], norm_up_vec.inputs[0])

    neg_forward = nodes.new("ShaderNodeVectorMath")
    neg_forward.operation = 'SCALE'
    neg_forward.location = (-600, y_base - 400)
    neg_forward.inputs["Scale"].default_value = -1.0
    links.new(norm_forward.outputs["Vector"], neg_forward.inputs[0])

    dot_d_right = nodes.new("ShaderNodeVectorMath")
    dot_d_right.operation = 'DOT_PRODUCT'
    dot_d_right.location = (-400, y_base - 200)
    links.new(normalize_node.outputs["Vector"], dot_d_right.inputs[0])
    links.new(norm_right.outputs["Vector"], dot_d_right.inputs[1])

    dot_d_up = nodes.new("ShaderNodeVectorMath")
    dot_d_up.operation = 'DOT_PRODUCT'
    dot_d_up.location = (-400, y_base - 300)
    links.new(normalize_node.outputs["Vector"], dot_d_up.inputs[0])
    links.new(norm_up_vec.outputs["Vector"], dot_d_up.inputs[1])

    dot_d_neg_fwd = nodes.new("ShaderNodeVectorMath")
    dot_d_neg_fwd.operation = 'DOT_PRODUCT'
    dot_d_neg_fwd.location = (-400, y_base - 400)
    links.new(normalize_node.outputs["Vector"], dot_d_neg_fwd.inputs[0])
    links.new(neg_forward.outputs["Vector"], dot_d_neg_fwd.inputs[1])

    cam_z_check = nodes.new("ShaderNodeMath")
    cam_z_check.operation = 'LESS_THAN'
    cam_z_check.location = (-200, y_base - 400)
    cam_z_check.inputs[1].default_value = 0.0
    links.new(dot_d_neg_fwd.outputs["Value"], cam_z_check.inputs[0])

    fov_half = nodes.new("ShaderNodeMath")
    fov_half.operation = 'MULTIPLY'
    fov_half.location = (-400, y_base - 500)
    fov_half.inputs[1].default_value = 0.5
    links.new(camera_fov_node.outputs[0], fov_half.inputs[0])

    filmw = nodes.new("ShaderNodeMath")
    filmw.operation = 'TANGENT'
    filmw.location = (-300, y_base - 500)
    links.new(fov_half.outputs[0], filmw.inputs[0])

    filmh = nodes.new("ShaderNodeMath")
    filmh.operation = 'DIVIDE'
    filmh.location = (-200, y_base - 500)
    links.new(filmw.outputs[0], filmh.inputs[0])
    links.new(camera_aspect_node.outputs[0], filmh.inputs[1])

    neg_cam_z = nodes.new("ShaderNodeMath")
    neg_cam_z.operation = 'MULTIPLY'
    neg_cam_z.location = (-200, y_base - 600)
    neg_cam_z.inputs[1].default_value = -1.0
    links.new(dot_d_neg_fwd.outputs["Value"], neg_cam_z.inputs[0])

    x_proj = nodes.new("ShaderNodeMath")
    x_proj.operation = 'DIVIDE'
    x_proj.location = (-100, y_base - 200)
    links.new(dot_d_right.outputs["Value"], x_proj.inputs[0])
    links.new(neg_cam_z.outputs[0], x_proj.inputs[1])

    y_proj = nodes.new("ShaderNodeMath")
    y_proj.operation = 'DIVIDE'
    y_proj.location = (-100, y_base - 300)
    links.new(dot_d_up.outputs["Value"], y_proj.inputs[0])
    links.new(neg_cam_z.outputs[0], y_proj.inputs[1])

    shift_x_amount = nodes.new("ShaderNodeMath")
    shift_x_amount.operation = 'MULTIPLY'
    shift_x_amount.inputs[1].default_value = (
        2.0 * float(camera.data.shift_x)
        if getattr(context.scene, 'sg_rework_projection_geometry', False) else 0.0
    )
    links.new(filmw.outputs[0], shift_x_amount.inputs[0])
    centered_x = nodes.new("ShaderNodeMath")
    centered_x.operation = 'SUBTRACT'
    links.new(x_proj.outputs[0], centered_x.inputs[0])
    links.new(shift_x_amount.outputs[0], centered_x.inputs[1])

    shift_y_amount = nodes.new("ShaderNodeMath")
    shift_y_amount.operation = 'MULTIPLY'
    shift_y_amount.inputs[1].default_value = (
        2.0 * float(camera.data.shift_y)
        if getattr(context.scene, 'sg_rework_projection_geometry', False) else 0.0
    )
    links.new(filmh.outputs[0], shift_y_amount.inputs[0])
    centered_y = nodes.new("ShaderNodeMath")
    centered_y.operation = 'SUBTRACT'
    links.new(y_proj.outputs[0], centered_y.inputs[0])
    links.new(shift_y_amount.outputs[0], centered_y.inputs[1])

    abs_x = nodes.new("ShaderNodeMath")
    abs_x.operation = 'ABSOLUTE'
    abs_x.location = (0, y_base - 200)
    links.new(centered_x.outputs[0], abs_x.inputs[0])

    x_in_bounds = nodes.new("ShaderNodeMath")
    x_in_bounds.operation = 'LESS_THAN'
    x_in_bounds.location = (100, y_base - 200)
    links.new(abs_x.outputs[0], x_in_bounds.inputs[0])
    links.new(filmw.outputs[0], x_in_bounds.inputs[1])

    abs_y = nodes.new("ShaderNodeMath")
    abs_y.operation = 'ABSOLUTE'
    abs_y.location = (0, y_base - 300)
    links.new(centered_y.outputs[0], abs_y.inputs[0])

    y_in_bounds = nodes.new("ShaderNodeMath")
    y_in_bounds.operation = 'LESS_THAN'
    y_in_bounds.location = (100, y_base - 300)
    links.new(abs_y.outputs[0], y_in_bounds.inputs[0])
    links.new(filmh.outputs[0], y_in_bounds.inputs[1])

    frustum_xy = nodes.new("ShaderNodeMath")
    frustum_xy.operation = 'MINIMUM'
    frustum_xy.location = (200, y_base - 250)
    links.new(x_in_bounds.outputs[0], frustum_xy.inputs[0])
    links.new(y_in_bounds.outputs[0], frustum_xy.inputs[1])

    frustum_ok = nodes.new("ShaderNodeMath")
    frustum_ok.operation = 'MINIMUM'
    frustum_ok.location = (300, y_base - 300)
    links.new(cam_z_check.outputs[0], frustum_ok.inputs[0])
    links.new(frustum_xy.outputs[0], frustum_ok.inputs[1])

    # ========================================
    # RAYCAST NODE (occlusion check)
    # ========================================
    raycast = nodes.new("ShaderNodeRaycast")
    raycast.location = (x_base + 200, y_base)
    links.new(add_camera_loc.outputs["Vector"], raycast.inputs["Position"])
    links.new(normalize_node.outputs["Vector"], raycast.inputs["Direction"])
    threshold_plus = nodes.new("ShaderNodeMath")
    threshold_plus.operation = 'ADD'
    threshold_plus.location = (x_base + 100, y_base + 100)
    ray_epsilon = _scene_ray_epsilon(context)
    threshold_plus.inputs[1].default_value = ray_epsilon * 10.0
    links.new(length_node.outputs["Value"], threshold_plus.inputs[0])
    links.new(threshold_plus.outputs[0], raycast.inputs["Length"])

    hit_dist_plus = nodes.new("ShaderNodeMath")
    hit_dist_plus.operation = 'ADD'
    hit_dist_plus.location = (x_base + 400, y_base + 50)
    hit_dist_plus.inputs[1].default_value = ray_epsilon
    links.new(raycast.outputs["Hit Distance"], hit_dist_plus.inputs[0])

    dist_check = nodes.new("ShaderNodeMath")
    dist_check.operation = 'LESS_THAN'
    dist_check.location = (x_base + 500, y_base + 50)
    links.new(length_node.outputs["Value"], dist_check.inputs[0])
    links.new(hit_dist_plus.outputs[0], dist_check.inputs[1])

    occlusion_ok = nodes.new("ShaderNodeMath")
    occlusion_ok.operation = 'MINIMUM'
    occlusion_ok.location = (x_base + 600, y_base + 50)
    links.new(raycast.outputs["Is Hit"], occlusion_ok.inputs[0])
    links.new(dist_check.outputs[0], occlusion_ok.inputs[1])

    # ========================================
    # ANGLE CHECK + WEIGHT
    # ========================================
    dot_d_normal = nodes.new("ShaderNodeVectorMath")
    dot_d_normal.operation = 'DOT_PRODUCT'
    dot_d_normal.location = (x_base + 300, y_base - 150)
    links.new(normalize_node.outputs["Vector"], dot_d_normal.inputs[0])
    links.new(geometry.outputs["Normal"], dot_d_normal.inputs[1])

    clamp_dot = nodes.new("ShaderNodeMath")
    clamp_dot.operation = 'MAXIMUM'
    clamp_dot.location = (x_base + 400, y_base - 150)
    clamp_dot.inputs[1].default_value = -1.0
    links.new(dot_d_normal.outputs["Value"], clamp_dot.inputs[0])

    clamp_dot2 = nodes.new("ShaderNodeMath")
    clamp_dot2.operation = 'MINIMUM'
    clamp_dot2.location = (x_base + 500, y_base - 150)
    clamp_dot2.inputs[1].default_value = 1.0
    links.new(clamp_dot.outputs[0], clamp_dot2.inputs[0])

    facing_dot = nodes.new("ShaderNodeMath")
    facing_dot.location = (x_base + 600, y_base - 150)
    if getattr(context.scene, 'sg_allow_backface_projection', False):
        facing_dot.operation = 'ABSOLUTE'
        links.new(clamp_dot2.outputs[0], facing_dot.inputs[0])
    else:
        # camera->surface dot outward-normal is negative for a front face.
        facing_dot.operation = 'MULTIPLY'
        facing_dot.inputs[1].default_value = -1.0
        links.new(clamp_dot2.outputs[0], facing_dot.inputs[0])

        facing_clamp = nodes.new("ShaderNodeMath")
        facing_clamp.operation = 'MAXIMUM'
        facing_clamp.location = (x_base + 650, y_base - 200)
        facing_clamp.inputs[1].default_value = 0.0
        links.new(facing_dot.outputs[0], facing_clamp.inputs[0])
        facing_dot = facing_clamp

    acos_node = nodes.new("ShaderNodeMath")
    acos_node.operation = 'ARCCOSINE'
    acos_node.location = (x_base + 700, y_base - 150)
    links.new(facing_dot.outputs[0], acos_node.inputs[0])

    to_degrees = nodes.new("ShaderNodeMath")
    to_degrees.operation = 'MULTIPLY'
    to_degrees.location = (x_base + 800, y_base - 150)
    to_degrees.inputs[1].default_value = 180.0 / pi
    links.new(acos_node.outputs[0], to_degrees.inputs[0])

    angle_check = nodes.new("ShaderNodeMath")
    angle_check.operation = 'LESS_THAN'
    angle_check.location = (x_base + 900, y_base - 150)
    angle_check.inputs[1].default_value = context.scene.discard_factor
    angle_check.label = f"AngleThreshold-{i}-{mat_id}"
    links.new(to_degrees.outputs[0], angle_check.inputs[0])

    power_node = nodes.new("ShaderNodeMath")
    power_node.operation = 'POWER'
    power_node.location = (x_base + 700, y_base - 250)
    power_node.inputs[1].default_value = context.scene.weight_exponent
    power_node.label = 'power_weight'
    links.new(facing_dot.outputs[0], power_node.inputs[0])

    # ========================================
    # COMBINE ALL: frustum_ok * occlusion_ok * angle_ok * weight
    # ========================================
    combine1 = nodes.new("ShaderNodeMath")
    combine1.operation = 'MULTIPLY'
    combine1.location = (x_base + 800, y_base)
    links.new(frustum_ok.outputs[0], combine1.inputs[0])
    links.new(occlusion_ok.outputs[0], combine1.inputs[1])

    combine2 = nodes.new("ShaderNodeMath")
    combine2.operation = 'MULTIPLY'
    combine2.location = (x_base + 900, y_base)
    links.new(combine1.outputs[0], combine2.inputs[0])
    links.new(angle_check.outputs[0], combine2.inputs[1])

    final_weight = nodes.new("ShaderNodeMath")
    final_weight.operation = 'MULTIPLY'
    final_weight.location = (x_base + 1000, y_base)
    final_weight.label = f"Angle-{i}-{mat_id}"
    links.new(combine2.outputs[0], final_weight.inputs[0])
    links.new(power_node.outputs[0], final_weight.inputs[1])

    return (final_weight, subtract, normalize_node, length_node,
            add_camera_loc, camera_fov_node, camera_aspect_node,
            camera_dir_node, camera_up_node)


def create_native_feather(nodes, links, normalize_node, camera_fov_node, camera_aspect_node,
                          camera_dir_node, camera_up_node, camera, context, i, mat_id):
    """
    Creates a native node equivalent of feather.osl for Blender 5.1+.
    Computes edge feathering based on distance to camera frustum border.
    
    Returns the feather output node (ShaderNodeMath).
    """
    x_base = -400
    y_base = (-800) * i - 400  # Offset below the raycast nodes

    # Build camera basis (reuse from raycast, but we need the forward/right/up)
    norm_forward = nodes.new("ShaderNodeVectorMath")
    norm_forward.operation = 'NORMALIZE'
    norm_forward.location = (x_base - 300, y_base)
    links.new(camera_dir_node.outputs["Vector"], norm_forward.inputs[0])

    cross_fwd_up = nodes.new("ShaderNodeVectorMath")
    cross_fwd_up.operation = 'CROSS_PRODUCT'
    cross_fwd_up.location = (x_base - 300, y_base - 100)
    links.new(norm_forward.outputs["Vector"], cross_fwd_up.inputs[0])
    links.new(camera_up_node.outputs["Vector"], cross_fwd_up.inputs[1])

    norm_right = nodes.new("ShaderNodeVectorMath")
    norm_right.operation = 'NORMALIZE'
    norm_right.location = (x_base - 200, y_base - 100)
    links.new(cross_fwd_up.outputs["Vector"], norm_right.inputs[0])

    cross_right_fwd = nodes.new("ShaderNodeVectorMath")
    cross_right_fwd.operation = 'CROSS_PRODUCT'
    cross_right_fwd.location = (x_base - 300, y_base - 200)
    links.new(norm_right.outputs["Vector"], cross_right_fwd.inputs[0])
    links.new(norm_forward.outputs["Vector"], cross_right_fwd.inputs[1])

    norm_up_vec = nodes.new("ShaderNodeVectorMath")
    norm_up_vec.operation = 'NORMALIZE'
    norm_up_vec.location = (x_base - 200, y_base - 200)
    links.new(cross_right_fwd.outputs["Vector"], norm_up_vec.inputs[0])

    neg_forward = nodes.new("ShaderNodeVectorMath")
    neg_forward.operation = 'SCALE'
    neg_forward.location = (x_base - 200, y_base - 300)
    neg_forward.inputs["Scale"].default_value = -1.0
    links.new(norm_forward.outputs["Vector"], neg_forward.inputs[0])

    # cam_x = dot(d, right)
    dot_d_right = nodes.new("ShaderNodeVectorMath")
    dot_d_right.operation = 'DOT_PRODUCT'
    dot_d_right.location = (x_base, y_base - 100)
    links.new(normalize_node.outputs["Vector"], dot_d_right.inputs[0])
    links.new(norm_right.outputs["Vector"], dot_d_right.inputs[1])

    # cam_y = dot(d, upVec)
    dot_d_up = nodes.new("ShaderNodeVectorMath")
    dot_d_up.operation = 'DOT_PRODUCT'
    dot_d_up.location = (x_base, y_base - 200)
    links.new(normalize_node.outputs["Vector"], dot_d_up.inputs[0])
    links.new(norm_up_vec.outputs["Vector"], dot_d_up.inputs[1])

    # cam_z = dot(d, -forward)
    dot_d_neg_fwd = nodes.new("ShaderNodeVectorMath")
    dot_d_neg_fwd.operation = 'DOT_PRODUCT'
    dot_d_neg_fwd.location = (x_base, y_base - 300)
    links.new(normalize_node.outputs["Vector"], dot_d_neg_fwd.inputs[0])
    links.new(neg_forward.outputs["Vector"], dot_d_neg_fwd.inputs[1])

    # cam_z >= 0 check (behind camera -> result = 0)
    cam_z_check = nodes.new("ShaderNodeMath")
    cam_z_check.operation = 'LESS_THAN'
    cam_z_check.location = (x_base + 100, y_base - 300)
    cam_z_check.inputs[1].default_value = 0.0
    links.new(dot_d_neg_fwd.outputs["Value"], cam_z_check.inputs[0])

    # filmw = tan(FOV * 0.5)
    fov_half = nodes.new("ShaderNodeMath")
    fov_half.operation = 'MULTIPLY'
    fov_half.location = (x_base, y_base - 400)
    fov_half.inputs[1].default_value = 0.5
    links.new(camera_fov_node.outputs[0], fov_half.inputs[0])

    filmw = nodes.new("ShaderNodeMath")
    filmw.operation = 'TANGENT'
    filmw.location = (x_base + 100, y_base - 400)
    links.new(fov_half.outputs[0], filmw.inputs[0])

    # filmh = filmw / CameraAspect
    filmh = nodes.new("ShaderNodeMath")
    filmh.operation = 'DIVIDE'
    filmh.location = (x_base + 200, y_base - 400)
    links.new(filmw.outputs[0], filmh.inputs[0])
    links.new(camera_aspect_node.outputs[0], filmh.inputs[1])

    # neg_cam_z = -cam_z
    neg_cam_z = nodes.new("ShaderNodeMath")
    neg_cam_z.operation = 'MULTIPLY'
    neg_cam_z.location = (x_base + 100, y_base - 500)
    neg_cam_z.inputs[1].default_value = -1.0
    links.new(dot_d_neg_fwd.outputs["Value"], neg_cam_z.inputs[0])

    # x_proj = cam_x / -cam_z
    x_proj = nodes.new("ShaderNodeMath")
    x_proj.operation = 'DIVIDE'
    x_proj.location = (x_base + 200, y_base - 100)
    links.new(dot_d_right.outputs["Value"], x_proj.inputs[0])
    links.new(neg_cam_z.outputs[0], x_proj.inputs[1])

    # y_proj = cam_y / -cam_z
    y_proj = nodes.new("ShaderNodeMath")
    y_proj.operation = 'DIVIDE'
    y_proj.location = (x_base + 200, y_base - 200)
    links.new(dot_d_up.outputs["Value"], y_proj.inputs[0])
    links.new(neg_cam_z.outputs[0], y_proj.inputs[1])

    shift_x_amount = nodes.new("ShaderNodeMath")
    shift_x_amount.operation = 'MULTIPLY'
    shift_x_amount.inputs[1].default_value = (
        2.0 * float(camera.data.shift_x)
        if getattr(context.scene, 'sg_rework_projection_geometry', False) else 0.0
    )
    links.new(filmw.outputs[0], shift_x_amount.inputs[0])
    centered_x = nodes.new("ShaderNodeMath")
    centered_x.operation = 'SUBTRACT'
    links.new(x_proj.outputs[0], centered_x.inputs[0])
    links.new(shift_x_amount.outputs[0], centered_x.inputs[1])

    shift_y_amount = nodes.new("ShaderNodeMath")
    shift_y_amount.operation = 'MULTIPLY'
    shift_y_amount.inputs[1].default_value = (
        2.0 * float(camera.data.shift_y)
        if getattr(context.scene, 'sg_rework_projection_geometry', False) else 0.0
    )
    links.new(filmh.outputs[0], shift_y_amount.inputs[0])
    centered_y = nodes.new("ShaderNodeMath")
    centered_y.operation = 'SUBTRACT'
    links.new(y_proj.outputs[0], centered_y.inputs[0])
    links.new(shift_y_amount.outputs[0], centered_y.inputs[1])

    # nx = abs(x_proj) / filmw
    abs_x = nodes.new("ShaderNodeMath")
    abs_x.operation = 'ABSOLUTE'
    abs_x.location = (x_base + 300, y_base - 100)
    links.new(centered_x.outputs[0], abs_x.inputs[0])

    nx = nodes.new("ShaderNodeMath")
    nx.operation = 'DIVIDE'
    nx.location = (x_base + 400, y_base - 100)
    links.new(abs_x.outputs[0], nx.inputs[0])
    links.new(filmw.outputs[0], nx.inputs[1])

    # ny = abs(y_proj) / filmh
    abs_y = nodes.new("ShaderNodeMath")
    abs_y.operation = 'ABSOLUTE'
    abs_y.location = (x_base + 300, y_base - 200)
    links.new(centered_y.outputs[0], abs_y.inputs[0])

    ny = nodes.new("ShaderNodeMath")
    ny.operation = 'DIVIDE'
    ny.location = (x_base + 400, y_base - 200)
    links.new(abs_y.outputs[0], ny.inputs[0])
    links.new(filmh.outputs[0], ny.inputs[1])

    # edge = max(nx, ny)
    edge = nodes.new("ShaderNodeMath")
    edge.operation = 'MAXIMUM'
    edge.location = (x_base + 500, y_base - 150)
    links.new(nx.outputs[0], edge.inputs[0])
    links.new(ny.outputs[0], edge.inputs[1])

    # Get feather parameters
    if context.scene.visibility_vignette:
        feather_val = context.scene.visibility_vignette_width
        gamma_val = context.scene.visibility_vignette_softness
    else:
        feather_val = 0.0
        gamma_val = 1.0

    feather_val = max(0.0, min(feather_val, 0.49))

    if feather_val > 0.0:
        # edge_factor = 1.0 - smoothstep(1.0 - feather, 1.0, edge)
        # Approximate smoothstep using a Map Range node with SMOOTHSTEP interpolation
        map_range = nodes.new("ShaderNodeMapRange")
        map_range.location = (x_base + 600, y_base - 150)
        map_range.interpolation_type = 'SMOOTHSTEP'
        map_range.inputs["From Min"].default_value = 1.0 - feather_val
        map_range.inputs["From Max"].default_value = 1.0
        map_range.inputs["To Min"].default_value = 1.0  # Invert: edge=low -> factor=1
        map_range.inputs["To Max"].default_value = 0.0  # edge=high -> factor=0
        links.new(edge.outputs[0], map_range.inputs["Value"])

        # edge_factor = pow(smoothstep_result, gamma)
        pow_node = nodes.new("ShaderNodeMath")
        pow_node.operation = 'POWER'
        pow_node.location = (x_base + 700, y_base - 150)
        pow_node.inputs[1].default_value = max(0.01, gamma_val)
        links.new(map_range.outputs[0], pow_node.inputs[0])

        # Multiply by cam_z_check (0 if behind camera)
        final_feather = nodes.new("ShaderNodeMath")
        final_feather.operation = 'MULTIPLY'
        final_feather.location = (x_base + 800, y_base - 150)
        final_feather.label = f"Feather-{i}-{mat_id}"
        links.new(cam_z_check.outputs[0], final_feather.inputs[0])
        links.new(pow_node.outputs[0], final_feather.inputs[1])
    else:
        # No feathering: result = cam_z_check (1 if in front, 0 behind)
        final_feather = cam_z_check
        final_feather.label = f"Feather-{i}-{mat_id}"

    return final_feather


_SG_VORONOI_PLACEHOLDER_NAME = "_SG_Voronoi_Placeholder"

def _get_voronoi_placeholder(color=(1.0, 0.0, 1.0)):
    """Get or create a 1x1 solid-colour image used as placeholder for
    non-generated cameras in Voronoi projection mode.  The colour is
    updated every call so it always matches the current setting."""
    img = bpy.data.images.get(_SG_VORONOI_PLACEHOLDER_NAME)
    if img is None:
        img = bpy.data.images.new(_SG_VORONOI_PLACEHOLDER_NAME, width=1, height=1)
        img.colorspace_settings.name = 'Non-Color'
        img.pack()
    img.pixels[:] = [color[0], color[1], color[2], 1.0]
    return img


def project_image(
    context,
    to_project,
    mat_id,
    stop_index=1000000,
    cameras=None,
    view_plan=None,
):
    """     
    Projects an image onto all mesh objects using UV Project Modifier.     
    """
    def build_mix_tree(shaders, weight_nodes, nodes_collection, links, last_node=None, level=0, stop_index=1000000):
        """
        Recursively builds a binary tree of mix shader nodes.
        The mix factor between two shader nodes is computed dynamically using
        the outputs of weight_nodes assumed to come from OSL/maths nodes.
        
        For each pair, the total weight is created using an ADD node.
        Then a DIVIDE node computes:
            mix_factor = weight_A / (weight_A + weight_B)
        which is connected to the Mix Shader node's Fac input.
        
        Params:
        shaders: list of shader nodes (e.g., Principled BSDF nodes).
        weight_nodes: list of nodes whose outputs provide the dynamic weight values.
                        (e.g. the less_than nodes outputs)
        nodes_collection: the node tree's nodes collection.
        links: the node tree's links collection.
        Returns:
        A tuple (final_shader, final_weight_node).
        """

        def _combine_angle_feather(nodes_col, lnk, cr_angle_out, cr_feather_out, loc):
            """Combine angle-ramp and feather-ramp outputs into a single
            visibility weight: ``angle_vis × feather_vis``.

            Returns the output node of the final MULTIPLY.
            """
            x, y = loc
            mult = nodes_col.new("ShaderNodeMath")
            mult.operation = 'MULTIPLY'
            mult.location = (x, y)
            lnk.new(cr_angle_out, mult.inputs[0])
            lnk.new(cr_feather_out, mult.inputs[1])
            return mult

        # Compute offsets based on recursion level
        is_local_edit = (context.scene.generation_method == 'local_edit' or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'local_edit'))
        if is_local_edit:
            last_node_x = last_node.location[0] if last_node else 0
            x_offset = 1000 + level * 800 + last_node_x + 200 if context.scene.early_priority else 1000 + level * 600 + last_node_x + 200
        else:
            x_offset = 1000 + level * 800 if context.scene.early_priority else 1000 + level * 600
        y_offset = 0
        if len(shaders) == 1:
            # Base-case: Mix single color with fallback magenta
            final_mix = nodes_collection.new("ShaderNodeMixRGB")
            final_mix.location = (x_offset - 200, y_offset)
            # Convert fallback_color from 3 to 4 components by appending alpha 1.0
            final_mix.inputs["Color2"].default_value = (*context.scene.fallback_color, 1.0)
            # Connect the single color output to Color1
            links.new(shaders[0].outputs[0], final_mix.inputs["Color1"])
            if last_node:
                # Connect the final mix node to the last node
                links.new(last_node.outputs[0], final_mix.inputs["Color2"])
            # Create a compare node to drive the mix factor
            
            if is_local_edit:
                w_node = weight_nodes[0]
                if isinstance(w_node, tuple):
                    angle_node = w_node[0]
                    feather_node = w_node[1]
                    ef_node = w_node[2] if len(w_node) >= 3 else None
                    
                    # Create Ramps (Output Visibility: 0=Invis, 1=Vis)
                    # Angle Ramp
                    cr_angle = nodes_collection.new("ShaderNodeValToRGB")
                    cr_angle.location = (x_offset - 500, y_offset)
                    cr_angle.color_ramp.elements[0].position = context.scene.refine_angle_ramp_pos_0 if context.scene.refine_angle_ramp_active else 0.0
                    cr_angle.color_ramp.elements[0].color = (0, 0, 0, 1) # Black (Invis)
                    cr_angle.color_ramp.elements[1].position = context.scene.refine_angle_ramp_pos_1 if context.scene.refine_angle_ramp_active else 0.0
                    cr_angle.color_ramp.elements[1].color = (1, 1, 1, 1) # White (Vis)
                    cr_angle.color_ramp.interpolation = 'LINEAR'
                    links.new(angle_node.outputs[0], cr_angle.inputs[0])
                    
                    # Feather Ramp
                    cr_feather = nodes_collection.new("ShaderNodeValToRGB")
                    cr_feather.location = (x_offset - 500, y_offset - 200)
                    cr_feather.color_ramp.elements[0].position = context.scene.refine_feather_ramp_pos_0 if context.scene.visibility_vignette else 0.0
                    cr_feather.color_ramp.elements[0].color = (0, 0, 0, 1) # Black (Invis)
                    cr_feather.color_ramp.elements[1].position = context.scene.refine_feather_ramp_pos_1 if context.scene.visibility_vignette else 0.0
                    cr_feather.color_ramp.elements[1].color = (1, 1, 1, 1) # White (Vis)
                    cr_feather.color_ramp.interpolation = 'LINEAR'
                    links.new(feather_node.outputs[0], cr_feather.inputs[0])
                    
                    # Combine angle and feather visibility
                    mult = _combine_angle_feather(
                        nodes_collection, links,
                        cr_angle.outputs[0], cr_feather.outputs[0],
                        loc=(x_offset - 300, y_offset)
                    )

                    # Optionally multiply by edge-feather mask
                    if ef_node is not None:
                        ef_mult = nodes_collection.new("ShaderNodeMath")
                        ef_mult.operation = 'MULTIPLY'
                        ef_mult.location = (x_offset - 200, y_offset - 80)
                        links.new(mult.outputs[0], ef_mult.inputs[0])
                        links.new(ef_node.outputs[0], ef_mult.inputs[1])
                        weight_out = ef_mult
                    else:
                        weight_out = mult
                    
                    # Invert (Convert Visibility to Mix Factor: 1=Vis->0=Proj, 0=Invis->1=Orig)
                    invert = nodes_collection.new("ShaderNodeMath")
                    invert.operation = 'SUBTRACT'
                    invert.location = (x_offset - 150, y_offset)
                    invert.inputs[0].default_value = 1.0
                    links.new(weight_out.outputs[0], invert.inputs[1])

                    compare_node = invert # For return
                    links.new(invert.outputs[0], final_mix.inputs["Fac"])
                else:
                    # Fallback if not tuple (e.g. stop_index logic used LessThan node)
                    # If it's a sum of processed weights (from multi-camera recursion), we need to invert it
                    # to get the Mix Factor (1 - TotalVis)
                    if is_local_edit:
                        invert = nodes_collection.new("ShaderNodeMath")
                        invert.operation = 'SUBTRACT'
                        invert.location = (x_offset - 150, y_offset)
                        invert.inputs[0].default_value = 1.0
                        links.new(w_node.outputs[0], invert.inputs[1])
                        links.new(invert.outputs[0], final_mix.inputs["Fac"])
                        compare_node = invert
                    else:
                        links.new(w_node.outputs[0], final_mix.inputs["Fac"])
                        compare_node = w_node
            else:
                compare_node = nodes_collection.new("ShaderNodeMath")
                compare_node.operation = 'COMPARE'
                # Value to compare against (Value 2)
                compare_node.inputs[1].default_value = 0.0
                # Epsilon for comparison
                compare_node.inputs[2].default_value = 0.0
                compare_node.location = (x_offset - 500, y_offset)
                # Connect final add node to compare node
                links.new(weight_nodes[0].outputs[0], compare_node.inputs[0])
                links.new(compare_node.outputs[0], final_mix.inputs["Fac"])
            
            return final_mix, compare_node

        # ── Helper: extract final weight output from a weight_node item ────
        def _get_weight_output(w_item, v_offset):
            """Process a weight_node item (tuple or plain node) through ramps
            and return the output socket.  Used by the normalization pre-pass."""
            if isinstance(w_item, tuple):
                angle_node = w_item[0]
                feather_node = w_item[1]
                ef_node = w_item[2] if len(w_item) >= 3 else None

                # Angle Ramp
                cr_angle = nodes_collection.new("ShaderNodeValToRGB")
                cr_angle.location = (x_offset - 1100, y_offset + v_offset)
                cr_angle.color_ramp.elements[0].position = context.scene.refine_angle_ramp_pos_0 if context.scene.refine_angle_ramp_active else 0.0
                cr_angle.color_ramp.elements[0].color = (0, 0, 0, 1)
                cr_angle.color_ramp.elements[1].position = context.scene.refine_angle_ramp_pos_1 if context.scene.refine_angle_ramp_active else 0.0
                cr_angle.color_ramp.elements[1].color = (1, 1, 1, 1)
                cr_angle.color_ramp.interpolation = 'LINEAR'
                links.new(angle_node.outputs[0], cr_angle.inputs[0])

                # Feather Ramp
                cr_feather = nodes_collection.new("ShaderNodeValToRGB")
                cr_feather.location = (x_offset - 1100, y_offset + v_offset - 200)
                cr_feather.color_ramp.elements[0].position = context.scene.refine_feather_ramp_pos_0 if context.scene.visibility_vignette else 0.0
                cr_feather.color_ramp.elements[0].color = (0, 0, 0, 1)
                cr_feather.color_ramp.elements[1].position = context.scene.refine_feather_ramp_pos_1 if context.scene.visibility_vignette else 0.0
                cr_feather.color_ramp.elements[1].color = (1, 1, 1, 1)
                cr_feather.color_ramp.interpolation = 'LINEAR'
                links.new(feather_node.outputs[0], cr_feather.inputs[0])

                # Combine angle × feather
                m = _combine_angle_feather(
                    nodes_collection, links,
                    cr_angle.outputs[0], cr_feather.outputs[0],
                    loc=(x_offset - 900, y_offset + v_offset)
                )

                # Optionally multiply by edge-feather mask
                if ef_node is not None:
                    ef_mult = nodes_collection.new("ShaderNodeMath")
                    ef_mult.operation = 'MULTIPLY'
                    ef_mult.location = (x_offset - 800, y_offset + v_offset - 80)
                    links.new(m.outputs[0], ef_mult.inputs[0])
                    links.new(ef_node.outputs[0], ef_mult.inputs[1])
                    return ef_mult.outputs[0]

                return m.outputs[0]
            else:
                return w_item.outputs[0]

        # ── Max-relative weight normalization (level 0 only) ──────────────
        # At high weight exponents, pow(cos(θ), exp) can underflow to 0 in
        # float32 for every camera, making the w_A/(w_A+w_B) ratio become
        # 0/0 → black.  To fix this robustly we:
        #   1. Temporarily set each per-camera Power to 1.0 so the weight
        #      outputs become the *base* values (binary_gates × |cos(θ)|)
        #      that never underflow.
        #   2. Normalize those base weights by the per-pixel maximum so the
        #      best camera gets 1.0.
        #   3. Re-apply the user exponent to the normalized [0,1] values
        #      where underflow is impossible.
        # Mathematically: pow(cos/max_cos, exp) == pow(cos, exp)/pow(max_cos, exp),
        # so the blending ratios are identical to the original formula.
        if level == 0 and len(shaders) > 1:
            # ── Step 1: Neutralize per-camera Power (set to 1.0) ──────
            user_exponent = None
            for idx in range(len(weight_nodes)):
                w = weight_nodes[idx]
                node = w[0] if isinstance(w, tuple) else w

                # Skip disabled cameras (stop_index → LESS_THAN always-0)
                if node.type == 'MATH' and node.operation == 'LESS_THAN':
                    continue

                # Native raycast path:
                #   final_weight (MULTIPLY, label Angle-*) →
                #       inputs[1] → power_node (POWER, label power_weight)
                if (node.type == 'MATH' and node.operation == 'MULTIPLY'
                        and node.label.startswith('Angle-')):
                    if node.inputs[1].links:
                        pn = node.inputs[1].links[0].from_node
                        if (pn.type == 'MATH' and pn.operation == 'POWER'
                                and pn.label == 'power_weight'):
                            if user_exponent is None:
                                user_exponent = pn.inputs[1].default_value
                            pn.inputs[1].default_value = 1.0

                # Runtime acquisition wrapper: angle output × projected
                # density.  Traverse to its angle source so normalization can
                # still neutralize the base exponent.
                elif (node.type == 'MATH' and node.operation == 'MULTIPLY'
                      and node.label.startswith('AcquisitionQuality-')):
                    pending = [node]
                    visited = set()
                    while pending:
                        upstream = pending.pop()
                        pointer = upstream.as_pointer()
                        if pointer in visited:
                            continue
                        visited.add(pointer)
                        if (upstream.type == 'MATH'
                                and upstream.operation == 'POWER'
                                and upstream.label == 'power_weight'):
                            if user_exponent is None:
                                user_exponent = upstream.inputs[1].default_value
                            upstream.inputs[1].default_value = 1.0
                            break
                        if upstream.type == 'SCRIPT' and "Power" in upstream.inputs:
                            if user_exponent is None:
                                user_exponent = upstream.inputs["Power"].default_value
                            upstream.inputs["Power"].default_value = 1.0
                            break
                        for socket in upstream.inputs:
                            pending.extend(link.from_node for link in socket.links)

                # OSL path: ShaderNodeScript with "Power" input
                elif node.type == 'SCRIPT' and "Power" in node.inputs:
                    if user_exponent is None:
                        user_exponent = node.inputs["Power"].default_value
                    node.inputs["Power"].default_value = 1.0

                # Masks/guards may wrap the acquisition node.  Always scan
                # upstream so every projection path reaches the same single
                # SharpW exponent stage.
                pending = [node]
                visited = set()
                while pending:
                    upstream = pending.pop()
                    pointer = upstream.as_pointer()
                    if pointer in visited:
                        continue
                    visited.add(pointer)
                    if (upstream.type == 'MATH'
                            and upstream.operation == 'POWER'
                            and upstream.label == 'power_weight'):
                        if user_exponent is None:
                            user_exponent = upstream.inputs[1].default_value
                        upstream.inputs[1].default_value = 1.0
                        break
                    if upstream.type == 'SCRIPT' and "Power" in upstream.inputs:
                        if user_exponent is None:
                            user_exponent = upstream.inputs["Power"].default_value
                        upstream.inputs["Power"].default_value = 1.0
                        break
                    for socket in upstream.inputs:
                        pending.extend(link.from_node for link in socket.links)

            if user_exponent is None:
                user_exponent = 1.0

            # ── Step 2: Get base weight outputs (now power = 1) ───────
            all_w_outputs = []
            for idx in range(len(weight_nodes)):
                v_off = -200 * idx
                all_w_outputs.append(_get_weight_output(weight_nodes[idx], v_off))

            # Keep these unmodified source-camera acquisition qualities for
            # ownership.  Initial-view priority is applied later only as
            # replacement resistance for a trusted owner; it must never turn
            # poor grazing coverage into trusted context.
            acquisition_outputs = list(all_w_outputs)

            # ── Step 3: Find per-pixel maximum ────────────────────────
            max_node = nodes_collection.new("ShaderNodeMath")
            max_node.operation = 'MAXIMUM'
            max_node.location = (x_offset - 1500, y_offset - 600)
            max_node.label = "WeightMax"
            links.new(all_w_outputs[0], max_node.inputs[0])
            if len(all_w_outputs) > 1:
                links.new(all_w_outputs[1], max_node.inputs[1])
            else:
                max_node.inputs[1].default_value = 0.0

            for idx in range(2, len(all_w_outputs)):
                next_max = nodes_collection.new("ShaderNodeMath")
                next_max.operation = 'MAXIMUM'
                next_max.location = (x_offset - 1500 + 150 * (idx - 1),
                                     y_offset - 600)
                next_max.label = "WeightMax"
                links.new(max_node.outputs[0], next_max.inputs[0])
                links.new(all_w_outputs[idx], next_max.inputs[1])
                max_node = next_max

            # Clamp max to epsilon to prevent 0/0 division
            safe_max = nodes_collection.new("ShaderNodeMath")
            safe_max.operation = 'MAXIMUM'
            safe_max.location = (max_node.location[0] + 150, y_offset - 600)
            safe_max.inputs[1].default_value = 1e-7
            safe_max.label = "SafeMax"
            links.new(max_node.outputs[0], safe_max.inputs[0])

            # ── Step 4: Normalize + re-apply exponent ─────────────────
            powered_nodes = []
            for idx, w_out in enumerate(all_w_outputs):
                # norm_w = base_w / max(base_w)
                div = nodes_collection.new("ShaderNodeMath")
                div.operation = 'DIVIDE'
                div.location = (safe_max.location[0] + 200,
                                y_offset - 120 * idx)
                div.label = f"NormW-{idx}"
                links.new(w_out, div.inputs[0])
                links.new(safe_max.outputs[0], div.inputs[1])

                # sharp_w = pow(norm_w, user_exponent)
                # (inputs are in [0, 1] → no underflow possible)
                # Always keep an explicit sharpening node. This makes the
                # post-generation reset deterministic even when the material
                # was initially built with exponent 1.
                pw = nodes_collection.new("ShaderNodeMath")
                pw.operation = 'POWER'
                pw.location = (div.location[0] + 200,
                               div.location[1])
                pw.label = f"SharpW-{idx}"
                pw.inputs[1].default_value = user_exponent
                links.new(div.outputs[0], pw.inputs[0])
                powered_nodes.append(pw)

            # Replace weight_nodes with sharpened-normalised nodes
            weight_nodes = powered_nodes

            # Binary Qwen write masks and per-texel ownership solve different
            # problems. Write masks prevent a later generated view from writing
            # protected context; ownership selects exactly one camera for
            # high-frequency detail and therefore creates harder territories.
            # Keep ownership genuinely opt-in so disabling the UI toggle
            # restores continuous weighted blending without disabling Qwen's
            # protection masks.
            use_confidence_ownership = bool(
                getattr(context.scene, 'sg_texel_ownership', False)
            )
            legacy_blend = getattr(
                context.scene, 'sg_ownership_blend_mode', 'OWNER'
            ) == 'LEGACY'
            if use_confidence_ownership and not legacy_blend:
                # Sequential single-owner detail.  Untrusted owners remain
                # replaceable; trusted owners lock unless explicit quality
                # upgrades are enabled.
                owner_color = shaders[0]
                owner_low = lowpass_tex_image_nodes[0]
                seam_low = owner_low
                owner_quality = acquisition_outputs[0]
                margin = max(
                    0.0,
                    float(getattr(context.scene, 'sg_texel_replacement_margin', 0.05)),
                )
                trust_threshold = float(getattr(
                    context.scene, 'qwen_trust_threshold', 0.7
                ))
                allow_upgrade = bool(getattr(
                    context.scene, 'sg_allow_trusted_replacement', False
                ))
                priority_strength = (
                    float(getattr(context.scene, 'early_priority_strength', 0.5))
                    if (getattr(context.scene, 'early_priority', False)
                        and getattr(context.scene, 'sg_rework_initial_view_priority', False))
                    else 0.0
                )

                owner_camera = nodes_collection.new("ShaderNodeValue")
                owner_camera.outputs[0].default_value = 0.0
                owner_camera.label = "SGOwnerCamera-0"
                owner_priority = nodes_collection.new("ShaderNodeValue")
                owner_priority.outputs[0].default_value = priority_strength * 0.25
                owner_priority.label = "SGOwnerPriority-0"

                for idx in range(1, len(shaders)):
                    new_quality = acquisition_outputs[idx]
                    old_owner_low = owner_low

                    owner_trusted = nodes_collection.new("ShaderNodeMath")
                    owner_trusted.operation = 'GREATER_THAN'
                    owner_trusted.inputs[1].default_value = trust_threshold - 1e-7
                    owner_trusted.label = f"OwnerTrusted-{idx}"
                    links.new(owner_quality, owner_trusted.inputs[0])

                    owner_untrusted = nodes_collection.new("ShaderNodeMath")
                    owner_untrusted.operation = 'SUBTRACT'
                    owner_untrusted.inputs[0].default_value = 1.0
                    links.new(owner_trusted.outputs[0], owner_untrusted.inputs[1])

                    candidate_valid = nodes_collection.new("ShaderNodeMath")
                    candidate_valid.operation = 'GREATER_THAN'
                    candidate_valid.inputs[1].default_value = 0.0
                    links.new(new_quality, candidate_valid.inputs[0])

                    replace_untrusted = nodes_collection.new("ShaderNodeMath")
                    replace_untrusted.operation = 'MULTIPLY'
                    links.new(owner_untrusted.outputs[0], replace_untrusted.inputs[0])
                    links.new(candidate_valid.outputs[0], replace_untrusted.inputs[1])

                    margin_node = nodes_collection.new("ShaderNodeMath")
                    margin_node.operation = 'ADD'
                    margin_node.location = (x_offset - 200, y_offset - 180 * idx)
                    margin_node.inputs[1].default_value = margin
                    margin_node.label = f"OwnerMargin-{idx}"
                    links.new(owner_quality, margin_node.inputs[0])

                    priority_margin = nodes_collection.new("ShaderNodeMath")
                    priority_margin.operation = 'ADD'
                    links.new(margin_node.outputs[0], priority_margin.inputs[0])
                    links.new(owner_priority.outputs[0], priority_margin.inputs[1])

                    replace = nodes_collection.new("ShaderNodeMath")
                    replace.operation = 'GREATER_THAN'
                    replace.location = (x_offset, y_offset - 180 * idx)
                    replace.label = f"OwnerReplace-{idx}"
                    links.new(new_quality, replace.inputs[0])
                    links.new(priority_margin.outputs[0], replace.inputs[1])

                    replace_trusted = nodes_collection.new("ShaderNodeMath")
                    replace_trusted.operation = 'MULTIPLY'
                    replace_trusted.inputs[1].default_value = 1.0 if allow_upgrade else 0.0
                    links.new(replace.outputs[0], replace_trusted.inputs[0])

                    replace_or_empty = nodes_collection.new("ShaderNodeMath")
                    replace_or_empty.operation = 'MAXIMUM'
                    replace_or_empty.location = (x_offset + 100, y_offset - 180 * idx)
                    replace_or_empty.label = f"OwnerAccept-{idx}"
                    links.new(replace_untrusted.outputs[0], replace_or_empty.inputs[0])
                    links.new(replace_trusted.outputs[0], replace_or_empty.inputs[1])

                    select_color = nodes_collection.new("ShaderNodeMixRGB")
                    select_color.location = (x_offset + 200, y_offset - 180 * idx)
                    select_color.label = f"TexelOwner-{idx}"
                    links.new(replace_or_empty.outputs[0], select_color.inputs["Fac"])
                    links.new(owner_color.outputs[0], select_color.inputs["Color1"])
                    links.new(shaders[idx].outputs[0], select_color.inputs["Color2"])

                    select_low = nodes_collection.new("ShaderNodeMixRGB")
                    select_low.label = f"LowFrequencyOwner-{idx}"
                    links.new(replace_or_empty.outputs[0], select_low.inputs["Fac"])
                    links.new(seam_low.outputs[0], select_low.inputs["Color1"])
                    links.new(lowpass_tex_image_nodes[idx].outputs[0], select_low.inputs["Color2"])

                    if getattr(context.scene, 'sg_ownership_blend_mode', 'OWNER') == 'OWNER':
                        difference = nodes_collection.new("ShaderNodeMath")
                        difference.operation = 'SUBTRACT'
                        links.new(new_quality, difference.inputs[0])
                        links.new(owner_quality, difference.inputs[1])
                        absolute_difference = nodes_collection.new("ShaderNodeMath")
                        absolute_difference.operation = 'ABSOLUTE'
                        links.new(difference.outputs[0], absolute_difference.inputs[0])
                        near_boundary = nodes_collection.new("ShaderNodeMath")
                        near_boundary.operation = 'LESS_THAN'
                        near_boundary.inputs[1].default_value = float(getattr(
                            context.scene, 'sg_ownership_seam_width', 0.08
                        ))
                        links.new(absolute_difference.outputs[0], near_boundary.inputs[0])
                        candidate_trusted = nodes_collection.new("ShaderNodeMath")
                        candidate_trusted.operation = 'GREATER_THAN'
                        candidate_trusted.inputs[1].default_value = trust_threshold - 1e-7
                        links.new(new_quality, candidate_trusted.inputs[0])
                        both_trusted = nodes_collection.new("ShaderNodeMath")
                        both_trusted.operation = 'MULTIPLY'
                        links.new(owner_trusted.outputs[0], both_trusted.inputs[0])
                        links.new(candidate_trusted.outputs[0], both_trusted.inputs[1])
                        seam_factor = nodes_collection.new("ShaderNodeMath")
                        seam_factor.operation = 'MULTIPLY'
                        links.new(near_boundary.outputs[0], seam_factor.inputs[0])
                        links.new(both_trusted.outputs[0], seam_factor.inputs[1])
                        low_average = nodes_collection.new("ShaderNodeMixRGB")
                        low_average.inputs["Fac"].default_value = 0.5
                        links.new(owner_low.outputs[0], low_average.inputs["Color1"])
                        links.new(lowpass_tex_image_nodes[idx].outputs[0], low_average.inputs["Color2"])
                        blended_seam_low = nodes_collection.new("ShaderNodeMixRGB")
                        blended_seam_low.label = f"LowFrequencySeam-{idx}"
                        links.new(seam_factor.outputs[0], blended_seam_low.inputs["Fac"])
                        links.new(select_low.outputs[0], blended_seam_low.inputs["Color1"])
                        links.new(low_average.outputs[0], blended_seam_low.inputs["Color2"])
                        next_seam_low = blended_seam_low
                    else:
                        next_seam_low = select_low

                    confidence_delta = nodes_collection.new("ShaderNodeMath")
                    confidence_delta.operation = 'SUBTRACT'
                    confidence_delta.location = (x_offset + 200, y_offset - 180 * idx - 90)
                    links.new(new_quality, confidence_delta.inputs[0])
                    links.new(owner_quality, confidence_delta.inputs[1])
                    selected_delta = nodes_collection.new("ShaderNodeMath")
                    selected_delta.operation = 'MULTIPLY'
                    selected_delta.location = (x_offset + 350, y_offset - 180 * idx - 90)
                    links.new(confidence_delta.outputs[0], selected_delta.inputs[0])
                    links.new(replace_or_empty.outputs[0], selected_delta.inputs[1])
                    selected_confidence = nodes_collection.new("ShaderNodeMath")
                    selected_confidence.operation = 'ADD'
                    selected_confidence.location = (x_offset + 500, y_offset - 180 * idx - 90)
                    selected_confidence.label = f"OwnerConfidence-{idx}"
                    links.new(owner_quality, selected_confidence.inputs[0])
                    links.new(selected_delta.outputs[0], selected_confidence.inputs[1])

                    camera_delta = nodes_collection.new("ShaderNodeMath")
                    camera_delta.operation = 'SUBTRACT'
                    camera_delta.inputs[0].default_value = float(idx)
                    links.new(owner_camera.outputs[0], camera_delta.inputs[1])
                    selected_camera_delta = nodes_collection.new("ShaderNodeMath")
                    selected_camera_delta.operation = 'MULTIPLY'
                    links.new(camera_delta.outputs[0], selected_camera_delta.inputs[0])
                    links.new(replace_or_empty.outputs[0], selected_camera_delta.inputs[1])
                    selected_camera = nodes_collection.new("ShaderNodeMath")
                    selected_camera.operation = 'ADD'
                    selected_camera.label = f"SGOwnerCamera-{idx}"
                    links.new(owner_camera.outputs[0], selected_camera.inputs[0])
                    links.new(selected_camera_delta.outputs[0], selected_camera.inputs[1])

                    candidate_priority = priority_strength * 0.25 * (
                        1.0 - idx / float(max(len(shaders) - 1, 1))
                    )
                    priority_delta = nodes_collection.new("ShaderNodeMath")
                    priority_delta.operation = 'SUBTRACT'
                    priority_delta.inputs[0].default_value = candidate_priority
                    links.new(owner_priority.outputs[0], priority_delta.inputs[1])
                    selected_priority_delta = nodes_collection.new("ShaderNodeMath")
                    selected_priority_delta.operation = 'MULTIPLY'
                    links.new(priority_delta.outputs[0], selected_priority_delta.inputs[0])
                    links.new(replace_or_empty.outputs[0], selected_priority_delta.inputs[1])
                    selected_priority = nodes_collection.new("ShaderNodeMath")
                    selected_priority.operation = 'ADD'
                    selected_priority.label = f"SGOwnerPriority-{idx}"
                    links.new(owner_priority.outputs[0], selected_priority.inputs[0])
                    links.new(selected_priority_delta.outputs[0], selected_priority.inputs[1])

                    owner_color = select_color
                    owner_low = nodes_collection.new("ShaderNodeMixRGB")
                    owner_low.label = f"SelectedOwnerLow-{idx}"
                    links.new(replace_or_empty.outputs[0], owner_low.inputs["Fac"])
                    links.new(old_owner_low.outputs[0], owner_low.inputs["Color1"])
                    links.new(lowpass_tex_image_nodes[idx].outputs[0], owner_low.inputs["Color2"])
                    seam_low = next_seam_low
                    owner_quality = selected_confidence.outputs[0]
                    owner_camera = selected_camera
                    owner_priority = selected_priority

                selected_confidence.label = "SGOwnerQuality"
                selected_camera.label = "SGOwnerCamera"

                if getattr(context.scene, 'sg_ownership_blend_mode', 'OWNER') == 'OWNER':
                    detail = nodes_collection.new("ShaderNodeMixRGB")
                    detail.blend_type = 'SUBTRACT'
                    detail.inputs["Fac"].default_value = 1.0
                    detail.label = "OwnedHighFrequencyDetail"
                    links.new(owner_color.outputs[0], detail.inputs["Color1"])
                    links.new(owner_low.outputs[0], detail.inputs["Color2"])
                    recompose = nodes_collection.new("ShaderNodeMixRGB")
                    recompose.blend_type = 'ADD'
                    recompose.inputs["Fac"].default_value = 1.0
                    recompose.use_clamp = True
                    recompose.label = "LowFrequencySeamComposite"
                    links.new(detail.outputs[0], recompose.inputs["Color1"])
                    links.new(seam_low.outputs[0], recompose.inputs["Color2"])
                    owner_color = recompose

                final_mix = nodes_collection.new("ShaderNodeMixRGB")
                final_mix.location = (x_offset + 450, y_offset)
                final_mix.inputs["Color2"].default_value = (
                    *context.scene.fallback_color, 1.0
                )
                links.new(owner_color.outputs[0], final_mix.inputs["Color1"])
                if last_node:
                    links.new(last_node.outputs[0], final_mix.inputs["Color2"])
                uncovered = nodes_collection.new("ShaderNodeMath")
                uncovered.operation = 'COMPARE'
                uncovered.inputs[1].default_value = 0.0
                uncovered.inputs[2].default_value = 0.0
                uncovered.label = "OwnerUncovered"
                links.new(owner_quality, uncovered.inputs[0])
                links.new(uncovered.outputs[0], final_mix.inputs["Fac"])
                return final_mix, selected_confidence

        new_shaders = []
        new_weight_nodes = []
        i = 0
        while i < len(shaders):
            if i + 1 < len(shaders):
                # Sum the weights
                vert_offset = -200 * (i // 2)
                sum_node = nodes_collection.new("ShaderNodeMath")
                sum_node.operation = 'ADD'
                legacy_priority = (
                    context.scene.early_priority
                    and not getattr(context.scene, 'sg_rework_initial_view_priority', False)
                )
                sum_node.location = (
                    (x_offset - 800, y_offset + vert_offset)
                    if legacy_priority else (x_offset - 600, y_offset + vert_offset)
                )
                
                def get_weight_output(w_item):
                    if isinstance(w_item, tuple):
                        angle_node = w_item[0]
                        feather_node = w_item[1]
                        ef_node = w_item[2] if len(w_item) >= 3 else None
                        
                        # Create Ramps (Output Visibility: 0=Invis, 1=Vis)
                        # Angle Ramp
                        cr_angle = nodes_collection.new("ShaderNodeValToRGB")
                        cr_angle.location = (x_offset - 1100, y_offset + vert_offset)
                        cr_angle.color_ramp.elements[0].position = context.scene.refine_angle_ramp_pos_0 if context.scene.refine_angle_ramp_active else 0.0
                        cr_angle.color_ramp.elements[0].color = (0, 0, 0, 1) # Black (Invis)
                        cr_angle.color_ramp.elements[1].position = context.scene.refine_angle_ramp_pos_1 if context.scene.refine_angle_ramp_active else 0.0
                        cr_angle.color_ramp.elements[1].color = (1, 1, 1, 1) # White (Vis)
                        cr_angle.color_ramp.interpolation = 'LINEAR'
                        links.new(angle_node.outputs[0], cr_angle.inputs[0])
                        
                        # Feather Ramp
                        cr_feather = nodes_collection.new("ShaderNodeValToRGB")
                        cr_feather.location = (x_offset - 1100, y_offset + vert_offset - 200)
                        cr_feather.color_ramp.elements[0].position = context.scene.refine_feather_ramp_pos_0 if context.scene.visibility_vignette else 0.0
                        cr_feather.color_ramp.elements[0].color = (0, 0, 0, 1) # Black (Invis)
                        cr_feather.color_ramp.elements[1].position = context.scene.refine_feather_ramp_pos_1 if context.scene.visibility_vignette else 0.0
                        cr_feather.color_ramp.elements[1].color = (1, 1, 1, 1) # White (Vis)
                        cr_feather.color_ramp.interpolation = 'LINEAR'
                        links.new(feather_node.outputs[0], cr_feather.inputs[0])
                        
                        # Combine angle and feather visibility
                        m = _combine_angle_feather(
                            nodes_collection, links,
                            cr_angle.outputs[0], cr_feather.outputs[0],
                            loc=(x_offset - 900, y_offset + vert_offset)
                        )

                        # Optionally multiply by edge-feather mask
                        if ef_node is not None:
                            ef_mult = nodes_collection.new("ShaderNodeMath")
                            ef_mult.operation = 'MULTIPLY'
                            ef_mult.location = (x_offset - 800, y_offset + vert_offset - 80)
                            links.new(m.outputs[0], ef_mult.inputs[0])
                            links.new(ef_node.outputs[0], ef_mult.inputs[1])
                            return ef_mult.outputs[0]

                        return m.outputs[0]
                    else:
                        return w_item.outputs[0]

                # Cache outputs to avoid creating duplicate nodes
                w_out_1 = get_weight_output(weight_nodes[i])
                w_out_2 = get_weight_output(weight_nodes[i+1])

                links.new(w_out_1, sum_node.inputs[0])
                links.new(w_out_2, sum_node.inputs[1])

                # Compute mix factor: weight_A / (weight_A+weight_B)
                div_node = nodes_collection.new("ShaderNodeMath")
                div_node.operation = 'DIVIDE'
                div_node.location = (
                    (x_offset - 600, y_offset + vert_offset)
                    if legacy_priority else (x_offset - 400, y_offset + vert_offset)
                )
                links.new(w_out_2, div_node.inputs[0])
                links.new(sum_node.outputs[0], div_node.inputs[1])

                if legacy_priority:
                    map_range_node = nodes_collection.new("ShaderNodeMapRange")
                    map_range_node.location = (x_offset - 400, y_offset + vert_offset)
                    map_range_node.inputs[1].default_value = context.scene.early_priority_strength
                    links.new(div_node.outputs[0], map_range_node.inputs[0])

                # Create a MixRGB node for color blending of the two inputs
                mix_node = nodes_collection.new("ShaderNodeMixRGB")
                mix_node.location = (x_offset - 200, y_offset + vert_offset)
                mix_node.use_clamp = True
                # Connect first color to Color1 and second color to Color2
                links.new(shaders[i].outputs[0], mix_node.inputs["Color1"])
                links.new(shaders[i+1].outputs[0], mix_node.inputs["Color2"])
                # Use the computed mix factor
                to_connect = map_range_node if legacy_priority else div_node
                links.new(to_connect.outputs[0], mix_node.inputs["Fac"])

                new_shaders.append(mix_node)
                # The new effective weight is the sum (stored in sum_node)
                new_weight_nodes.append(sum_node)
                i += 2
            else:
                new_shaders.append(shaders[i])
                new_weight_nodes.append(weight_nodes[i])
                i += 1
        return build_mix_tree(new_shaders, new_weight_nodes, nodes_collection, links, last_node, level+1)

    if view_plan is not None:
        cameras = []
        for record in view_plan:
            camera = record.camera
            if record.matrix_world:
                camera.matrix_world = Matrix(record.matrix_world)
            for name, value in dict(record.intrinsics or {}).items():
                if value is not None and hasattr(camera.data, name):
                    setattr(camera.data, name, value)
            cameras.append(camera)
    elif cameras is None:
        from .view_plan import order_cameras
        cameras = order_cameras(
            obj for obj in context.scene.objects if obj.type == 'CAMERA'
        )
    else:
        cameras = list(cameras)
    
    # Force refresh of the UI
    for area in context.screen.areas:
        area.tag_redraw()
    
    # Apply projection to all mesh objects for each camera
    for i, camera in enumerate(cameras):
        for x, obj in enumerate(to_project):
            # We can skip the UV Projection step in sequential mode for i > 0
            if context.scene.generation_method != 'sequential' or stop_index == 0:
                # Deselect all objects
                bpy.ops.object.select_all(action='DESELECT')
                # Select object as active (needed for applying the modifier)
                context.view_layer.objects.active = obj
                obj.select_set(True)
                    
                # Make object data single-user before applying modifier
                bpy.ops.object.make_single_user(object=True, obdata=True)

                # Check if the data is now single user
                if obj.data.users > 1:
                    # If not, we need to make it single user again
                    print("[StableGen] Warning: Cannot make object data single user. Making a copy.")
                    obj.data = obj.data.copy()
                    obj.data.name = f"{obj.name}_data"
                    obj.data.update()
                    # Check again
                    if obj.data.users > 1:
                        print("[StableGen] Error: Cannot make object data single user. Exiting.")
                        return Exception("Cannot make object data single user. Exiting.")

                # Create or reuse a buffer UV map for the UV Project modifier
                buffer_uv = obj.data.uv_layers.get(_SG_BUFFER_UV_NAME)
                if not buffer_uv:
                    buffer_uv = obj.data.uv_layers.new(name=_SG_BUFFER_UV_NAME)

                # Add the UV Project Modifier
                uv_project_mod = obj.modifiers.new(name="UVProject", type='UV_PROJECT')

                # Assign the active camera to the UV Project modifier
                if not camera:
                    return False

                uv_project_mod.projectors[0].object = camera

                # Set the buffer UV map for the modifier
                uv_project_mod.uv_layer = buffer_uv.name

                # Calculate and set the aspect ratio (per-camera if available)
                cam_res_x, cam_res_y = _get_camera_resolution(camera, context.scene)
                aspect_ratio = cam_res_x / cam_res_y
                uv_project_mod.aspect_x = aspect_ratio if aspect_ratio > 1 else 1
                uv_project_mod.aspect_y = 1 / aspect_ratio if aspect_ratio < 1 else 1

                # Set the buffer UV as active so the modifier writes to it
                # (Blender 5.0+ ignores uv_layer and writes to the active UV map)
                obj.data.uv_layers.active = buffer_uv

                # Apply the modifier
                bpy.ops.object.modifier_apply(modifier=uv_project_mod.name)

                # Copy buffer UV data to a named attribute (no UV slot limit)
                attr_name = f"ProjectionUV_{i}_{mat_id}"
                _copy_uv_to_attribute(obj, _SG_BUFFER_UV_NAME, attr_name)

                # Restore active UV to the first non-buffer UV layer
                original_uv_map = obj.data.uv_layers[0]
                if original_uv_map.name == _SG_BUFFER_UV_NAME and len(obj.data.uv_layers) > 1:
                    original_uv_map = obj.data.uv_layers[1]
                obj.data.uv_layers.active = original_uv_map

    # Clean up buffer UV map from all objects
    for obj in to_project:
        buffer_uv = obj.data.uv_layers.get(_SG_BUFFER_UV_NAME)
        if buffer_uv:
            obj.data.uv_layers.remove(buffer_uv)

    # ── Optional: bake per-vertex visibility weights ──────────────────
    use_baked_weights = (
        getattr(context.scene, 'bake_visibility_weights', False)
        or (getattr(context.scene, 'sg_rework_projection_geometry', False)
            and any(camera.data.type == 'ORTHO' for camera in cameras))
    )
    if use_baked_weights:
        # Always compute weights for ALL cameras (ignore stop_index).
        #
        # Needs further testing, maybe will need to revert this.

        # In sequential mode reuse the first bake, but never assume it exists:
        # regeneration, project-only runs, and orthographic imports can enter
        # at an arbitrary stop index.
        is_sequential = context.scene.generation_method == 'sequential'
        missing_bake = any(
            obj.data.attributes.get(f"_SG_VisWeight_{cam_idx}_{mat_id}") is None
            for obj in to_project for cam_idx in range(len(cameras))
        )
        if not is_sequential or stop_index == 0 or missing_bake:
            _bake_visibility_weights(context, to_project, cameras, mat_id)

    # Switch to Cycles (needed for Raycast shader node on all versions)
    if context.scene.render.engine != 'CYCLES':
        context.scene.render.engine = 'CYCLES'
    # Force CPU + OSL only for Blender < 5.1 (native Raycast nodes don't need it)
    if bpy.app.version < (5, 1, 0):
        context.scene.cycles.device = 'CPU'
        if hasattr(context.scene.cycles, 'shading_system'):
            context.scene.cycles.shading_system = True
        else:
            context.scene.cycles.use_osl = True

    processed_materials = set()

    for obj in to_project:

        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = obj
        obj.select_set(True)

        # Create the material
        to_switch = False
        is_local_edit = (context.scene.generation_method == 'local_edit' or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'local_edit'))
        if is_local_edit and not context.scene.overwrite_material:
            # Copy active material
            mat = obj.active_material.copy()
            obj.data.materials.append(mat)
            to_switch = True
        elif obj.active_material and (context.scene.overwrite_material or is_local_edit \
                                      or (context.scene.generation_method == 'sequential' and stop_index > 0) or context.scene.generation_mode == 'regenerate_selected'):
            # Use active material
            mat = obj.active_material
        else:
            mat = bpy.data.materials.new(name="ProjectionMaterial")
            obj.data.materials.append(mat)
            # Mark as active material
            obj.active_material_index = obj.material_slots.find(mat.name)
            to_switch = True
        
        if to_switch:
            original_materials = obj.data.materials[:]
            # Clear existing materials
            obj.data.materials.clear()
            # Add the new material
            obj.data.materials.append(mat)
            # Add the rest of the original materials
            for original_mat in original_materials:
                if original_mat != mat:
                    obj.data.materials.append(original_mat)

        # Enable use of nodes
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        previous_node = None
        output = None

        # Get the original UV map (needed for bake path).
        # If the mesh has no UV maps at all, create one so baking / node
        # setup doesn't crash.
        if len(obj.data.uv_layers) == 0:
            obj.data.uv_layers.new(name="UVMap")
        original_uv_map = obj.data.uv_layers[0]
        if original_uv_map.name == _SG_BUFFER_UV_NAME and len(obj.data.uv_layers) > 1:
            original_uv_map = obj.data.uv_layers[1]


        rework_requires_rebuild = any((
            getattr(context.scene, 'sg_rework_projection_rebuild', False),
            getattr(context.scene, 'sg_rework_projection_guard', False),
            getattr(context.scene, 'sg_texel_ownership', False),
            getattr(context.scene, 'qwen_use_projection_write_masks', False),
            getattr(context.scene, 'sg_rework_sdxl_registration', False),
        ))
        legacy_incremental_update = (
            ((context.scene.generation_method == 'sequential' and stop_index > 0)
             or context.scene.generation_mode == 'regenerate_selected')
            and not rework_requires_rebuild
        )

        if legacy_incremental_update:
            # Exact rigging-branch behavior: patch the newly generated image
            # into the existing graph and remove that view's stop-index gate.
            script_node = None
            for node in nodes:
                if ((node.type == 'SCRIPT'
                     or (node.type == 'MATH' and node.operation == 'MULTIPLY'))
                        and (node.label == f"Angle-{stop_index}-{mat_id}"
                             or node.label == f"{stop_index}-{mat_id}")):
                    script_node = node
                    break
            # New feature nodes may sit between Angle and the temporary gate.
            # Identify the gate explicitly and reconnect its actual input source;
            # retaining the old direct-child search keeps old saved graphs valid.
            stop_gate = next(
                (node for node in nodes
                 if node.label == f"StopGate-{stop_index}-{mat_id}"),
                None,
            )
            if stop_gate is None and script_node:
                stop_gate = next(
                    (link.to_node for link in script_node.outputs[0].links
                     if link.to_node.type == 'MATH'
                     and link.to_node.operation == 'LESS_THAN'),
                    None,
                )
            if stop_gate is not None:
                gate_source = (
                    stop_gate.inputs[0].links[0].from_socket
                    if stop_gate.inputs[0].links else None
                )
                gate_destinations = [
                    link.to_socket for link in stop_gate.outputs[0].links
                ]
                nodes.remove(stop_gate)
                if gate_source is not None:
                    for output_socket in gate_destinations:
                        links.new(gate_source, output_socket)
            for node in nodes:
                if node.type == 'TEX_IMAGE' and node.label == f"{stop_index}-{mat_id}":
                    image_path = get_file_path(
                        context, "generated", camera_id=stop_index, material_id=mat_id
                    )
                    if ((context.scene.generation_method in ('sequential', 'separate'))
                            and context.scene.sequential_ipadapter
                            and context.scene.sequential_ipadapter_regenerate
                            and not context.scene.use_ipadapter
                            and stop_index == 0
                            and context.scene.sequential_ipadapter_mode == 'first'):
                        image_path = image_path.replace(".png", "_ipadapter.png")
                    image = get_or_load_image(
                        image_path,
                        force_reload=context.scene.overwrite_material,
                    )
                    if image:
                        node.image = image
                    break
            continue
        elif not is_local_edit:
            # Rebuild mode is required for per-view assets that only exist
            # after generation (write masks, guards, and ownership inputs).
            # Clear existing nodes
            for node in nodes:
                nodes.remove(node)
        else:
            # Find the node connected to the output node
            for node in nodes:
                if node.type == 'OUTPUT_MATERIAL':
                    output = node
                    break
            if output:
                # Save the node connected to the output node into a variable
                if output.inputs[0].links[0].from_node.type == 'BSDF_PRINCIPLED':
                    previous_node = output.inputs[0].links[0].from_node.inputs[0].links[0].from_node
                else:
                    previous_node = output.inputs[0].links[0].from_node
            
        if not is_local_edit:
            output = nodes.new("ShaderNodeOutputMaterial")
            output.location = (3000, 0)
        
        geometry = nodes.new("ShaderNodeNewGeometry")
        geometry.location = (-600, 0)

        use_native_raycast = bpy.app.version >= (5, 1, 0)

        tex_image_nodes = []
        lowpass_tex_image_nodes = []
        uv_map_nodes = []
        subtract_nodes = []
        normalize_nodes = []
        script_nodes = []
        script_nodes_outputs = []
        add_camera_loc_nodes = []
        length_nodes = []
        camera_fov_nodes = []
        camera_aspect_ratio_nodes = []
        camera_direction_nodes = []
        camera_up_nodes = []

        def _create_edge_feather_weight(cam_index, mat_id_inner):
            """Load the edge-feathered visibility mask for camera *cam_index*
            and return a shader Math node whose output[0] gives a 0..1 float
            weight (0 = edge/invisible, 1 = interior/full), or *None* on failure.

            The mask is UV-projected through the same camera UV attribute
            that projects the generated image.
            """
            vis_dir = get_dir_path(context, "inpaint")["visibility"]
            ef_path = os.path.join(vis_dir, f"render{cam_index}_edgefeather.png")
            if not os.path.exists(ef_path):
                print(f"[StableGen] Edge-feather mask not found: {ef_path}")
                return None
            ef_image = get_or_load_image(ef_path, force_reload=context.scene.overwrite_material)
            if ef_image is None:
                return None

            # Image Texture node to sample the mask via camera UV
            mask_tex = nodes.new("ShaderNodeTexImage")
            mask_tex.image = ef_image
            mask_tex.extension = 'CLIP'
            mask_tex.label = f"EdgeFeather-{cam_index}-{mat_id_inner}"
            mask_tex.location = (-500, -200 * cam_index - 150)

            # Connect to the same camera UV attribute
            uv_node = uv_map_nodes[cam_index]
            uv_out = "Vector" if uv_node.type == 'ATTRIBUTE' else "UV"
            links.new(uv_node.outputs[uv_out], mask_tex.inputs["Vector"])

            # Color→Float conversion via a pass-through Math node
            to_float = nodes.new("ShaderNodeMath")
            to_float.operation = 'MULTIPLY'
            to_float.inputs[1].default_value = 1.0
            to_float.location = (-300, -200 * cam_index - 150)
            to_float.label = f"EF_Weight-{cam_index}"
            links.new(mask_tex.outputs[0], to_float.inputs[0])
            return to_float

        def _create_write_mask_weight(cam_index, mat_id_inner, binary=False):
            """Sample the authoritative per-view sequential write mask."""
            mask_path = get_file_path(
                context,
                "inpaint",
                subtype="visibility",
                camera_id=cam_index,
            )
            if not os.path.exists(mask_path):
                return None
            mask_image = get_or_load_image(
                mask_path,
                force_reload=context.scene.overwrite_material,
            )
            if mask_image is None:
                return None
            mask_image.colorspace_settings.name = 'Non-Color'

            mask_tex = nodes.new("ShaderNodeTexImage")
            mask_tex.image = mask_image
            mask_tex.extension = 'CLIP'
            mask_tex.interpolation = 'Closest' if binary else 'Linear'
            mask_tex.label = f"SequentialWriteMask-{cam_index}-{mat_id_inner}"
            mask_tex.location = (-500, -200 * cam_index - 300)

            uv_node = uv_map_nodes[cam_index]
            uv_out = "Vector" if uv_node.type == 'ATTRIBUTE' else "UV"
            links.new(uv_node.outputs[uv_out], mask_tex.inputs["Vector"])

            to_float = nodes.new("ShaderNodeMath")
            to_float.operation = 'MULTIPLY'
            to_float.inputs[1].default_value = 1.0
            to_float.location = (-300, -200 * cam_index - 300)
            to_float.label = f"SequentialWriteWeight-{cam_index}-{mat_id_inner}"
            links.new(mask_tex.outputs[0], to_float.inputs[0])
            return to_float

        def _create_projection_guard_weight(cam_index, mat_id_inner):
            """Sample per-pixel depth/object/material silhouette rejection."""
            if not getattr(context.scene, 'sg_rework_projection_guard', False):
                return None
            guard_path = os.path.join(
                get_dir_path(context, "inpaint")["visibility"],
                f"projection_guard{cam_index}.png",
            )
            if not os.path.exists(guard_path):
                return None
            guard_image = get_or_load_image(
                guard_path, force_reload=context.scene.overwrite_material
            )
            if guard_image is None:
                return None
            guard_image.colorspace_settings.name = 'Non-Color'
            guard_tex = nodes.new("ShaderNodeTexImage")
            guard_tex.image = guard_image
            guard_tex.extension = 'CLIP'
            guard_tex.interpolation = 'Linear'
            guard_tex.label = f"ProjectionGuard-{cam_index}-{mat_id_inner}"
            uv_node = uv_map_nodes[cam_index]
            uv_out = "Vector" if uv_node.type == 'ATTRIBUTE' else "UV"
            links.new(uv_node.outputs[uv_out], guard_tex.inputs["Vector"])
            guard_weight = nodes.new("ShaderNodeMath")
            guard_weight.operation = 'MULTIPLY'
            guard_weight.inputs[1].default_value = 1.0
            guard_weight.label = f"ProjectionGuardWeight-{cam_index}-{mat_id_inner}"
            links.new(guard_tex.outputs[0], guard_weight.inputs[0])
            return guard_weight

        # Voronoi mode flag: keep natural weights for non-generated cameras
        voronoi_active = (context.scene.model_architecture in ('qwen_image_edit', 'flux2_klein')
                          and getattr(context.scene, 'qwen_voronoi_mode', False)
                          and context.scene.generation_method == 'sequential')

        for i, camera in enumerate(cameras):
            # Add image texture node
            tex_image = nodes.new("ShaderNodeTexImage")
            if i <= stop_index:
                image_path = get_file_path(context, "generated", camera_id=i, material_id=mat_id)
                if (context.scene.generation_method == 'sequential' or context.scene.generation_method == 'separate') and context.scene.sequential_ipadapter and context.scene.sequential_ipadapter_regenerate \
                and not context.scene.use_ipadapter and i == 0 and context.scene.sequential_ipadapter_mode == 'first':
                    image_path = get_file_path(context, "generated", camera_id=i, material_id=mat_id).replace(".png", "_ipadapter.png")

                image = get_or_load_image(image_path, force_reload=context.scene.overwrite_material)
                if image:
                    tex_image.image = image
            elif voronoi_active:
                # Voronoi mode: assign a solid placeholder matching the Qwen
                # guidance fallback colour so non-generated cameras contribute
                # that colour instead of black.
                voronoi_color = tuple(context.scene.qwen_guidance_fallback_color)
                tex_image.image = _get_voronoi_placeholder(voronoi_color)
                tex_image.interpolation = 'Closest'

            tex_image.location = (0, -200 * i)
            tex_image.extension = 'CLIP'
            tex_image.label = f"{i}-{mat_id}"
            tex_image_nodes.append(tex_image)

            lowpass_node = tex_image
            if (getattr(context.scene, 'sg_texel_ownership', False)
                    and getattr(context.scene, 'sg_ownership_blend_mode', 'OWNER') == 'OWNER'
                    and i <= stop_index and tex_image.image is not None):
                source_path = bpy.path.abspath(tex_image.image.filepath)
                lowpass_path = _ensure_lowpass_texture(
                    source_path,
                    getattr(context.scene, 'sg_ownership_lowpass_sigma', 8.0),
                )
                if lowpass_path:
                    lowpass_image = get_or_load_image(
                        lowpass_path,
                        force_reload=context.scene.overwrite_material,
                    )
                    if lowpass_image:
                        lowpass_node = nodes.new("ShaderNodeTexImage")
                        lowpass_node.image = lowpass_image
                        lowpass_node.extension = 'CLIP'
                        lowpass_node.label = f"LowPass-{i}-{mat_id}"
                        lowpass_node.location = (200, -200 * i)
            lowpass_tex_image_nodes.append(lowpass_node)
                
            # Add UV map / attribute node
            # Use ShaderNodeAttribute to read from the corner-domain attribute
            uv_map_node = nodes.new("ShaderNodeAttribute")
            uv_map_node.attribute_name = f"ProjectionUV_{i}_{mat_id}"
            uv_map_node.attribute_type = 'GEOMETRY'
            uv_map_node.location = (-200, -200 * (i+1))
            uv_map_nodes.append(uv_map_node)
            
            # Compute the dot product of the direction vector and geometry normal

            if use_baked_weights:
                # ── Baked-weights path (transform-stable) ──
                # Read pre-computed per-vertex weight from mesh attribute.
                # A pass-through Math node is used so outputs[0] provides the
                # scalar float (ShaderNodeAttribute outputs[0] is Color).
                vis_attr = nodes.new("ShaderNodeAttribute")
                vis_attr.attribute_name = f"_SG_VisWeight_{i}_{mat_id}"
                vis_attr.attribute_type = 'GEOMETRY'
                vis_attr.location = (-400, (-800) * i)
                vis_attr.label = f"BakedVis-{i}-{mat_id}"

                power_weight = nodes.new("ShaderNodeMath")
                power_weight.operation = 'POWER'
                power_weight.inputs[1].default_value = context.scene.weight_exponent
                power_weight.location = (-300, (-800) * i)
                power_weight.label = 'power_weight'
                links.new(vis_attr.outputs["Fac"], power_weight.inputs[0])

                passthrough = nodes.new("ShaderNodeMath")
                passthrough.operation = 'MULTIPLY'
                passthrough.inputs[0].default_value = 1.0
                passthrough.location = (-200, (-800) * i)
                passthrough.label = f"Angle-{i}-{mat_id}"
                links.new(power_weight.outputs[0], passthrough.inputs[1])

                scripts_to_connect = []
                final_weight_node = passthrough

                script_nodes.append(scripts_to_connect)

                if i > stop_index and not voronoi_active:
                    less_than = nodes.new("ShaderNodeMath")
                    less_than.operation = 'LESS_THAN'
                    less_than.label = f"StopGate-{i}-{mat_id}"
                    less_than.location = (-100, (-800) * i)
                    less_than.inputs[1].default_value = -1
                    links.new(passthrough.outputs[0], less_than.inputs[0])
                    script_nodes_outputs.append(less_than)
                else:
                    script_nodes_outputs.append(final_weight_node)

                subtract_nodes.append(None)
                normalize_nodes.append(None)
                add_camera_loc_nodes.append(None)
                length_nodes.append(None)
                camera_fov_nodes.append(None)
                camera_aspect_ratio_nodes.append(None)
                camera_direction_nodes.append(None)
                camera_up_nodes.append(None)

            elif use_native_raycast:
                # ── Native Raycast path (Blender 5.1+) ──
                result = create_native_raycast_visibility(
                    nodes, links, camera, geometry, context, i, mat_id, stop_index
                )
                angle_weight = result[0]
                # Destructure result for create_native_feather
                _, _, normalize_node, length_node, camera_loc_node, camera_fov_node, camera_aspect_node, camera_dir_node, camera_up_node = result

                scripts_to_connect = []  # No OSL scripts to connect
                final_weight_node = None

                is_local_edit_mode = (context.scene.generation_method == 'local_edit' or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'local_edit'))
                if is_local_edit_mode:
                    feather_weight = create_native_feather(
                        nodes, links, normalize_node, camera_fov_node, camera_aspect_node,
                        camera_dir_node, camera_up_node, camera, context, i, mat_id
                    )
                    # Optionally add edge-feather mask as third multiplier
                    if context.scene.refine_edge_feather_projection and i <= stop_index:
                        ef_node = _create_edge_feather_weight(i, mat_id)
                        if ef_node is not None:
                            final_weight_node = (angle_weight, feather_weight, ef_node)
                        else:
                            final_weight_node = (angle_weight, feather_weight)
                    else:
                        final_weight_node = (angle_weight, feather_weight)
                else:
                    final_weight_node = angle_weight
                    if (getattr(context.scene, 'sg_rework_acquisition_quality', False)
                            and context.scene.generation_method == 'sequential'
                            and context.scene.visibility_vignette):
                        frame_weight = create_native_feather(
                            nodes, links, normalize_node, camera_fov_node,
                            camera_aspect_node, camera_dir_node, camera_up_node,
                            camera, context, i, mat_id,
                        )
                        frame_quality = nodes.new("ShaderNodeMath")
                        frame_quality.operation = 'MULTIPLY'
                        frame_quality.label = f"FrameEdgeQuality-{i}-{mat_id}"
                        links.new(angle_weight.outputs[0], frame_quality.inputs[0])
                        links.new(frame_weight.outputs[0], frame_quality.inputs[1])
                        final_weight_node = frame_quality

                script_nodes.append(scripts_to_connect)

                if i > stop_index and not voronoi_active:
                    less_than = nodes.new("ShaderNodeMath")
                    less_than.operation = 'LESS_THAN'
                    less_than.label = f"StopGate-{i}-{mat_id}"
                    less_than.location = (-200, (-800) * i)
                    less_than.inputs[1].default_value = -1
                    source_node = final_weight_node[0] if isinstance(final_weight_node, tuple) else final_weight_node
                    links.new(source_node.outputs[0], less_than.inputs[0])
                    script_nodes_outputs.append(less_than)
                else:
                    script_nodes_outputs.append(final_weight_node)

                # Append None to node lists (connections handled inside create functions)
                subtract_nodes.append(None)
                normalize_nodes.append(None)
                add_camera_loc_nodes.append(None)
                length_nodes.append(length_node)
                camera_fov_nodes.append(camera_fov_node)
                camera_aspect_ratio_nodes.append(camera_aspect_node)
                camera_direction_nodes.append(camera_dir_node)
                camera_up_nodes.append(camera_up_node)
            else:
                # ── OSL path (Blender < 5.1) ──
                subtract = nodes.new("ShaderNodeVectorMath")
                subtract.operation = 'SUBTRACT'
                subtract.location = (-400, -300 + (-800) * i)
                subtract.inputs[1].default_value = _camera_world_location(camera)
                subtract_nodes.append(subtract)

                normalize = nodes.new("ShaderNodeVectorMath")
                normalize.operation = 'NORMALIZE'
                normalize.location = (-400, -500 + (-800) * (i))
                normalize_nodes.append(normalize)

                # Add script nodes (Angle)
                script_angle = nodes.new("ShaderNodeScript")
                script_angle.location = (-400, (-800) * i)
                
                # Load OSL script into internal text block for portability
                raycast_osl_text = get_or_create_osl_text("raycast.osl")
                if raycast_osl_text:
                    script_angle.mode = 'INTERNAL'
                    script_angle.script = raycast_osl_text
                else:
                    script_angle.mode = 'EXTERNAL'
                    script_angle.filepath = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)), "raycast.osl"
                    )

                script_angle.inputs["AngleThreshold"].default_value = context.scene.discard_factor
                script_angle.inputs["Power"].default_value = context.scene.weight_exponent
                if "AllowBackface" in script_angle.inputs:
                    script_angle.inputs["AllowBackface"].default_value = float(
                        getattr(context.scene, 'sg_allow_backface_projection', False)
                    )
                if "RayEpsilon" in script_angle.inputs:
                    script_angle.inputs["RayEpsilon"].default_value = _scene_ray_epsilon(context)
                if "CameraShiftX" in script_angle.inputs:
                    script_angle.inputs["CameraShiftX"].default_value = (
                        camera.data.shift_x
                        if getattr(context.scene, 'sg_rework_projection_geometry', False) else 0.0
                    )
                if "CameraShiftY" in script_angle.inputs:
                    script_angle.inputs["CameraShiftY"].default_value = (
                        camera.data.shift_y
                        if getattr(context.scene, 'sg_rework_projection_geometry', False) else 0.0
                    )
                script_angle.label = f"Angle-{i}-{mat_id}"

                scripts_to_connect = [script_angle]
                final_weight_node = None

                is_local_edit_mode = (
                    context.scene.generation_method == 'local_edit'
                    or (context.scene.model_architecture.startswith('qwen')
                        and context.scene.qwen_generation_method == 'local_edit')
                )
                needs_frame_quality = (
                    context.scene.generation_method == 'sequential'
                    and context.scene.visibility_vignette
                )
                if is_local_edit_mode or needs_frame_quality:
                    # Add Feather script
                    script_feather = nodes.new("ShaderNodeScript")
                    script_feather.location = (-400, (-800) * i - 200) # Offset slightly

                    # Load OSL script into internal text block for portability
                    feather_osl_text = get_or_create_osl_text("feather.osl")
                    if feather_osl_text:
                        script_feather.mode = 'INTERNAL'
                        script_feather.script = feather_osl_text
                    else:
                        script_feather.mode = 'EXTERNAL'
                        script_feather.filepath = os.path.join(
                            os.path.dirname(os.path.dirname(__file__)), "feather.osl"
                        )

                    if context.scene.visibility_vignette: # Only set if feathering is active
                        script_feather.inputs["EdgeFeather"].default_value = context.scene.visibility_vignette_width
                        script_feather.inputs["EdgeGamma"].default_value = context.scene.visibility_vignette_softness
                    else:
                        script_feather.inputs["EdgeFeather"].default_value = 0.0
                        script_feather.inputs["EdgeGamma"].default_value = 1.0
                    script_feather.label = f"Feather-{i}-{mat_id}"
                    if "CameraShiftX" in script_feather.inputs:
                        script_feather.inputs["CameraShiftX"].default_value = (
                            camera.data.shift_x
                            if getattr(context.scene, 'sg_rework_projection_geometry', False) else 0.0
                        )
                    if "CameraShiftY" in script_feather.inputs:
                        script_feather.inputs["CameraShiftY"].default_value = (
                            camera.data.shift_y
                            if getattr(context.scene, 'sg_rework_projection_geometry', False) else 0.0
                        )
                    scripts_to_connect.append(script_feather)

                    # Optionally add edge-feather mask as third multiplier
                    if is_local_edit_mode and context.scene.refine_edge_feather_projection and i <= stop_index:
                        ef_node = _create_edge_feather_weight(i, mat_id)
                        if ef_node is not None:
                            final_weight_node = (script_angle, script_feather, ef_node)
                        else:
                            final_weight_node = (script_angle, script_feather)
                    elif is_local_edit_mode:
                        final_weight_node = (script_angle, script_feather)
                    elif getattr(context.scene, 'sg_rework_acquisition_quality', False):
                        frame_quality = nodes.new("ShaderNodeMath")
                        frame_quality.operation = 'MULTIPLY'
                        frame_quality.label = f"FrameEdgeQuality-{i}-{mat_id}"
                        links.new(script_angle.outputs[0], frame_quality.inputs[0])
                        links.new(script_feather.outputs[0], frame_quality.inputs[1])
                        final_weight_node = frame_quality
                    else:
                        final_weight_node = script_angle
                else:
                    # Just use Angle script output directly
                    final_weight_node = script_angle

                script_nodes.append(scripts_to_connect)

                if i > stop_index and not voronoi_active:
                    # Connect a temporary less than node to the script node
                    less_than = nodes.new("ShaderNodeMath")
                    less_than.operation = 'LESS_THAN'
                    less_than.label = f"StopGate-{i}-{mat_id}"
                    less_than.location = (-200, (-800) * i)
                    less_than.inputs[1].default_value = -1
                    
                    # For stop_index check, we use the Angle script output
                    # (or the first element if it's a tuple)
                    source_node = final_weight_node[0] if isinstance(final_weight_node, tuple) else final_weight_node
                    links.new(source_node.outputs[0], less_than.inputs[0])
                    
                    script_nodes_outputs.append(less_than)
                else:
                    script_nodes_outputs.append(final_weight_node)

                # Add additional add node, which will contain camera's FOV in first default value, and camera's aspect ratio in second default value
                camera_fov = nodes.new("ShaderNodeValue")
                camera_fov.location = (-600, 200 + 300 * i)
                cam_res_x, cam_res_y = _get_camera_resolution(camera, context.scene)
                fov = camera.data.angle_x
                # Correct the FOV for vertical aspect ratio
                if cam_res_y > cam_res_x:
                    fov = 2 * atan(tan(fov / 2) * cam_res_x / cam_res_y)
                camera_fov.outputs[0].default_value = fov
                camera_fov_nodes.append(camera_fov)
                camera_aspect_ratio = nodes.new("ShaderNodeValue")
                camera_aspect_ratio.location = (-600, 200 + 300 * i)
                
                # Add camera direction and up nodes (combine XYZ)
                camera_direction = nodes.new("ShaderNodeCombineXYZ")
                camera_direction.location = (-600, 200 + 300 * i)
                # Get the camera direction vector
                camera_direction.inputs[0].default_value = (camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))).x
                camera_direction.inputs[1].default_value = (camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))).y
                camera_direction.inputs[2].default_value = (camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))).z
                camera_direction_nodes.append(camera_direction)
                camera_up = nodes.new("ShaderNodeCombineXYZ")
                camera_up.location = (-600, 200 + 300 * i)
                # Get the camera up vector
                camera_up.inputs[0].default_value = (camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))).x
                camera_up.inputs[1].default_value = (camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))).y
                camera_up.inputs[2].default_value = (camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))).z
                camera_up_nodes.append(camera_up)
                
                camera_aspect_ratio.outputs[0].default_value = cam_res_x / cam_res_y
                camera_aspect_ratio_nodes.append(camera_aspect_ratio)

                # Add combine XYZ node
                add_camera_loc = nodes.new("ShaderNodeCombineXYZ")
                add_camera_loc.location = (-600, 200 + 300 * i)
                camera_location = _camera_world_location(camera)
                add_camera_loc.inputs[0].default_value = camera_location.x
                add_camera_loc.inputs[1].default_value = camera_location.y
                add_camera_loc.inputs[2].default_value = camera_location.z
                add_camera_loc_nodes.append(add_camera_loc)

                # Add length node
                length = nodes.new("ShaderNodeVectorMath")
                length.operation = 'LENGTH'
                length.location = (-400, 200 * (i+1))
                length_nodes.append(length)

            # Add projected pixel density to live native/OSL paths.  Baked
            # acquisition attributes already contain this term.
            if (getattr(context.scene, 'sg_rework_acquisition_quality', False)
                    and not use_baked_weights and script_nodes_outputs
                    and not isinstance(script_nodes_outputs[-1], tuple)
                    and length_nodes[-1] is not None):
                cam_res_x, cam_res_y = _get_camera_resolution(camera, context.scene)
                fov_for_density = camera.data.angle_x
                if cam_res_y > cam_res_x:
                    fov_for_density = 2 * atan(
                        tan(fov_for_density / 2) * cam_res_x / cam_res_y
                    )
                focal_pixels = cam_res_x / max(2.0 * tan(fov_for_density / 2), 1e-7)
                density_numerator = (
                    _scene_projection_scale(context) * focal_pixels
                    / max(0.5 * min(cam_res_x, cam_res_y), 1.0)
                )
                density = nodes.new("ShaderNodeMath")
                density.operation = 'DIVIDE'
                density.inputs[0].default_value = density_numerator
                density.label = f"ProjectedDensity-{i}-{mat_id}"
                links.new(length_nodes[-1].outputs[0], density.inputs[1])
                clamp_density = nodes.new("ShaderNodeMath")
                clamp_density.operation = 'MINIMUM'
                clamp_density.inputs[1].default_value = 1.0
                links.new(density.outputs[0], clamp_density.inputs[0])
                acquisition = nodes.new("ShaderNodeMath")
                acquisition.operation = 'MULTIPLY'
                acquisition.label = f"AcquisitionQuality-{i}-{mat_id}"
                links.new(script_nodes_outputs[-1].outputs[0], acquisition.inputs[0])
                links.new(clamp_density.outputs[0], acquisition.inputs[1])
                script_nodes_outputs[-1] = acquisition

            use_qwen_write_mask = (
                i > 0
                and i <= stop_index
                and context.scene.model_architecture == 'qwen_image_edit'
                and context.scene.generation_method == 'sequential'
                and context.scene.qwen_generation_method == 'generate'
                and context.scene.qwen_context_render_mode in {'REPLACE_STYLE', 'ADDITIONAL'}
                and getattr(context.scene, 'qwen_use_projection_write_masks', False)
            )
            use_sdxl_write_mask = (
                getattr(context.scene, 'sg_rework_sdxl_registration', False)
                and
                i > 0
                and i <= stop_index
                and context.scene.model_architecture == 'sdxl'
                and context.scene.generation_method == 'sequential'
            )
            if (use_qwen_write_mask or use_sdxl_write_mask) and script_nodes_outputs:
                current_weight = script_nodes_outputs[-1]
                if not isinstance(current_weight, tuple):
                    write_weight = _create_write_mask_weight(
                        i, mat_id, binary=use_qwen_write_mask
                    )
                    if write_weight is not None:
                        masked_weight = nodes.new("ShaderNodeMath")
                        masked_weight.operation = 'MULTIPLY'
                        masked_weight.location = (0, (-800) * i + 100)
                        masked_weight.label = f"QwenWriteMasked-{i}-{mat_id}"
                        links.new(current_weight.outputs[0], masked_weight.inputs[0])
                        links.new(write_weight.outputs[0], masked_weight.inputs[1])
                        script_nodes_outputs[-1] = masked_weight

            if (use_sdxl_write_mask and script_nodes_outputs
                    and not isinstance(script_nodes_outputs[-1], tuple)):
                alpha_weighted = nodes.new("ShaderNodeMath")
                alpha_weighted.operation = 'MULTIPLY'
                alpha_weighted.label = f"SDXLFeatherAlpha-{i}-{mat_id}"
                links.new(script_nodes_outputs[-1].outputs[0], alpha_weighted.inputs[0])
                links.new(tex_image_nodes[i].outputs["Alpha"], alpha_weighted.inputs[1])
                script_nodes_outputs[-1] = alpha_weighted

            if script_nodes_outputs and not isinstance(script_nodes_outputs[-1], tuple):
                guard_weight = _create_projection_guard_weight(i, mat_id)
                if guard_weight is not None:
                    # A projection guard marks depth/object/material boundaries
                    # as unsafe for ownership, but it must not delete the last
                    # available coverage. Complex character meshes can have
                    # dense ID boundaries around hair, facial parts, fingers,
                    # clothing openings, and adjacent mesh pieces. If every
                    # camera is hard-zeroed at the same surface point, the
                    # ownership graph falls through to the material's black
                    # fallback. Keep a tiny, untrusted confidence floor: safe
                    # views still beat it overwhelmingly, while the best weak
                    # source fills a texel only when no guarded source exists.
                    guard_floor = nodes.new("ShaderNodeMath")
                    guard_floor.operation = 'MAXIMUM'
                    guard_floor.inputs[1].default_value = 0.001
                    guard_floor.label = f"ProjectionGuardFloor-{i}-{mat_id}"
                    links.new(guard_weight.outputs[0], guard_floor.inputs[0])

                    guarded = nodes.new("ShaderNodeMath")
                    guarded.operation = 'MULTIPLY'
                    guarded.label = f"DepthPartGuarded-{i}-{mat_id}"
                    links.new(script_nodes_outputs[-1].outputs[0], guarded.inputs[0])
                    links.new(guard_floor.outputs[0], guarded.inputs[1])
                    script_nodes_outputs[-1] = guarded

        # Build mix shader tree
        if mat.name in processed_materials and is_local_edit:
            # If we already processed this material, we don't need to rebuild the mix tree
            # But we still need to connect the nodes for the current object (done in the loop below)
            # We need to find the existing mix node to position the output node
            # The mix node is connected to the output node
            mix_node = output.inputs["Surface"].links[0].from_node
        else:
            mix_node, _ = build_mix_tree(tex_image_nodes, script_nodes_outputs, nodes, links, previous_node, stop_index=stop_index)
            links.new(mix_node.outputs[0], output.inputs["Surface"])
            processed_materials.add(mat.name)

        # Move output node right to the mix_node
        output.location = (mix_node.location[0] + 400, mix_node.location[1])

        for i, camera in enumerate(cameras):
            tex_image = tex_image_nodes[i]
            uv_map_node = uv_map_nodes[i]

            # Connect UV → Texture (needed for both paths)
            uv_output = "Vector" if uv_map_node.type == 'ATTRIBUTE' else "UV"
            links.new(uv_map_node.outputs[uv_output], tex_image.inputs["Vector"])
            lowpass_tex = lowpass_tex_image_nodes[i]
            if lowpass_tex is not tex_image:
                links.new(uv_map_node.outputs[uv_output], lowpass_tex.inputs["Vector"])

            if use_baked_weights or use_native_raycast:
                # Baked-weights path: all links already made during creation.
                # Native raycast path: all links already made inside
                # create_native_raycast_visibility / create_native_feather.
                pass
            else:
                # OSL path: connect subtract, normalize, script inputs, length
                subtract = subtract_nodes[i] 
                normalize = normalize_nodes[i] 
                scripts = script_nodes[i]
                add_camera_loc = add_camera_loc_nodes[i] 
                length = length_nodes[i] 
                camera_fov = camera_fov_nodes[i]
                camera_aspect_ratio = camera_aspect_ratio_nodes[i]
                camera_direction = camera_direction_nodes[i]
                camera_up = camera_up_nodes[i]

                links.new(geometry.outputs["Position"], subtract.inputs[0])
                links.new(subtract.outputs["Vector"], normalize.inputs[0])
                
                for script in scripts:
                    links.new(normalize.outputs["Vector"], script.inputs["Direction"])
                    
                    if "Origin" in script.inputs:
                        links.new(add_camera_loc.outputs["Vector"], script.inputs["Origin"])
                    
                    if "threshold" in script.inputs:
                        links.new(length.outputs["Value"], script.inputs["threshold"])
                        
                    if "SurfaceNormal" in script.inputs:
                        links.new(geometry.outputs["Normal"], script.inputs["SurfaceNormal"])
                    links.new(camera_fov.outputs[0], script.inputs["CameraFOV"])
                    links.new(camera_aspect_ratio.outputs[0], script.inputs["CameraAspect"])
                    links.new(camera_direction.outputs[0], script.inputs["CameraDir"])
                    links.new(camera_up.outputs[0], script.inputs["CameraUp"])
                
                links.new(subtract.outputs["Vector"], length.inputs[0])

        # Add material index node (subtract node)
        subtract_node = nodes.new("ShaderNodeMath")
        subtract_node.operation = 'SUBTRACT'
        subtract_node.inputs[0].default_value = mat_id
        subtract_node.location = (-1000, 0)
    return True
    

def get_or_create_osl_text(filename):
    """
    Loads an OSL script into a Blender internal text block.
    This ensures the script is embedded in the .blend file (portable).
    """
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), filename)
    if not os.path.exists(filepath):
        print(f"[StableGen] Error: OSL script not found at {filepath}")
        return None

    # Check if text block exists
    text = bpy.data.texts.get(filename)
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        if text:
            # Update content to match the current file on disk
            text.clear()
            text.write(content)
        else:
            text = bpy.data.texts.new(name=filename)
            text.write(content)
    except Exception as e:
        print(f"[StableGen] Error reading OSL file {filepath}: {e}")
        return None
        
    return text


def get_or_load_image(filepath, force_reload=False):
    """
    Prevents duplicate image datablocks by default.
    If force_reload is True, it finds the existing datablock 
    and reloads it from the specified filepath.
    """
    if not filepath:
        print("[StableGen] Error: No filepath provided to get_or_load_image.")
        return None

    filename = os.path.basename(filepath)
    image = bpy.data.images.get(filename)
    
    # Verify if the found image actually points to the requested file
    if image:
        # Normalize paths for comparison (handle // prefix and OS separators)
        try:
            # bpy.path.abspath resolves // relative paths to absolute paths
            img_path = os.path.normpath(bpy.path.abspath(image.filepath))
            req_path = os.path.normpath(bpy.path.abspath(filepath))
            
            if img_path != req_path:
                # Name collision: found an image with the same name but different path.
                # This is NOT the image we want.
                image = None
                
                # Try to find if the correct image is already loaded under a different name
                for img in bpy.data.images:
                    if img.filepath:
                        try:
                            if os.path.normpath(bpy.path.abspath(img.filepath)) == req_path:
                                image = img
                                break
                        except:
                            continue
        except Exception as e:
            print(f"[StableGen] Warning: Error comparing image paths: {e}")
            image = None
    
    if image and force_reload:
        # Image exists, but we are forced to reload (overwrite).
        try:
            # IMPORTANT: Update the filepath property of the existing
            # datablock to the new file path.
            image.filepath = filepath
            
            # Reload the image data from that path.
            image.reload()
        except RuntimeError as e:
            # Reload can fail if the file isn't found, etc.
            print(f"[StableGen] Reload failed for {filename}. Removing old datablock. Error: {e}")
            # Remove the bad datablock so we can try loading it fresh.
            bpy.data.images.remove(image)
            image = None # Set to None to trigger the load block below

    if image is None:
        # Image does not exist in .data, or it failed to reload.
        try:
            image = bpy.data.images.load(filepath)
            # Only set the name if it's a new datablock to avoid renaming existing ones
            # Blender will handle naming collisions (e.g. .001) automatically
            if not bpy.data.images.get(filename):
                image.name = filename 
        except RuntimeError as e:
            # Load can fail if the file isn't found.
            print(f"[StableGen] Warning: Could not load image file: {filepath}. Error: {e}")
            return None
            
    return image


def reset_projection_material_nodes(material, angle_threshold, exponent):
    """Reset display controls without changing stored ownership confidence."""
    if not material or not material.use_nodes:
        return False
    nodes = material.node_tree.nodes
    has_sharp = any(
        node.type == 'MATH' and node.operation == 'POWER'
        and node.label.startswith('SharpW-') for node in nodes
    )
    for node in nodes:
        if node.type == 'SCRIPT' and 'AngleThreshold' in node.inputs:
            node.inputs['AngleThreshold'].default_value = angle_threshold
        elif (node.type == 'MATH' and node.operation == 'LESS_THAN'
              and node.label.startswith('AngleThreshold-')):
            node.inputs[1].default_value = angle_threshold
        elif (node.type == 'MATH' and node.operation == 'POWER'
              and node.label.startswith('SharpW-')):
            node.inputs[1].default_value = exponent
        elif node.type == 'SCRIPT' and 'Power' in node.inputs:
            node.inputs['Power'].default_value = 1.0 if has_sharp else exponent
        elif (node.type == 'MATH' and node.operation == 'POWER'
              and node.label == 'power_weight'):
            node.inputs[1].default_value = 1.0 if has_sharp else exponent
    return has_sharp


def reinstate_compare_nodes(context, to_project, stop_id_mat_id_pairs):
    """
    Reinstates the 'LESS_THAN' compare nodes that were removed for sequential generation.
    This will esentially revert given views to not-generated state for viewpoint regeneration.
    """

    for obj in to_project:
        if not obj.active_material:
            continue

        mat = obj.active_material
        if not mat.use_nodes:
            continue

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        for stop_id, mat_id in stop_id_mat_id_pairs:
            script_node = None
            # Find the script node with the specific label
            # Also match native MATH MULTIPLY nodes (Blender 5.1+ native raycast path)
            for node in nodes:
                if (node.type == 'SCRIPT' or (node.type == 'MATH' and node.operation == 'MULTIPLY')) and node.label == f"{stop_id}-{mat_id}":
                    script_node = node
                    break
                # Also check Angle-prefixed labels
                if (node.type == 'SCRIPT' or (node.type == 'MATH' and node.operation == 'MULTIPLY')) and node.label == f"Angle-{stop_id}-{mat_id}":
                    script_node = node
                    break
            
            if not script_node:
                continue

            # Store links to disconnect and reconnect later
            links_to_recreate = []
            for link in list(script_node.outputs[0].links):
                links_to_recreate.append((link.from_socket, link.to_socket))
                links.remove(link)

            # For each original connection, insert a 'LESS_THAN' node
            for from_socket, to_socket in links_to_recreate:
                # Create a new 'LESS_THAN' math node
                less_than_node = nodes.new(type='ShaderNodeMath')
                less_than_node.operation = 'LESS_THAN'
                less_than_node.inputs[1].default_value = -1
                # Position it between the script node and its original destination
                less_than_node.location = (script_node.location.x + 200, script_node.location.y)

                # Connect script_node -> less_than_node -> original destination
                links.new(from_socket, less_than_node.inputs[0])
                links.new(less_than_node.outputs[0], to_socket)


# ── PBR Projection ────────────────────────────────────────────────────
