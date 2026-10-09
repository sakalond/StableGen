from ..feature_flags import experimental_enabled
import bpy
import os
from math import atan, tan, pi

from ..utils import get_last_material_index, get_file_path, get_dir_path
from .rendering import prepare_baking, bake_texture, unwrap
from ..cameras.geometry import _get_camera_resolution
from mathutils import Matrix, Vector
from .svm_budget import SVM_STACK_SIZE, estimate_svm_stack


_SG_BUFFER_UV_NAME = "_SG_ProjectionBuffer"
# The estimate matches Cycles; warn a little before the hard limit.
_SVM_STACK_WARNING = SVM_STACK_SIZE - 16


# Cycles compiles an SVM shader in sweeps over the (inlined) nodes, and a node
# reading another socket type waits for a conversion node that Cycles creates
# after all nodes, i.e. for the next sweep. Everything a camera computes before
# such a wait stays on the 255-float SVM stack until the wait is over, for all
# cameras at once. The raycast's direction is such a conversion, so projection
# graphs keep the values that wait for it few, give masks explicit float
# conversions and read the masks' UVs through the camera's raycast result.
# svm_budget.estimate_svm_stack() replays the compiler to check the result.

def _mask_value(nodes, links, texture):
    """Grey mask texture as a float, without an implicit Cycles conversion."""
    value = nodes.new('ShaderNodeRGBToBW')
    value.label = texture.label + ' value'
    value.hide = True
    links.new(texture.outputs['Color'], value.inputs[0])
    return value


def _absorb_unused_raycast_outputs(nodes, links, raycast, visible):
    """Return ``visible`` with the raycast's unused outputs read but ignored.

    Cycles' Raycast node writes all five outputs to the SVM stack and frees an
    output only after reading it, so unread outputs leak 7 floats per camera.
    Each is read by a node of its exact type (no conversion) and folded in as
    max(visible, min(..., 0)), which equals ``visible`` (>= 0) and ignores NaN.
    """
    position = nodes.new('ShaderNodeTexGradient')
    position.label = 'Read hit position'
    links.new(raycast.outputs['Hit Position'], position.inputs['Vector'])
    normal = nodes.new('ShaderNodeFresnel')
    normal.label = 'Read hit normal'
    links.new(raycast.outputs['Hit Normal'], normal.inputs['Normal'])
    hit = _projection_feather_math(nodes, links, 'MINIMUM', 'Unused raycast outputs',
                                   raycast.outputs['Self Hit'], position.outputs['Fac'])
    surface = _projection_feather_math(nodes, links, 'MINIMUM', 'Unused raycast outputs',
                                       hit.outputs[0], normal.outputs['Fac'])
    zero = _projection_feather_math(nodes, links, 'MINIMUM', 'Unused raycast outputs (zero)',
                                    surface.outputs[0], 0.0)
    absorbed = _projection_feather_math(nodes, links, 'MAXIMUM', 'Visible from camera',
                                        visible, zero.outputs[0])
    return absorbed, (position, normal, hit, surface, zero)


def _late_projection_uv(nodes, links, uv_node, visible, camera_index, mat_id):
    """Projection UV that Cycles reads only after the camera's raycast.

    Masks and the generated image only depend on the UV, so Cycles would
    sample them for every camera in the sweep before any raycast and keep all
    those values on the stack. min(uv, max(visible, 1e30)) equals the UV but
    depends on the raycast, so each camera's textures follow its raycast.
    """
    def new(node_type, label):
        node = nodes.new(node_type)
        node.label = f"{label}-{camera_index}-{mat_id}"
        node.hide = True
        node['sg_layout_stage'] = 'Image and UV'
        return node

    bound = new('ShaderNodeMath', 'ProjectionUVAfterRaycastBound')
    bound.operation = 'MAXIMUM'
    bound.inputs[1].default_value = 1e30
    links.new(visible, bound.inputs[0])
    bounds = new('ShaderNodeCombineXYZ', 'ProjectionUVAfterRaycastBounds')
    for axis in range(3):
        links.new(bound.outputs[0], bounds.inputs[axis])
    late = new('ShaderNodeVectorMath', 'ProjectionUVAfterRaycast')
    late.operation = 'MINIMUM'
    links.new(_projection_uv_output(uv_node), late.inputs[0])
    links.new(bounds.outputs[0], late.inputs[1])
    return late


# Materials whose estimated SVM stack use is near or past Cycles' limit,
# reported by the generation operators once they finish.
_svm_stack_warnings = {}


def check_projection_svm_stack(material, estimates=None):
    """Estimate the material's Cycles SVM stack use and warn when too big.

    Cycles only logs an SVM stack overflow and renders the material black,
    which also blanks every Qwen context render made from it. ``estimates``
    caches results for materials with the same topology within one build.
    """
    tree = material.node_tree
    key = (len(tree.nodes), len(tree.links))
    if estimates is not None and key in estimates:
        peak, details = estimates[key]
    else:
        peak, details = estimate_svm_stack(material)
        if estimates is not None:
            estimates[key] = (peak, details)
    material['sg_svm_stack_estimate'] = peak
    if peak >= _SVM_STACK_WARNING:
        holders = ', '.join(f'{name} ({size})' for name, size in details.get('holders', ())[:4])
        print(f"[StableGen] WARNING: projection material '{material.name}' needs about "
              f"{peak} of Cycles' {SVM_STACK_SIZE} SVM stack slots and may render black "
              f"in Cycles. Largest live values: {holders}")
        _svm_stack_warnings[material.name] = peak
    else:
        _svm_stack_warnings.pop(material.name, None)
    return peak


def pop_svm_stack_warning():
    """Return a one-line user warning for oversized materials, then reset."""
    if not _svm_stack_warnings:
        return None
    names = sorted(_svm_stack_warnings)
    peak = max(_svm_stack_warnings.values())
    _svm_stack_warnings.clear()
    shown = ', '.join(names[:3]) + (f' and {len(names) - 3} more' if len(names) > 3 else '')
    return (f"Projection material {shown} is near Cycles' shader size limit "
            f"(~{peak}/{SVM_STACK_SIZE}) and may render black. Use fewer cameras "
            f"or disable some Precise+ features.")


def _is_projection_mesh(obj):
    """Consistent render/visibility filter for production geometry."""
    if obj.type != 'MESH' or obj.hide_get():
        return False
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


def _multiply_projection_weight_by_alpha(
    nodes, links, weight_node, image_node, label
):
    """Return a projection weight multiplied by an image's alpha channel."""
    alpha_weighted = nodes.new("ShaderNodeMath")
    alpha_weighted.operation = 'MULTIPLY'
    alpha_weighted.label = label
    links.new(weight_node.outputs[0], alpha_weighted.inputs[0])
    links.new(image_node.outputs["Alpha"], alpha_weighted.inputs[1])
    return alpha_weighted





def _camera_world_location(camera):
    """Return camera position in world space (including parent transforms)."""
    return camera.matrix_world.translation.copy()


def _scene_ray_epsilon(context):
    """Scale ray tolerances to the scene instead of assuming metre-sized assets."""
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


def _projection_uvw_name(camera_index, mat_id):
    return f"ProjectionUVW_{camera_index}_{mat_id}"


def _write_projection_uvw(obj, camera, uv_attr_name, uvw_attr_name):
    """Store homogeneous projection coordinates ``(u*w, v*w, w)`` per corner.

    ``ProjectionUV_*`` holds the UV Project modifier's per-vertex result, which
    is already divided by depth.  Interpolating it linearly across a face is
    not perspective-correct, so large faces (a 4 m floor quad) receive the
    image visibly shifted.  ``u*w``, ``v*w`` and ``w`` are affine in position
    and interpolate exactly; the material divides per shading point.

    Only perspective cameras need this.  Returns True when the attribute was
    written; otherwise any stale attribute is removed and the material falls
    back to ``ProjectionUV_*``.
    """
    import numpy as np

    mesh = obj.data
    stale = mesh.attributes.get(uvw_attr_name)
    if stale is not None:
        mesh.attributes.remove(stale)
    uv_attr = mesh.attributes.get(uv_attr_name)
    if camera.data.type != 'PERSP' or uv_attr is None or not mesh.loops:
        return False

    positions = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", positions)
    positions = positions.reshape((-1, 3))
    # The modifier projects through the normalized (unscaled) camera matrix.
    to_camera = np.array(
        camera.matrix_world.normalized().inverted() @ obj.matrix_world,
        dtype=np.float64,
    )
    camera_space = positions @ to_camera[:3, :3].T + to_camera[:3, 3]

    loop_vertices = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertices)
    uv = np.empty(len(mesh.loops) * 3, dtype=np.float32)
    uv_attr.data.foreach_get("vector", uv)
    uv = uv.reshape((-1, 3))[:, :2].astype(np.float64)

    corner_space = camera_space[loop_vertices]
    w = -corner_space[:, 2]

    # In front of the camera the modifier's UV is A * x / w + B per axis.
    # Fit A and B there and emit A * x + B * w for every corner: the modifier
    # mirrors corners behind the camera, while the homogeneous form stays
    # exact for faces crossing the camera plane.  A poor fit means shape keys
    # or skeletal parenting fed the modifier other positions than the base
    # mesh, so the legacy path is safer.
    usable = w > 1e-4
    homogeneous = np.empty((len(w), 3), dtype=np.float64)
    homogeneous[:, 2] = w
    for axis in range(2):
        ratio = corner_space[usable, axis] / w[usable]
        if ratio.size < 2 or np.ptp(ratio) < 1e-6:
            return False
        basis = np.column_stack((ratio, np.ones_like(ratio)))
        target = uv[usable, axis]
        scale, offset = np.linalg.lstsq(basis, target, rcond=None)[0]
        error = np.abs(basis @ (scale, offset) - target) / np.maximum(np.abs(target), 1.0)
        if error.max() > 1e-4:
            print(
                f"[StableGen] Warning: projection depth for {obj.name} does not "
                f"match its projection UVs (error {error.max():.2e}); keeping "
                "per-vertex UV interpolation for this camera."
            )
            return False
        homogeneous[:, axis] = scale * corner_space[:, axis] + offset * w

    attr = mesh.attributes.new(name=uvw_attr_name, type='FLOAT_VECTOR', domain='CORNER')
    attr.data.foreach_set("vector", homogeneous.astype(np.float32).ravel())
    mesh.update()
    return True


def _create_projection_uv_nodes(nodes, links, camera_index, mat_id, location):
    """Build the per-camera projection UV and return its output node.

    The result's ``Vector`` output is ``(u, v, 0)``.  When the mesh carries
    ``ProjectionUVW_*`` the divide happens per shading point, which keeps the
    mapping exact on faces of any size.  Meshes without it (older
    projections, orthographic cameras, joined objects) read zero ``w`` and
    use the per-vertex ``ProjectionUV_*`` instead.
    """
    x, y = location

    def new(node_type, label, offset_x, offset_y=0):
        node = nodes.new(node_type)
        node.label = f"{label}-{camera_index}-{mat_id}"
        node.location = (x + offset_x, y + offset_y)
        node['sg_layout_stage'] = 'Image and UV'
        return node

    legacy = new("ShaderNodeAttribute", "ProjectionUVLinear", -800, -150)
    legacy.attribute_name = f"ProjectionUV_{camera_index}_{mat_id}"
    legacy.attribute_type = 'GEOMETRY'

    homogeneous = new("ShaderNodeAttribute", "ProjectionUVW", -800)
    homogeneous.attribute_name = _projection_uvw_name(camera_index, mat_id)
    homogeneous.attribute_type = 'GEOMETRY'

    # w = dot(uvw, Z). Cycles' Separate XYZ would keep the unused X and Y on
    # the SVM stack for the rest of the shader and reads through a conversion.
    w = new("ShaderNodeVectorMath", "ProjectionW", -600)
    w.operation = 'DOT_PRODUCT'
    w.inputs[1].default_value = (0.0, 0.0, 1.0)
    links.new(homogeneous.outputs["Vector"], w.inputs[0])

    inverse_w = new("ShaderNodeMath", "PerspectiveDivide", -450)
    inverse_w.operation = 'DIVIDE'
    inverse_w.inputs[0].default_value = 1.0
    links.new(w.outputs["Value"], inverse_w.inputs[1])

    divided = new("ShaderNodeVectorMath", "PerspectiveUV", -300)
    divided.operation = 'SCALE'
    links.new(homogeneous.outputs["Vector"], divided.inputs[0])
    links.new(inverse_w.outputs[0], divided.inputs["Scale"])

    missing = new("ShaderNodeMath", "ProjectionUVWMissing", -450, -150)
    missing.operation = 'COMPARE'
    missing.inputs[1].default_value = 0.0
    missing.inputs[2].default_value = 0.0
    links.new(w.outputs["Value"], missing.inputs[0])

    # divided + missing * (legacy - divided); plain vector math so graph
    # walkers never mistake the selector for a colour mix.
    difference = new("ShaderNodeVectorMath", "ProjectionUVFallbackDelta", -300, -150)
    difference.operation = 'SUBTRACT'
    links.new(legacy.outputs["Vector"], difference.inputs[0])
    links.new(divided.outputs["Vector"], difference.inputs[1])

    select = new("ShaderNodeVectorMath", "ProjectionUVSelect", -150)
    select.operation = 'MULTIPLY_ADD'
    links.new(difference.outputs["Vector"], select.inputs[0])
    links.new(missing.outputs[0], select.inputs[1])
    links.new(divided.outputs["Vector"], select.inputs[2])

    flatten = new("ShaderNodeVectorMath", "ProjectionUV", 0)
    flatten.operation = 'MULTIPLY'
    flatten.inputs[1].default_value = (1.0, 1.0, 0.0)
    links.new(select.outputs["Vector"], flatten.inputs[0])
    return flatten


_COARSE_FACE_FRACTION = 0.1


def _warn_coarse_projection_faces(objects, cameras, mat_id, reason):
    """Warn when per-vertex projection data cannot resolve large faces.

    The UV itself is exact per pixel, but baked visibility weights and
    Surface Blend sample visibility only at vertices, so occlusion inside a
    face spanning a sizeable part of a view is smeared across it.  Returns
    ``{object name: coarse face count}``.
    """
    import numpy as np

    coarse = {}
    for obj in objects:
        mesh = obj.data
        if not mesh.polygons:
            continue
        face_starts = np.empty(len(mesh.polygons), dtype=np.int32)
        mesh.polygons.foreach_get("loop_start", face_starts)
        flagged = np.zeros(len(mesh.polygons), dtype=bool)
        for i, camera in enumerate(cameras):
            uv_attr = mesh.attributes.get(f"ProjectionUV_{i}_{mat_id}")
            if uv_attr is None:
                continue
            uv = np.empty(len(mesh.loops) * 3, dtype=np.float32)
            uv_attr.data.foreach_get("vector", uv)
            uv = uv.reshape((-1, 3))
            in_front = np.ones(len(mesh.loops), dtype=bool)
            uvw_attr = mesh.attributes.get(_projection_uvw_name(i, mat_id))
            if uvw_attr is not None:
                uvw = np.empty(len(mesh.loops) * 3, dtype=np.float32)
                uvw_attr.data.foreach_get("vector", uvw)
                in_front = uvw.reshape((-1, 3))[:, 2] > 0
            elif camera.data.type != 'ORTHO':
                continue
            u, v = uv[:, 0], uv[:, 1]
            min_u = np.minimum.reduceat(u, face_starts)
            max_u = np.maximum.reduceat(u, face_starts)
            min_v = np.minimum.reduceat(v, face_starts)
            max_v = np.maximum.reduceat(v, face_starts)
            all_front = np.logical_and.reduceat(in_front, face_starts)
            visible = (
                all_front
                & (max_u >= 0.0) & (min_u <= 1.0)
                & (max_v >= 0.0) & (min_v <= 1.0)
            )
            extent = np.maximum(max_u - min_u, max_v - min_v)
            # Faces crossing the camera plane get zero weight at the corners
            # behind it, however small their visible part is.
            straddling = np.logical_or.reduceat(in_front, face_starts) & ~all_front
            flagged |= (visible & (extent > _COARSE_FACE_FRACTION)) | straddling
        if flagged.any():
            coarse[obj.name] = int(flagged.sum())
    if coarse:
        listed = ", ".join(f"{name} ({count})" for name, count in sorted(coarse.items()))
        print(
            f"[StableGen] Warning: {reason} samples visibility per vertex, but "
            f"some faces span more than {_COARSE_FACE_FRACTION:.0%} of a camera "
            f"view: {listed}. Occlusion inside those faces will be smeared; "
            "subdivide them for accurate projection."
        )
    return coarse


def _projection_uv_output(node):
    """UV socket of a projection UV node (attribute, UV map or built chain)."""
    return node.outputs["Vector"] if "Vector" in node.outputs else node.outputs["UV"]


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
        )
        ortho_half_h = max(float(camera.data.ortho_scale) * 0.5, 1e-8)
        ortho_half_w = ortho_half_h * aspect

        cam_data.append((
            np.array(cam_loc, dtype=np.float64),
            np.array(cam_dir, dtype=np.float64),
            np.array(cam_right, dtype=np.float64),
            np.array(cam_up_vec, dtype=np.float64),
            filmw, filmh,
            float(camera.data.shift_x),
            float(camera.data.shift_y),
            cam_loc,  # keep mathutils Vector for BVH calls
            camera_type, ortho_half_w, ortho_half_h,
        ))

    # Build scene-wide BVH for occlusion (matches ShaderNodeRaycast behaviour,
    # which checks ALL visible scene geometry, not just the current object).
    use_corrected_geometry = True
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
            # Freeze receiver depth alongside visibility so moving/posing a
            # baked projection cannot change its foreground exemption.
            depth_name = f"_SG_ProjectionDepth_{cam_idx}_{mat_id}"
            depth_attr = obj.data.attributes.get(depth_name)
            if depth_attr is None:
                depth_attr = obj.data.attributes.new(name=depth_name, type='FLOAT', domain='POINT')
            depth_attr.data.foreach_set('value', np.einsum('ij,j->i', to_vert, cam_dir_np))
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
                        20.0
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
                                20.0
                            ))
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


def create_native_raycast_visibility(nodes, links, camera, geometry, context, i, mat_id, stop_index,
                                     pre_factors=None, extra=None):
    """
    Creates a visibility weight node group using the native Raycast shader node (Blender 5.1+).
    Replicates the OSL raycast.osl shader logic:
      1. Frustum check (is the surface within the camera FOV)
      2. Occlusion check (Raycast from camera, compare Hit Distance with expected distance)
      3. Angle check (surface normal vs ray direction)
      4. Weight = pow(orthogonality, Power)

    Cycles compiles the raycast one sweep after the camera math (its direction
    is converted to a normal), so the raycast-free terms are multiplied first
    and the occlusion test last: one float per camera waits for the raycast.
    ``pre_factors(weight_socket, parts)`` may multiply further raycast-free
    terms in. ``extra`` (a dict) receives the 'visible' socket and the
    'view_depth' sockets (camera-axis depth = product of the two).

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
    visible, raycast_sinks = _absorb_unused_raycast_outputs(
        nodes, links, raycast, occlusion_ok.outputs[0])

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
    clamp_dot2.label = f'FacingSignedDot-{i}-{mat_id}'
    clamp_dot2.location = (x_base + 500, y_base - 150)
    clamp_dot2.inputs[1].default_value = 1.0
    links.new(clamp_dot.outputs[0], clamp_dot2.inputs[0])

    facing_dot = nodes.new("ShaderNodeMath")
    facing_dot.location = (x_base + 600, y_base - 150)
    facing_raw = facing_dot
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

    facing_dot.label = f'FacingWeight-{i}-{mat_id}'

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
    # COMBINE ALL: frustum_ok * angle_ok * weight [* pre_factors] * occlusion_ok
    # ========================================
    combine1 = nodes.new("ShaderNodeMath")
    combine1.operation = 'MULTIPLY'
    combine1.location = (x_base + 800, y_base)
    links.new(frustum_ok.outputs[0], combine1.inputs[0])
    links.new(angle_check.outputs[0], combine1.inputs[1])

    combine2 = nodes.new("ShaderNodeMath")
    combine2.operation = 'MULTIPLY'
    combine2.location = (x_base + 900, y_base)
    links.new(combine1.outputs[0], combine2.inputs[0])
    links.new(power_node.outputs[0], combine2.inputs[1])
    weighted = combine2.outputs[0]
    if pre_factors is not None:
        weighted = pre_factors(weighted, dict(
            normalize=normalize_node, length=length_node, fov=camera_fov_node,
            aspect=camera_aspect_node, direction=camera_dir_node, up=camera_up_node))

    final_weight = nodes.new("ShaderNodeMath")
    final_weight.operation = 'MULTIPLY'
    final_weight.location = (x_base + 1000, y_base)
    final_weight.label = f"Angle-{i}-{mat_id}"
    links.new(weighted, final_weight.inputs[0])
    links.new(visible.outputs[0], final_weight.inputs[1])
    if extra is not None:
        extra['visible'] = visible.outputs[0]
        extra['view_depth'] = (neg_cam_z.outputs[0], length_node.outputs["Value"])

    # Presentation metadata only: keep the native graph's large camera math
    # readable without changing any operation, input or connection.
    stage_names = {
        'Camera basis and frustum': '''
            add_camera_loc subtract normalize_node length_node camera_fov_node
            camera_aspect_node camera_dir_node camera_up_node norm_forward
            cross_fwd_up norm_right cross_right_fwd norm_up_vec neg_forward
            dot_d_right dot_d_up dot_d_neg_fwd cam_z_check fov_half filmw
            filmh neg_cam_z x_proj y_proj shift_x_amount centered_x
            shift_y_amount centered_y abs_x x_in_bounds abs_y y_in_bounds
            frustum_xy frustum_ok''',
        'Raycast visibility': '''
            raycast threshold_plus hit_dist_plus dist_check occlusion_ok visible''',
        'Facing and angle': '''
            dot_d_normal clamp_dot clamp_dot2 facing_raw facing_dot acos_node
            to_degrees angle_check power_node combine1 combine2 final_weight''',
    }
    readable_names = {
        'add_camera_loc': 'Camera position',
        'subtract': 'Camera to surface',
        'normalize_node': 'View direction',
        'length_node': 'Camera distance',
        'camera_fov_node': 'Camera field of view',
        'camera_aspect_node': 'Camera aspect',
        'camera_dir_node': 'Camera forward',
        'camera_up_node': 'Camera up',
        'frustum_ok': 'Inside camera frame',
        'raycast': 'Raycast to surface',
        'threshold_plus': 'Ray length tolerance',
        'hit_dist_plus': 'Hit distance tolerance',
        'dist_check': 'Target depth matches hit',
        'occlusion_ok': 'Hit is the surface',
        'dot_d_normal': 'View facing angle',
        'facing_raw': 'Apply backface setting',
        'combine1': 'Frustum and discard angle',
        'combine2': 'Apply facing weight',
    }
    available = locals()
    for stage, names in stage_names.items():
        for name in names.split():
            node = available[name]
            node['sg_layout_stage'] = stage
            if not node.label:
                node.label = readable_names.get(name, name.replace('_', ' ').capitalize())
    for node in raycast_sinks:
        node['sg_layout_stage'] = 'Raycast visibility'
    if pre_factors is not None:
        # The final weight now follows the frame-edge and density terms; keep
        # it in their frame so the graph still reads left to right.
        final_weight['sg_layout_stage'] = 'Acquisition quality'

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
    before = {node.as_pointer() for node in nodes}
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
        # Clamp explicitly: Cycles turns the Map Range clamp option into a
        # node created after all others, which delays the frame weight to the
        # raycast's compile sweep and keeps its inputs on the SVM stack.
        map_range.clamp = False
        clamped = nodes.new("ShaderNodeClamp")
        clamped.clamp_type = 'RANGE'
        clamped.inputs["Min"].default_value = 1.0
        clamped.inputs["Max"].default_value = 0.0
        links.new(map_range.outputs[0], clamped.inputs["Value"])

        # edge_factor = pow(smoothstep_result, gamma)
        pow_node = nodes.new("ShaderNodeMath")
        pow_node.operation = 'POWER'
        pow_node.location = (x_base + 700, y_base - 150)
        pow_node.inputs[1].default_value = max(0.01, gamma_val)
        links.new(clamped.outputs[0], pow_node.inputs[0])

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

    for node in nodes:
        if node.as_pointer() not in before:
            node['sg_layout_stage'] = 'Frame edge and vignette'
            if not node.label:
                node.label = 'Frame edge ' + (
                    node.operation.replace('_', ' ').lower()
                    if hasattr(node, 'operation') else node.bl_label.lower())

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


def _projection_feather_math(nodes, links, operation, label, a, b):
    node = nodes.new('ShaderNodeMath')
    node.operation = operation
    node.label = label
    if operation == 'COMPARE':
        node.inputs[2].default_value = 0.0
    for i, value in enumerate((a, b)):
        if isinstance(value, (int, float)):
            node.inputs[i].default_value = value
        else:
            links.new(value, node.inputs[i])
    return node


def _apply_guard_coverage_rescue(nodes, links, guarded, raw, guards, feathers, threshold,
                               occlusion_feathers=None, occlusion_influences=None,
                               feather_method=None):
    """Ease rejection continuously when no comparable trusted view survives.

    Work with raw quality before exponentiation, retaining raycast, alpha and
    write-mask validity. Confidence continues using the original guarded graph.
    """
    def math(op, label, a, b):
        return _projection_feather_math(nodes, links, op, label, a, b).outputs[0]
    def protect_occlusions(fades):
        if occlusion_feathers is None:
            return fades
        if feather_method == 'BOUNDARY_CROSSFADE':
            # Restore the original crossfade handoff: each camera retains its
            # own rescued guard feather, limited by its occlusion feather.
            # Lifting every competitor exposed a hard guard edge on the cheek.
            return [math('MINIMUM', f'GuardOcclusionFade-{i}',
                         fade if fade is not None else 1.0, occlusion)
                    if occlusion is not None else fade
                    for i, (fade, occlusion) in enumerate(zip(fades, occlusion_feathers))]
        # A handoff needs an unfaded receiving view. Suppressing that view
        # with the older guard feather rescales/compresses the occlusion ramp.
        # Share the occlusion transition region across ALL visible competitors,
        # not just the departing view. Retain part-seam feathering elsewhere,
        # returning to it smoothly over the occlusion field's outer shoulder.
        influence = 1.0 if occlusion_influences is None else 0.0
        for i, field in enumerate(occlusion_influences or []):
            if field is not None:
                visible = math('GREATER_THAN', f'OcclusionHandoffVisible-{i}', raw[i], 0.0)
                active = math('MULTIPLY', f'OcclusionHandoffActive-{i}', visible, field)
                influence = math('MAXIMUM', f'OcclusionHandoffRegion-{i}', influence, active)
        result = []
        for i, (fade, occlusion) in enumerate(zip(fades, occlusion_feathers)):
            if occlusion is None:
                result.append(fade)
                continue
            legacy = fade if fade is not None else 1.0
            lifted = math('ADD', f'OcclusionHandoffLegacy-{i}', legacy,
                          math('MULTIPLY', f'OcclusionHandoffLift-{i}', influence,
                               math('SUBTRACT', f'OcclusionHandoffMissing-{i}', 1.0, legacy)))
            result.append(math('MINIMUM', f'GuardOcclusionFade-{i}', lifted, occlusion))
        return result
    if not any(g is not None for g in guards):
        return guarded, protect_occlusions(feathers)
    best, trusted = 0.0, 0.0
    for i, (quality, guard, feather) in enumerate(zip(raw, guards, feathers)):
        best = math('MAXIMUM', f'GuardRescueBest-{i}', best, quality)
        reliability = guard if guard is not None else 1.0
        if feather is not None:
            reliability = math('MULTIPLY', f'GuardRescueReliability-{i}', reliability, feather)
        accepted = math('MULTIPLY', f'GuardRescueAccepted-{i}', quality, reliability)
        trusted = math('MAXIMUM', f'GuardRescueTrusted-{i}', trusted, accepted)
    threshold_product = math('MULTIPLY', 'GuardRescueThreshold', best, threshold)
    denominator = math('MAXIMUM', 'GuardRescueSafeThreshold', threshold_product, 1e-8)
    fraction = math('DIVIDE', 'GuardRescueFraction', trusted, denominator)
    t = math('MINIMUM', 'GuardRescueClamped', fraction, 1.0)
    smooth = math('MULTIPLY', 'GuardRescueSmooth', math('MULTIPLY', 'GuardRescueSquare', t, t),
                  math('SUBTRACT', 'GuardRescueCurve', 3.0, math('MULTIPLY', 'GuardRescueTwice', 2.0, t)))
    # Threshold zero means rescue is OFF, not full rescue. Keep the graph
    # present so its threshold can later be changed from one shader control.
    active = math('GREATER_THAN', 'GuardRescueActive', threshold, 0.0)
    strength = math('SUBTRACT', 'GuardRescueStrength', 1.0,
                    math('MULTIPLY', 'GuardRescueReduction', active,
                         math('SUBTRACT', 'GuardRescueMissing', 1.0, smooth)))
    restored = math('SUBTRACT', 'GuardRescueRestored', 1.0, strength)
    outputs, fades = [], []
    for i, (quality, guard, feather) in enumerate(zip(raw, guards, feathers)):
        if guard is None:
            outputs.append(guarded[i]); fades.append(feather); continue
        effective = math('ADD', f'GuardRescueEffective-{i}', restored,
                         math('MULTIPLY', f'GuardRescueRetained-{i}', strength, guard))
        floor = math('MAXIMUM', f'GuardRescueFloor-{i}', effective, 0.001)
        outputs.append(math('MULTIPLY', f'GuardCoverageWeight-{i}', quality, floor))
        fades.append(math('ADD', f'GuardRescueFeather-{i}', restored,
                          math('MULTIPLY', f'GuardRescueFade-{i}', strength, feather))
                     if feather is not None else None)
    # Rescue may restore a conservative guard, but cannot restore the fade
    # toward a real occluder. Apply this limit after rescue, before the common
    # post-exponent feather stage. Raw raycast/alpha validity stays untouched.
    return outputs, protect_occlusions(fades)


def _crossfade_projection_feathers(nodes, links, weights, feathers, bases):
    """Integrate normalized camera combinations over feather thresholds.

    For two views with feathers f and 1 this is f * mix(A, B) + (1-f) * B.
    The same construction handles overlapping fades without independent weight
    caps. Equal thresholds share one interval; invisible views cannot open one.
    Normalize by the best feather to retain coverage when every view is unsafe.
    """
    def math(op, label, a, b):
        return _projection_feather_math(nodes, links, op, label, a, b).outputs[0]

    levels = []
    maximum = 0.0
    for i, (base, feather) in enumerate(zip(bases, feathers)):
        visible = math('GREATER_THAN', f'CrossfadeVisible-{i}', base, 0.0)
        floor = math('MAXIMUM', f'CrossfadeFloor-{i}', feather if feather is not None else 1.0, 0.001)
        level = math('MULTIPLY', f'CrossfadeLevel-{i}', floor, visible)
        levels.append(level)
        maximum = math('MAXIMUM', f'CrossfadeMaxLevel-{i}', maximum, level)

    result = [0.0] * len(weights)
    for i, level in enumerate(levels):
        lower, ties, best = 0.0, 0.0, 0.0
        selected = []
        for j, (other, base) in enumerate(zip(levels, bases)):
            below = math('GREATER_THAN', f'CrossfadeBelow-{i}-{j}', level, other)
            lower = math('MAXIMUM', f'CrossfadeLower-{i}-{j}', lower,
                         math('MULTIPLY', f'CrossfadeLowerCandidate-{i}-{j}', other, below))
            ties = math('ADD', f'CrossfadeTies-{i}-{j}', ties,
                        math('COMPARE', f'CrossfadeEqual-{i}-{j}', level, other))
            include = math('SUBTRACT', f'CrossfadeInclude-{i}-{j}', 1.0, below)
            quality = math('MULTIPLY', f'CrossfadeQuality-{i}-{j}', base, include)
            selected.append(quality)
            best = math('MAXIMUM', f'CrossfadeBest-{i}-{j}', best, quality)
        interval = math('DIVIDE', f'CrossfadeInterval-{i}',
                        math('SUBTRACT', f'CrossfadeGap-{i}', level, lower),
                        math('MULTIPLY', f'CrossfadeIntervalScale-{i}', maximum, ties))
        # Rescale each subset BEFORE sharpening: a weak but sole remaining
        # camera must not underflow to black at high exponents.
        powered, total = [], 0.0
        for j, quality in enumerate(selected):
            normalized = math('DIVIDE', f'CrossfadeNormalized-{i}-{j}', quality, best)
            power = math('POWER', f'SharpW-Feather-{i}-{j}', normalized, weights[j].inputs[1].default_value)
            powered.append(power)
            total = math('ADD', f'CrossfadeTotal-{i}-{j}', total, power)
        for j, power in enumerate(powered):
            share = math('DIVIDE', f'CrossfadeShare-{i}-{j}', power, total)
            result[j] = math('ADD', f'CrossfadeAccumulated-{i}-{j}', result[j],
                             math('MULTIPLY', f'CrossfadeContribution-{i}-{j}', interval, share))
    return [_projection_feather_math(nodes, links, 'MULTIPLY', f'PostExponentFeather-{i}', value, 1.0)
            for i, value in enumerate(result)]


def _apply_projection_feathers(nodes, links, weights, feathers, bases, influences, method):
    """Apply a continuous, post-exponent fade without replacing view preference.

    Keep the rejected core dark in the texture. A small constant floor retains
    last-source coverage without the white-core discontinuity that caused halos.
    """
    if not any(feather is not None for feather in feathers):
        return weights
    if method == 'BOUNDARY_CROSSFADE':
        return _crossfade_projection_feathers(nodes, links, weights, feathers, bases)
    if method == 'LOCAL_UNSHARPEN':
        def math(op, label, a, b):
            return _projection_feather_math(nodes, links, op, label, a, b).outputs[0]
        influence = 0.0
        for i, (base, field) in enumerate(zip(bases, influences)):
            if field is None:
                continue
            visible = math('GREATER_THAN', f'FeatherVisible-{i}', base, 0.0)
            active = math('MULTIPLY', f'FeatherActive-{i}', field, visible)
            influence = math('MAXIMUM', f'FeatherInfluence-{i}', influence, active)
        sharp_fraction = math('SUBTRACT', 'FeatherSharpFraction', 1.0, influence)
        weights = [
            _projection_feather_math(nodes, links, 'ADD', f'FeatherPreference-{i}',
                math('MULTIPLY', f'FeatherSharp-{i}', weight.outputs[0], sharp_fraction),
                math('MULTIPLY', f'FeatherSoft-{i}', base, influence))
            for i, (weight, base) in enumerate(zip(weights, bases))]
    result = []
    for i, (weight, feather) in enumerate(zip(weights, feathers)):
        if feather is None:
            result.append(weight)
            continue
        floor = nodes.new('ShaderNodeMath')
        floor.operation = 'MAXIMUM'
        floor.label = f'ProjectionFeatherFloor-{i}'
        floor.inputs[1].default_value = 0.001
        links.new(feather, floor.inputs[0])
        faded = nodes.new('ShaderNodeMath')
        faded.operation = 'MULTIPLY'
        faded.label = f'PostExponentFeather-{i}'
        links.new(weight.outputs[0], faded.inputs[0])
        links.new(floor.outputs[0], faded.inputs[1])
        result.append(faded)
    return result


def project_image(
    context,
    to_project,
    mat_id,
    stop_index=1000000,
    cameras=None,
    view_plan=None,
    force_rebuild=False,
    reuse_projection_uv=False,
):
    """     
    Projects an image onto all mesh objects using UV Project Modifier.     
    """
    surface_blend = (getattr(context.scene, 'sg_guard_feather_method', 'ORIGINAL') == 'SURFACE_BLEND'
                     and getattr(context.scene, 'sg_rework_projection_guard', False)
                     and getattr(context.scene, 'sg_silhouette_feather', 0) > 0)
    if surface_blend:
        if (context.scene.generation_method in {'local_edit', 'refine'}
                or (context.scene.model_architecture.startswith('qwen')
                    and context.scene.qwen_generation_method == 'local_edit')):
            raise ValueError('Surface Blend preview supports base multi-view projection; use Local Unsharpen for local refinement')
        # A newly generated view changes the joint solution. Do not reuse a
        # sequential material containing the previous camera set's field.
        force_rebuild = True
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

        if level == 0 and surface_blend and len(shaders) > 1:
            from .surface_blend import build
            key = build(context, to_project, cameras, shaders, mat, mat_id, stop_index)
            solved = []
            for idx, original in enumerate(weight_nodes):
                if isinstance(original, tuple):
                    raise ValueError('Surface Blend preview requires ordinary multi-view projection')
                raw = original
                if raw.label.startswith('DepthPartFeathered-'):
                    raw = raw.inputs[0].links[0].from_node
                if raw.label.startswith('DepthPartGuarded-'):
                    raw = raw.inputs[0].links[0].from_node
                field = nodes_collection.new('ShaderNodeAttribute')
                field.attribute_name = key + f'Weight{idx}'
                field.label = f'SurfaceBlendWeight-{idx}'
                valid = _projection_feather_math(nodes_collection, links, 'GREATER_THAN',
                                                 f'SurfaceBlendVisible-{idx}', raw.outputs[0], 0.)
                solved.append(_projection_feather_math(nodes_collection, links, 'MULTIPLY',
                              f'SurfaceBlendFinal-{idx}', field.outputs['Fac'], valid.outputs[0]))
            return build_mix_tree(shaders, solved, nodes_collection, links, last_node, level=1,
                                  stop_index=stop_index)

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
            feather_outputs = []
            influence_outputs = []
            raw_outputs = []
            guard_outputs = []
            occlusion_outputs = []
            occlusion_influence_outputs = []
            for idx in range(len(weight_nodes)):
                v_off = -200 * idx
                influence = nodes_collection.get(f"SGFeatherInfluence-{idx}-{mat_id}")
                influence_outputs.append(influence.outputs[0] if influence else None)
                occlusion_influence = nodes_collection.get(f"SGOcclusionInfluence-{idx}-{mat_id}")
                occlusion_influence_outputs.append(occlusion_influence.outputs[0] if occlusion_influence else None)
                if occlusion_influence is not None:
                    influence_outputs[-1] = _projection_feather_math(
                        nodes_collection, links, 'MAXIMUM', f'OcclusionCombinedInfluence-{idx}',
                        influence_outputs[-1] if influence_outputs[-1] is not None else 0.0,
                        occlusion_influence.outputs[0]).outputs[0]
                occlusion = nodes_collection.get(f"SGOcclusionFeather-{idx}-{mat_id}")
                occlusion_outputs.append(occlusion.outputs[0] if occlusion else None)
                weight = weight_nodes[idx]
                if (not isinstance(weight, tuple)
                        and weight.label.startswith('DepthPartFeathered-')):
                    # Preserve the existing hard guard and source preference.
                    # Apply only the extra continuous fade after sharpening.
                    all_w_outputs.append(weight.inputs[0].links[0].from_socket)
                    feather_outputs.append(weight.inputs[1].links[0].from_socket)
                else:
                    all_w_outputs.append(_get_weight_output(weight, v_off))
                    feather_outputs.append(None)
                guard_node = all_w_outputs[-1].node
                if guard_node.label.startswith('DepthPartGuarded-'):
                    raw_outputs.append(guard_node.inputs[0].links[0].from_socket)
                    guard_floor = guard_node.inputs[1].links[0].from_node
                    guard_outputs.append(guard_floor.inputs[0].links[0].from_socket)
                else:
                    raw_outputs.append(all_w_outputs[-1])
                    guard_outputs.append(None)
            all_w_outputs, feather_outputs = _apply_guard_coverage_rescue(
                nodes_collection, links, all_w_outputs, raw_outputs, guard_outputs, feather_outputs,
                getattr(context.scene, 'sg_guard_coverage_rescue', 0.0), occlusion_outputs,
                occlusion_influence_outputs,
                getattr(context.scene, 'sg_guard_feather_method', 'ORIGINAL'))

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
            normalized_outputs = []
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
                normalized_outputs.append(div.outputs[0])

            # Replace weight_nodes with sharpened-normalised nodes
            weight_nodes = _apply_projection_feathers(
                nodes_collection, links, powered_nodes, feather_outputs,
                normalized_outputs, influence_outputs,
                getattr(context.scene, 'sg_guard_feather_method', 'ORIGINAL'))

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
                    map_range_node.label = f'EarlyPriorityMap-{level}-{i}'
                    map_range_node.location = (x_offset - 400, y_offset + vert_offset)
                    map_range_node.inputs[1].default_value = context.scene.early_priority_strength
                    links.new(div_node.outputs[0], map_range_node.inputs[0])

                # Create a MixRGB node for color blending of the two inputs
                mix_node = nodes_collection.new("ShaderNodeMixRGB")
                mix_node.label = f'CameraMix-{level}-{i}'
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

    if surface_blend:
        from .surface_blend import validate
        validate(context, to_project, cameras)
    
    # Force refresh of the UI
    for area in context.screen.areas:
        area.tag_redraw()
    
    # Apply projection to all mesh objects for each camera. Interactive
    # diagnostics can reuse the attributes created by the original generation;
    # this keeps a live preview from repeatedly applying UV Project modifiers.
    if reuse_projection_uv:
        missing = [
            f"{obj.name}:ProjectionUV_{i}_{mat_id}"
            for i, _camera in enumerate(cameras)
            for obj in to_project
            if obj.data.attributes.get(f"ProjectionUV_{i}_{mat_id}") is None
        ]
        if missing:
            sample = ", ".join(missing[:3])
            raise RuntimeError(
                "Existing projection UV data is missing (" + sample + "). "
                "Run texture generation or Reproject Textures first."
            )

    # Apply projection to all mesh objects for each camera.
    for i, camera in enumerate(cameras):
        for x, obj in enumerate(to_project):
            if reuse_projection_uv:
                continue
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
                _write_projection_uvw(obj, camera, attr_name,
                                      _projection_uvw_name(i, mat_id))

                # Restore active UV to the first non-buffer UV layer
                original_uv_map = obj.data.uv_layers[0]
                if original_uv_map.name == _SG_BUFFER_UV_NAME and len(obj.data.uv_layers) > 1:
                    original_uv_map = obj.data.uv_layers[1]
                obj.data.uv_layers.active = original_uv_map

    # Clean up the temporary buffer only when this call created projection UVs.
    if not reuse_projection_uv:
        for obj in to_project:
            buffer_uv = obj.data.uv_layers.get(_SG_BUFFER_UV_NAME)
            if buffer_uv:
                obj.data.uv_layers.remove(buffer_uv)
        if surface_blend:
            _warn_coarse_projection_faces(to_project, cameras, mat_id, "Surface Blend")

    # ── Optional: bake per-vertex visibility weights ──────────────────
    use_baked_weights = (
        getattr(context.scene, 'bake_visibility_weights', False)
        or (any((camera.data.type == 'ORTHO' for camera in cameras)))
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
            or (getattr(context.scene, 'sg_rework_projection_guard', False)
                and getattr(context.scene, 'sg_silhouette_feather', 0) > 0
                and obj.data.attributes.get(f"_SG_ProjectionDepth_{cam_idx}_{mat_id}") is None)
            for obj in to_project for cam_idx in range(len(cameras))
        )
        if reuse_projection_uv and missing_bake:
            # Saved projection materials from before transform-stable weights
            # can legitimately have reusable ProjectionUV attributes without
            # the newer _SG_VisWeight attributes.  An interactive preview must
            # stay non-destructive, so use the dynamic raycast graph rather
            # than baking new mesh data merely to inspect the context.
            use_baked_weights = False
        if (not reuse_projection_uv
                and (not is_sequential or stop_index == 0 or missing_bake)):
            _warn_coarse_projection_faces(
                to_project, cameras, mat_id, "Baked visibility weighting")
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
    svm_estimates = {}

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

        if surface_blend and mat.name in processed_materials:
            continue

        # Enable use of nodes
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        layout_existing = ({node.as_pointer() for node in nodes}
                           if is_local_edit else set())
        layout_camera_nodes = {}
        
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
            force_rebuild,
            getattr(context.scene, 'sg_rework_projection_guard', False),
            getattr(context.scene, 'qwen_use_confidence_context', False),
            experimental_enabled(context.scene, 'qwen_use_projection_write_masks'),
            context.scene.model_architecture == 'sdxl',
            context.scene.model_architecture == 'qwen_image_21',
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
            # after generation (write masks, guards, and confidence inputs).
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
        
        use_native_raycast = bpy.app.version >= (5, 1, 0)

        tex_image_nodes = []
        uv_map_nodes = []
        subtract_nodes = []
        normalize_nodes = []
        script_nodes = []
        script_nodes_outputs = []
        confidence_quality_nodes = []
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
            links.new(_projection_uv_output(uv_node), mask_tex.inputs["Vector"])

            # Color→Float conversion via a pass-through Math node
            to_float = nodes.new("ShaderNodeMath")
            to_float.operation = 'MULTIPLY'
            to_float.inputs[1].default_value = 1.0
            to_float.location = (-300, -200 * cam_index - 150)
            to_float.label = f"EF_Weight-{cam_index}"
            links.new(_mask_value(nodes, links, mask_tex).outputs[0], to_float.inputs[0])
            return to_float

        def _create_write_mask_weight(
            cam_index,
            mat_id_inner,
            binary=False,
            label_prefix="SequentialWrite",
        ):
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
            mask_tex.label = f"{label_prefix}Mask-{cam_index}-{mat_id_inner}"
            mask_tex.location = (-500, -200 * cam_index - 300)

            uv_node = uv_map_nodes[cam_index]
            links.new(_projection_uv_output(uv_node), mask_tex.inputs["Vector"])

            to_float = nodes.new("ShaderNodeMath")
            to_float.operation = 'MULTIPLY'
            to_float.inputs[1].default_value = 1.0
            to_float.location = (-300, -200 * cam_index - 300)
            to_float.label = f"{label_prefix}Weight-{cam_index}-{mat_id_inner}"
            links.new(_mask_value(nodes, links, mask_tex).outputs[0], to_float.inputs[0])
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
            links.new(_projection_uv_output(uv_node), guard_tex.inputs["Vector"])
            guard_weight = nodes.new("ShaderNodeMath")
            guard_weight.operation = 'MULTIPLY'
            guard_weight.inputs[1].default_value = 1.0
            guard_weight.label = f"ProjectionGuardWeight-{cam_index}-{mat_id_inner}"
            links.new(_mask_value(nodes, links, guard_tex).outputs[0], guard_weight.inputs[0])
            return guard_weight

        def _acquisition_quality(weight, length_node, camera, cam_index):
            """Multiply a projection weight by the view's projected pixel density."""
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
            density.label = f"ProjectedDensity-{cam_index}-{mat_id}"
            links.new(length_node.outputs["Value"], density.inputs[1])
            clamp_density = nodes.new("ShaderNodeMath")
            clamp_density.operation = 'MINIMUM'
            clamp_density.inputs[1].default_value = 1.0
            links.new(density.outputs[0], clamp_density.inputs[0])
            acquisition = nodes.new("ShaderNodeMath")
            acquisition.operation = 'MULTIPLY'
            acquisition.label = f"AcquisitionQuality-{cam_index}-{mat_id}"
            links.new(weight, acquisition.inputs[0])
            links.new(clamp_density.outputs[0], acquisition.inputs[1])
            return acquisition

        # Voronoi mode flag: keep natural weights for non-generated cameras
        voronoi_active = (context.scene.model_architecture in ('qwen_image_edit', 'qwen_image_21', 'flux2_klein')
                          and getattr(context.scene, 'qwen_voronoi_mode', False)
                          and context.scene.generation_method == 'sequential')
        confidence_context_active = (
            getattr(context.scene, 'qwen_use_confidence_context', False)
            and context.scene.model_architecture.startswith('qwen')
            and context.scene.generation_method == 'sequential'
            and context.scene.qwen_generation_method == 'generate'
            and context.scene.qwen_context_render_mode
            in {'REPLACE_STYLE', 'ADDITIONAL'}
        )

        geometry_nodes = []
        for i, camera in enumerate(cameras):
            camera_node_start = {node.as_pointer() for node in nodes}
            native_extra = {}
            # Geometry is evaluated at the current shading point. Keeping an
            # instance in each view avoids links spanning every camera row.
            geometry = nodes.new("ShaderNodeNewGeometry")
            geometry_nodes.append(geometry)
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

            # Perspective-correct projection UV from the corner attributes
            uv_map_node = _create_projection_uv_nodes(
                nodes, links, i, mat_id, (-200, -200 * (i+1)))
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
                is_local_edit_mode = (context.scene.generation_method == 'local_edit' or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'local_edit'))
                native_pre_factors = None
                if (not is_local_edit_mode
                        and getattr(context.scene, 'sg_rework_acquisition_quality', False)):
                    # Frame-edge and density terms do not depend on the
                    # raycast; they are combined before it (see there).
                    def native_pre_factors(weight, parts, camera=camera, i=i):
                        if (context.scene.generation_method == 'sequential'
                                and context.scene.visibility_vignette):
                            frame_weight = create_native_feather(
                                nodes, links, parts['normalize'], parts['fov'],
                                parts['aspect'], parts['direction'], parts['up'],
                                camera, context, i, mat_id,
                            )
                            frame_quality = nodes.new("ShaderNodeMath")
                            frame_quality.operation = 'MULTIPLY'
                            frame_quality.label = f"FrameEdgeQuality-{i}-{mat_id}"
                            links.new(weight, frame_quality.inputs[0])
                            links.new(frame_weight.outputs[0], frame_quality.inputs[1])
                            weight = frame_quality.outputs[0]
                        return _acquisition_quality(weight, parts['length'], camera, i).outputs[0]
                result = create_native_raycast_visibility(
                    nodes, links, camera, geometry, context, i, mat_id, stop_index,
                    pre_factors=native_pre_factors, extra=native_extra,
                )
                angle_weight = result[0]
                # Destructure result for create_native_feather
                _, _, normalize_node, length_node, camera_loc_node, camera_fov_node, camera_aspect_node, camera_dir_node, camera_up_node = result
                uv_map_nodes[i] = _late_projection_uv(
                    nodes, links, uv_map_node, native_extra['visible'], i, mat_id)

                scripts_to_connect = []  # No OSL scripts to connect
                final_weight_node = None

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
                    )
                if "CameraShiftY" in script_angle.inputs:
                    script_angle.inputs["CameraShiftY"].default_value = (
                        camera.data.shift_y
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
                        )
                    if "CameraShiftY" in script_feather.inputs:
                        script_feather.inputs["CameraShiftY"].default_value = (
                            camera.data.shift_y
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

            # Add projected pixel density to live OSL paths (the native path
            # adds it before its raycast).  Baked acquisition attributes
            # already contain this term.
            if (getattr(context.scene, 'sg_rework_acquisition_quality', False)
                    and not use_baked_weights and not use_native_raycast
                    and script_nodes_outputs
                    and not isinstance(script_nodes_outputs[-1], tuple)
                    and length_nodes[-1] is not None):
                script_nodes_outputs[-1] = _acquisition_quality(
                    script_nodes_outputs[-1].outputs[0], length_nodes[-1], camera, i)

            use_qwen_write_mask = (
                i > 0
                and i <= stop_index
                and context.scene.model_architecture.startswith('qwen')
                and context.scene.generation_method == 'sequential'
                and context.scene.qwen_generation_method == 'generate'
                and context.scene.qwen_context_render_mode in {'REPLACE_STYLE', 'ADDITIONAL'}
                and experimental_enabled(context.scene, 'qwen_use_projection_write_masks')
            )
            qwen_write_weight = None
            use_sdxl_write_mask = (
                i > 0 and i <= stop_index and (context.scene.model_architecture == 'sdxl') and (context.scene.generation_method == 'sequential')
            )
            use_qwen21_generated_alpha = (
                i <= stop_index
                and context.scene.model_architecture == 'qwen_image_21'
            )
            if (use_qwen_write_mask or use_sdxl_write_mask) and script_nodes_outputs:
                current_weight = script_nodes_outputs[-1]
                if not isinstance(current_weight, tuple):
                    write_weight = _create_write_mask_weight(
                        i, mat_id, binary=use_qwen_write_mask
                    )
                    if use_qwen_write_mask:
                        qwen_write_weight = write_weight
                    if write_weight is not None:
                        masked_weight = nodes.new("ShaderNodeMath")
                        masked_weight.operation = 'MULTIPLY'
                        masked_weight.location = (0, (-800) * i + 100)
                        masked_weight.label = f"QwenWriteMasked-{i}-{mat_id}"
                        links.new(current_weight.outputs[0], masked_weight.inputs[0])
                        links.new(write_weight.outputs[0], masked_weight.inputs[1])
                        script_nodes_outputs[-1] = masked_weight

            if ((use_sdxl_write_mask or use_qwen21_generated_alpha)
                    and script_nodes_outputs
                    and not isinstance(script_nodes_outputs[-1], tuple)):
                script_nodes_outputs[-1] = _multiply_projection_weight_by_alpha(
                    nodes,
                    links,
                    script_nodes_outputs[-1],
                    tex_image_nodes[i],
                    (
                        f"Qwen21GeneratedAlpha-{i}-{mat_id}"
                        if use_qwen21_generated_alpha
                        else f"SDXLFeatherAlpha-{i}-{mat_id}"
                    ),
                )

            if script_nodes_outputs and not isinstance(script_nodes_outputs[-1], tuple):
                guard_weight = _create_projection_guard_weight(i, mat_id)
                if guard_weight is not None:
                    # A projection guard marks depth/object/material boundaries
                    # as unsafe for projection, but it must not delete the last
                    # available coverage. Complex character meshes can have
                    # dense ID boundaries around hair, facial parts, fingers,
                    # clothing openings, and adjacent mesh pieces. If every
                    # camera is hard-zeroed at the same surface point, the
                    # weighted graph falls through to the material's black
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
                    if (len(cameras) > 1
                            and getattr(context.scene, 'sg_silhouette_feather', 0) > 0):
                        feather_path = os.path.join(
                            get_dir_path(context, "inpaint")["visibility"],
                            f"projection_guard_feather{i}.png",
                        )
                        if os.path.isfile(feather_path):
                            feather_image = get_or_load_image(
                                feather_path, force_reload=context.scene.overwrite_material)
                            if feather_image is not None:
                                feather_image.colorspace_settings.name = 'Non-Color'
                                feather_tex = nodes.new("ShaderNodeTexImage")
                                feather_tex.image = feather_image
                                feather_tex.extension = 'CLIP'
                                feather_tex.interpolation = 'Linear'
                                feather_tex.label = f"ProjectionGuardFeatherMask-{i}-{mat_id}"
                                feather_tex['sg_layout_stage'] = 'Guard feather'
                                uv = uv_map_nodes[i]
                                links.new(_projection_uv_output(uv),
                                          feather_tex.inputs['Vector'])
                                # The float value carries the feather label that
                                # graph controls and diagnostics look up.
                                feather_value = _mask_value(nodes, links, feather_tex)
                                feather_value.label = f"ProjectionGuardFeather-{i}-{mat_id}"
                                feathered = nodes.new("ShaderNodeMath")
                                feathered.operation = 'MULTIPLY'
                                feathered.label = f"DepthPartFeathered-{i}-{mat_id}"
                                links.new(guarded.outputs[0], feathered.inputs[0])
                                links.new(feather_value.outputs[0], feathered.inputs[1])
                                script_nodes_outputs[-1] = feathered
                                occlusion_path = os.path.join(os.path.dirname(feather_path),
                                                              f"projection_occlusion_geometry{i}.png")
                                calibration_path = os.path.splitext(occlusion_path)[0] + '.json'
                                if os.path.isfile(occlusion_path) and os.path.isfile(calibration_path):
                                    import json
                                    with open(calibration_path) as stream:
                                        distance_scale = float(json.load(stream)['distance_scale'])
                                    occlusion_image = get_or_load_image(
                                        occlusion_path, force_reload=context.scene.overwrite_material)
                                    occlusion_image.colorspace_settings.name = 'Non-Color'
                                    occlusion_tex = nodes.new("ShaderNodeTexImage")
                                    occlusion_tex.label = f"ProjectionOcclusionFeather-{i}-{mat_id}"
                                    occlusion_tex.image = occlusion_image
                                    occlusion_tex.extension = 'CLIP'
                                    occlusion_tex.interpolation = 'Linear'
                                    links.new(feather_tex.inputs['Vector'].links[0].from_socket,
                                              occlusion_tex.inputs['Vector'])
                                    channels = nodes.new('ShaderNodeSeparateColor')
                                    links.new(occlusion_tex.outputs['Color'], channels.inputs[0])
                                    # Cycles' Z pass is camera-axis depth, not
                                    # Euclidean distance along the perspective ray.
                                    if use_baked_weights:
                                        receiver = nodes.new('ShaderNodeAttribute')
                                        receiver.attribute_name = f'_SG_ProjectionDepth_{i}_{mat_id}'
                                        depth_socket = receiver.outputs['Fac']
                                        divisor = distance_scale
                                    elif 'view_depth' in native_extra:
                                        # (view direction . forward) * distance
                                        # is the camera-axis depth; scalars
                                        # keep the SVM stack small.
                                        depth_socket = _projection_feather_math(
                                            nodes, links, 'MULTIPLY',
                                            f'OcclusionReceiverCameraDepth-{i}',
                                            *native_extra['view_depth']).outputs[0]
                                        divisor = distance_scale
                                    else:
                                        coordinates = nodes.new('ShaderNodeTexCoord')
                                        coordinates.object = camera
                                        camera_space = nodes.new('ShaderNodeSeparateXYZ')
                                        links.new(coordinates.outputs['Object'], camera_space.inputs[0])
                                        depth_socket = camera_space.outputs['Z']
                                        divisor = -distance_scale / max(camera.matrix_world.to_scale().z, 1e-12)
                                    normalized = _projection_feather_math(nodes, links, 'DIVIDE',
                                        f'OcclusionReceiverDepth-{i}', depth_socket, divisor)
                                    far = _projection_feather_math(nodes, links, 'GREATER_THAN',
                                        f'OcclusionReceiverFar-{i}', normalized.outputs[0], channels.outputs['Blue'])
                                    rejected = _projection_feather_math(nodes, links, 'SUBTRACT',
                                        f'OcclusionInverseFade-{i}', 1.0, channels.outputs['Red'])
                                    rejected = _projection_feather_math(nodes, links, 'MULTIPLY',
                                        f'OcclusionFarFade-{i}', far.outputs[0], rejected.outputs[0])
                                    protected = _projection_feather_math(nodes, links, 'SUBTRACT',
                                        f'OcclusionProtectedFade-{i}', 1.0, rejected.outputs[0])
                                    protected.name = f'SGOcclusionFeather-{i}-{mat_id}'
                                    # Merge the receiver-aware shoulder into local
                                    # unsharpen after loading the ordinary guard field.
                                    occlusion_shoulder = _projection_feather_math(nodes, links, 'MULTIPLY',
                                        f'OcclusionInfluence-{i}', far.outputs[0], channels.outputs['Green'])
                                    occlusion_shoulder.name = f'SGOcclusionInfluence-{i}-{mat_id}'
                                if getattr(context.scene, 'sg_guard_feather_method', 'ORIGINAL') == 'LOCAL_UNSHARPEN':
                                    influence_path = os.path.join(os.path.dirname(feather_path),
                                                                  f"projection_guard_influence{i}.png")
                                    if os.path.isfile(influence_path):
                                        influence_image = get_or_load_image(
                                            influence_path, force_reload=context.scene.overwrite_material)
                                        influence_image.colorspace_settings.name = 'Non-Color'
                                        influence_tex = nodes.new("ShaderNodeTexImage")
                                        influence_tex.name = f"SGFeatherInfluenceMask-{i}-{mat_id}"
                                        influence_tex.label = f"ProjectionGuardInfluence-{i}-{mat_id}"
                                        influence_tex.image = influence_image
                                        influence_tex.extension = 'CLIP'
                                        influence_tex.interpolation = 'Linear'
                                        links.new(feather_tex.inputs['Vector'].links[0].from_socket,
                                                  influence_tex.inputs['Vector'])
                                        _mask_value(nodes, links, influence_tex).name = (
                                            f"SGFeatherInfluence-{i}-{mat_id}")

            # Confidence is deliberately independent from colour ownership.
            # Preserve the unnormalised source-camera quality before the mix
            # tree turns it into relative colour weights. Later Qwen cameras
            # may acquire confidence only inside the binary repair/write mask;
            # protected RGB copied through a full-frame edit cannot be
            # laundered into high-quality coverage.
            if (confidence_context_active and script_nodes_outputs
                    and not isinstance(script_nodes_outputs[-1], tuple)):
                confidence_quality = script_nodes_outputs[-1]
                if i > stop_index:
                    # Voronoi generation intentionally leaves future camera
                    # placeholders active for colour segmentation. They are
                    # not acquired observations and must never contribute to
                    # source confidence.
                    confidence_quality = nodes.new("ShaderNodeValue")
                    confidence_quality.outputs[0].default_value = 0.0
                    confidence_quality.label = f"ConfidenceFuture-{i}-{mat_id}"
                elif i > 0 and qwen_write_weight is None:
                    confidence_write = _create_write_mask_weight(
                        i,
                        mat_id,
                        binary=True,
                        label_prefix="ConfidenceWrite",
                    )
                    if confidence_write is None:
                        print(
                            "[StableGen] WARNING: confidence write mask "
                            f"missing for camera {i}; refusing to upgrade "
                            "source confidence for that view."
                        )
                        confidence_quality = nodes.new("ShaderNodeValue")
                        confidence_quality.outputs[0].default_value = 0.0
                        confidence_quality.label = (
                            f"ConfidenceWriteMissing-{i}-{mat_id}"
                        )
                    else:
                        accepted_quality = nodes.new("ShaderNodeMath")
                        accepted_quality.operation = 'MULTIPLY'
                        accepted_quality.label = (
                            f"ConfidenceAccepted-{i}-{mat_id}"
                        )
                        links.new(
                            confidence_quality.outputs[0],
                            accepted_quality.inputs[0],
                        )
                        links.new(
                            confidence_write.outputs[0],
                            accepted_quality.inputs[1],
                        )
                        confidence_quality = accepted_quality
                confidence_quality_nodes.append(confidence_quality)

            # Record ownership only for layout. A local edit can append another
            # complete view pass without touching the older material graph.
            for camera_node in nodes:
                pointer = camera_node.as_pointer()
                if pointer not in camera_node_start:
                    layout_camera_nodes[pointer] = i

        # Expose a confidence-only max chain for context rendering.  This
        # node does not mix RGB. Continuous weighted blending retains a separate
        # best-source trust signal to Qwen.
        if confidence_context_active and confidence_quality_nodes:
            best_quality = nodes.new("ShaderNodeMath")
            best_quality.operation = 'MULTIPLY'
            best_quality.inputs[1].default_value = 1.0
            links.new(
                confidence_quality_nodes[0].outputs[0],
                best_quality.inputs[0],
            )
            for index, quality_node in enumerate(confidence_quality_nodes[1:], 1):
                next_best = nodes.new("ShaderNodeMath")
                next_best.operation = 'MAXIMUM'
                next_best.label = f"SGBestSourceQualityStep-{index}"
                links.new(best_quality.outputs[0], next_best.inputs[0])
                links.new(quality_node.outputs[0], next_best.inputs[1])
                best_quality = next_best
            best_quality.label = "SGBestSourceQuality"

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
            geometry = geometry_nodes[i]

            # Connect UV → Texture (needed for both paths)
            links.new(_projection_uv_output(uv_map_node), tex_image.inputs["Vector"])

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
        subtract_node.label = 'SGMaterialIndex'
        subtract_node.location = (-1000, 0)
        from .graph_controls import install_projection_controls
        install_projection_controls(
            mat, context.scene,
            [node for node in nodes if node.as_pointer() not in layout_existing],
            layout_camera_nodes,
        )
        from .node_layout import organize_projection_material
        layout_nodes = [node for node in nodes
                        if node.as_pointer() not in layout_existing]
        if is_local_edit and output not in layout_nodes:
            # The output is reused by appended edit passes. Move its frame to
            # the latest pass so the visible graph still ends at the output.
            layout_nodes.append(output)
        organize_projection_material(
            mat, layout_nodes,
            layout_camera_nodes, cameras, stop_index, local_edit=is_local_edit,
            feather_method=getattr(context.scene, 'sg_guard_feather_method', 'ORIGINAL'),
        )
        check_projection_svm_stack(mat, svm_estimates)
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
    """Reset display controls without changing stored source confidence."""
    if not material or not material.use_nodes:
        return False
    if material.get('sg_surface_blend') and any(n.label.startswith('SurfaceBlendWeight-') for n in material.node_tree.nodes):
        from .surface_blend import update
        update(material, angle_threshold, exponent)
    nodes = material.node_tree.nodes
    for key, value in (('discard_angle', angle_threshold), ('exponent', exponent)):
        control = nodes.get('SG Control | ' + key)
        if control is not None and control.type == 'VALUE':
            control.outputs[0].default_value = value
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
