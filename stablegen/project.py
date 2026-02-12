import bpy
import os
from math import atan, tan, pi

from .utils import get_last_material_index, get_file_path, get_dir_path
from .render_tools import prepare_baking, bake_texture, unwrap
from mathutils import Vector

_SG_BUFFER_UV_NAME = "_SG_ProjectionBuffer"


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
    add_camera_loc.inputs[0].default_value = camera.location.x
    add_camera_loc.inputs[1].default_value = camera.location.y
    add_camera_loc.inputs[2].default_value = camera.location.z

    # --- Direction: Geometry.Position - CameraLoc ---
    subtract = nodes.new("ShaderNodeVectorMath")
    subtract.operation = 'SUBTRACT'
    subtract.location = (x_base, -300 + y_base)
    subtract.inputs[1].default_value = camera.location

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
    fov = camera.data.angle_x
    if context.scene.render.resolution_y > context.scene.render.resolution_x:
        fov = 2 * atan(tan(fov / 2) * context.scene.render.resolution_x / context.scene.render.resolution_y)
    camera_fov_node.outputs[0].default_value = fov

    # --- Camera aspect ratio ---
    camera_aspect_node = nodes.new("ShaderNodeValue")
    camera_aspect_node.location = (-800, 100 + 300 * i)
    camera_aspect_node.outputs[0].default_value = context.scene.render.resolution_x / context.scene.render.resolution_y

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

    abs_x = nodes.new("ShaderNodeMath")
    abs_x.operation = 'ABSOLUTE'
    abs_x.location = (0, y_base - 200)
    links.new(x_proj.outputs[0], abs_x.inputs[0])

    x_in_bounds = nodes.new("ShaderNodeMath")
    x_in_bounds.operation = 'LESS_THAN'
    x_in_bounds.location = (100, y_base - 200)
    links.new(abs_x.outputs[0], x_in_bounds.inputs[0])
    links.new(filmw.outputs[0], x_in_bounds.inputs[1])

    abs_y = nodes.new("ShaderNodeMath")
    abs_y.operation = 'ABSOLUTE'
    abs_y.location = (0, y_base - 300)
    links.new(y_proj.outputs[0], abs_y.inputs[0])

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
    threshold_plus.inputs[1].default_value = 0.01
    links.new(length_node.outputs["Value"], threshold_plus.inputs[0])
    links.new(threshold_plus.outputs[0], raycast.inputs["Length"])

    hit_dist_plus = nodes.new("ShaderNodeMath")
    hit_dist_plus.operation = 'ADD'
    hit_dist_plus.location = (x_base + 400, y_base + 50)
    hit_dist_plus.inputs[1].default_value = 0.001
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

    abs_dot = nodes.new("ShaderNodeMath")
    abs_dot.operation = 'ABSOLUTE'
    abs_dot.location = (x_base + 600, y_base - 150)
    links.new(clamp_dot2.outputs[0], abs_dot.inputs[0])

    acos_node = nodes.new("ShaderNodeMath")
    acos_node.operation = 'ARCCOSINE'
    acos_node.location = (x_base + 700, y_base - 150)
    links.new(abs_dot.outputs[0], acos_node.inputs[0])

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
    links.new(abs_dot.outputs[0], power_node.inputs[0])

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
                          camera_dir_node, camera_up_node, context, i, mat_id):
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

    # nx = abs(x_proj) / filmw
    abs_x = nodes.new("ShaderNodeMath")
    abs_x.operation = 'ABSOLUTE'
    abs_x.location = (x_base + 300, y_base - 100)
    links.new(x_proj.outputs[0], abs_x.inputs[0])

    nx = nodes.new("ShaderNodeMath")
    nx.operation = 'DIVIDE'
    nx.location = (x_base + 400, y_base - 100)
    links.new(abs_x.outputs[0], nx.inputs[0])
    links.new(filmw.outputs[0], nx.inputs[1])

    # ny = abs(y_proj) / filmh
    abs_y = nodes.new("ShaderNodeMath")
    abs_y.operation = 'ABSOLUTE'
    abs_y.location = (x_base + 300, y_base - 200)
    links.new(y_proj.outputs[0], abs_y.inputs[0])

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

def project_image(context, to_project, mat_id, stop_index=1000000):
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
        # Compute offsets based on recursion level
        if context.scene.refine_preserve:
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
            
            if (context.scene.generation_method == 'refine' or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'refine')) and context.scene.refine_preserve:
                w_node = weight_nodes[0]
                if isinstance(w_node, tuple):
                    angle_node, feather_node = w_node
                    
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
                    
                    # Multiply (Intersection of Visibility)
                    mult = nodes_collection.new("ShaderNodeMath")
                    mult.operation = 'MULTIPLY'
                    mult.location = (x_offset - 300, y_offset)
                    links.new(cr_angle.outputs[0], mult.inputs[0])
                    links.new(cr_feather.outputs[0], mult.inputs[1])
                    
                    # Invert (Convert Visibility to Mix Factor: 1=Vis->0=Proj, 0=Invis->1=Orig)
                    invert = nodes_collection.new("ShaderNodeMath")
                    invert.operation = 'SUBTRACT'
                    invert.location = (x_offset - 150, y_offset)
                    invert.inputs[0].default_value = 1.0
                    links.new(mult.outputs[0], invert.inputs[1])

                    compare_node = invert # For return
                    links.new(invert.outputs[0], final_mix.inputs["Fac"])
                else:
                    # Fallback if not tuple (e.g. stop_index logic used LessThan node)
                    # If it's a sum of processed weights (from multi-camera recursion), we need to invert it
                    # to get the Mix Factor (1 - TotalVis)
                    if (context.scene.generation_method == 'refine' or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'refine')) and context.scene.refine_preserve:
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
            
            if not context.scene.apply_bsdf:
                return final_mix, compare_node
                 
            # Add a principle shader for the mixed color
            should_add_principled = True
            if (context.scene.generation_method == 'refine' or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'refine')) and context.scene.refine_preserve:
                # Final principled is at last_node's output if it is BSDF
                final_principled = last_node.outputs[0].links[0].to_node
                if final_principled.type == 'BSDF_PRINCIPLED':
                    should_add_principled = False
            
            if should_add_principled:
                final_principled = nodes_collection.new("ShaderNodeBsdfPrincipled")
                final_principled.location = (x_offset, y_offset)
                final_principled.inputs["Roughness"].default_value = 1.0
                
            links.new(final_mix.outputs[0], final_principled.inputs[0])
            return final_principled, compare_node

        new_shaders = []
        new_weight_nodes = []
        i = 0
        while i < len(shaders):
            if i + 1 < len(shaders):
                # Sum the weights
                vert_offset = -200 * (i // 2)
                sum_node = nodes_collection.new("ShaderNodeMath")
                sum_node.operation = 'ADD'
                sum_node.location = (x_offset - 800, y_offset + vert_offset) if context.scene.early_priority else (x_offset - 600, y_offset + vert_offset)
                
                def get_weight_output(w_item):
                    if isinstance(w_item, tuple):
                        angle_node, feather_node = w_item
                        
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
                        
                        # Multiply (Intersection of Visibility)
                        m = nodes_collection.new("ShaderNodeMath")
                        m.operation = 'MULTIPLY'
                        m.location = (x_offset - 900, y_offset + vert_offset)
                        links.new(cr_angle.outputs[0], m.inputs[0])
                        links.new(cr_feather.outputs[0], m.inputs[1])
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
                div_node.location = (x_offset - 600, y_offset + vert_offset) if context.scene.early_priority else (x_offset - 400, y_offset + vert_offset)
                links.new(w_out_2, div_node.inputs[0])
                links.new(sum_node.outputs[0], div_node.inputs[1])

                if context.scene.early_priority:
                    # Add map range node
                    map_range_node = nodes_collection.new("ShaderNodeMapRange")
                    map_range_node.location = (x_offset - 400, y_offset + vert_offset)
                    # to_min, to_max, from_max are set to 0.0 and 1.0 by default
                    # set to prioritize earlier images
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
                to_connect = div_node if not context.scene.early_priority else map_range_node
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

    cameras = [obj for obj in context.scene.objects if obj.type == 'CAMERA']
    cameras.sort(key=lambda x: x.name)
    
    # Force refresh of the UI
    for area in context.screen.areas:
        area.tag_redraw()
    
    # Apply projection to all mesh objects for each camera
    for i, camera in enumerate(cameras):
        for x, obj in enumerate(to_project):
            # We can skip the UV Projection step in sequential mode for i > 0
            if context.scene.generation_method != 'sequential' or stop_index == 0 or context.scene.bake_texture:
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
                    print("Warning: Cannot make object data single user. Making a copy.")
                    obj.data = obj.data.copy()
                    obj.data.name = f"{obj.name}_data"
                    obj.data.update()
                    # Check again
                    if obj.data.users > 1:
                        print("Error: Cannot make object data single user. Exiting.")
                        return Exception("Cannot make object data single user. Exiting.")

                # Create or reuse a buffer UV map for the UV Project modifier
                buffer_uv = obj.data.uv_layers.get(_SG_BUFFER_UV_NAME)
                if not buffer_uv:
                    buffer_uv = obj.data.uv_layers.new(name=_SG_BUFFER_UV_NAME)

                # If the object has no UV map other than the buffer and we are baking, create one
                non_buffer_uvs = [uv for uv in obj.data.uv_layers if uv.name != _SG_BUFFER_UV_NAME]
                if not non_buffer_uvs and context.scene.bake_texture:
                    obj.data.uv_layers.new(name="BakeUV")

                # Add the UV Project Modifier
                uv_project_mod = obj.modifiers.new(name="UVProject", type='UV_PROJECT')

                # Assign the active camera to the UV Project modifier
                if not camera:
                    return False

                uv_project_mod.projectors[0].object = camera

                # Set the buffer UV map for the modifier
                uv_project_mod.uv_layer = buffer_uv.name

                # Calculate and set the aspect ratio
                render = context.scene.render
                aspect_ratio = render.resolution_x / render.resolution_y
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

                # If we are running in sequential mode, we already have baked textures for i < stop_index
            if context.scene.bake_texture:
                if (stop_index > 0 and context.scene.generation_method == 'sequential'):
                    # Deselect all objects
                    bpy.ops.object.select_all(action='DESELECT')
                    # Select object as active (needed for applying the modifier)
                    context.view_layer.objects.active = obj
                    obj.select_set(True)
                if i <= stop_index and (not context.scene.generation_method == 'sequential' or i == stop_index):
                    simple_project_bake(context, i, obj, mat_id)
                # Remove the projection attribute (data is now baked into texture)
                attr = obj.data.attributes.get(f"ProjectionUV_{i}_{mat_id}")
                if attr:
                    obj.data.attributes.remove(attr)

    # Clean up buffer UV map from all objects
    for obj in to_project:
        buffer_uv = obj.data.uv_layers.get(_SG_BUFFER_UV_NAME)
        if buffer_uv:
            obj.data.uv_layers.remove(buffer_uv)

    # Switch to Cycles (needed for Raycast shader node on all versions)
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
        if (context.scene.generation_method == "refine" or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'refine')) and context.scene.refine_preserve and not context.scene.overwrite_material:
            # Copy active material
            mat = obj.active_material.copy()
            obj.data.materials.append(mat)
            to_switch = True
        elif obj.active_material and (context.scene.overwrite_material or ((context.scene.generation_method == "refine" or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'refine')) and context.scene.refine_preserve) \
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


        if (context.scene.generation_method == 'sequential' and stop_index > 0) or context.scene.generation_mode == 'regenerate_selected':
            # We just need to remove the compare nodes which are connected to script node at stop_index
            script_node = None
            # First find all script nodes with label Angle-{stop_index}-{mat_id} or {stop_index}-{mat_id} (legacy)
            # Also match native MATH MULTIPLY nodes (Blender 5.1+ native raycast path)
            for node in nodes:
                if (node.type == 'SCRIPT' or (node.type == 'MATH' and node.operation == 'MULTIPLY')) and (node.label == f"Angle-{stop_index}-{mat_id}" or node.label == f"{stop_index}-{mat_id}"):
                    script_node = node
                    break
            compare_output_sockets = set()
            if script_node:
                # Find all compare nodes connected to the script node
                for link in script_node.outputs[0].links:
                    if link.to_node.type == 'MATH' and link.to_node.operation == 'LESS_THAN':
                        # Save all outputs (nodes connected to) of compare nodes
                        for link2 in link.to_node.outputs[0].links:
                            compare_output_sockets.add(link2.to_socket)
                        # Remove the compare node
                        nodes.remove(link.to_node)
            # Connect the script node to all outputs
            if script_node:
                for output in compare_output_sockets:
                    links.new(script_node.outputs[0], output)
            # We also need to set the generated image to the texture node with label {stop_index}-{mat_id}
            for node in nodes:
                if node.type == 'TEX_IMAGE' and node.label == f"{stop_index}-{mat_id}":
                    if not context.scene.bake_texture:
                        image_path = get_file_path(context, "generated", camera_id=stop_index, material_id=mat_id)
                        if (context.scene.generation_method == 'sequential' or context.scene.generation_method == 'separate') and context.scene.sequential_ipadapter and context.scene.sequential_ipadapter_regenerate \
                        and not context.scene.use_ipadapter and stop_index == 0 and context.scene.sequential_ipadapter_mode == 'first':
                            image_path = get_file_path(context, "generated", camera_id=stop_index, material_id=mat_id).replace(".png", "_ipadapter.png")
                    else:
                        # Use baked texture
                        image_path = get_file_path(context, "generated_baked", camera_id=stop_index, material_id=mat_id, object_name=obj.name)

                    image = get_or_load_image(image_path, force_reload=context.scene.overwrite_material)
                    if image:
                        node.image = image
                    break
            # Now we can continue to the next object
            continue
                        
        elif not context.scene.refine_preserve or not (context.scene.generation_method == 'refine' or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'refine')):
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
            
        if not ((context.scene.generation_method == 'refine' or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'refine')) and context.scene.refine_preserve):
            output = nodes.new("ShaderNodeOutputMaterial")
            output.location = (3000, 0)
        
        geometry = nodes.new("ShaderNodeNewGeometry")
        geometry.location = (-600, 0)

        use_native_raycast = bpy.app.version >= (5, 1, 0)

        tex_image_nodes = []
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

        for i, camera in enumerate(cameras):
            # Add image texture node
            tex_image = nodes.new("ShaderNodeTexImage")
            if i <= stop_index:
                if not context.scene.bake_texture:
                    image_path = get_file_path(context, "generated", camera_id=i, material_id=mat_id)
                    if (context.scene.generation_method == 'sequential' or context.scene.generation_method == 'separate') and context.scene.sequential_ipadapter and context.scene.sequential_ipadapter_regenerate \
                    and not context.scene.use_ipadapter and i == 0 and context.scene.sequential_ipadapter_mode == 'first':
                        image_path = get_file_path(context, "generated", camera_id=i, material_id=mat_id).replace(".png", "_ipadapter.png")
                else:
                    # Use baked texture
                    image_path = get_file_path(context, "generated_baked", camera_id=i, material_id=mat_id, object_name=obj.name)

                image = get_or_load_image(image_path, force_reload=context.scene.overwrite_material)
                if image:
                    tex_image.image = image

            tex_image.location = (0, -200 * i)
            tex_image.extension = 'CLIP'
            tex_image.label = f"{i}-{mat_id}"
            tex_image_nodes.append(tex_image)
                
            # Add UV map / attribute node
            if not context.scene.bake_texture:
                # Use ShaderNodeAttribute to read from the corner-domain attribute
                uv_map_node = nodes.new("ShaderNodeAttribute")
                uv_map_node.attribute_name = f"ProjectionUV_{i}_{mat_id}"
                uv_map_node.attribute_type = 'GEOMETRY'
            else:
                # Baked textures use the original UV map
                uv_map_node = nodes.new("ShaderNodeUVMap")
                uv_map_node.uv_map = original_uv_map.name
            uv_map_node.location = (-200, -200 * (i+1))
            uv_map_nodes.append(uv_map_node)
            
            # Compute the dot product of the direction vector and geometry normal

            if use_native_raycast:
                # ── Native Raycast path (Blender 5.1+) ──
                result = create_native_raycast_visibility(
                    nodes, links, camera, geometry, context, i, mat_id, stop_index
                )
                angle_weight = result[0]
                # Destructure result for create_native_feather
                _, _, normalize_node, length_node, camera_loc_node, camera_fov_node, camera_aspect_node, camera_dir_node, camera_up_node = result

                scripts_to_connect = []  # No OSL scripts to connect
                final_weight_node = None

                is_refine = (context.scene.generation_method == 'refine' or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'refine')) and context.scene.refine_preserve
                if is_refine:
                    feather_weight = create_native_feather(
                        nodes, links, normalize_node, camera_fov_node, camera_aspect_node,
                        camera_dir_node, camera_up_node, context, i, mat_id
                    )
                    final_weight_node = (angle_weight, feather_weight)
                else:
                    final_weight_node = angle_weight

                script_nodes.append(scripts_to_connect)

                if i > stop_index:
                    less_than = nodes.new("ShaderNodeMath")
                    less_than.operation = 'LESS_THAN'
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
                length_nodes.append(None)
                camera_fov_nodes.append(None)
                camera_aspect_ratio_nodes.append(None)
                camera_direction_nodes.append(None)
                camera_up_nodes.append(None)
            else:
                # ── OSL path (Blender < 5.1) ──
                subtract = nodes.new("ShaderNodeVectorMath")
                subtract.operation = 'SUBTRACT'
                subtract.location = (-400, -300 + (-800) * i)
                subtract.inputs[1].default_value = camera.location
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
                    script_angle.filepath = os.path.join(os.path.dirname(__file__), "raycast.osl")

                script_angle.inputs["AngleThreshold"].default_value = context.scene.discard_factor
                script_angle.inputs["Power"].default_value = context.scene.weight_exponent
                script_angle.label = f"Angle-{i}-{mat_id}"

                scripts_to_connect = [script_angle]
                final_weight_node = None

                if (context.scene.generation_method == 'refine' or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'refine')) and context.scene.refine_preserve:
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
                        script_feather.filepath = os.path.join(os.path.dirname(__file__), "feather.osl")
                    
                    if context.scene.visibility_vignette: # Only set if feathering is active
                        script_feather.inputs["EdgeFeather"].default_value = context.scene.visibility_vignette_width
                        script_feather.inputs["EdgeGamma"].default_value = context.scene.visibility_vignette_softness
                    else:
                        script_feather.inputs["EdgeFeather"].default_value = 0.0
                        script_feather.inputs["EdgeGamma"].default_value = 1.0
                    script_feather.label = f"Feather-{i}-{mat_id}"
                    scripts_to_connect.append(script_feather)

                    # Pass tuple of (Angle, Feather) to build_mix_tree
                    final_weight_node = (script_angle, script_feather)
                else:
                    # Just use Angle script output directly
                    final_weight_node = script_angle

                script_nodes.append(scripts_to_connect)

                if i > stop_index:
                    # Connect a temporary less than node to the script node
                    less_than = nodes.new("ShaderNodeMath")
                    less_than.operation = 'LESS_THAN'
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
                fov = camera.data.angle_x
                # Correct the FOV for vertical aspect ratio
                if context.scene.render.resolution_y > context.scene.render.resolution_x:
                    fov = 2 * atan(tan(fov / 2) * context.scene.render.resolution_x / context.scene.render.resolution_y)
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
                
                camera_aspect_ratio.outputs[0].default_value = context.scene.render.resolution_x / context.scene.render.resolution_y
                camera_aspect_ratio_nodes.append(camera_aspect_ratio)

                # Add combine XYZ node
                add_camera_loc = nodes.new("ShaderNodeCombineXYZ")
                add_camera_loc.location = (-600, 200 + 300 * i)
                add_camera_loc.inputs[0].default_value = camera.location.x
                add_camera_loc.inputs[1].default_value = camera.location.y
                add_camera_loc.inputs[2].default_value = camera.location.z
                add_camera_loc_nodes.append(add_camera_loc)

                # Add length node
                length = nodes.new("ShaderNodeVectorMath")
                length.operation = 'LENGTH'
                length.location = (-400, 200 * (i+1))
                length_nodes.append(length)

        # Build mix shader tree
        if not context.scene.bake_texture and mat.name in processed_materials and (context.scene.generation_method == 'refine' or (context.scene.model_architecture.startswith('qwen') and context.scene.qwen_generation_method == 'refine')) and context.scene.refine_preserve:
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

            if not use_native_raycast:
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
    
def simple_project_bake(context, camera_id, obj, mat_id):
    # Create a temporary material for the projection
    mat = bpy.data.materials.new(name="ProjectionMaterialTemp")
    obj.data.materials.append(mat)

    # Switch the active material to the new material (Switch to edit mode, select all, assign the material)
    obj.active_material_index = len(obj.material_slots) - 1
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.object.material_slot_assign()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Unwrap (only for the first camera)
    if camera_id == 0:
        unwrap(obj, context.scene.bake_unwrap_method, context.scene.bake_unwrap_overlap_only)

    # Enable use of nodes
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # Add image texture node
    tex_image = nodes.new("ShaderNodeTexImage")
    
    file_path = get_file_path(context, "generated", camera_id=camera_id, material_id=mat_id)
    if (context.scene.generation_method == 'sequential' or context.scene.generation_method == 'separate') and context.scene.sequential_ipadapter and context.scene.sequential_ipadapter_regenerate \
        and not context.scene.use_ipadapter and camera_id == 0 and context.scene.sequential_ipadapter_mode == 'first':
        file_path = get_file_path(context, "generated", camera_id=camera_id, material_id=mat_id).replace(".png", "_ipadapter.png")
    
    image = get_or_load_image(file_path, force_reload=context.scene.overwrite_material)
    if image:
        tex_image.image = image

    # Add UV attribute node (reads from corner-domain attribute)
    uv_map_node = nodes.new("ShaderNodeAttribute")
    uv_map_node.attribute_name = f"ProjectionUV_{camera_id}_{mat_id}"
    uv_map_node.attribute_type = 'GEOMETRY'
    
    # Add BSDF node
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 1.0

    # Add output node
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(uv_map_node.outputs["Vector"], tex_image.inputs["Vector"])
    # Connect the nodes
    if context.scene.apply_bsdf:
        links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    else:
        links.new(tex_image.outputs["Color"], output.inputs["Surface"])
 
    # Bake texture using BakeTextures.bake_texture
    texture_size = context.scene.bake_texture_size
    original_engine = context.scene.render.engine
    prepare_baking(context)
    
    # If the object has no UV map, create one
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")
    
    bake_texture(context, obj, texture_size, suffix=f"{camera_id}-{mat_id}", output_dir=get_dir_path(context, "generated_baked"))
    context.scene.render.engine = original_engine

    # Remove the temporary material
    bpy.ops.object.material_slot_remove()


def get_or_create_osl_text(filename):
    """
    Loads an OSL script into a Blender internal text block.
    This ensures the script is embedded in the .blend file (portable).
    """
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(filepath):
        print(f"Error: OSL script not found at {filepath}")
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
        print(f"Error reading OSL file {filepath}: {e}")
        return None
        
    return text


def get_or_load_image(filepath, force_reload=False):
    """
    Prevents duplicate image datablocks by default.
    If force_reload is True, it finds the existing datablock 
    and reloads it from the specified filepath.
    """
    if not filepath:
        print("Error: No filepath provided to get_or_load_image.")
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
            print(f"Warning: Error comparing image paths: {e}")
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
            print(f"Reload failed for {filename}. Removing old datablock. Error: {e}")
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
            print(f"Warning: Could not load image file: {filepath}. Error: {e}")
            return None
            
    return image


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
