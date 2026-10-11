"""
Debug / diagnostic operators for StableGen.

These operators help visualise the projection, blending and visibility
behaviour without running any AI generation.  They are gated behind the
*Enable Debug Settings* toggle in the addon preferences.

IMPORTANT: The projection operators (Solid Colors, Grid Pattern, Visibility
Material) call the *real* project_image() and export_visibility() functions
so the debug output is guaranteed to match the actual generation pipeline.
"""

import bpy
import os
import colorsys
import glob
import json
import numpy as np
import sys
import tempfile
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
from mathutils import Vector
from bpy_extras import view3d_utils

from .texturing.projection import (
    project_image,
    create_native_raycast_visibility,
    create_native_feather,
    _SG_BUFFER_UV_NAME,
    _copy_uv_to_attribute,
    _projection_uvw_name,
    _write_projection_uvw,
)
from .texturing.rendering import export_emit_image, export_visibility
from .cameras.geometry import _get_camera_resolution, _SGCameraResolution
from .utils import (
    get_file_path,
    get_generation_dirs,
    ensure_dirs_exist,
    sg_modal_active,
)
from .texturing.view_plan import (
    generation_cameras,
    order_cameras,
    order_cameras_from_manifest,
)
from .mesh_gen.rig_display import (
    convert_armature_bones_to_fortune,
    fortune_conversion_risks,
    selected_armatures_for_display,
)


# ─── Shared helpers ──────────────────────────────────────────────────────

def _get_cameras(context):
    """Return scene cameras sorted by name."""
    cams = [o for o in context.scene.objects if o.type == 'CAMERA']
    cams.sort(key=lambda x: x.name)
    return cams


def _get_target_meshes(context):
    """Selected meshes, or all meshes in the scene."""
    sel = [o for o in context.selected_objects if o.type == 'MESH']
    return sel if sel else [o for o in context.scene.objects if o.type == 'MESH']


def _common_poll(cls, context):
    """Shared poll for all debug operators."""
    prefs = context.preferences.addons.get(__package__)
    if not prefs or not prefs.preferences.enable_debug:
        return False
    if not _get_target_meshes(context):
        return False
    if not _get_cameras(context):
        return False
    # Block while another heavy modal is running
    if sg_modal_active(context):
        return False
    return True


class SG_OT_DebugForceGenerationIdle(bpy.types.Operator):
    """Clear stale in-process texturing state after an interrupted operation."""

    bl_idname = "stablegen.debug_force_generation_idle"
    bl_label = "Force Generation Idle"
    bl_description = (
        "Force StableGen's saved generation state back to Idle and close any "
        "stale generation WebSocket; use only when no generation is actually running"
    )
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        # The button itself is debug-gated. Keep the recovery operator callable
        # through Search/Python even when a stale file also lost its UI state.
        return context.scene is not None

    def execute(self, context):
        # Avoid importing generator.py here: debug_tools is imported by the
        # add-on registration graph before the generator operator. If it is
        # already loaded, also clear its process-local sentinels.
        generator_modules = [
            module
            for name, module in tuple(sys.modules.items())
            if name.endswith(".texturing.generator")
        ]
        closed_socket = False
        for module in generator_modules:
            generator = getattr(module, "ComfyUIGenerate", None)
            if generator is None:
                continue
            websocket = getattr(generator, "_active_ws", None)
            if websocket is not None:
                try:
                    websocket.close()
                    closed_socket = True
                except Exception:
                    pass
            generator._active_ws = None
            generator._is_running = False

        context.scene.generation_status = 'idle'
        context.scene.generation_progress = 0.0
        if hasattr(context.scene, "sg_last_gen_error"):
            context.scene.sg_last_gen_error = False

        message = "Forced StableGen generation state to Idle"
        if closed_socket:
            message += " and closed the stale WebSocket"
        self.report({'WARNING'}, message)
        return {'FINISHED'}


def _saturated_colors(n):
    """Return *n* maximally-saturated, evenly-spaced RGB tuples."""
    return [colorsys.hsv_to_rgb(i / max(n, 1), 1.0, 1.0) for i in range(n)]


_SKIN_WEIGHT_COLOR_ATTRIBUTE = "SG_SkinWeights"
_SKIN_WEIGHT_PREVIOUS_ATTRIBUTE = "sg_skin_debug_previous_color_attribute"


def _mesh_armature_for_skin_debug(obj):
    """Return the armature that owns *obj*'s deform groups, if available."""
    if obj.parent and obj.parent.type == 'ARMATURE':
        return obj.parent
    for modifier in obj.modifiers:
        if (
            modifier.type == 'ARMATURE'
            and modifier.object
            and modifier.object.type == 'ARMATURE'
        ):
            return modifier.object
    return None


def _get_skinning_target_meshes(context):
    """Resolve meshes for the skin-weight diagnostic without requiring cameras.

    Selected meshes take priority. If an armature is selected, all scene meshes
    driven by it are used. With no useful selection, all rigged scene meshes are
    considered. This matches the common post-auto-rig workflow where the
    imported armature, rather than every individual garment mesh, is selected.
    """
    selected_meshes = [
        obj for obj in context.selected_objects
        if obj.type == 'MESH' and obj.vertex_groups
    ]
    if selected_meshes:
        return selected_meshes

    selected_armatures = {
        obj for obj in context.selected_objects if obj.type == 'ARMATURE'
    }
    if context.active_object and context.active_object.type == 'ARMATURE':
        selected_armatures.add(context.active_object)

    candidates = [
        obj for obj in context.scene.objects
        if obj.type == 'MESH' and obj.vertex_groups
    ]
    if selected_armatures:
        return [
            obj for obj in candidates
            if _mesh_armature_for_skin_debug(obj) in selected_armatures
        ]
    return [obj for obj in candidates if _mesh_armature_for_skin_debug(obj)]


def _skinning_debug_poll(context):
    prefs = context.preferences.addons.get(__package__)
    if not prefs or not prefs.preferences.enable_debug:
        return False
    if sg_modal_active(context):
        return False
    return bool(_get_skinning_target_meshes(context))


def _active_skinning_bone_name(context):
    """Resolve the bone/group whose individual influence should be displayed."""
    active_pose_bone = getattr(context, "active_pose_bone", None)
    if active_pose_bone is not None:
        return active_pose_bone.name
    active_edit_bone = getattr(context, "active_bone", None)
    if active_edit_bone is not None:
        return active_edit_bone.name

    active_object = context.active_object
    if active_object and active_object.type == 'ARMATURE':
        active_bone = active_object.data.bones.active
        if active_bone is not None:
            return active_bone.name
    if active_object and active_object.type == 'MESH':
        active_group = active_object.vertex_groups.active
        if active_group is not None:
            return active_group.name

    for obj in context.selected_objects:
        if obj.type == 'ARMATURE' and obj.data.bones.active is not None:
            return obj.data.bones.active.name
        if obj.type == 'MESH' and obj.vertex_groups.active is not None:
            return obj.vertex_groups.active.name
    return None


def _stable_skinning_palette(names):
    """Return stable, visually separated sRGB colors for bone/group names."""
    golden_ratio = 0.618033988749895
    colors = []
    for index, _name in enumerate(names):
        hue = (0.03 + index * golden_ratio) % 1.0
        saturation = 0.78 + 0.10 * (index % 3)
        value = 1.0
        colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
    return np.asarray(colors, dtype=np.float32)


def _skinning_palette_for_object(obj):
    """Return (bone names, colors, vertex-group-index -> palette-index)."""
    armature = _mesh_armature_for_skin_debug(obj)
    if armature:
        names = [bone.name for bone in armature.data.bones if bone.use_deform]
    else:
        # Useful for imported/debug meshes whose Armature modifier was removed.
        # StableGen marker groups are not skin influences and are excluded.
        names = sorted(
            group.name for group in obj.vertex_groups
            if not group.name.startswith("SG_")
        )
    palette = _stable_skinning_palette(names)
    group_to_palette = {}
    for palette_index, name in enumerate(names):
        group = obj.vertex_groups.get(name)
        if group is not None:
            group_to_palette[group.index] = palette_index
    return names, palette, group_to_palette


def _max_weight_heat_color(value):
    """Blue -> cyan -> yellow -> red diagnostic ramp for [0, 1]."""
    value = float(np.clip(value, 0.0, 1.0))
    positions = np.asarray((0.0, 0.33, 0.66, 1.0), dtype=np.float32)
    colors = np.asarray(
        ((0.05, 0.05, 0.45), (0.0, 0.9, 1.0),
         (1.0, 0.95, 0.0), (1.0, 0.0, 0.0)),
        dtype=np.float32,
    )
    return np.asarray([
        np.interp(value, positions, colors[:, channel])
        for channel in range(3)
    ], dtype=np.float32)


def _influence_heat_color(value):
    """SkinTokens-like pale-blue -> cyan -> green -> yellow -> red ramp."""
    value = float(np.clip(value, 0.0, 1.0))
    positions = np.asarray((0.0, 0.2, 0.45, 0.7, 1.0), dtype=np.float32)
    colors = np.asarray(
        ((0.34, 0.56, 0.78), (0.05, 0.82, 0.95),
         (0.10, 0.82, 0.36), (1.0, 0.88, 0.0), (0.92, 0.05, 0.02)),
        dtype=np.float32,
    )
    return np.asarray([
        np.interp(value, positions, colors[:, channel])
        for channel in range(3)
    ], dtype=np.float32)


def _skinning_colors_for_object(obj, mode='BLENDED', bone_name=None):
    """Evaluate the current vertex groups into a SkinTokens-style color map.

    ``ACTIVE_BONE`` matches the SkinTokens paper view: a blue-to-red heatmap of
    one selected bone's influence over the complete mesh. ``BLENDED`` displays
    the normalized weighted blend of per-bone colors, ``DOMINANT`` shows hard
    ownership, and ``MAX_WEIGHT`` visualizes assignment certainty.
    """
    names, palette, group_to_palette = _skinning_palette_for_object(obj)
    colors = np.zeros((len(obj.data.vertices), 4), dtype=np.float32)
    colors[:, 3] = 1.0
    unweighted = 0

    if mode == 'ACTIVE_BONE':
        colors[:, :3] = _influence_heat_color(0.0)
        group = obj.vertex_groups.get(bone_name) if bone_name else None
        if group is not None:
            for vertex in obj.data.vertices:
                weight = next(
                    (
                        float(item.weight) for item in vertex.groups
                        if item.group == group.index
                    ),
                    0.0,
                )
                colors[vertex.index, :3] = _influence_heat_color(weight)
        return colors, {
            "bone_count": len(names),
            "mapped_group_count": int(group is not None),
            "vertex_count": len(obj.data.vertices),
            "unweighted_count": 0,
            "selected_bone": bone_name,
        }

    for vertex in obj.data.vertices:
        influences = [
            (float(item.weight), group_to_palette[item.group])
            for item in vertex.groups
            if item.group in group_to_palette and item.weight > 1e-8
        ]
        total = sum(weight for weight, _palette_index in influences)
        if total <= 1e-8:
            unweighted += 1
            continue

        if mode == 'DOMINANT':
            _weight, palette_index = max(influences, key=lambda item: item[0])
            colors[vertex.index, :3] = palette[palette_index]
        elif mode == 'MAX_WEIGHT':
            strongest = max(weight for weight, _palette_index in influences)
            colors[vertex.index, :3] = _max_weight_heat_color(strongest / total)
        else:
            blended = np.zeros(3, dtype=np.float32)
            for weight, palette_index in influences:
                blended += palette[palette_index] * weight
            colors[vertex.index, :3] = blended / total

    return colors, {
        "bone_count": len(names),
        "mapped_group_count": len(group_to_palette),
        "vertex_count": len(obj.data.vertices),
        "unweighted_count": unweighted,
    }


def _apply_skinning_weight_colors(obj, mode='BLENDED', bone_name=None):
    """Write the diagnostic POINT color attribute and return its statistics."""
    mesh = obj.data
    current = mesh.color_attributes.get(_SKIN_WEIGHT_COLOR_ATTRIBUTE)
    if current is None:
        previous = mesh.color_attributes.active_color
        mesh[_SKIN_WEIGHT_PREVIOUS_ATTRIBUTE] = (
            previous.name if previous is not None else ""
        )
    elif current.domain != 'POINT' or current.data_type != 'FLOAT_COLOR':
        mesh.color_attributes.remove(current)
        current = None

    if current is None:
        current = mesh.color_attributes.new(
            name=_SKIN_WEIGHT_COLOR_ATTRIBUTE,
            type='FLOAT_COLOR',
            domain='POINT',
        )

    colors, stats = _skinning_colors_for_object(
        obj, mode=mode, bone_name=bone_name)
    try:
        current.data.foreach_set("color_srgb", colors.ravel())
    except (AttributeError, TypeError):
        # Blender 3.x exposes only the scene-linear color accessor.
        current.data.foreach_set("color", colors.ravel())
    mesh.color_attributes.active_color = current
    mesh.update()
    return stats


def _clear_skinning_weight_colors(obj):
    """Remove only StableGen's diagnostic attribute and restore color focus."""
    mesh = obj.data
    attribute = mesh.color_attributes.get(_SKIN_WEIGHT_COLOR_ATTRIBUTE)
    if attribute is not None:
        mesh.color_attributes.remove(attribute)
    previous_name = mesh.get(_SKIN_WEIGHT_PREVIOUS_ATTRIBUTE, "")
    previous = mesh.color_attributes.get(previous_name) if previous_name else None
    if previous is not None:
        mesh.color_attributes.active_color = previous
    if _SKIN_WEIGHT_PREVIOUS_ATTRIBUTE in mesh:
        del mesh[_SKIN_WEIGHT_PREVIOUS_ATTRIBUTE]
    mesh.update()
    return attribute is not None


def _set_skinning_debug_viewport(context, enabled):
    """Select vertex-color solid shading without touching object materials."""
    screen = getattr(context, "screen", None)
    if screen is None:
        return
    for area in screen.areas:
        if area.type != 'VIEW_3D':
            continue
        for space in area.spaces:
            if space.type != 'VIEW_3D':
                continue
            shading = space.shading
            shading.type = 'SOLID'
            if enabled:
                shading.color_type = 'VERTEX'
                try:
                    shading.light = 'FLAT'
                except (AttributeError, TypeError):
                    pass
            elif shading.color_type == 'VERTEX':
                shading.color_type = 'MATERIAL'


def _save_solid_image(path, width, height, color_rgba):
    """Create and save a solid-colour PNG at *path*."""
    img = bpy.data.images.new("_sg_dbg_tmp", width=width, height=height, alpha=True)
    n = width * height
    px = np.empty(n * 4, dtype=np.float32)
    px[0::4] = color_rgba[0]
    px[1::4] = color_rgba[1]
    px[2::4] = color_rgba[2]
    px[3::4] = color_rgba[3]
    img.pixels.foreach_set(px)
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)


def _save_grid_image(path, width, height, color_rgb, grid_size=64):
    """Create and save a checkerboard grid PNG at *path*."""
    img = bpy.data.images.new("_sg_dbg_tmp", width=width, height=height, alpha=True)
    r, g, b = color_rgb
    px = np.empty((height, width, 4), dtype=np.float32)
    # Build checkerboard mask
    ys = np.arange(height)[:, None]  # (H, 1)
    xs = np.arange(width)[None, :]   # (1, W)
    checker = ((xs // grid_size) % 2) == ((ys // grid_size) % 2)
    px[:, :, 0] = np.where(checker, r, r * 0.4)
    px[:, :, 1] = np.where(checker, g, g * 0.4)
    px[:, :, 2] = np.where(checker, b, b * 0.4)
    px[:, :, 3] = 1.0
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)


def _apply_uv_projections(context, cameras, meshes, mat_id):
    """Create UV projections for each camera on each mesh, storing them
    as named attributes.  Returns True on success."""
    for i, camera in enumerate(cameras):
        for obj in meshes:
            bpy.ops.object.select_all(action='DESELECT')
            context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.make_single_user(object=True, obdata=True)
            if obj.data.users > 1:
                obj.data = obj.data.copy()
                obj.data.name = f"{obj.name}_data"
                obj.data.update()

            buffer_uv = obj.data.uv_layers.get(_SG_BUFFER_UV_NAME)
            if not buffer_uv:
                buffer_uv = obj.data.uv_layers.new(name=_SG_BUFFER_UV_NAME)

            mod = obj.modifiers.new(name="UVProject", type='UV_PROJECT')
            mod.projectors[0].object = camera
            mod.uv_layer = buffer_uv.name
            cam_res_x, cam_res_y = _get_camera_resolution(camera, context.scene)
            aspect = cam_res_x / cam_res_y
            mod.aspect_x = aspect if aspect > 1 else 1
            mod.aspect_y = 1 / aspect if aspect < 1 else 1
            obj.data.uv_layers.active = buffer_uv
            bpy.ops.object.modifier_apply(modifier=mod.name)

            attr_name = f"ProjectionUV_{i}_{mat_id}"
            _copy_uv_to_attribute(obj, _SG_BUFFER_UV_NAME, attr_name)
            _write_projection_uvw(obj, camera, attr_name, _projection_uvw_name(i, mat_id))

            original_uv = obj.data.uv_layers[0]
            if original_uv.name == _SG_BUFFER_UV_NAME and len(obj.data.uv_layers) > 1:
                original_uv = obj.data.uv_layers[1]
            obj.data.uv_layers.active = original_uv

    # Clean up buffer UV
    for obj in meshes:
        buf = obj.data.uv_layers.get(_SG_BUFFER_UV_NAME)
        if buf:
            obj.data.uv_layers.remove(buf)
    return True


def _get_active_camera(context):
    """Return the camera the user intends to preview and its index in the
    sorted camera list.  Detection: selected camera → scene camera → first."""
    cameras = _get_cameras(context)
    # Active object is a camera
    if context.active_object and context.active_object.type == 'CAMERA' and context.active_object in cameras:
        cam = context.active_object
        return cam, cameras.index(cam)
    # Any selected object is a camera
    for obj in context.selected_objects:
        if obj.type == 'CAMERA' and obj in cameras:
            return obj, cameras.index(obj)
    # Scene active camera
    if context.scene.camera and context.scene.camera in cameras:
        return context.scene.camera, cameras.index(context.scene.camera)
    return cameras[0], 0


def _get_selected_cameras(context):
    """Return cameras among selected objects, falling back to all cameras."""
    cameras = _get_cameras(context)
    selected = [o for o in context.selected_objects if o.type == 'CAMERA' and o in cameras]
    return selected if selected else cameras


def _hide_unselected_cameras(context, keep_cameras):
    """Temporarily unlink cameras not in *keep_cameras* from scene collections.
    Returns a mapping {camera: [collections]} for _restore_cameras()."""
    all_cams = [o for o in context.scene.objects if o.type == 'CAMERA']
    hidden = {}
    for cam in all_cams:
        if cam not in keep_cameras:
            hidden[cam] = list(cam.users_collection)
            for col in hidden[cam]:
                col.objects.unlink(cam)
    return hidden


def _restore_cameras(hidden):
    """Re-link cameras that were hidden by _hide_unselected_cameras()."""
    for cam, collections in hidden.items():
        for col in collections:
            if cam.name not in col.objects:
                col.objects.link(cam)


@contextmanager
def _debug_projection_output(context):
    """Keep diagnostic textures separate from generated images and GPU caches."""
    scene = context.scene
    source = get_generation_dirs(context)['revision']
    debug_root = os.path.join(source, 'debug')
    os.makedirs(debug_root, exist_ok=True)
    output = tempfile.mkdtemp(prefix='projection_', dir=debug_root)
    manifest = os.path.join(source, 'view_plan.json')
    if os.path.isfile(manifest):
        shutil.copy2(manifest, os.path.join(output, 'view_plan.json'))
    # Keep the same per-view masks as the generation being diagnosed.
    visibility = os.path.join(source, 'inpaint', 'visibility')
    if os.path.isdir(visibility):
        shutil.copytree(visibility, os.path.join(output, 'inpaint', 'visibility'))
    previous = scene.get('_sg_debug_output')
    scene['_sg_debug_output'] = output
    try:
        yield
    finally:
        if previous is None:
            del scene['_sg_debug_output']
        else:
            scene['_sg_debug_output'] = previous


@contextmanager
def _debug_projection_build_state(scene):
    """Build diagnostics with the finished material's display controls.

    Sequential generation may use a restrictive angle and then reset only the
    material nodes for final display.  Rebuilding a debug material from the
    generation configuration made the new 70-degree graph look substantially
    sharper than the actual 90-degree final graph (and than legacy diagnostics,
    which mutated the scene value itself).  Preserve the generation settings,
    use the recorded material state when a completed run exists, and restore
    everything after the diagnostic graph is built.
    """
    original = {
        "overwrite_material": scene.overwrite_material,
        "generation_method": scene.generation_method,
        "generation_mode": scene.generation_mode,
        "discard_factor": scene.discard_factor,
        "weight_exponent": scene.weight_exponent,
        "sequential_ipadapter_regenerate": scene.sequential_ipadapter_regenerate,
    }
    if getattr(scene, "output_timestamp", ""):
        scene.discard_factor = getattr(
            scene, "sg_material_discard_factor", scene.discard_factor
        )
        scene.weight_exponent = getattr(
            scene, "sg_material_weight_exponent", scene.weight_exponent
        )
    scene.overwrite_material = True
    scene.generation_method = 'separate'
    scene.generation_mode = 'standard'
    # Diagnostic images replace every camera, including the initial reference.
    scene.sequential_ipadapter_regenerate = False
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(scene, name, value)


# ═════════════════════════════════════════════════════════════════════════
# 1. Draw Solid Colors  (calls the real project_image)
# ═════════════════════════════════════════════════════════════════════════

class SG_OT_DebugSolidColors(bpy.types.Operator):
    """Project solid saturated colours (one per camera) through the real
    project_image() pipeline.  Useful for understanding how views are
    being blended.  No ComfyUI connection required"""
    bl_idname = "stablegen.debug_solid_colors"
    bl_label = "Draw Solid Colors"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _common_poll(cls, context)

    def execute(self, context):
        with _debug_projection_output(context):
            return self._execute_projection(context)

    def _execute_projection(self, context):
        cameras = _interactive_generation_cameras(context)
        meshes = _get_target_meshes(context)
        n_cams = len(cameras)
        colors = _saturated_colors(n_cams)

        # Ensure output dirs exist
        dirs = get_generation_dirs(context)
        ensure_dirs_exist(dirs)

        mat_id = 0

        # Write a solid-colour PNG for every camera at the path project_image expects
        for i, cam in enumerate(cameras):
            path = get_file_path(context, "generated", camera_id=i, material_id=mat_id)
            rx, ry = _get_camera_resolution(cam, context.scene)
            _save_solid_image(path, rx, ry, (*colors[i], 1.0))

        # Replay the real sequential material lifecycle.  A one-shot
        # ``separate`` build cannot detect failures in the incremental path
        # (for example, a later camera image loading while its stop gate stays
        # closed).  stop_index=0 creates the UV projections/material; every
        # later call patches or rebuilds it exactly as generation does.
        with _debug_projection_build_state(context.scene):
            context.scene.generation_method = 'sequential'
            for stop_index in range(n_cams):
                project_image(
                    context,
                    meshes,
                    mat_id,
                    stop_index=stop_index,
                    cameras=cameras,
                )

        validation_errors = []
        inspected_materials = set()
        for obj in meshes:
            mat = obj.active_material
            if mat is None or mat.as_pointer() in inspected_materials:
                continue
            inspected_materials.add(mat.as_pointer())
            nodes = mat.node_tree.nodes
            for i in range(n_cams):
                image_node = next(
                    (node for node in nodes
                     if node.type == 'TEX_IMAGE' and node.label == f"{i}-{mat_id}"),
                    None,
                )
                expected_path = os.path.normcase(os.path.abspath(get_file_path(
                    context, "generated", camera_id=i, material_id=mat_id
                )))
                actual_path = (
                    os.path.normcase(os.path.abspath(bpy.path.abspath(image_node.image.filepath)))
                    if image_node is not None and image_node.image is not None else ""
                )
                if actual_path != expected_path:
                    validation_errors.append(
                        f"camera {i}: expected {os.path.basename(expected_path)}, "
                        f"got {os.path.basename(actual_path) or 'no image'}"
                    )
                if any(node.label == f"StopGate-{i}-{mat_id}" for node in nodes):
                    validation_errors.append(f"camera {i}: sequential stop gate is still closed")

        if validation_errors:
            details = "; ".join(validation_errors[:4])
            print(f"[StableGen] Solid-color sequential validation FAILED: {details}")
            self.report({'ERROR'}, f"Sequential projection validation failed: {details}")
            return {'CANCELLED'}

        # Rename materials for easy identification / cleanup
        for obj in meshes:
            if obj.active_material:
                obj.active_material.name = "SG_Debug_SolidColors"

        for camera, color in zip(cameras, colors):
            camera['_sg_debug_projection_color'] = color
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        self.report(
            {'INFO'},
            f"Solid-colour sequential projection validated ({n_cams} cameras)",
        )
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════
# 2. Projection Grid Pattern  (calls the real project_image)
# ═════════════════════════════════════════════════════════════════════════

class SG_OT_DebugGridPattern(bpy.types.Operator):
    """Project a numbered checkerboard grid (one per camera) through the real
    project_image() pipeline to reveal UV distortion, stretching and alignment"""
    bl_idname = "stablegen.debug_grid_pattern"
    bl_label = "Projection Grid Pattern"
    bl_options = {'REGISTER', 'UNDO'}

    grid_size: bpy.props.IntProperty(
        name="Grid Cell Size",
        description="Size of each checkerboard cell in pixels",
        default=64, min=8, max=512
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        return _common_poll(cls, context)

    def execute(self, context):
        with _debug_projection_output(context):
            return self._execute_projection(context)

    def _execute_projection(self, context):
        cameras = _interactive_generation_cameras(context)
        meshes = _get_target_meshes(context)
        n_cams = len(cameras)
        colors = _saturated_colors(n_cams)

        dirs = get_generation_dirs(context)
        ensure_dirs_exist(dirs)

        mat_id = 0

        for i, cam in enumerate(cameras):
            path = get_file_path(context, "generated", camera_id=i, material_id=mat_id)
            rx, ry = _get_camera_resolution(cam, context.scene)
            _save_grid_image(path, rx, ry, colors[i], self.grid_size)

        with _debug_projection_build_state(context.scene):
            project_image(context, meshes, mat_id)

        for obj in meshes:
            if obj.active_material:
                obj.active_material.name = "SG_Debug_GridPattern"

        self.report({'INFO'}, f"Grid pattern projection applied ({n_cams} cameras)")
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════
# 3. Apply Visibility Material  (calls the real export_visibility)
# ═════════════════════════════════════════════════════════════════════════

class SG_OT_DebugVisibilityMaterial(bpy.types.Operator):
    """Project solid colours then apply export_visibility() to visualise
    which areas each camera can see.  Uses selected cameras (or all if
    none are selected).  No ComfyUI connection required"""
    bl_idname = "stablegen.debug_visibility_material"
    bl_label = "Apply Visibility Material"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _common_poll(cls, context)

    def execute(self, context):
        selected_cams = _get_selected_cameras(context)
        meshes = _get_target_meshes(context)

        # Temporarily hide unselected cameras so project_image only sees
        # the ones the user cares about.
        hidden = _hide_unselected_cameras(context, selected_cams)

        try:
            # Re-query cameras (only the kept ones are visible now)
            active_cams = _get_cameras(context)
            n_cams = len(active_cams)
            colors = _saturated_colors(n_cams)

            dirs = get_generation_dirs(context)
            ensure_dirs_exist(dirs)

            mat_id = 0

            for i, cam in enumerate(active_cams):
                path = get_file_path(context, "generated",
                                     camera_id=i, material_id=mat_id)
                rx, ry = _get_camera_resolution(cam, context.scene)
                _save_solid_image(path, rx, ry, (*colors[i], 1.0))

            with _debug_projection_build_state(context.scene):
                project_image(context, meshes, mat_id)

            # Transform projection material into visibility mask
            result = export_visibility(context, meshes, prepare_only=True)
            if not result:
                self.report({'WARNING'}, "export_visibility failed.")
                return {'CANCELLED'}
        finally:
            _restore_cameras(hidden)

        self.report({'INFO'},
                    f"Visibility material applied ({n_cams} camera(s)). "
                    "Inspect in Material Preview or Rendered mode.")
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════
# 4. Project Coverage Heatmap
# ═════════════════════════════════════════════════════════════════════════

class SG_OT_DebugCoverageHeatmap(bpy.types.Operator):
    """Create a heatmap material showing total angle-based weight coverage
    across all cameras.  Red = low coverage, green = high coverage"""
    bl_idname = "stablegen.debug_coverage_heatmap"
    bl_label = "Project Coverage Heatmap"
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('AVERAGE', "Average",
             "Sum all camera weights and divide by camera count. "
             "Shows mean coverage across all cameras"),
            ('MAX', "Best Camera",
             "Take the maximum single-camera weight at each point. "
             "Shows the best coverage available, ignoring overlap"),
        ],
        default='AVERAGE',
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        return _common_poll(cls, context)

    def execute(self, context):
        cameras = _get_cameras(context)
        meshes = _get_target_meshes(context)
        mat_id = "dbg_heatmap"

        # Switch to Cycles for Raycast nodes
        context.scene.render.engine = 'CYCLES'
        if bpy.app.version < (5, 1, 0):
            context.scene.cycles.device = 'CPU'
            if hasattr(context.scene.cycles, 'shading_system'):
                context.scene.cycles.shading_system = True
            else:
                context.scene.cycles.use_osl = True

        _apply_uv_projections(context, cameras, meshes, mat_id)

        for obj in meshes:
            mat = bpy.data.materials.new(name="SG_Debug_Heatmap")
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            for n in list(nodes):
                nodes.remove(n)

            output = nodes.new("ShaderNodeOutputMaterial")
            output.location = (2000, 0)
            geometry = nodes.new("ShaderNodeNewGeometry")
            geometry.location = (-600, 0)

            use_native = bpy.app.version >= (5, 1, 0)

            # Accumulate angle weights from all cameras
            accum_node = None
            for i, cam in enumerate(cameras):
                if use_native:
                    result = create_native_raycast_visibility(
                        nodes, links, cam, geometry, context, i, mat_id, 1000000)
                    w_node = result[0]
                else:
                    w_node = nodes.new("ShaderNodeValue")
                    w_node.outputs[0].default_value = 1.0
                    w_node.location = (-400, -800 * i)

                if accum_node is None:
                    accum_node = w_node
                else:
                    combine = nodes.new("ShaderNodeMath")
                    combine.operation = ('MAXIMUM'
                                         if self.mode == 'MAX' else 'ADD')
                    combine.location = (600 + i * 100, -100 * i)
                    links.new(accum_node.outputs[0], combine.inputs[0])
                    links.new(w_node.outputs[0], combine.inputs[1])
                    accum_node = combine

            # Normalise: divide by camera count (average) or clamp to 1 (max)
            if self.mode == 'MAX':
                norm = nodes.new("ShaderNodeMath")
                norm.operation = 'MINIMUM'
                norm.location = (1200, 0)
                norm.inputs[1].default_value = 1.0
                links.new(accum_node.outputs[0], norm.inputs[0])
            else:
                norm = nodes.new("ShaderNodeMath")
                norm.operation = 'DIVIDE'
                norm.location = (1200, 0)
                norm.inputs[1].default_value = max(len(cameras), 1)
                links.new(accum_node.outputs[0], norm.inputs[0])

            # ── Log-scale to spread out low-coverage differences ──
            # Formula: log(1 + value * 9) / log(10)
            # Maps 0→0, 1→1 but stretches the low end significantly.
            scale9 = nodes.new("ShaderNodeMath")
            scale9.operation = 'MULTIPLY'
            scale9.location = (1300, -100)
            scale9.inputs[1].default_value = 9.0
            links.new(norm.outputs[0], scale9.inputs[0])

            add1 = nodes.new("ShaderNodeMath")
            add1.operation = 'ADD'
            add1.location = (1400, -100)
            add1.inputs[1].default_value = 1.0
            links.new(scale9.outputs[0], add1.inputs[0])

            log_node = nodes.new("ShaderNodeMath")
            log_node.operation = 'LOGARITHM'
            log_node.location = (1500, -100)
            log_node.inputs[1].default_value = 10.0
            links.new(add1.outputs[0], log_node.inputs[0])

            # ── Detect zero coverage (sum == 0) for distinct color ──
            is_zero = nodes.new("ShaderNodeMath")
            is_zero.operation = 'LESS_THAN'
            is_zero.location = (1300, 200)
            is_zero.inputs[1].default_value = 0.001
            links.new(accum_node.outputs[0], is_zero.inputs[0])

            # Non-linear color ramp with more low-end stops:
            # 0.0  → dark red   (very low coverage)
            # 0.15 → red
            # 0.30 → orange
            # 0.50 → yellow
            # 0.75 → yellow-green
            # 1.0  → green      (excellent coverage)
            ramp = nodes.new("ShaderNodeValToRGB")
            ramp.location = (1700, 0)
            ramp.color_ramp.interpolation = 'LINEAR'
            elements = ramp.color_ramp.elements
            # Stop 0: dark red
            elements[0].position = 0.0
            elements[0].color = (0.4, 0.0, 0.0, 1.0)
            # Stop 1: bright red
            elements[1].position = 0.15
            elements[1].color = (1.0, 0.0, 0.0, 1.0)
            # Stop 2: orange
            s2 = elements.new(0.30)
            s2.color = (1.0, 0.5, 0.0, 1.0)
            # Stop 3: yellow
            s3 = elements.new(0.50)
            s3.color = (1.0, 1.0, 0.0, 1.0)
            # Stop 4: yellow-green
            s4 = elements.new(0.75)
            s4.color = (0.5, 1.0, 0.0, 1.0)
            # Stop 5 (already at 1.0 from elements[1] which got pushed):
            # Ensure last element is green at 1.0
            last = ramp.color_ramp.elements[-1]
            last.position = 1.0
            last.color = (0.0, 1.0, 0.0, 1.0)

            links.new(log_node.outputs[0], ramp.inputs["Fac"])

            # Mix: if zero coverage → magenta, else → ramp color
            zero_color = nodes.new("ShaderNodeRGB")
            zero_color.location = (1700, 300)
            zero_color.outputs[0].default_value = (0.8, 0.0, 0.8, 1.0)  # magenta

            mix = nodes.new("ShaderNodeMixRGB")
            mix.location = (1900, 100)
            mix.blend_type = 'MIX'
            links.new(is_zero.outputs[0], mix.inputs["Fac"])
            links.new(ramp.outputs["Color"], mix.inputs["Color1"])
            links.new(zero_color.outputs[0], mix.inputs["Color2"])

            # Emission so it's visible in solid mode too
            emit = nodes.new("ShaderNodeEmission")
            emit.location = (2100, 0)
            links.new(mix.outputs["Color"], emit.inputs["Color"])
            links.new(emit.outputs[0], output.inputs["Surface"])

        mode_label = 'best-camera' if self.mode == 'MAX' else 'average'
        self.report({'INFO'},
                    f"Coverage heatmap applied ({len(cameras)} cameras, "
                    f"{mode_label} mode)")
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════
# 5. Per-Camera Weight Preview
# ═════════════════════════════════════════════════════════════════════════

class SG_OT_DebugPerCameraWeight(bpy.types.Operator):
    """Show the selected camera's raw angle weight as grayscale emission.
    White = full weight, black = zero weight.
    Select a camera object first, or the active scene camera is used"""
    bl_idname = "stablegen.debug_per_camera_weight"
    bl_label = "Per-Camera Weight Preview"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _common_poll(cls, context)

    def execute(self, context):
        cam, idx = _get_active_camera(context)
        meshes = _get_target_meshes(context)
        mat_id = "dbg_weight"

        context.scene.render.engine = 'CYCLES'
        if bpy.app.version < (5, 1, 0):
            context.scene.cycles.device = 'CPU'
            if hasattr(context.scene.cycles, 'shading_system'):
                context.scene.cycles.shading_system = True
            else:
                context.scene.cycles.use_osl = True

        use_native = bpy.app.version >= (5, 1, 0)

        for obj in meshes:
            mat = bpy.data.materials.new(name=f"SG_Debug_Weight_{cam.name}")
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            for n in list(nodes):
                nodes.remove(n)

            output = nodes.new("ShaderNodeOutputMaterial")
            output.location = (1400, 0)
            geometry = nodes.new("ShaderNodeNewGeometry")
            geometry.location = (-600, 0)

            if use_native:
                result = create_native_raycast_visibility(
                    nodes, links, cam, geometry, context, idx, mat_id, 1000000)
                w_node = result[0]
            else:
                w_node = nodes.new("ShaderNodeValue")
                w_node.outputs[0].default_value = 1.0

            emit = nodes.new("ShaderNodeEmission")
            emit.location = (1200, 0)
            links.new(w_node.outputs[0], emit.inputs["Strength"])
            emit.inputs["Color"].default_value = (1, 1, 1, 1)
            links.new(emit.outputs[0], output.inputs["Surface"])

        context.scene.camera = cam
        self.report({'INFO'}, f"Weight preview for {cam.name}")
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════
# 6. Feather / Vignette Preview
# ═════════════════════════════════════════════════════════════════════════

class SG_OT_DebugFeatherPreview(bpy.types.Operator):
    """Show the selected camera's feather / vignette falloff as grayscale.
    White = full (centre), black = feathered-out (edge).
    Select a camera object first, or the active scene camera is used"""
    bl_idname = "stablegen.debug_feather_preview"
    bl_label = "Feather / Vignette Preview"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _common_poll(cls, context)

    def execute(self, context):
        cam, idx = _get_active_camera(context)
        meshes = _get_target_meshes(context)
        mat_id = "dbg_feather"

        context.scene.render.engine = 'CYCLES'
        if bpy.app.version < (5, 1, 0):
            context.scene.cycles.device = 'CPU'
            if hasattr(context.scene.cycles, 'shading_system'):
                context.scene.cycles.shading_system = True
            else:
                context.scene.cycles.use_osl = True

        use_native = bpy.app.version >= (5, 1, 0)

        for obj in meshes:
            mat = bpy.data.materials.new(name=f"SG_Debug_Feather_{cam.name}")
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            for n in list(nodes):
                nodes.remove(n)

            output = nodes.new("ShaderNodeOutputMaterial")
            output.location = (1400, 0)
            geometry = nodes.new("ShaderNodeNewGeometry")
            geometry.location = (-600, 0)

            if use_native:
                result = create_native_raycast_visibility(
                    nodes, links, cam, geometry, context, idx, mat_id, 1000000)
                _, _, normalize_node, _, _, camera_fov_node, camera_aspect_node, camera_dir_node, camera_up_node = result
                feather_node = create_native_feather(
                    nodes, links, normalize_node, camera_fov_node,
                    camera_aspect_node, camera_dir_node, camera_up_node,
                    cam, context, idx, mat_id)
            else:
                feather_node = nodes.new("ShaderNodeValue")
                feather_node.outputs[0].default_value = 1.0

            emit = nodes.new("ShaderNodeEmission")
            emit.location = (1200, 0)
            links.new(feather_node.outputs[0], emit.inputs["Strength"])
            emit.inputs["Color"].default_value = (1, 1, 1, 1)
            links.new(emit.outputs[0], output.inputs["Surface"])

        context.scene.camera = cam
        self.report({'INFO'}, f"Feather preview for {cam.name}")
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════
# 7. UV Seam Visualiser
# ═════════════════════════════════════════════════════════════════════════

class SG_OT_DebugUVSeamViz(bpy.types.Operator):
    """Visualize UV projection overlap and seam boundaries.
    Each camera's projected area is a unique color; overlapping regions
    appear as a combined/lighter shade"""
    bl_idname = "stablegen.debug_uv_seam_viz"
    bl_label = "UV Seam Visualizer"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _common_poll(cls, context)

    def execute(self, context):
        cameras = _get_cameras(context)
        meshes = _get_target_meshes(context)
        mat_id = "dbg_seam"

        context.scene.render.engine = 'CYCLES'
        if bpy.app.version < (5, 1, 0):
            context.scene.cycles.device = 'CPU'
            if hasattr(context.scene.cycles, 'shading_system'):
                context.scene.cycles.shading_system = True
            else:
                context.scene.cycles.use_osl = True

        _apply_uv_projections(context, cameras, meshes, mat_id)

        use_native = bpy.app.version >= (5, 1, 0)
        colors = _saturated_colors(len(cameras))

        for obj in meshes:
            mat = bpy.data.materials.new(name="SG_Debug_UVSeams")
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            for n in list(nodes):
                nodes.remove(n)

            output = nodes.new("ShaderNodeOutputMaterial")
            output.location = (2400, 0)
            geometry = nodes.new("ShaderNodeNewGeometry")
            geometry.location = (-600, 0)

            # For each camera: create a visibility mask, scale by unique colour,
            # then ADD all together.
            sum_node = None
            for i, cam in enumerate(cameras):
                if use_native:
                    result = create_native_raycast_visibility(
                        nodes, links, cam, geometry, context, i, mat_id, 1000000)
                    w = result[0]
                else:
                    w = nodes.new("ShaderNodeValue")
                    w.outputs[0].default_value = 1.0
                    w.location = (-400, -800 * i)

                # Multiply weight by the camera's unique colour
                colour_node = nodes.new("ShaderNodeRGB")
                colour_node.location = (800, -300 * i)
                colour_node.outputs[0].default_value = (*colors[i], 1.0)

                scale = nodes.new("ShaderNodeMixRGB")
                scale.blend_type = 'MULTIPLY'
                scale.location = (1000, -300 * i)
                scale.inputs["Fac"].default_value = 1.0
                links.new(colour_node.outputs[0], scale.inputs["Color1"])
                # Weight as grayscale colour
                w_rgb = nodes.new("ShaderNodeCombineColor")
                w_rgb.location = (900, -300 * i - 100)
                links.new(w.outputs[0], w_rgb.inputs[0])
                links.new(w.outputs[0], w_rgb.inputs[1])
                links.new(w.outputs[0], w_rgb.inputs[2])
                links.new(w_rgb.outputs[0], scale.inputs["Color2"])

                if sum_node is None:
                    sum_node = scale
                else:
                    add = nodes.new("ShaderNodeMixRGB")
                    add.blend_type = 'ADD'
                    add.location = (1400 + i * 200, -150 * i)
                    add.inputs["Fac"].default_value = 1.0
                    links.new(sum_node.outputs[0], add.inputs["Color1"])
                    links.new(scale.outputs[0], add.inputs["Color2"])
                    sum_node = add

            emit = nodes.new("ShaderNodeEmission")
            emit.location = (2100, 0)
            if sum_node:
                links.new(sum_node.outputs[0], emit.inputs["Color"])
            links.new(emit.outputs[0], output.inputs["Surface"])

        self.report({'INFO'}, f"UV seam overlay applied ({len(cameras)} cameras)")
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════
# 8. Skinning Weight Colors
# ═════════════════════════════════════════════════════════════════════════

class SG_OT_DebugConvertFortuneRig(bpy.types.Operator):
    """Reorient an existing generated rig using Fortune joint directions."""

    bl_idname = "stablegen.debug_convert_fortune_rig"
    bl_label = "Convert Existing Rig to Fortune"
    bl_description = (
        "Rebuild an existing SkinTokens rig's rest-pose bone directions so "
        "tails point toward child joints; weights and joint heads are preserved"
    )
    bl_options = {'REGISTER', 'UNDO'}

    force_unsafe: bpy.props.BoolProperty(
        name="Convert Animated/Posed Rig Anyway",
        description=(
            "Rest-axis conversion may change existing animations, constraints, "
            "drivers, or the current pose"
        ),
        default=False,
    )  # type: ignore

    _risks = None

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            cls.poll_message_set("Another StableGen operation is in progress")
            return False
        if not selected_armatures_for_display(context):
            cls.poll_message_set("Select a rigged mesh or armature")
            return False
        return True

    def invoke(self, context, _event):
        armatures = selected_armatures_for_display(context)
        self._risks = {
            armature.name: fortune_conversion_risks(armature)
            for armature in armatures
            if fortune_conversion_risks(armature)
        }
        if self._risks:
            return context.window_manager.invoke_props_dialog(self, width=560)
        return self.execute(context)

    def draw(self, _context):
        layout = self.layout
        layout.label(
            text="This rewrites rest-pose bone axes and is safe by default only for unanimated rigs.",
            icon='ERROR',
        )
        for armature_name, risks in (self._risks or {}).items():
            layout.label(text=f"{armature_name}: {', '.join(risks)}")
        layout.prop(self, "force_unsafe")

    def execute(self, context):
        armatures = selected_armatures_for_display(context)
        blocked = {
            armature.name: fortune_conversion_risks(armature)
            for armature in armatures
            if fortune_conversion_risks(armature)
        }
        if blocked and not self.force_unsafe:
            details = "; ".join(
                f"{name}: {', '.join(risks)}"
                for name, risks in blocked.items()
            )
            self.report(
                {'ERROR'},
                "Conversion cancelled. Enable the confirmation checkbox to "
                f"override: {details}",
            )
            return {'CANCELLED'}

        total = 0
        for armature in armatures:
            stats = convert_armature_bones_to_fortune(armature)
            total += stats["changed"]
            print(
                "[StableGen Debug] Fortune-converted "
                f"{armature.name}: {stats}"
            )
        self.report(
            {'INFO'},
            f"Converted {len(armatures)} rig(s), reoriented {total} bones",
        )
        return {'FINISHED'}

class SG_OT_DebugSkinningWeights(bpy.types.Operator):
    """Color rigged meshes from their actual deform vertex-group weights.

    The default view matches the SkinTokens comparison: the selected bone's
    individual influence is shown from blue (zero) to red (one). Alternative
    modes expose blended per-bone colors, dominant ownership, and assignment
    confidence. Materials and weights are left untouched; only a named mesh
    color attribute and viewport shading mode are changed.
    """
    bl_idname = "stablegen.debug_skinning_weights"
    bl_label = "Skinning Weight Colors"
    bl_description = (
        "Visualize generated skin weights, defaulting to a selected-bone "
        "blue-to-red influence heatmap"
    )
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(
        name="Display",
        items=[
            (
                'ACTIVE_BONE',
                "Selected Bone Heatmap",
                "SkinTokens paper-style view: blue is zero influence and red "
                "is full influence for the active bone or vertex group",
            ),
            (
                'BLENDED',
                "Blended Bone Colors",
                "SkinTokens-style map: blend stable per-bone colors using "
                "the normalized skin weights",
            ),
            (
                'DOMINANT',
                "Dominant Bone",
                "Show only the strongest deform bone on every vertex",
            ),
            (
                'MAX_WEIGHT',
                "Maximum Weight",
                "Heatmap the strongest normalized influence: blue is weak "
                "and red is fully assigned",
            ),
        ],
        default='ACTIVE_BONE',
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        return _skinning_debug_poll(context)

    def execute(self, context):
        meshes = _get_skinning_target_meshes(context)
        bone_name = (
            _active_skinning_bone_name(context)
            if self.mode == 'ACTIVE_BONE' else None
        )
        if self.mode == 'ACTIVE_BONE' and not bone_name:
            self.report(
                {'ERROR'},
                "Select a pose/edit bone or make a deform vertex group active",
            )
            return {'CANCELLED'}
        processed_data = set()
        totals = {
            "meshes": 0,
            "vertices": 0,
            "groups": 0,
            "unweighted": 0,
        }
        skipped = []

        for obj in meshes:
            data_pointer = obj.data.as_pointer()
            if data_pointer in processed_data:
                skipped.append(obj.name)
                continue
            processed_data.add(data_pointer)

            stats = _apply_skinning_weight_colors(
                obj, mode=self.mode, bone_name=bone_name)
            if (
                self.mode != 'ACTIVE_BONE'
                and not stats["mapped_group_count"]
            ):
                _clear_skinning_weight_colors(obj)
                skipped.append(obj.name)
                continue
            totals["meshes"] += 1
            totals["vertices"] += stats["vertex_count"]
            totals["groups"] += stats["mapped_group_count"]
            totals["unweighted"] += stats["unweighted_count"]

        if not totals["meshes"] or (
            self.mode == 'ACTIVE_BONE' and not totals["groups"]
        ):
            for obj in meshes:
                _clear_skinning_weight_colors(obj)
            self.report(
                {'ERROR'},
                (
                    f"No vertex group named '{bone_name}' was found on the target meshes"
                    if self.mode == 'ACTIVE_BONE'
                    else "No deform-bone vertex groups were found on the target meshes"
                ),
            )
            return {'CANCELLED'}

        _set_skinning_debug_viewport(context, True)
        print(
            "[StableGen Debug] Skinning colors: "
            f"{totals['meshes']} mesh(es), {totals['vertices']} vertices, "
            f"{totals['groups']} mapped deform groups, "
            f"{totals['unweighted']} unweighted vertices, mode={self.mode}, "
            f"bone={bone_name or 'all'}."
        )
        if skipped:
            print(
                "[StableGen Debug] Skinning color targets skipped: "
                + ", ".join(skipped)
            )
        self.report(
            {'INFO'},
            (
                f"Skinning heatmap for {bone_name} applied to "
                f"{totals['meshes']} mesh(es)"
                if self.mode == 'ACTIVE_BONE'
                else f"Skinning colors applied to {totals['meshes']} mesh(es); "
                f"{totals['unweighted']} unweighted vertices shown black"
            ),
        )
        return {'FINISHED'}


class SG_OT_DebugClearSkinningWeights(bpy.types.Operator):
    """Remove StableGen skin-weight colors without changing materials/weights."""
    bl_idname = "stablegen.debug_clear_skinning_weights"
    bl_label = "Clear Skinning Weight Colors"
    bl_description = "Remove the StableGen skin-weight color attribute"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        prefs = context.preferences.addons.get(__package__)
        if not prefs or not prefs.preferences.enable_debug:
            return False
        if sg_modal_active(context):
            return False
        return any(
            obj.type == 'MESH'
            and obj.data.color_attributes.get(_SKIN_WEIGHT_COLOR_ATTRIBUTE)
            for obj in context.scene.objects
        )

    def execute(self, context):
        selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
        meshes = selected or [
            obj for obj in context.scene.objects
            if obj.type == 'MESH'
            and obj.data.color_attributes.get(_SKIN_WEIGHT_COLOR_ATTRIBUTE)
        ]
        processed_data = set()
        removed = 0
        for obj in meshes:
            data_pointer = obj.data.as_pointer()
            if data_pointer in processed_data:
                continue
            processed_data.add(data_pointer)
            removed += int(_clear_skinning_weight_colors(obj))
        _set_skinning_debug_viewport(context, False)
        self.report({'INFO'}, f"Cleared skinning colors from {removed} mesh(es)")
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════
# 9. Restore Original Material
# ═════════════════════════════════════════════════════════════════════════

class SG_OT_DebugRestoreMaterial(bpy.types.Operator):
    """Remove all debug materials from selected (or all) mesh objects and
    leave the object without a material so the user can re-assign"""
    bl_idname = "stablegen.debug_restore_material"
    bl_label = "Remove Debug Materials"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        prefs = context.preferences.addons.get(__package__)
        if not prefs or not prefs.preferences.enable_debug:
            return False
        if sg_modal_active(context):
            return False
        return bool(_get_target_meshes(context))

    def execute(self, context):
        meshes = _get_target_meshes(context)
        removed = 0
        for obj in meshes:
            for slot in obj.material_slots:
                if slot.material and slot.material.name.startswith("SG_Debug_"):
                    bpy.data.materials.remove(slot.material)
                    removed += 1
            obj.data.materials.clear()
        self.report({'INFO'}, f"Removed {removed} debug material(s)")
        for camera in _get_cameras(context):
            if '_sg_debug_projection_color' in camera:
                del camera['_sg_debug_projection_color']
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════
# 10. Live Confidence Diagnostics
# ═════════════════════════════════════════════════════════════════════════

class SG_OT_DebugConfidenceDiagnostic(bpy.types.Operator):
    """Visualize values from the actual projection confidence graph.

    The material is copied before its output is replaced, so the diagnostic
    cannot alter the production graph. Undo restores the original slots.
    """
    bl_idname = "stablegen.debug_confidence_diagnostic"
    bl_label = "Confidence Diagnostic"
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(
        name="Diagnostic",
        items=[
            ('QUALITY', "Best Source Quality", "Resolved acquisition quality of accepted sources"),
            ('ACQUISITION', "Camera Quality", "Raw acquisition quality for the active camera"),
            ('TRUSTED', "Trusted Context", "White where source quality reaches the trust threshold"),
            ('REPAIR', "Repair Mask", "White where the current view must regenerate context"),
            ('WRITE', "Write Mask", "Authoritative generated-pixel write mask for the active camera"),
            ('GUARD', "Depth/Part Guard", "Rear-surface rejection mask for the active camera"),
        ],
        default='QUALITY',
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        return _common_poll(cls, context)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    @staticmethod
    def _find_node(nodes, exact=None, prefix=None):
        for node in nodes:
            label = getattr(node, "label", "")
            if exact is not None and label == exact:
                return node
            if prefix is not None and label.startswith(prefix):
                return node
        return None

    def execute(self, context):
        meshes = _get_target_meshes(context)
        cameras = _get_cameras(context)
        _cam, camera_index = _get_active_camera(context)
        material_copies = {}
        visualized = 0

        for obj in meshes:
            for slot in obj.material_slots:
                source_mat = slot.material
                if not source_mat or not source_mat.use_nodes:
                    continue
                key = source_mat.as_pointer()
                mat = material_copies.get(key)
                if mat is None:
                    mat = source_mat.copy()
                    mat.name = f"SG_Debug_Diagnostic_{source_mat.name}"
                    material_copies[key] = mat
                    nodes = mat.node_tree.nodes
                    links = mat.node_tree.links

                    source_quality = self._find_node(nodes, exact="SGBestSourceQuality")
                    if self.mode == 'QUALITY':
                        source = source_quality
                    elif self.mode == 'ACQUISITION':
                        source = self._find_node(
                            nodes, prefix=f"AcquisitionQuality-{camera_index}-"
                        )
                    elif self.mode in {'TRUSTED', 'REPAIR'}:
                        source = source_quality
                    elif self.mode == 'WRITE':
                        source = self._find_node(
                            nodes, prefix=f"SequentialWriteWeight-{camera_index}-"
                        )
                    elif self.mode == 'GUARD':
                        source = self._find_node(
                            nodes, prefix=f"ProjectionGuardWeight-{camera_index}-"
                        )
                    else:
                        source = None

                    if source is None or not source.outputs:
                        bpy.data.materials.remove(mat)
                        material_copies[key] = None
                        continue

                    value_socket = source.outputs[0]
                    if self.mode in {'TRUSTED', 'REPAIR'}:
                        compare = nodes.new("ShaderNodeMath")
                        compare.operation = 'GREATER_THAN'
                        compare.label = "SGDebugTrusted"
                        compare.inputs[1].default_value = getattr(
                            context.scene, 'qwen_trust_threshold', 0.55
                        )
                        links.new(value_socket, compare.inputs[0])
                        value_socket = compare.outputs[0]
                        if self.mode == 'REPAIR':
                            invert = nodes.new("ShaderNodeMath")
                            invert.operation = 'SUBTRACT'
                            invert.inputs[0].default_value = 1.0
                            links.new(value_socket, invert.inputs[1])
                            value_socket = invert.outputs[0]

                    ramp = nodes.new("ShaderNodeValToRGB")
                    ramp.label = f"SGDebug-{self.mode}"
                    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
                    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
                    links.new(value_socket, ramp.inputs['Fac'])

                    emission = nodes.new("ShaderNodeEmission")
                    emission.label = "SGDebugEmission"
                    links.new(ramp.outputs['Color'], emission.inputs['Color'])
                    output = next(
                        (n for n in nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output),
                        None,
                    )
                    if output is None:
                        output = nodes.new("ShaderNodeOutputMaterial")
                    for link in list(output.inputs['Surface'].links):
                        links.remove(link)
                    links.new(emission.outputs[0], output.inputs['Surface'])

                if mat is not None:
                    slot.material = mat
                    visualized += 1

        if not visualized:
            self.report({'WARNING'}, "The selected diagnostic is not present in these projection materials")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Applied {self.mode.lower()} diagnostic to {visualized} material slot(s)")
        return {'FINISHED'}


class SG_OT_DebugDPPGMaterial(bpy.types.Operator):
    """Color the actual DPPG guard and feather signals on the projected mesh.

    Red is hard-guard attenuation; blue is feather attenuation. Only views
    with a nonzero pre-guard source weight are counted, so clipped or alpha-zero
    images cannot paint misleading rejection outside their valid footprint.
    """
    bl_idname = 'stablegen.debug_dppg_material'
    bl_label = 'DPPG Mask Material'
    bl_description = 'Visualize hard-guard attenuation in red and feather attenuation in blue'
    bl_options = {'REGISTER', 'UNDO'}

    scope: bpy.props.EnumProperty(
        name='Cameras',
        items=[('ALL', 'All cameras', 'Show the strongest effect from any valid camera'),
               ('ACTIVE', 'Active camera', 'Show only the selected or scene camera')],
        default='ALL',
    )  # type: ignore
    stage: bpy.props.EnumProperty(
        name='Stage',
        items=[('CURRENT', 'Current effective masks',
                'Show guards and feathering after coverage rescue and occlusion limits'),
               ('BEFORE_RESCUE', 'Before coverage rescue',
                'Show the original DPPG guard and feather fields')],
        default='CURRENT',
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        return _common_poll(cls, context)

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, 'scope')
        layout.prop(self, 'stage')

    @staticmethod
    def _indexed_nodes(nodes, prefix):
        found = {}
        for node in nodes:
            label = node.label
            if not label.startswith(prefix):
                continue
            index = label[len(prefix):].split('-', 1)[0]
            if index.isdigit():
                found[int(index)] = node
        return found

    @staticmethod
    def _math(nodes, links, operation, label, a, b):
        node = nodes.new('ShaderNodeMath')
        node.operation = operation
        node.label = label
        for index, value in enumerate((a, b)):
            if isinstance(value, (float, int)):
                node.inputs[index].default_value = value
            else:
                links.new(value, node.inputs[index])
        return node.outputs[0]

    def _build_material(self, source_mat, selected_index):
        mat = source_mat.copy()
        mat.name = f'SG_Debug_DPPG_{source_mat.name}'
        mat['sg_dppg_source_material'] = source_mat.name
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        guards = self._indexed_nodes(nodes, 'ProjectionGuardWeight-')
        guard_switches = self._indexed_nodes(nodes, 'ProjectionGuardWeightSwitch-')
        guarded = self._indexed_nodes(nodes, 'DepthPartGuarded-')
        raw_feathers = self._indexed_nodes(nodes, 'ProjectionGuardFeather-')
        feather_switches = self._indexed_nodes(nodes, 'ProjectionGuardFeatherSwitch-')
        rescue_floors = self._indexed_nodes(nodes, 'GuardRescueFloor-')
        guard_floors = self._indexed_nodes(nodes, 'ProjectionGuardFloor-')
        rescue_feathers = self._indexed_nodes(nodes, 'GuardRescueFeather-')
        occlusion_feathers = self._indexed_nodes(nodes, 'GuardOcclusionFade-')
        indices = sorted(guards)
        if self.scope == 'ACTIVE':
            indices = [index for index in indices if index == selected_index]
        red, blue = None, None
        used = 0
        for index in indices:
            guarded_node = guarded.get(index)
            if (guarded_node is None or not guarded_node.inputs[0].links):
                continue
            raw_source = guarded_node.inputs[0].links[0].from_socket
            valid = self._math(nodes, links, 'GREATER_THAN',
                               f'SGDebugDPPGValid-{index}', raw_source, 0.000001)
            if self.stage == 'CURRENT':
                guard_node = (rescue_floors.get(index) or guard_floors.get(index)
                              or guard_switches.get(index) or guards[index])
                feather_node = (occlusion_feathers.get(index)
                                or rescue_feathers.get(index) or feather_switches.get(index)
                                or raw_feathers.get(index))
            else:
                guard_node = guard_switches.get(index) or guards[index]
                feather_node = feather_switches.get(index) or raw_feathers.get(index)
            guard_loss = self._math(nodes, links, 'SUBTRACT',
                                    f'SGDebugDPPGGuardLoss-{index}', 1.0,
                                    guard_node.outputs[0])
            guard_loss = self._math(nodes, links, 'MULTIPLY',
                                    f'SGDebugDPPGGuardValid-{index}', valid, guard_loss)
            if feather_node is not None:
                feather_loss = self._math(nodes, links, 'SUBTRACT',
                                          f'SGDebugDPPGFeatherLoss-{index}', 1.0,
                                          feather_node.outputs[0])
                feather_loss = self._math(nodes, links, 'MULTIPLY',
                                          f'SGDebugDPPGFeatherValid-{index}', valid, feather_loss)
            else:
                feather_loss = None
            red = (self._math(nodes, links, 'MAXIMUM', 'SGDebugDPPGMaxGuard', red, guard_loss)
                   if red is not None else guard_loss)
            if feather_loss is not None:
                blue = (self._math(nodes, links, 'MAXIMUM', 'SGDebugDPPGMaxFeather', blue, feather_loss)
                        if blue is not None else feather_loss)
            used += 1
        if not used:
            bpy.data.materials.remove(mat)
            return None, 0
        # A small neutral floor keeps unaffected mesh visible. Signal intensity
        # is otherwise proportional to the actual attenuation (0..1).
        red = self._math(nodes, links, 'MULTIPLY', 'SGDebugDPPGRed', red, 0.92)
        red = self._math(nodes, links, 'ADD', 'SGDebugDPPGRedBase', red, 0.08)
        if blue is not None:
            blue = self._math(nodes, links, 'MULTIPLY', 'SGDebugDPPGBlue', blue, 0.92)
            blue = self._math(nodes, links, 'ADD', 'SGDebugDPPGBlueBase', blue, 0.08)
        color = nodes.new('ShaderNodeCombineXYZ')
        color.label = 'SGDebugDPPGColor: red guard, blue feather'
        links.new(red, color.inputs['X'])
        color.inputs['Y'].default_value = 0.08
        if blue is not None:
            links.new(blue, color.inputs['Z'])
        else:
            color.inputs['Z'].default_value = 0.08
        emission = nodes.new('ShaderNodeEmission')
        emission.label = 'SGDebugDPPGEmission'
        links.new(color.outputs[0], emission.inputs['Color'])
        output = next((node for node in nodes
                       if node.type == 'OUTPUT_MATERIAL' and node.is_active_output), None)
        if output is None:
            output = nodes.new('ShaderNodeOutputMaterial')
        links.new(emission.outputs[0], output.inputs['Surface'])
        return mat, used

    def execute(self, context):
        cameras = _interactive_generation_cameras(context)
        selected = _interactive_selected_camera(context, cameras)
        camera = selected or (context.scene.camera if context.scene.camera in cameras else None)
        selected_index = cameras.index(camera) if camera in cameras else 0
        material_copies = {}
        assignments = []
        cameras_seen = 0
        for obj in _get_target_meshes(context):
            for slot in obj.material_slots:
                source = slot.material
                if source is None or not source.use_nodes:
                    continue
                original_name = source.get('sg_dppg_source_material')
                if original_name:
                    source = bpy.data.materials.get(original_name) or source
                key = source.as_pointer()
                if key not in material_copies:
                    material_copies[key] = self._build_material(source, selected_index)
                clone, count = material_copies[key]
                if clone is not None:
                    assignments.append((slot, clone))
                    cameras_seen = max(cameras_seen, count)
        if not assignments:
            self.report({'WARNING'}, 'No DPPG guard fields found in the target projection materials; Reproject with DPPG enabled')
            return {'CANCELLED'}
        for slot, clone in assignments:
            slot.material = clone
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        self.report({'INFO'},
                    f'DPPG mask material applied to {len(assignments)} slot(s); '
                    f'{cameras_seen} camera field(s), red=guard, blue=feather. View in Material Preview.')
        return {'FINISHED'}


class SG_OT_DebugRestoreDPPGMaterial(bpy.types.Operator):
    """Restore the projection materials saved by DPPG Mask Material."""
    bl_idname = 'stablegen.debug_restore_dppg_material'
    bl_label = 'Restore Projection Material'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not _common_poll(cls, context):
            return False
        return any(slot.material and slot.material.get('sg_dppg_source_material')
                   for obj in _get_target_meshes(context) for slot in obj.material_slots)

    def execute(self, context):
        restored = 0
        old_materials = set()
        for obj in _get_target_meshes(context):
            for slot in obj.material_slots:
                debug_mat = slot.material
                if debug_mat is None:
                    continue
                source_name = debug_mat.get('sg_dppg_source_material')
                source = bpy.data.materials.get(source_name) if source_name else None
                if source is None:
                    continue
                slot.material = source
                old_materials.add(debug_mat)
                restored += 1
        for material in old_materials:
            if material.users == 0:
                bpy.data.materials.remove(material)
        self.report({'INFO'}, f'Restored {restored} projection material slot(s)')
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════
# 11. Debug Make Solid
# ═════════════════════════════════════════════════════════════════════════

class SG_OT_DebugMakeSolid(bpy.types.Operator):
    """Run the cylinder-based raycast solidification and hole-filling algorithm
    in-place on the active mesh object.  Requires the 'SG_PrintPreview' color attribute."""
    bl_idname = "stablegen.debug_make_solid"
    bl_label = "Debug Make Solid"
    bl_options = {'REGISTER', 'UNDO'}

    fill_gaps: bpy.props.BoolProperty(
        name="Fill Gaps",
        description="Run boundary loop tracing and hole filling",
        default=True,
    )

    keep_largest_island: bpy.props.BoolProperty(
        name="Keep Largest Island",
        description="After the visibility cleanup, keep only the largest connected piece of geometry and discard any smaller disconnected islands (e.g. interior skin the raycast missed). Runs before the gap-filling step",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        prefs = context.preferences.addons.get(__package__)
        if not prefs or not prefs.preferences.enable_debug:
            return False
        if sg_modal_active(context):
            return False
        return context.active_object and context.active_object.type == 'MESH'

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        from .texturing.print_export import (
            _make_solid_mesh_object,
            _match_colors_hsv,
            _smooth_colors_vectorized,
            linear_to_srgb_numpy,
            srgb_to_linear_numpy,
            classify_filaments,
            decompose_subtractive,
            solve_color_mixing_numpy,
        )

        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "No active mesh object.")
            return {'CANCELLED'}

        if not obj.data.color_attributes.get("SG_PrintPreview"):
            self.report({'ERROR'}, "Make Solid requires the 'SG_PrintPreview' color layer. Run 'Preview Colors' first.")
            return {'CANCELLED'}

        # Calculate cylinder bounds for visualization using current mesh
        mesh = obj.data
        verts_co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", verts_co)
        verts_co = verts_co.reshape((-1, 3))
        
        xmin, ymin, zmin = verts_co.min(axis=0)
        xmax, ymax, zmax = verts_co.max(axis=0)
        
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        radius = max(xmax - xmin, ymax - ymin) * 0.5 * 1.25
        z_top = zmax + (zmax - zmin) * 0.25
        z_bottom = zmin
        depth = z_top - z_bottom
        z_center = (z_top + z_bottom) * 0.5

        try:
            # Delete any existing debug cylinder first
            existing_cyl = context.scene.objects.get("SG_Debug_Solid_Cylinder")
            if existing_cyl:
                bpy.data.objects.remove(existing_cyl, do_unlink=True)
                
            # Create cylinder visualization
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=32,
                radius=radius,
                depth=depth,
                location=(cx, cy, z_center),
                end_fill_type='NOTHING'
            )
            cyl_obj = context.active_object
            cyl_obj.name = "SG_Debug_Solid_Cylinder"
            cyl_obj.display_type = 'WIRE'
            
            # Restore selection to the original object
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj

            solid_mesh = _make_solid_mesh_object(obj, fill_gaps=self.fill_gaps, keep_largest_island=self.keep_largest_island)

            # Replace the mesh of the object
            old_mesh = obj.data
            obj.data = solid_mesh

            # Remove the old mesh from memory
            bpy.data.meshes.remove(old_mesh)

            # ── Re-run the export color pipeline so the viewport shows exactly  ──
            # ── what filament/color the 3MF exporter will assign to each tri.   ──
            scene = context.scene
            palette_rgbs_linear = np.array([item.color for item in scene.stablegen_print_palette], dtype=np.float32)
            palette_rgbs = [tuple(c) for c in linear_to_srgb_numpy(palette_rgbs_linear)]

            color_layer = solid_mesh.color_attributes.get("SG_PrintPreview")
            if palette_rgbs and color_layer:
                solid_mesh.calc_loop_triangles()

                n_loops = len(solid_mesh.loops)
                loop_colors = np.empty(n_loops * 4, dtype=np.float32)
                color_layer.data.foreach_get("color", loop_colors)
                loop_colors = loop_colors.reshape((-1, 4))

                n_tris = len(solid_mesh.loop_triangles)
                tri_loops = np.empty(n_tris * 3, dtype=np.int32)
                solid_mesh.loop_triangles.foreach_get("loops", tri_loops)
                tri_loops = tri_loops.reshape((-1, 3))

                face_colors_rgb = loop_colors[tri_loops].mean(axis=1)[:, :3]
                if color_layer.data_type == 'BYTE_COLOR':
                    face_colors_rgb = srgb_to_linear_numpy(face_colors_rgb)
                
                # Convert linear face colors to sRGB for matching/mixing
                face_colors_srgb = linear_to_srgb_numpy(face_colors_rgb)

                is_dithered = getattr(scene, "stablegen_print_dithered", True)
                if is_dithered:
                    mode, channel_indices = classify_filaments(palette_rgbs)
                    if mode == 'solid':
                        fil_rgbs = np.array(palette_rgbs, dtype=np.float32)
                        demands = solve_color_mixing_numpy(fil_rgbs, face_colors_srgb)
                        
                        # Reconstruct mixed color using the chosen mixing model
                        model_formula = getattr(scene, "stablegen_print_solver_formula", 'KUBELKA_MUNK')
                        if model_formula == 'KUBELKA_MUNK':
                            scat_weight = getattr(scene, "stablegen_print_scattering_weight", 10.0)
                            fil_val = fil_rgbs.max(axis=1)
                            fil_min = fil_rgbs.min(axis=1)
                            fil_sat = np.zeros_like(fil_val)
                            mask = fil_val > 1e-5
                            fil_sat[mask] = (fil_val[mask] - fil_min[mask]) / fil_val[mask]
                            S_fil = 1.0 + scat_weight * (1.0 - fil_sat) * fil_val
                            S_fil = S_fil[:, None]
                            R_clip = np.clip(fil_rgbs, 0.001, 0.999)
                            K_fil = S_fil * ((1.0 - R_clip)**2) / (2.0 * R_clip)
                            
                            K_mix = np.dot(demands, K_fil)
                            S_mix = np.dot(demands, S_fil)
                            ratio = K_mix / S_mix
                            mixed_srgb = 1.0 + ratio - np.sqrt(ratio**2 + 2.0 * ratio)
                        elif model_formula == 'ADDITIVE':
                            mixed_srgb = np.dot(demands, fil_rgbs)
                        else:  # SUBTRACTIVE
                            mixed_srgb = 1.0 - np.dot(demands, 1.0 - fil_rgbs)
                    else:
                        mixed_srgb = np.empty_like(face_colors_srgb)
                        fil_rgbs = np.array([palette_rgbs[idx] for idx in channel_indices], dtype=np.float32)
                        for i in range(len(face_colors_srgb)):
                            r, g, b = face_colors_srgb[i] * 255.0
                            fractions = decompose_subtractive(r, g, b, mode, palette_rgbs, channel_indices)
                            mixed_srgb[i] = np.clip(1.0 - sum(fractions[j] * (1.0 - fil_rgbs[j]) for j in range(len(fractions))), 0.0, 1.0)
                    
                    mapped_colors = srgb_to_linear_numpy(mixed_srgb)
                else:
                    # Solid Mode Quantization
                    palette_arr = np.array(palette_rgbs, dtype=np.float32)
                    chroma_threshold = getattr(scene, "stablegen_print_chroma_threshold", 0.35)
                    material_indices = _match_colors_hsv(face_colors_srgb, palette_rgbs, chroma_threshold=chroma_threshold)

                    smoothing_passes = getattr(scene, "stablegen_print_smoothing", 2)
                    if smoothing_passes > 0:
                        tri_verts = np.empty(n_tris * 3, dtype=np.int32)
                        solid_mesh.loop_triangles.foreach_get("vertices", tri_verts)
                        tri_verts = tri_verts.reshape((-1, 3))
                        material_indices = _smooth_colors_vectorized(
                            tri_verts, material_indices + 1,
                            len(solid_mesh.vertices), len(palette_rgbs),
                            iterations=smoothing_passes
                        ) - 1

                    mapped_srgb = palette_arr[material_indices]
                    mapped_colors = srgb_to_linear_numpy(mapped_srgb)

                # Map quantized colors and write back into SG_PrintPreview
                if color_layer.data_type != 'FLOAT_COLOR':
                    solid_mesh.color_attributes.remove(color_layer)
                    color_layer = solid_mesh.color_attributes.new(
                        name="SG_PrintPreview",
                        type='FLOAT_COLOR',
                        domain='CORNER',
                    )

                mapped_rgba = np.ones((n_tris, 4), dtype=np.float32)
                mapped_rgba[:, :3] = mapped_colors

                all_loop_colors = np.ones((n_loops, 4), dtype=np.float32)
                all_loop_colors[tri_loops[:, 0]] = mapped_rgba
                all_loop_colors[tri_loops[:, 1]] = mapped_rgba
                all_loop_colors[tri_loops[:, 2]] = mapped_rgba
                color_layer.data.foreach_set("color", all_loop_colors.ravel())

                # Switch viewport shading to show the updated color attribute
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                try:
                                    space.shading.type = 'SOLID'
                                    space.shading.color_type = 'VERTEX'
                                    try:
                                        space.shading.light = 'FLAT'
                                    except Exception:
                                        pass
                                    solid_mesh.color_attributes.active = color_layer
                                    if hasattr(solid_mesh, "attributes"):
                                        solid_mesh.attributes.active_color = color_layer
                                except TypeError:
                                    try:
                                        space.shading.type = 'SOLID'
                                        space.shading.color_type = 'ATTRIBUTE'
                                        space.shading.attribute_name = "SG_PrintPreview"
                                        try:
                                            space.shading.light = 'FLAT'
                                        except Exception:
                                            pass
                                    except Exception:
                                        pass
                                except Exception:
                                    pass

                print(f"[StableGen Debug] Export color preview: {n_tris} tris, {len(palette_rgbs)} palette colors.")

            self.report({'INFO'}, f"Successfully solidified {obj.name} in-place. Viewport shows export color assignment.")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Solidification failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


# ═════════════════════════════════════════════════════════════════════════
# Interactive discard / Qwen context check
# ═════════════════════════════════════════════════════════════════════════

_INTERACTIVE_DISCARD_SETTINGS = (
    'discard_factor',
    'qwen_prompt_gray_background',
    'qwen_guidance_background_color',
    'qwen_guidance_fallback_color',
    'qwen_use_confidence_context',
    'qwen_trust_threshold',
    'qwen_context_fallback_dilation',
    'qwen_context_cleanup',
    'qwen_context_cleanup_hue_tolerance',
    'qwen_context_cleanup_value_adjust',
)

_INTERACTIVE_DISCARD_BUILD_SETTINGS = (
    'overwrite_material',
    'generation_method',
    'generation_mode',
    'qwen_generation_method',
    'qwen_voronoi_mode',
)

_MAGENTA_EDIT_ARCHITECTURES = {
    'qwen_image_edit',
    'qwen_image_21',
    'flux2_klein',
}


def _copy_scene_value(value):
    """Copy scalar and Blender vector values into assignment-safe Python data."""
    try:
        if not isinstance(value, (str, bytes)) and hasattr(value, '__len__'):
            return tuple(value)
    except TypeError:
        pass
    return value


def _interactive_generation_root(context, meshes, material_id):
    """Resolve the source run from projection images, independent of Save As."""
    roots = {}
    for obj in meshes:
        material = obj.active_material
        if not material or not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type != 'TEX_IMAGE' or node.image is None:
                continue
            parts = node.label.split('-')
            if (len(parts) != 2 or not parts[0].isdigit()
                    or parts[1] != str(material_id)):
                continue
            path = os.path.normpath(bpy.path.abspath(
                node.image.filepath, library=node.image.library
            ))
            if not os.path.isfile(path):
                continue
            folder = os.path.dirname(path)
            if os.path.basename(folder).lower() != 'generated':
                continue
            root = os.path.dirname(folder)
            roots[os.path.normcase(root)] = root
    if len(roots) > 1:
        raise RuntimeError('Selected projection materials reference multiple generation runs')
    return next(iter(roots.values()), get_generation_dirs(context)['revision'])


def _interactive_generation_cameras(context, revision=None):
    """Resolve the same persisted camera order used by the latest generation."""
    scene = context.scene
    cameras = list(generation_cameras(scene))
    order_names = []
    if scene.sg_use_custom_camera_order and len(scene.sg_camera_order) > 0:
        order_names = [item.name for item in scene.sg_camera_order]
    cameras = order_cameras(cameras, order_names)

    manifest_path = os.path.join(
        revision or get_generation_dirs(context)['revision'], 'view_plan.json'
    )
    try:
        with open(manifest_path, 'r', encoding='utf-8') as stream:
            cameras = list(order_cameras_from_manifest(cameras, json.load(stream)))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        pass
    return cameras


def _interactive_selected_camera(context, cameras):
    selected = [
        obj for obj in context.selected_objects
        if obj.type == 'CAMERA' and obj in cameras
    ]
    if len(selected) != 1:
        return None
    return selected[0]


def _interactive_preview_layout(context_size, generated_size, max_width,
                                max_height, gap, title_height):
    """Return image rectangles, fitting both cards while preserving aspect."""
    context_aspect = context_size[0] / max(1, context_size[1])
    generated_aspect = generated_size[0] / max(1, generated_size[1])
    if context_aspect > 1 or generated_aspect > 1:
        # The generated view sits below the live context; both share a width.
        width = max(1, min(max_width,
                    (max_height - gap - 2 * title_height)
                    / (1 / context_aspect + 1 / generated_aspect)))
        context_height = width / context_aspect
        generated_height = width / generated_aspect
        return ((0, generated_height + title_height + gap, width, context_height),
                (0, 0, width, generated_height))
    height = max(1, min(max_height - title_height,
                       (max_width - gap) / (context_aspect + generated_aspect)))
    context_width = height * context_aspect
    return ((0, 0, context_width, height),
            (context_width + gap, 0, height * generated_aspect, height))


class SG_OT_InteractiveDiscardCheck(bpy.types.Operator):
    """Preview the sequential context passed to an image-edit architecture."""

    bl_idname = 'stablegen.interactive_discard_check'
    bl_label = 'Interactive Discard Check'
    bl_description = (
        "Preview the selected camera's magenta context while tuning discard "
        "and Context Render Options; Apply keeps the values and Cancel restores them"
    )
    bl_options = {'REGISTER'}

    _active = None
    status = ''
    preview_image_name = ''

    def _load_generated_preview(self, context):
        self._generated_preview_image_name = ''
        self._generated_preview_texture = None
        path = get_file_path(context, 'generated', camera_id=self._camera_index,
                             material_id=self._material_id)
        if not os.path.isfile(path):
            return
        try:
            # Own a separate datablock so its display color space cannot alter
            # the image used by the actual projection material.
            image = bpy.data.images.load(path, check_existing=False)
            image.name = 'SG Investigated Camera Last Generated'
            image.colorspace_settings.name = 'Non-Color'
            self._generated_preview_image_name = image.name
        except RuntimeError as error:
            print(f'[StableGen] Could not load comparison image: {error}')

    def _start_preview_overlay(self, context):
        area = context.area
        if area is None or area.type != 'VIEW_3D':
            area = next((area for area in context.window.screen.areas
                         if area.type == 'VIEW_3D'), None)
        self._preview_area_pointer = area.as_pointer() if area else None
        self._preview_texture = None
        self._load_generated_preview(context)
        self._preview_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_preview_overlay, (), 'WINDOW', 'POST_PIXEL'
        )

    def _stop_preview_overlay(self):
        handle = getattr(self, '_preview_draw_handle', None)
        if handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
            self._preview_draw_handle = None
        self._preview_texture = None
        self._generated_preview_texture = None
        image = bpy.data.images.get(getattr(self, '_generated_preview_image_name', ''))
        if image is not None:
            bpy.data.images.remove(image)
        self._generated_preview_image_name = ''

    def _draw_preview_overlay(self):
        context = bpy.context
        area, region = context.area, context.region
        if (type(self)._active is not self or area is None or region is None
                or area.as_pointer() != self._preview_area_pointer
                or region.type != 'WINDOW'):
            return
        image = bpy.data.images.get(type(self).preview_image_name)
        if image is None or not all(image.size):
            return

        import blf
        import gpu
        from .texturing.gallery import _draw_rect_2d, _draw_texture_2d

        # Leave room for overlapping toolbars and the settings sidebar. Refit
        # on every redraw so resizing the viewport keeps the image in view.
        left, right = 0, region.width
        for other in area.regions:
            if other.type in {'TOOLS', 'UI'} and other.width > 1:
                x = other.x - region.x
                if x <= 0:
                    left = max(left, x + other.width)
                elif x < region.width:
                    right = min(right, x)
        margin = 16 * context.preferences.system.ui_scale
        available_width = right - left - 2 * margin
        available_height = region.height - 2 * margin - 28
        if available_width <= 0 or available_height <= 0:
            return
        generated = bpy.data.images.get(self._generated_preview_image_name)
        generated_size = generated.size if generated and all(generated.size) else image.size
        gap = 16 * context.preferences.system.ui_scale
        title_height = 26
        rectangles = _interactive_preview_layout(
            image.size, generated_size, available_width * 0.45,
            available_height * 0.80, gap, title_height,
        )
        x, y = left + margin, margin
        if self._preview_texture is None:
            self._preview_texture = gpu.texture.from_image(image)
        if generated is not None and self._generated_preview_texture is None:
            self._generated_preview_texture = gpu.texture.from_image(generated)
        old_blend = gpu.state.blend_get()
        old_depth = gpu.state.depth_test_get()
        try:
            gpu.state.depth_test_set('NONE')
            cards = ((self._preview_texture, 'Context preview'),
                     (self._generated_preview_texture, 'Last generated'))
            for (dx, dy, width, height), (texture, title) in zip(rectangles, cards):
                card_x, card_y = x + dx, y + dy
                _draw_rect_2d(card_x - 4, card_y - 4, card_x + width + 4,
                              card_y + height + title_height,
                              (0.025, 0.025, 0.025, 0.95))
                if texture is not None:
                    _draw_texture_2d(texture, card_x, card_y,
                                     card_x + width, card_y + height)
                else:
                    blf.size(0, 12)
                    blf.color(0, 0.8, 0.8, 0.8, 1)
                    blf.position(0, card_x + 8, card_y + height / 2, 0)
                    blf.draw(0, 'No generated image')
                blf.size(0, 13)
                blf.color(0, 1, 1, 1, 1)
                blf.position(0, card_x + 4, card_y + height + 7, 0)
                blf.draw(0, title)
        finally:
            gpu.state.depth_test_set(old_depth)
            gpu.state.blend_set(old_blend)

    @classmethod
    def poll(cls, context):
        scene = context.scene
        if cls._active is not None:
            cls.poll_message_set('Interactive Discard Check is already running')
            return False
        if scene.model_architecture not in _MAGENTA_EDIT_ARCHITECTURES:
            cls.poll_message_set('Requires an image-edit architecture with magenta context')
            return False
        if scene.generation_method != 'sequential':
            cls.poll_message_set('Requires Sequential generation')
            return False
        if scene.qwen_generation_method != 'generate':
            cls.poll_message_set('Requires the Qwen Generate workflow')
            return False
        if scene.qwen_context_render_mode not in {'REPLACE_STYLE', 'ADDITIONAL'}:
            cls.poll_message_set('Enable a magenta Context Render mode first')
            return False
        if not scene.output_timestamp:
            cls.poll_message_set('No completed generation is available')
            return False
        if sg_modal_active(context):
            cls.poll_message_set('Another StableGen operation is running')
            return False
        cameras = _interactive_generation_cameras(context)
        if _interactive_selected_camera(context, cameras) is None:
            cls.poll_message_set('Select exactly one generated camera')
            return False
        return bool(cameras)

    def _tag_redraw(self, context):
        for window in context.window_manager.windows:
            if not window.screen:
                continue
            for area in window.screen.areas:
                area.tag_redraw()

    def _settings_key(self, scene):
        result = []
        for name in _INTERACTIVE_DISCARD_SETTINGS:
            value = getattr(scene, name)
            result.append((name, _copy_scene_value(value)))
        return tuple(result)

    def _snapshot_materials(self):
        self._material_states = {}
        preview_materials = {}
        for obj in self._meshes:
            materials = [slot.material for slot in obj.material_slots]
            self._material_states[obj] = (materials, obj.active_material_index)
            if not materials:
                continue
            active_index = min(obj.active_material_index, len(materials) - 1)
            original = materials[active_index]
            if original is None:
                continue
            preview = preview_materials.get(original)
            if preview is None:
                preview = original.copy()
                preview.name = f'SG_DiscardPreview_{original.name}'
                preview_materials[original] = preview
            obj.material_slots[active_index].material = preview
        self._preview_materials = list(preview_materials.values())

    def _restore_materials(self):
        for obj, (materials, active_index) in self._material_states.items():
            if obj.name not in bpy.data.objects:
                continue
            obj.data.materials.clear()
            for material in materials:
                obj.data.materials.append(material)
            if materials:
                obj.active_material_index = min(active_index, len(materials) - 1)
        for material in self._preview_materials:
            if material and material.name in bpy.data.materials:
                bpy.data.materials.remove(material)

    def _artifact_paths(self, context):
        context_path = get_file_path(
            context, 'inpaint', subtype='render', camera_id=self._camera_index
        )
        mask_path = get_file_path(
            context, 'inpaint', subtype='visibility', camera_id=self._camera_index
        )
        stem, extension = os.path.splitext(mask_path)
        return (
            context_path,
            mask_path,
            f'{stem}_repair{extension}',
            f'{stem}_trusted{extension}',
        )

    def _snapshot_artifacts(self, context):
        self._artifact_states = {}
        for path in self._artifact_paths(context):
            try:
                with open(path, 'rb') as stream:
                    self._artifact_states[path] = stream.read()
            except FileNotFoundError:
                self._artifact_states[path] = None

    def _restore_artifacts(self):
        for path, content in self._artifact_states.items():
            if content is None:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as stream:
                    stream.write(content)

    def _reload_preview_image(self, context, path):
        image = bpy.data.images.get(type(self).preview_image_name)
        if image is None:
            image = bpy.data.images.load(path, check_existing=False)
            image.name = 'SG Interactive Context Preview'
            type(self).preview_image_name = image.name
        else:
            image.filepath = path
            image.reload()
        # The POST_PIXEL image shader displays the PNG's encoded RGB directly,
        # matching the gallery overlay and avoiding a second scene view transform.
        image.colorspace_settings.name = 'Non-Color'
        self._preview_texture = None
        self._tag_redraw(context)

    def _refresh_preview(self, context):
        scene = context.scene
        type(self).status = (
            f'Updating {self._camera.name} '
            f'({self._camera_index + 1}/{len(self._cameras)})…'
        )
        self._tag_redraw(context)

        # These values affect how project_image builds the diagnostic graph,
        # but are infrastructure rather than user choices in this tool.
        scene.overwrite_material = True
        scene.generation_method = 'sequential'
        scene.generation_mode = 'standard'
        scene.qwen_generation_method = 'generate'
        scene.qwen_voronoi_mode = False
        scene.camera = self._camera

        project_image(
            context,
            self._meshes,
            self._material_id,
            stop_index=self._camera_index - 1,
            cameras=self._cameras,
            force_rebuild=True,
            reuse_projection_uv=True,
        )

        visibility_camera = (
            self._cameras[self._camera_index - 1]
            if self._camera_index > 0 else self._camera
        )
        fallback = tuple(scene.qwen_guidance_fallback_color)
        background = tuple(scene.qwen_guidance_background_color)
        if scene.model_architecture == 'qwen_image_21':
            background = (0.5, 0.5, 0.5)

        # Blender's write_still render also writes to scene.render.filepath in
        # addition to the compositor target. Route that incidental output to a
        # private sink so preview refreshes never overwrite the user's render.
        original_render_filepath = scene.render.filepath
        scene.render.filepath = self._render_sink
        try:
            with _SGCameraResolution(context, self._camera):
                export_visibility(
                    context,
                    self._meshes,
                    camera_visibility=visibility_camera,
                    camera_render=self._camera,
                    camera_id=self._camera_index,
                )
                export_emit_image(
                    context,
                    self._meshes,
                    camera_id=self._camera_index,
                    bg_color=background,
                    fallback_color=fallback,
                )
        finally:
            scene.render.filepath = original_render_filepath
            self._clean_render_sink()

        # Reuse the production post-processing implementation without writing
        # confidence-manifest metadata during a temporary inspection.
        from .texturing.generator import ComfyUIGenerate
        helper = SimpleNamespace(
            _uploaded_images_cache={},
            _record_confidence_artifacts=lambda *args, **kwargs: None,
        )
        if scene.qwen_use_confidence_context:
            ComfyUIGenerate._apply_qwen_binary_context_mask(
                helper, context, self._camera_index, fallback
            )
        else:
            ComfyUIGenerate._dilate_qwen_context_fallback(
                helper, context, self._camera_index, fallback
            )

        context_path = get_file_path(
            context, 'inpaint', subtype='render', camera_id=self._camera_index
        )
        if not os.path.exists(context_path):
            raise RuntimeError('Context preview render was not created')
        self._reload_preview_image(context, context_path)
        type(self).status = (
            f'{self._camera.name}: cameras 1–{self._camera_index} projected; '
            f'camera {self._camera_index + 1} and later blanked'
            if self._camera_index > 0 else
            f'{self._camera.name}: every generated camera is blanked'
        )
        self._tag_redraw(context)

    def _restore_selection(self, context):
        try:
            bpy.ops.object.select_all(action='DESELECT')
        except Exception:
            pass
        for obj in self._selected_objects:
            if obj.name in bpy.data.objects:
                obj.select_set(True)
        if self._active_object and self._active_object.name in bpy.data.objects:
            context.view_layer.objects.active = self._active_object

    def _restore_render_state(self, scene):
        scene.render.engine = self._render_snapshot['engine']
        scene.render.filepath = self._render_snapshot['filepath']
        cycles = getattr(scene, 'cycles', None)
        if cycles is None:
            return
        for name, value in self._render_snapshot['cycles'].items():
            if value is not None and hasattr(cycles, name):
                setattr(cycles, name, value)

    def _remove_temporary_images(self):
        for image in list(bpy.data.images):
            if image.as_pointer() in self._image_pointers:
                continue
            if image.source == 'FILE' and image.users == 0:
                bpy.data.images.remove(image)

    def _clean_render_sink(self):
        for path in glob.glob(f'{self._render_sink}*'):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def _finish(self, context, apply):
        self._stop_preview_overlay()
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        if not apply:
            for name, value in self._setting_snapshot.items():
                setattr(context.scene, name, value)
        for name, value in self._build_snapshot.items():
            setattr(context.scene, name, value)

        context.scene.camera = self._original_camera
        self._restore_materials()
        try:
            self._restore_artifacts()
        finally:
            self._restore_output_scope(context.scene)
        self._restore_render_state(context.scene)
        self._restore_selection(context)
        self._clean_render_sink()

        image = bpy.data.images.get(type(self).preview_image_name)
        if image is not None:
            bpy.data.images.remove(image)
        self._remove_temporary_images()
        type(self).preview_image_name = ''
        type(self)._active = None
        type(self).status = ''

        if apply:
            from .ui.presets import update_parameters
            update_parameters(None, context)
        else:
            context.scene.stablegen_preset = self._original_preset
            context.scene.active_preset = self._original_active_preset
        self._tag_redraw(context)

    def _restore_output_scope(self, scene):
        previous = getattr(self, '_generation_output_previous', None)
        if previous is None:
            if '_sg_discard_output' in scene:
                del scene['_sg_discard_output']
        else:
            scene['_sg_discard_output'] = previous

    def invoke(self, context, _event):
        cameras = _interactive_generation_cameras(context)
        camera = _interactive_selected_camera(context, cameras)
        if camera is None:
            self.report({'ERROR'}, 'Select exactly one generated camera')
            return {'CANCELLED'}

        selected_meshes = [
            obj for obj in context.selected_objects
            if obj.type == 'MESH' and not obj.hide_get()
        ]
        meshes = selected_meshes or [
            obj for obj in context.view_layer.objects
            if obj.type == 'MESH' and not obj.hide_get()
        ]
        if not meshes:
            self.report({'ERROR'}, 'No visible mesh objects to preview')
            return {'CANCELLED'}

        projection_ids = set()
        for obj in meshes:
            for attribute in obj.data.attributes:
                if not attribute.name.startswith('ProjectionUV_'):
                    continue
                try:
                    projection_ids.add(int(attribute.name.rsplit('_', 1)[1]))
                except (ValueError, IndexError):
                    pass
        material_id = max(projection_ids, default=-1)
        if material_id < 0:
            self.report({'ERROR'}, 'No StableGen projection material was found')
            return {'CANCELLED'}

        meshes = [
            obj for obj in meshes
            if obj.active_material is not None
            and obj.data.attributes.get(f'ProjectionUV_0_{material_id}') is not None
        ]
        if not meshes:
            self.report({'ERROR'}, 'No meshes use the latest StableGen projection')
            return {'CANCELLED'}

        try:
            revision = _interactive_generation_root(context, meshes, material_id)
        except RuntimeError as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        # Use this run's manifest as well as its images; the copied .blend's
        # name may point at a different or nonexistent output directory.
        cameras = _interactive_generation_cameras(context, revision=revision)
        camera_index = cameras.index(camera)
        for index in range(camera_index):
            path = os.path.join(revision, 'generated',
                                f'generated_image{index}-{material_id}-0001.png')
            if not os.path.isfile(path):
                self.report({'ERROR'}, f'Missing generated image for camera {index + 1}: {path}')
                return {'CANCELLED'}

        self._cameras = tuple(cameras)
        self._camera = camera
        self._camera_index = camera_index
        self._meshes = tuple(meshes)
        self._material_id = material_id
        self._setting_snapshot = {
            name: _copy_scene_value(getattr(context.scene, name))
            for name in _INTERACTIVE_DISCARD_SETTINGS
        }
        self._build_snapshot = {
            name: _copy_scene_value(getattr(context.scene, name))
            for name in _INTERACTIVE_DISCARD_BUILD_SETTINGS
        }
        self._original_preset = context.scene.stablegen_preset
        self._original_active_preset = context.scene.active_preset
        self._original_camera = context.scene.camera
        cycles = getattr(context.scene, 'cycles', None)
        self._render_snapshot = {
            'engine': context.scene.render.engine,
            'filepath': context.scene.render.filepath,
            'cycles': {
                name: getattr(cycles, name, None) if cycles is not None else None
                for name in ('device', 'shading_system', 'use_osl')
            },
        }
        self._image_pointers = {
            image.as_pointer() for image in bpy.data.images
        }
        sink_handle, self._render_sink = tempfile.mkstemp(
            prefix='stablegen_interactive_discard_', suffix='.png'
        )
        os.close(sink_handle)
        os.remove(self._render_sink)
        self._selected_objects = tuple(context.selected_objects)
        self._active_object = context.view_layer.objects.active
        self._last_settings_key = None
        self._finish_requested = None
        self._timer = None

        self._generation_output_previous = context.scene.get('_sg_discard_output')
        context.scene['_sg_discard_output'] = revision
        try:
            self._snapshot_materials()
            self._snapshot_artifacts(context)
        except Exception as error:
            self._restore_output_scope(context.scene)
            self.report({'ERROR'}, f'Could not prepare discard preview: {error}')
            try:
                self._restore_materials()
            except Exception:
                pass
            return {'CANCELLED'}

        type(self)._active = self
        type(self).status = 'Preparing context preview…'
        self._start_preview_overlay(context)
        self._timer = context.window_manager.event_timer_add(
            0.35, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        self._tag_redraw(context)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        """Start the same modal session when called through EXEC_DEFAULT."""
        return self.invoke(context, None)

    def modal(self, context, event):
        if self._finish_requested is not None:
            apply = self._finish_requested == 'APPLY'
            self._finish(context, apply)
            self.report(
                {'INFO'},
                'Interactive discard settings applied' if apply
                else 'Interactive discard preview cancelled',
            )
            return {'FINISHED'} if apply else {'CANCELLED'}

        if event.type in {'ESC', 'RIGHTMOUSE'}:
            self._finish(context, False)
            self.report({'INFO'}, 'Interactive discard preview cancelled')
            return {'CANCELLED'}

        # Blender 5.1's Event RNA has no timer attribute. Accept TIMER events
        # there; the settings key prevents duplicate renders from other timers.
        if event.type == 'TIMER' and getattr(event, 'timer', self._timer) == self._timer:
            settings_key = self._settings_key(context.scene)
            if settings_key != self._last_settings_key:
                self._last_settings_key = settings_key
                try:
                    self._refresh_preview(context)
                except Exception as error:
                    type(self).status = f'Preview error: {error}'
                    import traceback
                    traceback.print_exc()
                    self._tag_redraw(context)
        return {'PASS_THROUGH'}

    def cancel(self, context):
        if type(self)._active is self:
            self._finish(context, False)


class SG_OT_InteractiveDiscardApply(bpy.types.Operator):
    bl_idname = 'stablegen.interactive_discard_apply'
    bl_label = 'Apply Settings'
    bl_description = 'Keep the previewed settings and restore the original scene materials'

    @classmethod
    def poll(cls, _context):
        return SG_OT_InteractiveDiscardCheck._active is not None

    def execute(self, _context):
        SG_OT_InteractiveDiscardCheck._active._finish_requested = 'APPLY'
        return {'FINISHED'}


class SG_OT_InteractiveDiscardCancel(bpy.types.Operator):
    bl_idname = 'stablegen.interactive_discard_cancel'
    bl_label = 'Cancel'
    bl_description = 'Restore the settings and scene state from before the preview'

    @classmethod
    def poll(cls, _context):
        return SG_OT_InteractiveDiscardCheck._active is not None

    def execute(self, _context):
        SG_OT_InteractiveDiscardCheck._active._finish_requested = 'CANCEL'
        return {'FINISHED'}


# ─── Registration list ───────────────────────────────────────────────────

class SG_OT_DebugProjectionLine(bpy.types.Operator):
    """Draw a line over the projected model and export reproducible view-weight diagnostics."""

    bl_idname = 'stablegen.debug_projection_line'
    bl_label = 'Projection Line Diagnostics'
    bl_description = ('Click two points in the 3D viewport; compare ordinary '
                      'camera weighting, DPPG guards, and coverage rescue along the line')
    bl_options = {'REGISTER'}

    _active = None
    status = ''
    last_report = ''

    @classmethod
    def poll(cls, context):
        return _common_poll(cls, context) and cls._active is None

    def _viewport(self, context, x, y):
        for area in context.window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for region in area.regions:
                if (region.type == 'WINDOW' and region.x <= x < region.x + region.width
                        and region.y <= y < region.y + region.height):
                    return area, region
        return None, None

    def _draw_line(self):
        if len(self._points) == 0:
            return
        import gpu
        from gpu_extras.batch import batch_for_shader
        area = bpy.context.area
        if area is None or area.as_pointer() != self._area_ptr:
            return
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        if region is None:
            return
        points = self._points + [self._mouse]
        coords = [(p[0] - region.x, p[1] - region.y) for p in points]
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINE_STRIP', {'pos': coords})
        gpu.state.line_width_set(4)
        shader.bind()
        shader.uniform_float('color', (1, 1, 1, 1))
        batch.draw(shader)
        gpu.state.line_width_set(1)

    def _cleanup(self, context):
        if getattr(self, '_draw', None) is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw, 'WINDOW')
            self._draw = None
        if getattr(self, '_timer', None) is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        for area in context.window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        if self._active is self:
            type(self)._active = None

    def invoke(self, context, _event):
        if any((slot.material and slot.material.use_nodes and
                (slot.material.get('sg_surface_blend') or any(
                    node.label.startswith('SurfaceBlendWeight-')
                    for node in slot.material.node_tree.nodes)))
               for obj in context.scene.objects if obj.type == 'MESH' and not obj.hide_render
               for slot in obj.material_slots):
            self.report({'ERROR'}, 'Current material uses baked Surface Blend weights; reproject with Boundary Crossfade or Local Unsharpen for this comparison')
            return {'CANCELLED'}
        if context.scene.generation_method == 'local_edit' or (
                context.scene.model_architecture.startswith('qwen')
                and context.scene.qwen_generation_method == 'local_edit'):
            self.report({'ERROR'}, 'Line diagnostics currently require an ordinary multi-view projection, not a local edit pass')
            return {'CANCELLED'}
        if not any(_projection_line_has_sources(obj) for obj in context.scene.objects
                   if obj.type == 'MESH' and not obj.hide_render):
            self.report({'ERROR'}, 'No StableGen projected material found on visible meshes')
            return {'CANCELLED'}
        self._points = []
        self._mouse = (0, 0)
        self._area_ptr = None
        self._draw = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_line, (), 'WINDOW', 'POST_PIXEL')
        self._timer = None
        self._process = None
        self._log = None
        type(self)._active = self
        type(self).status = 'Click two points in a 3D viewport; Esc cancels'
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _capture(self, context):
        area = next(a for a in context.window.screen.areas
                    if a.as_pointer() == self._area_ptr)
        region = next(r for r in area.regions if r.type == 'WINDOW')
        space = area.spaces.active
        depsgraph = context.evaluated_depsgraph_get()
        count = 192
        p0, p1 = self._points
        samples = []
        for i in range(count):
            t = i / (count - 1)
            x = p0[0] + t * (p1[0] - p0[0]) - region.x
            y = p0[1] + t * (p1[1] - p0[1]) - region.y
            origin = view3d_utils.region_2d_to_origin_3d(region, space.region_3d, (x, y))
            direction = view3d_utils.region_2d_to_vector_3d(region, space.region_3d, (x, y))
            hit, point, normal, face, obj, _matrix = context.scene.ray_cast(
                depsgraph, origin, direction, distance=1e7)
            samples.append({'t': t, 'screen': [x, y],
                            'point': list(point) if hit and obj and _projection_line_has_sources(obj) else None,
                            'normal': list(normal) if hit and obj and _projection_line_has_sources(obj) else None,
                            'face': int(face) if hit else None,
                            'object': obj.name if hit and obj else None})
        if not any(s['point'] for s in samples):
            raise RuntimeError('The line did not cross a visible StableGen projected mesh')
        cameras = _interactive_generation_cameras(context)
        if not cameras:
            raise RuntimeError('No generation cameras found')
        meshes = [o for o in context.scene.objects if o.type == 'MESH'
                  and _projection_line_has_sources(o)]
        material_id = _projection_line_material_id(meshes[0])
        root = _interactive_generation_root(context, meshes, material_id)
        run_root = Path(root) / 'debug' / ('projection_line_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        run_root.mkdir(parents=True, exist_ok=False)
        replay_source = Path(__file__).parent / 'texturing' / 'projection_line_diagnostics.py'
        shutil.copy2(replay_source, run_root / 'replay.py')
        view = space.region_3d
        matrix = view.view_matrix.inverted()
        manifest = {
            'schema': 'stablegen.projection-line.v1',
            'snapshot': 'snapshot.blend',
            'source_blend': bpy.data.filepath,
            'frame': context.scene.frame_current,
            'cameras': [c.name for c in cameras],
            'line_screen_global': [list(p) for p in self._points],
            'view': {'matrix_world': [list(row) for row in matrix],
                     'lens': float(space.lens), 'perspective': view.view_perspective,
                     'view_distance': float(view.view_distance),
                     'viewport_size': [region.width, region.height]},
            'render_width': 1024,
            'render_height': max(256, round(1024 * region.height / max(1, region.width))),
            'samples': samples,
            'settings': {name: _copy_scene_value(getattr(context.scene, name))
                         for name in (
                             'weight_exponent', 'discard_angle',
                             'sg_rework_projection_guard', 'sg_guard_feather_method',
                             'sg_silhouette_margin', 'sg_silhouette_feather',
                             'sg_projection_guard_scale', 'sg_guard_directional',
                             'sg_guard_foreground_margin', 'sg_guard_coverage_rescue')
                         if hasattr(context.scene, name)},
            'comparison': 'current material graph versus the same graph with DPPG guard/feather outputs bypassed',
        }
        (run_root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        # Make linked images resolve from the copy without changing the user's
        # scene. Save As(copy=True) leaves the live document at its original path.
        old_paths = []
        try:
            for image in bpy.data.images:
                if image.source == 'FILE' and image.filepath and image.library is None:
                    old_paths.append((image, image.filepath))
                    image.filepath = bpy.path.abspath(image.filepath, library=image.library)
            bpy.ops.wm.save_as_mainfile(filepath=str(run_root / manifest['snapshot']), copy=True)
        finally:
            for image, filepath in old_paths:
                image.filepath = filepath
        return run_root

    def modal(self, context, event):
        if self._process is not None:
            if event.type == 'ESC' and event.value == 'PRESS':
                self._process.terminate()
                self._process.wait(timeout=5)
                self._log.close()
                self._cleanup(context)
                type(self).status = 'Diagnostic cancelled; files kept in ' + str(self._root)
                return {'CANCELLED'}
            if event.type != 'TIMER' or getattr(event, 'timer', self._timer) != self._timer:
                return {'RUNNING_MODAL'}
            code = self._process.poll()
            if code is None:
                return {'RUNNING_MODAL'}
            self._log.close()
            report = self._root / 'index.html'
            self._cleanup(context)
            if code == 0 and report.is_file():
                type(self).last_report = str(report)
                type(self).status = 'Report ready: ' + str(report)
                self.report({'INFO'}, 'Projection line report ready: ' + str(report))
                return {'FINISHED'}
            type(self).status = 'Diagnostic failed; see ' + str(self._root / 'run.log')
            self.report({'ERROR'}, type(self).status)
            return {'CANCELLED'}
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._cleanup(context)
            type(self).status = ''
            return {'CANCELLED'}
        self._mouse = (event.mouse_x, event.mouse_y)
        area, _region = self._viewport(context, *self._mouse)
        if area and self._area_ptr is None:
            self._area_ptr = area.as_pointer()
        if area:
            area.tag_redraw()
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and area:
            if area.as_pointer() != self._area_ptr:
                return {'RUNNING_MODAL'}
            self._points.append(self._mouse)
            if len(self._points) < 2:
                return {'RUNNING_MODAL'}
            try:
                self._root = self._capture(context)
                self._draw and bpy.types.SpaceView3D.draw_handler_remove(self._draw, 'WINDOW')
                self._draw = None
                self._log = (self._root / 'run.log').open('w', encoding='utf-8')
                self._process = subprocess.Popen(
                    [bpy.app.binary_path, '--background', str(self._root / 'snapshot.blend'),
                     '--python', str(self._root / 'replay.py'), '--',
                     str(self._root / 'manifest.json')],
                    stdout=self._log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                self._timer = context.window_manager.event_timer_add(.5, window=context.window)
                type(self).status = 'Rendering projection contributions in background…'
            except Exception as error:
                self._cleanup(context)
                type(self).status = str(error)
                self.report({'ERROR'}, str(error))
                return {'CANCELLED'}
            return {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}


def _projection_line_has_sources(obj):
    if obj.type != 'MESH':
        return False
    for slot in obj.material_slots:
        if not slot.material or not slot.material.use_nodes:
            continue
        for node in slot.material.node_tree.nodes:
            pieces = node.label.split('-') if node.type == 'TEX_IMAGE' else []
            if (len(pieces) == 2 and all(part.isdigit() for part in pieces)
                    and node.image is not None):
                return True
    return False


def _projection_line_material_id(obj):
    for slot in obj.material_slots:
        if slot.material and slot.material.use_nodes:
            for node in slot.material.node_tree.nodes:
                pieces = node.label.split('-') if node.type == 'TEX_IMAGE' else []
                if (len(pieces) == 2 and all(part.isdigit() for part in pieces)
                        and node.image is not None):
                    return int(pieces[1])
    raise RuntimeError('Projected material ID not found')


class SG_OT_DebugOpenProjectionLine(bpy.types.Operator):
    bl_idname = 'stablegen.debug_open_projection_line'
    bl_label = 'Open Projection Line Report'
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, _context):
        return bool(SG_OT_DebugProjectionLine.last_report
                    and Path(SG_OT_DebugProjectionLine.last_report).is_file())

    def execute(self, _context):
        bpy.ops.wm.url_open(url=Path(SG_OT_DebugProjectionLine.last_report).as_uri())
        return {'FINISHED'}

debug_classes = [
    SG_OT_DebugForceGenerationIdle,
    SG_OT_DebugSolidColors,
    SG_OT_DebugGridPattern,
    SG_OT_DebugVisibilityMaterial,
    SG_OT_DebugCoverageHeatmap,
    SG_OT_DebugPerCameraWeight,
    SG_OT_DebugFeatherPreview,
    SG_OT_DebugUVSeamViz,
    SG_OT_DebugConvertFortuneRig,
    SG_OT_DebugSkinningWeights,
    SG_OT_DebugClearSkinningWeights,
    SG_OT_DebugRestoreMaterial,
    SG_OT_DebugConfidenceDiagnostic,
    SG_OT_DebugDPPGMaterial,
    SG_OT_DebugRestoreDPPGMaterial,
    SG_OT_DebugMakeSolid,
    SG_OT_InteractiveDiscardCheck,
    SG_OT_InteractiveDiscardApply,
    SG_OT_InteractiveDiscardCancel,
    SG_OT_DebugProjectionLine,
    SG_OT_DebugOpenProjectionLine,
]
