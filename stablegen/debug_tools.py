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
import numpy as np
from mathutils import Vector

from .texturing.projection import (
    project_image,
    create_native_raycast_visibility,
    create_native_feather,
    _SG_BUFFER_UV_NAME,
    _copy_uv_to_attribute,
)
from .texturing.rendering import export_visibility
from .cameras.geometry import _get_camera_resolution
from .utils import get_file_path, get_generation_dirs, ensure_dirs_exist, sg_modal_active


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


def _saturated_colors(n):
    """Return *n* maximally-saturated, evenly-spaced RGB tuples."""
    return [colorsys.hsv_to_rgb(i / max(n, 1), 1.0, 1.0) for i in range(n)]


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
        cameras = _get_cameras(context)
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

        # Override scene settings so project_image() performs a full
        # fresh build: UV projections + new material tree.  Without this,
        # sequential mode skips UV projections and enters the
        # "update-existing" branch which produces a blank material.
        orig_overwrite   = context.scene.overwrite_material
        orig_gen_method  = context.scene.generation_method
        orig_gen_mode    = context.scene.generation_mode
        context.scene.overwrite_material  = True
        context.scene.generation_method   = 'separate'
        context.scene.generation_mode     = 'standard'
        try:
            project_image(context, meshes, mat_id)
        finally:
            context.scene.overwrite_material  = orig_overwrite
            context.scene.generation_method   = orig_gen_method
            context.scene.generation_mode     = orig_gen_mode

        # Rename materials for easy identification / cleanup
        for obj in meshes:
            if obj.active_material:
                obj.active_material.name = "SG_Debug_SolidColors"

        self.report({'INFO'}, f"Solid-colour projection applied ({n_cams} cameras)")
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
        cameras = _get_cameras(context)
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

        orig_overwrite   = context.scene.overwrite_material
        orig_gen_method  = context.scene.generation_method
        orig_gen_mode    = context.scene.generation_mode
        context.scene.overwrite_material  = True
        context.scene.generation_method   = 'separate'
        context.scene.generation_mode     = 'standard'
        try:
            project_image(context, meshes, mat_id)
        finally:
            context.scene.overwrite_material  = orig_overwrite
            context.scene.generation_method   = orig_gen_method
            context.scene.generation_mode     = orig_gen_mode

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

            orig_overwrite  = context.scene.overwrite_material
            orig_gen_method = context.scene.generation_method
            orig_gen_mode   = context.scene.generation_mode
            context.scene.overwrite_material = True
            context.scene.generation_method  = 'separate'
            context.scene.generation_mode    = 'standard'
            try:
                project_image(context, meshes, mat_id)
            finally:
                context.scene.overwrite_material = orig_overwrite
                context.scene.generation_method  = orig_gen_method
                context.scene.generation_mode    = orig_gen_mode

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
                    context, idx, mat_id)
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
# 8. Restore Original Material
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
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════
# 9. Live Ownership / Confidence Diagnostics
# ═════════════════════════════════════════════════════════════════════════

class SG_OT_DebugOwnershipDiagnostic(bpy.types.Operator):
    """Visualize values from the actual projection ownership graph.

    The material is copied before its output is replaced, so the diagnostic
    cannot alter the production graph. Undo restores the original slots.
    """
    bl_idname = "stablegen.debug_ownership_diagnostic"
    bl_label = "Ownership Diagnostic"
    bl_options = {'REGISTER', 'UNDO'}

    mode: bpy.props.EnumProperty(
        name="Diagnostic",
        items=[
            ('OWNER', "Owner Camera", "Hue identifies the selected source camera"),
            ('QUALITY', "Owner Quality", "Resolved acquisition quality of the owner"),
            ('ACQUISITION', "Camera Quality", "Raw acquisition quality for the active camera"),
            ('TRUSTED', "Trusted Context", "White where owner quality reaches the trust threshold"),
            ('REPAIR', "Repair Mask", "White where the current view must regenerate context"),
            ('WRITE', "Write Mask", "Authoritative generated-pixel write mask for the active camera"),
            ('GUARD', "Depth/Part Guard", "Rear-surface rejection mask for the active camera"),
            ('FALLBACK', "Fallback Only", "White where no trusted owner exists"),
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

                    owner_quality = self._find_node(nodes, exact="SGOwnerQuality")
                    if self.mode == 'OWNER':
                        source = self._find_node(nodes, exact="SGOwnerCamera")
                    elif self.mode == 'QUALITY':
                        source = owner_quality
                    elif self.mode == 'ACQUISITION':
                        source = self._find_node(
                            nodes, prefix=f"AcquisitionQuality-{camera_index}-"
                        )
                    elif self.mode in {'TRUSTED', 'REPAIR'}:
                        source = owner_quality
                    elif self.mode == 'WRITE':
                        source = self._find_node(
                            nodes, prefix=f"SequentialWriteWeight-{camera_index}-"
                        )
                    elif self.mode == 'GUARD':
                        source = self._find_node(
                            nodes, prefix=f"ProjectionGuardWeight-{camera_index}-"
                        )
                    else:
                        source = self._find_node(nodes, exact="OwnerUncovered")

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
                    if self.mode == 'OWNER':
                        normalize = nodes.new("ShaderNodeMath")
                        normalize.operation = 'DIVIDE'
                        normalize.inputs[1].default_value = max(len(cameras) - 1, 1)
                        links.new(value_socket, normalize.inputs[0])
                        value_socket = normalize.outputs[0]
                        ramp.color_ramp.elements[0].color = (1.0, 0.0, 0.0, 1.0)
                        ramp.color_ramp.elements[1].color = (0.0, 0.2, 1.0, 1.0)
                        middle = ramp.color_ramp.elements.new(0.5)
                        middle.color = (0.0, 1.0, 0.1, 1.0)
                    else:
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


# ═════════════════════════════════════════════════════════════════════════
# 10. Debug Make Solid
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


# ─── Registration list ───────────────────────────────────────────────────

debug_classes = [
    SG_OT_DebugSolidColors,
    SG_OT_DebugGridPattern,
    SG_OT_DebugVisibilityMaterial,
    SG_OT_DebugCoverageHeatmap,
    SG_OT_DebugPerCameraWeight,
    SG_OT_DebugFeatherPreview,
    SG_OT_DebugUVSeamViz,
    SG_OT_DebugRestoreMaterial,
    SG_OT_DebugOwnershipDiagnostic,
    SG_OT_DebugMakeSolid,
]
