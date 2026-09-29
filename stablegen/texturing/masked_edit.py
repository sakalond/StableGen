"""Painted surface edits through existing cameras, with reversible material layers.

All Blender mutations and renders run on the modal/main thread. Only uploads and
ComfyUI execution run in a worker. A frozen UV selection gates every projection;
screen-space masks guide generation but never grant permission to edit elsewhere.
"""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import json
import threading
import time
import traceback
import uuid

import bpy
import numpy as np

from ..core import ADDON_PKG
from ..utils import sg_modal_active, get_file_path
from .rendering import _RenderStateTransaction

MASK_UV = 'SG Edit Mask UV'
COVERAGE = 'SG Masked Edit Coverage'
FROZEN = 'SG Masked Edit Selection'
_paint_viewports = []
_paint_keymaps = []
DEFAULT_QWEN_MASKED_PROMPT = (
    'In Image1, replace the visual content inside the magenta-highlighted region shown in Image2 '
    'so that it becomes: {main_prompt}\n\nKeep the camera, geometry, and all content outside that '
    'highlighted region unchanged. Image2 is a location guide only; do not render magenta.'
)


def unified_paint(scene):
    settings = scene.tool_settings
    # Blender 5.1 moved unified settings into each paint mode.
    return (settings.image_paint.unified_paint_settings
            if hasattr(settings.image_paint, 'unified_paint_settings')
            else settings.unified_paint_settings)


def qwen_edit_instruction(instruction, overlay, template=None):
    instruction = instruction.strip()
    if template is not None:
        try:
            return template.format(main_prompt=instruction)
        except (KeyError, ValueError) as exc:
            raise ValueError('Invalid Masked Edit prompt template; use {main_prompt} for the Edit text') from exc
    if overlay:
        return DEFAULT_QWEN_MASKED_PROMPT.format(main_prompt=instruction)
    return instruction


@contextmanager
def scene_settings(scene, **values):
    saved = {name: getattr(scene, name) for name in values}
    try:
        for name, value in values.items():
            setattr(scene, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(scene, name, value)


def output_node(material):
    return next((n for n in material.node_tree.nodes
                 if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)


def color_socket(material):
    """Support StableGen colour graphs, baked Principled materials and emission."""
    nodes, links = material.node_tree.nodes, material.node_tree.links
    out = output_node(material)
    if not out or not out.inputs['Surface'].links:
        raise ValueError('The material has no connected surface output')
    socket = out.inputs['Surface'].links[0].from_socket
    source = socket.node
    if source.type in {'BSDF_PRINCIPLED', 'EMISSION'}:
        color = source.inputs['Base Color' if source.type == 'BSDF_PRINCIPLED' else 'Color']
        if color.links:
            return color.links[0].from_socket
        rgb = nodes.new('ShaderNodeRGB')
        rgb.outputs[0].default_value = color.default_value
        return rgb.outputs[0]
    if socket.type not in {'RGBA', 'VALUE'}:
        raise ValueError('Masked Edit needs a StableGen, Principled, or emission material')
    return socket


def mask_socket(material, image, label=FROZEN):
    nodes, links = material.node_tree.nodes, material.node_tree.links
    uv = nodes.new('ShaderNodeUVMap')
    uv.uv_map = MASK_UV
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = image
    tex.label = label
    tex.extension = 'CLIP'
    links.new(uv.outputs['UV'], tex.inputs['Vector'])
    return tex.outputs['Color']


def math_node(material, operation, a, b):
    node = material.node_tree.nodes.new('ShaderNodeMath')
    node.operation = operation
    for index, value in enumerate((a, b)):
        if isinstance(value, (float, int)):
            node.inputs[index].default_value = value
        else:
            material.node_tree.links.new(value, node.inputs[index])
    return node.outputs[0]


def selection_socket(material, image, remaining=False):
    tex = next((n for n in material.node_tree.nodes if n.label == FROZEN), None)
    selected = tex.outputs['Color'] if tex else mask_socket(material, image)
    coverage = material.node_tree.nodes.get(COVERAGE)
    if remaining and coverage:
        # A small overlap remains only at partial-coverage boundaries.
        uncovered = math_node(material, 'SUBTRACT', 1.0, coverage.outputs[0])
        return math_node(material, 'MULTIPLY', selected, uncovered)
    return selected


def eligible_objects(context):
    objects = [o for o in context.view_layer.objects if o.type == 'MESH'
               and not o.hide_render and not o.hide_get() and not o.get('_sg_mask_paint_proxy')]
    if context.scene.texture_objects == 'selected':
        objects = [o for o in objects if o.select_get()]
    return objects


def validate_object(obj):
    if not obj.active_material or not obj.active_material.use_nodes:
        raise ValueError(f'{obj.name}: assign a textured material first')
    used = {p.material_index for p in obj.data.polygons}
    if len(used) > 1 or (used and next(iter(used)) != obj.active_material_index):
        raise ValueError(f'{obj.name}: Masked Edit currently requires one material across the mesh')
    test = obj.active_material.copy()
    try:
        color_socket(test)
    finally:
        bpy.data.materials.remove(test)


def ensure_mask(context, obj):
    validate_object(obj)
    size = context.scene.sg_masked_resolution
    if obj.sg_edit_mask and MASK_UV in obj.data.uv_layers:
        image = obj.sg_edit_mask
        if tuple(image.size) != (size, size):
            image.scale(size, size)
            image.update()
            image.pack()
        return obj.sg_edit_mask
    if len(obj.data.uv_layers) >= 8:
        raise ValueError(f'{obj.name}: no free UV slot for the edit mask')
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    selected = list(context.selected_objects)
    active = context.view_layer.objects.active
    old_uv = obj.data.uv_layers.active
    old_uv_name = old_uv.name if old_uv else None
    try:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj
        uv = obj.data.uv_layers.get(MASK_UV) or obj.data.uv_layers.new(name=MASK_UV)
        obj.data.uv_layers.active = uv
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(island_margin=0.02)
        bpy.ops.object.mode_set(mode='OBJECT')
        image = bpy.data.images.new(f'{obj.name} Edit Mask', width=size, height=size)
        image.generated_color = (0, 0, 0, 1)
        image.colorspace_settings.name = 'Non-Color'
        image.pack()
        obj.sg_edit_mask = image
        return image
    finally:
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        if old_uv_name and old_uv_name in obj.data.uv_layers:
            obj.data.uv_layers.active = obj.data.uv_layers[old_uv_name]
        bpy.ops.object.select_all(action='DESELECT')
        for item in selected:
            item.select_set(True)
        context.view_layer.objects.active = active


def is_painting(scene):
    return any(o.get('_sg_mask_paint_proxy') or o.sg_mask_paint_original for o in scene.objects)


def paint_session_valid(context):
    """Return whether the temporary multi-mesh paint proxy is still intact."""
    proxies = [o for o in context.scene.objects if o.get('_sg_mask_paint_proxy')]
    if context.mode != 'PAINT_TEXTURE' or len(proxies) != 1 or context.object != proxies[0]:
        return False
    proxy = proxies[0]
    if not proxy.select_get() or not proxy.data.materials:
        return False
    for material in proxy.data.materials:
        if not material or not material.node_tree:
            return False
        node = material.node_tree.nodes.active
        if not node or node.type != 'TEX_IMAGE' or not node.image:
            return False
    return True


def is_masked_edit_mode(scene):
    if scene.model_architecture.startswith('qwen'):
        return scene.qwen_generation_method == 'masked_edit'
    return scene.generation_method == 'masked_edit'


def make_paint_proxy(context, objects):
    """One disposable paint mesh, with a separate mask canvas per source material.

    Blender's texture paint only hits the active object. Joining evaluated copies
    lets a stroke cross the character's component meshes without joining originals.
    MATERIAL paint mode resolves each face's canvas and explicit mask UV map.
    """
    scene = context.scene
    scene['_sg_mask_paint_selected'] = json.dumps([o.name for o in context.selected_objects])
    if context.view_layer.objects.active:
        scene['_sg_mask_paint_active'] = context.view_layer.objects.active
    for obj in objects:
        validate_object(obj)
    for obj in objects:
        ensure_mask(context, obj)
    depsgraph = context.evaluated_depsgraph_get()
    copies = []
    for obj in objects:
        mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph), preserve_all_data_layers=True, depsgraph=depsgraph)
        proxy = bpy.data.objects.new('SG Paint Selection', mesh)
        proxy['_sg_mask_paint_proxy'] = True
        scene.collection.objects.link(proxy)
        proxy.matrix_world = obj.matrix_world.copy()
        copies.append(proxy)
        material = obj.active_material.copy()
        material.name = obj.name + ' Selection Preview'
        mesh.materials.clear()
        mesh.materials.append(material)
        for polygon in mesh.polygons:
            polygon.material_index = 0
        source = color_socket(material)
        selected = mask_socket(material, obj.sg_edit_mask)
        nodes, links = material.node_tree.nodes, material.node_tree.links
        mix = nodes.new('ShaderNodeMixRGB')
        links.new(math_node(material, 'MULTIPLY', selected, 0.55), mix.inputs[0])
        links.new(source, mix.inputs[1])
        mix.inputs[2].default_value = (1, 0, 0.5, 1)
        emit = nodes.new('ShaderNodeEmission')
        links.new(mix.outputs[0], emit.inputs[0])
        links.new(emit.outputs[0], output_node(material).inputs['Surface'])
        for node in nodes:
            node.select = node == selected.node
        nodes.active = selected.node
        mesh.uv_layers.active = mesh.uv_layers[MASK_UV]
        # Keep implicit texture coordinates in the original shader on its old UV;
        # the paint slot explicitly points at MASK_UV below.
        mesh.use_paint_mask = False
        mesh.use_paint_mask_vertex = False
        obj['_sg_mask_paint_hidden'] = obj.hide_get()
        obj['_sg_mask_paint_render_hidden'] = obj.hide_render
        obj.hide_set(True)
        obj.hide_render = True
    bpy.ops.object.select_all(action='DESELECT')
    for proxy in copies:
        proxy.select_set(True)
    context.view_layer.objects.active = copies[0]
    if len(copies) > 1:
        temporary_meshes = [copy.data for copy in copies]
        bpy.ops.object.join()
        for mesh in temporary_meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    proxy = context.object
    proxy.name = 'SG Paint Selection (temporary)'
    proxy['_sg_mask_paint_count'] = len(objects)
    proxy.data.uv_layers.active = proxy.data.uv_layers[MASK_UV]
    return proxy


def finish_paint(context):
    proxies = [o for o in context.scene.objects if o.get('_sg_mask_paint_proxy')]
    if proxies and context.object and context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for proxy in proxies:
        mesh = proxy.data
        materials = list(mesh.materials)
        bpy.data.objects.remove(proxy, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        for material in materials:
            if material and material.users == 0:
                bpy.data.materials.remove(material)
    for obj in context.scene.objects:
        if '_sg_mask_paint_hidden' in obj:
            obj.hide_set(obj['_sg_mask_paint_hidden'])
            obj.hide_render = obj['_sg_mask_paint_render_hidden']
            del obj['_sg_mask_paint_hidden']
            del obj['_sg_mask_paint_render_hidden']
            if obj.sg_edit_mask:
                obj.sg_edit_mask.pack()
    if '_sg_mask_paint_selected' in context.scene:
        selected = json.loads(context.scene['_sg_mask_paint_selected'])
        for obj in context.view_layer.objects:
            obj.select_set(obj.name in selected)
        context.view_layer.objects.active = context.scene.get('_sg_mask_paint_active')
        del context.scene['_sg_mask_paint_selected']
        if '_sg_mask_paint_active' in context.scene:
            del context.scene['_sg_mask_paint_active']
    # Recover previews left by versions that painted only the active object.
    for obj in context.scene.objects:
        if not obj.sg_mask_paint_original:
            continue
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        preview = obj.active_material
        obj.active_material = obj.sg_mask_paint_original
        obj.sg_mask_paint_original = None
        if obj.sg_edit_mask:
            obj.sg_edit_mask.pack()
        old_uv = obj.get('_sg_mask_old_uv', '')
        if old_uv in obj.data.uv_layers:
            obj.data.uv_layers.active = obj.data.uv_layers[old_uv]
        if preview and preview.users == 0:
            bpy.data.materials.remove(preview)
    paint = context.scene.tool_settings.image_paint
    if '_sg_mask_paint_mode' in context.scene:
        paint.mode = context.scene['_sg_mask_paint_mode']
        paint.canvas = context.scene.sg_mask_previous_canvas
        context.scene.sg_mask_previous_canvas = None
        del context.scene['_sg_mask_paint_mode']
    unified = unified_paint(context.scene)
    if '_sg_mask_brush_color' in context.scene:
        unified.color = context.scene['_sg_mask_brush_color']
        unified.secondary_color = context.scene['_sg_mask_brush_secondary']
        unified.use_unified_color = context.scene['_sg_mask_brush_unified']
        for key in ('_sg_mask_brush_color', '_sg_mask_brush_secondary', '_sg_mask_brush_unified'):
            del context.scene[key]
    for space, shading, color_type in _paint_viewports:
        try:
            space.shading.type = shading
            space.shading.color_type = color_type
        except ReferenceError:
            pass
    _paint_viewports.clear()


class SG_OT_PaintEditMask(bpy.types.Operator):
    bl_idname = 'stablegen.paint_edit_mask'
    bl_label = 'Paint Selection'
    bl_description = 'Paint the selection mask directly: white selects a surface; black erases. Finish Painting restores the textured view'

    @classmethod
    def poll(cls, context):
        return bool(eligible_objects(context) and not sg_modal_active(context)
                    and not any(o.sg_mask_original for o in context.scene.objects))

    def execute(self, context):
        try:
            finish_paint(context)
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            objects = eligible_objects(context)
            if not objects:
                raise ValueError('No visible meshes in the current texture object scope')
            proxy = make_paint_proxy(context, objects)
            paint = context.scene.tool_settings.image_paint
            context.scene['_sg_mask_paint_mode'] = paint.mode
            context.scene.sg_mask_previous_canvas = paint.canvas
            paint.mode = 'MATERIAL'
            bpy.ops.object.mode_set(mode='TEXTURE_PAINT')
            context.view_layer.update()
            for material in proxy.data.materials:
                mask = material.node_tree.nodes.active.image
                indices = [i for i, image in enumerate(material.texture_paint_images) if image == mask]
                if not indices:
                    raise RuntimeError(f'Blender did not expose the mask paint slot for {material.name}')
                index = indices[0]
                material.paint_active_slot = index
                material.texture_paint_slots[index].uv_layer = MASK_UV
            if context.area and context.area.type == 'VIEW_3D':
                bpy.ops.wm.tool_set_by_id(name='builtin.brush')
            unified = unified_paint(context.scene)
            context.scene['_sg_mask_brush_color'] = tuple(unified.color)
            context.scene['_sg_mask_brush_secondary'] = tuple(unified.secondary_color)
            context.scene['_sg_mask_brush_unified'] = unified.use_unified_color
            unified.use_unified_color = True
            unified.color = (1, 1, 1)
            unified.secondary_color = (0, 0, 0)
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    space = area.spaces.active
                    _paint_viewports.append((space, space.shading.type, space.shading.color_type))
                    # Workbench reads the active paint canvas directly and updates
                    # during a stroke. Cycles can retain a stale mask texture, while
                    # Eevee cannot display the original Cycles raycast/OSL graph.
                    space.shading.type = 'SOLID'
                    space.shading.color_type = 'TEXTURE'
            # Texture Paint uses the global undo stack. Without a boundary the
            # first Ctrl+Z can undo the proxy join/mode setup instead of a stroke.
            bpy.ops.ed.undo_push(message='StableGen Mask Paint Session')
            self.report({'INFO'}, f'Painting selection across {len(objects)} meshes. White selects; black erases')
            return {'FINISHED'}
        except Exception as exc:
            finish_paint(context)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class SG_OT_MaskPaintUndo(bpy.types.Operator):
    bl_idname = 'stablegen.mask_paint_undo'
    bl_label = 'Undo Mask Stroke'
    bl_description = 'Undo a mask-paint stroke without undoing StableGen paint-session setup'

    @classmethod
    def poll(cls, context):
        return context.mode == 'PAINT_TEXTURE' and is_painting(context.scene)

    def execute(self, context):
        if not bpy.ops.ed.undo.poll():
            return {'CANCELLED'}
        bpy.ops.ed.undo()
        current = bpy.context
        if paint_session_valid(current):
            return {'FINISHED'}
        # Crossing the boundary can split the joined proxy, reveal source
        # objects, or revert Texture Paint mode. Roll that step forward again.
        if bpy.ops.ed.redo.poll():
            bpy.ops.ed.redo()
        current = bpy.context
        if paint_session_valid(current):
            self.report({'INFO'}, 'Reached the beginning of this mask-paint session')
            return {'FINISHED'}
        # Last-resort recovery leaves the real meshes usable even if Blender's
        # redo stack was unavailable or an external operator altered the proxy.
        finish_paint(current)
        current.scene.sg_masked_status = 'Mask painting was safely closed after an undo conflict.'
        self.report({'WARNING'}, current.scene.sg_masked_status)
        return {'CANCELLED'}


class SG_OT_FinishEditMask(bpy.types.Operator):
    bl_idname = 'stablegen.finish_edit_mask'
    bl_label = 'Finish Painting'

    def execute(self, context):
        finish_paint(context)
        return {'FINISHED'}


class SG_OT_ClearEditMask(bpy.types.Operator):
    bl_idname = 'stablegen.clear_edit_mask'
    bl_label = 'Clear Mask'
    bl_description = 'Clear the painted selection mask for all current paint targets'
    bl_options = {'REGISTER', 'UNDO'}

    @staticmethod
    def images(context):
        images = []
        seen = set()
        proxies = [o for o in context.scene.objects if o.get('_sg_mask_paint_proxy')]
        if proxies:
            for proxy in proxies:
                for material in proxy.data.materials:
                    if not material or not material.use_nodes:
                        continue
                    node = material.node_tree.nodes.active
                    image = node.image if node and node.type == 'TEX_IMAGE' else None
                    if image and image.as_pointer() not in seen:
                        seen.add(image.as_pointer())
                        images.append(image)
            return images
        for obj in eligible_objects(context):
            image = obj.sg_edit_mask
            if image and image.as_pointer() not in seen:
                seen.add(image.as_pointer())
                images.append(image)
        return images

    @classmethod
    def poll(cls, context):
        return (not sg_modal_active(context) and not any(o.sg_mask_original for o in context.scene.objects)
                and bool(cls.images(context)))

    def execute(self, context):
        images = self.images(context)
        for image in images:
            pixels = np.zeros((image.size[1], image.size[0], 4), dtype=np.float32)
            pixels[:, :, 3] = 1
            image.pixels.foreach_set(pixels.ravel())
            image.update()
            image.pack()
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        self.report({'INFO'}, f'Cleared {len(images)} selection mask{"s" if len(images) != 1 else ""}')
        return {'FINISHED'}


def render_view(context, objects, camera, path, selection=False, remaining=False, quality=False):
    """Render clean colour or a visible surface mask, including scene occluders."""
    all_meshes = [o for o in context.scene.objects if o.type == 'MESH']
    with _RenderStateTransaction(context, all_meshes):
        scene = context.scene
        scene.camera = camera
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 4
        scene.render.use_compositing = False
        scene.render.film_transparent = False
        scene.render.resolution_x = int(camera.get('sg_res_x', scene.render.resolution_x))
        scene.render.resolution_y = int(camera.get('sg_res_y', scene.render.resolution_y))
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = str(path)
        scene.render.dither_intensity = 0
        scene.view_settings.view_transform = 'Raw' if selection else 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0
        scene.view_settings.gamma = 1
        context.view_layer.material_override = None
        world = bpy.data.worlds.new('Masked Edit Render World')
        world.use_nodes = True
        world.node_tree.nodes.get('Background').inputs['Color'].default_value = (0, 0, 0, 1) if selection else (0.5, 0.5, 0.5, 1)
        scene.world = world
        for obj in all_meshes:
            if obj.hide_render:
                continue
            if not selection and obj not in objects:
                continue
            material = obj.active_material.copy() if obj.active_material else bpy.data.materials.new('Mask Occluder')
            material.use_nodes = True
            nodes, links = material.node_tree.nodes, material.node_tree.links
            out = output_node(material) or nodes.new('ShaderNodeOutputMaterial')
            if selection:
                socket = selection_socket(material, obj.sg_edit_mask, remaining) if obj in objects else None
                if socket is not None and quality:
                    geometry = nodes.new('ShaderNodeNewGeometry')
                    dot = nodes.new('ShaderNodeVectorMath')
                    dot.operation = 'DOT_PRODUCT'
                    links.new(geometry.outputs['Normal'], dot.inputs[0])
                    links.new(geometry.outputs['Incoming'], dot.inputs[1])
                    facing = math_node(material, 'MAXIMUM', dot.outputs['Value'], 0.0)
                    socket = math_node(material, 'MULTIPLY', socket, facing)
            else:
                socket = color_socket(material)
            emit = nodes.new('ShaderNodeEmission')
            emit.inputs[0].default_value = (0, 0, 0, 1)
            if socket is not None:
                links.new(socket, emit.inputs[0])
            links.new(emit.outputs[0], out.inputs['Surface'])
            # Data is single-user for targets; other meshes are only black occluders.
            for slot in obj.material_slots:
                slot.material = material
            if not obj.material_slots:
                obj.data.materials.append(material)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.render.render(write_still=True)


def render_depth_view(context, camera, directory):
    """Render the current camera's normalized Cycles depth map for Qwen Image3."""
    import shutil
    from types import SimpleNamespace
    from .generator import ComfyUIGenerate
    from ..utils import get_file_path

    scene = context.scene
    meshes = [obj for obj in scene.objects if obj.type == 'MESH']
    previous_output = scene.get('_sg_masked_output')
    with _RenderStateTransaction(context, meshes):
        scene.camera = camera
        scene.render.resolution_x = int(camera.get('sg_res_x', scene.render.resolution_x))
        scene.render.resolution_y = int(camera.get('sg_res_y', scene.render.resolution_y))
        scene.render.resolution_percentage = 100
        beauty = Path(directory) / 'depth-beauty.png'
        scene.render.filepath = str(beauty)
        scene['_sg_masked_output'] = str(directory)
        try:
            dummy = SimpleNamespace(_record_confidence_artifacts=lambda *args, **kwargs: None)
            ComfyUIGenerate._export_depthmap_impl(dummy, context, camera_id=0)
            rendered = Path(get_file_path(context, 'controlnet', subtype='depth', camera_id=0))
            output = Path(directory) / 'depth-map.png'
            shutil.copyfile(rendered, output)
            beauty.unlink(missing_ok=True)
            return str(output)
        finally:
            if previous_output is None:
                del scene['_sg_masked_output']
            else:
                scene['_sg_masked_output'] = previous_output


def mask_pixels(path):
    from PIL import Image
    with Image.open(path) as image:
        return np.asarray(image.convert('L')).copy()


def prepare_guides(render_path, mask_path, directory, feather, expand=0):
    from PIL import Image
    import cv2
    mask = mask_pixels(mask_path)
    if expand:
        radius = int(expand)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.dilate(mask, kernel)
    # Feather inward from the expanded boundary. Expansion at least as large
    # as feathering gives full projection weight at the hand-painted edge.
    if feather:
        distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        mask = np.minimum(mask, np.clip(distance / feather, 0, 1) * 255).astype(np.uint8)
    write_path = Path(directory) / 'write-mask.png'
    Image.fromarray(mask).save(write_path)
    expanded = cv2.dilate(mask, np.ones((9, 9), dtype=np.uint8))
    generation = Path(directory) / 'generation-mask.png'
    Image.fromarray(expanded).save(generation)
    with Image.open(render_path) as image:
        rgb = np.asarray(image.convert('RGB')).astype(np.float32)
    alpha = mask[:, :, None].astype(np.float32) / 255 * 0.55
    overlay = np.clip(rgb * (1-alpha) + np.array([255, 0, 128]) * alpha, 0, 255).astype(np.uint8)
    overlay_path = Path(directory) / 'selection-guide.png'
    Image.fromarray(overlay).save(overlay_path)
    return str(write_path), str(generation), str(overlay_path)


def begin_layers(objects):
    for obj in objects:
        obj.sg_mask_original = obj.active_material
        obj.sg_mask_original_mesh = obj.data
        obj.data = obj.data.copy()
        material = obj.active_material.copy()
        material.name = obj.active_material.name + ' Masked Edit'
        obj.active_material = material
        for node in material.node_tree.nodes:
            if node.label == FROZEN:
                node.label = FROZEN + ' Previous'
            if node.name == COVERAGE:
                node.name = COVERAGE + ' Previous'
        frozen = obj.sg_edit_mask.copy()
        frozen.name = obj.name + ' Edit Selection Snapshot'
        frozen.pack()
        mask_socket(material, frozen)
        # Give the legacy local projector a colour node, retaining original shader
        # separately so its lighting/PBR inputs can be restored around each layer.
        source = color_socket(material)
        before = output_node(material).inputs['Surface'].links[0].from_node
        material['_sg_mask_shader'] = before.name if before.type in {'BSDF_PRINCIPLED', 'EMISSION'} else ''
        if before.type in {'BSDF_PRINCIPLED', 'EMISSION'}:
            material.node_tree.links.new(source, before.inputs['Base Color' if before.type == 'BSDF_PRINCIPLED' else 'Color'])


def resolve_layers(scene, apply):
    for obj in scene.objects:
        if not obj.sg_mask_original:
            continue
        original_mesh = obj.sg_mask_original_mesh
        edited_mesh = obj.data
        edited_material = obj.active_material
        if not apply:
            obj.data = original_mesh
            obj.active_material = obj.sg_mask_original
        obj.sg_mask_original = None
        obj.sg_mask_original_mesh = None
        if not apply:
            if edited_mesh.users == 0:
                bpy.data.meshes.remove(edited_mesh)
            if edited_material.users == 0:
                bpy.data.materials.remove(edited_material)
    if '_sg_mask_original_engine' in scene:
        if not apply:
            scene.render.engine = scene['_sg_mask_original_engine']
        del scene['_sg_mask_original_engine']


def project_view(context, objects, camera, image_path, write_path, material_id, expand=0):
    """Reuse raycast projection, then gate its weight with the frozen surface mask."""
    import shutil
    from .projection import project_image
    from ..utils import get_dir_path
    generated = get_file_path(context, 'generated', camera_id=0, material_id=material_id)
    Path(generated).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image_path, generated)
    edge = Path(get_dir_path(context, 'inpaint')['visibility']) / 'render0_edgefeather.png'
    edge.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(write_path, edge)
    previous_coverage = {obj: obj.active_material.node_tree.nodes.get(COVERAGE) for obj in objects}
    for obj in objects:
        material = obj.active_material
        material.node_tree.links.new(color_socket(material), output_node(material).inputs['Surface'])
    with scene_settings(context.scene, generation_method='local_edit', qwen_generation_method='local_edit',
                        overwrite_material=True, refine_edge_feather_projection=True,
                        refine_angle_ramp_active=False, visibility_vignette=False,
                        bake_visibility_weights=False, use_camera_prompts=False):
        project_image(context, objects, material_id, cameras=[camera])
    for obj in objects:
        material = obj.active_material
        nodes, links = material.node_tree.nodes, material.node_tree.links
        mix = output_node(material).inputs['Surface'].links[0].from_node
        invert = mix.inputs[0].links[0].from_node
        weight = invert.inputs[1].links[0].from_socket
        selected = selection_socket(material, obj.sg_edit_mask)
        # With expansion enabled, the camera-space write mask defines the
        # intentional outward region. At zero, retain the strict painted UV gate.
        accepted = weight if expand else math_node(material, 'MULTIPLY', weight, selected)
        links.new(accepted, invert.inputs[1])
        # Track coverage separately from brush softness so repeated views cannot
        # progressively strengthen a softly painted selection.
        old = previous_coverage[obj]
        if old:
            old.name = COVERAGE + ' Previous'
        coverage = math_node(material, 'MAXIMUM', old.outputs[0] if old else 0.0, weight)
        coverage.node.name = COVERAGE
        original_shader = nodes.get(material.get('_sg_mask_shader', ''))
        if original_shader:
            links.new(mix.outputs[0], original_shader.inputs['Base Color' if original_shader.type == 'BSDF_PRINCIPLED' else 'Color'])
            links.new(original_shader.outputs[0], output_node(material).inputs['Surface'])
        # Embed input assets so saving the blend preserves the edit preview.
        for node in nodes:
            if node.type == 'TEX_IMAGE' and node.image and not node.image.packed_file:
                node.image.pack()


class SG_OT_MaskedEdit(bpy.types.Operator):
    bl_idname = 'stablegen.masked_edit'
    bl_label = 'Generate Masked Edit'
    bl_description = 'Edit the painted surface sequentially through existing cameras, then preview and Apply or Discard'
    _active = None

    @classmethod
    def poll(cls, context):
        return (cls._active is None and not sg_modal_active(context)
                and context.scene.model_architecture in {'sdxl', 'qwen_image_edit', 'qwen_image_21'}
                and not any(o.sg_mask_original for o in context.scene.objects))

    def execute(self, context):
        self.timer = None
        self.objects = []
        try:
            finish_paint(context)
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            prefs = context.preferences.addons[ADDON_PKG].preferences
            if not prefs.server_address or not bpy.app.online_access:
                raise ValueError('Connect ComfyUI and enable Blender online access first')
            if not prefs.output_dir or not Path(prefs.output_dir).is_dir():
                raise ValueError('Set a valid output directory in StableGen preferences')
            if not context.scene.sg_masked_prompt.strip():
                raise ValueError('Describe the requested edit first')
            self.objects = [o for o in eligible_objects(context) if o.sg_edit_mask and MASK_UV in o.data.uv_layers]
            if not self.objects:
                raise ValueError('Paint a selection on a target mesh first')
            for obj in self.objects:
                validate_object(obj)
            from .view_plan import order_cameras
            self.cameras = list(order_cameras(o for o in context.scene.objects
                                             if o.type == 'CAMERA' and not o.hide_render))
            if not self.cameras:
                raise ValueError('No existing cameras are available')
            self.scene = context.scene
            self.window = context.window
            self.saved_camera = self.scene.camera
            self.saved_active = context.view_layer.objects.active
            self.saved_selected = list(context.selected_objects)
            self.saved_timestamp = self.scene.output_timestamp
            self.saved_engine = self.scene.render.engine
            self.scene['_sg_mask_original_engine'] = self.saved_engine
            self.scene.output_timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '-masked-' + uuid.uuid4().hex[:6]
            from ..utils import get_generation_dirs
            self.directory = Path(get_generation_dirs(context)['revision'])
            self.directory.mkdir(parents=True, exist_ok=True)
            self.root = str(self.directory)
            self.report_rows = []
            self.ranked = []
            self.index = 0
            self.phase = 'scan'
            self.stop_requested = False
            self.worker = None
            self._error = ''
            self._stage = 'Checking Visible Selections'
            self._progress = 0
            self._current_image = 0
            self._total_images = len(self.cameras)
            self._overall_progress = 0
            self._overall_label = f'Camera scan: 0/{len(self.cameras)}'
            from ..utils import get_last_material_index
            self._material_id = max(get_last_material_index(o) for o in self.objects) + 1
            self._view_plan = ()
            self._cameras = tuple(self.cameras)
            self.next_tick = 0.0
            begin_layers(self.objects)
            type(self)._active = self
            self.scene.generation_status = 'running'
            self.scene.generation_progress = 0.0
            self.scene.sg_masked_status = 'Checking visible selections'
            self.timer = context.window_manager.event_timer_add(0.25, window=context.window)
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        except Exception as exc:
            if hasattr(self, 'scene'):
                resolve_layers(self.scene, False)
                self.cleanup(context)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

    def write_report(self):
        (self.directory / 'masked-edit.json').write_text(json.dumps({
            'prompt': self.scene.sg_masked_prompt, 'architecture': self.scene.model_architecture,
            'expand_selection_px': int(self.scene.sg_masked_expand),
            'edge_softness_px': int(self.scene.sg_masked_feather),
            'use_depth': bool(self.scene.sg_masked_use_depth),
            'custom_prompt': bool(self.scene.sg_masked_use_custom_prompt),
            'prompt_template': (self.scene.sg_masked_custom_prompt
                                if self.scene.sg_masked_use_custom_prompt else None),
            'status': self.scene.sg_masked_status, 'views': self.report_rows,
        }, indent=2), encoding='utf-8')

    def cleanup(self, context):
        if self.timer:
            context.window_manager.event_timer_remove(self.timer)
            self.timer = None
        self.scene.camera = self.saved_camera
        self.scene.output_timestamp = self.saved_timestamp
        if not any(o.sg_mask_original for o in self.scene.objects):
            self.scene.render.engine = self.saved_engine
        for obj in context.view_layer.objects:
            obj.select_set(obj in self.saved_selected)
        context.view_layer.objects.active = self.saved_active
        self.scene.generation_status = 'idle'
        self.scene.generation_progress = 0.0
        type(self)._active = None

    def sync_progress(self):
        """Expose masked-edit progress through StableGen's normal UI fields."""
        if self.phase == 'scan':
            total = max(len(self.cameras), 1)
            completed = min(self.index, total)
            self._progress = completed / total * 100.0
            self._overall_progress = self._progress * 0.1
            self._overall_label = f'Camera scan: {completed}/{total}'
        else:
            total = len(self.ranked)
            if total:
                current = self._progress / 100.0 if self.phase == 'generate' else 0.0
                completed = min(self.index, total)
                fraction = min((completed + current) / total, 1.0)
                self._overall_progress = 10.0 + fraction * 90.0
                self._overall_label = f'Overall: View {min(completed + 1, total)}/{total}'
            else:
                self._overall_progress = 100.0 if self.phase == 'prepare' else 10.0
                self._overall_label = 'No visible camera views'
        self.scene.generation_progress = self._overall_progress

    def _get_uploaded_image_info(self, context, file_type, **kwargs):
        return self.uploads.get('render') if file_type == 'inpaint' else None

    def generate(self, context, render, generation, overlay, depth, result):
        from .._generator_utils import upload_image_to_comfyui
        from ..workflows import WorkflowManager
        try:
            server = context.preferences.addons[ADDON_PKG].preferences.server_address
            self.uploads = {}
            paths = {'render': render, 'mask': generation}
            if context.scene.model_architecture.startswith('qwen') and context.scene.sg_masked_overlay:
                paths['overlay'] = overlay
            if depth:
                paths['depth'] = depth
            for key, path in paths.items():
                uploaded = upload_image_to_comfyui(server, str(path))
                if not uploaded:
                    raise RuntimeError(f'Could not upload {key}')
                subfolder = uploaded.get('subfolder', '')
                if subfolder:
                    uploaded['name'] = subfolder.rstrip('/') + '/' + uploaded['name']
                self.uploads[key] = uploaded
            manager = WorkflowManager(self)
            if context.scene.model_architecture.startswith('qwen'):
                data = manager.generate_qwen_refine(context, camera_id=0, masked_edit={
                    'prompt': context.scene.sg_masked_prompt,
                    'overlay': self.uploads.get('overlay'),
                    'depth': self.uploads.get('depth'),
                    'prompt_template': (context.scene.sg_masked_custom_prompt
                                        if context.scene.sg_masked_use_custom_prompt else None),
                })
            else:
                data = manager.generate_masked_sdxl(context, self.uploads['render'], self.uploads['mask'], context.scene.sg_masked_prompt)
            if not isinstance(data, bytes):
                raise RuntimeError(self._error or 'ComfyUI returned no image')
            Path(result).write_bytes(data)
        except Exception as exc:
            self._error = str(exc)
            traceback.print_exc()

    def tick(self, context):
        if self.stop_requested and self.phase != 'generate':
            self.scene.sg_masked_status = 'Stopped. Apply or discard the partial edit.'
            if not any(row['status'] == 'edited' for row in self.report_rows):
                resolve_layers(self.scene, False)
                self.scene.sg_masked_status = 'Stopped before any edits were applied.'
            return True
        if self.phase == 'scan':
            camera = self.cameras[self.index]
            path = self.directory / f'scan-{self.index:03d}.png'
            self.scene.sg_masked_status = f'Checking {self.index+1}/{len(self.cameras)}: {camera.name}'
            render_view(context, self.objects, camera, path, selection=True, quality=True)
            scan = mask_pixels(path)
            pixels = int(np.count_nonzero(scan > 8))
            if pixels >= 16:
                self.ranked.append((float(scan.sum()) / 255, self.index, camera))
            else:
                self.report_rows.append({'camera': camera.name, 'status': 'skipped', 'reason': 'No visible selection'})
            self.index += 1
            if self.index == len(self.cameras):
                self.ranked.sort(key=lambda row: (-row[0], row[1]))
                self.index = 0
                self.phase = 'prepare'
                self._total_images = len(self.ranked)
                self._current_image = 0
                self._progress = 0
            return False
        if self.phase == 'prepare':
            if self.index == len(self.ranked):
                count = sum(row['status'] == 'edited' for row in self.report_rows)
                self.scene.sg_masked_status = (f'{count} views edited. Preview, then Apply or Discard.' if count
                                               else 'No visible selection in the existing cameras. Adjust the selection.')
                if not count:
                    resolve_layers(self.scene, False)
                self._stage = 'Masked Edit Complete'
                self._progress = 100
                self._overall_progress = 100
                return True
            camera = self.ranked[self.index][2]
            self._current_image = self.index
            self._stage = 'Preparing Camera'
            self._progress = 0
            self.current_camera = camera
            self.current_dir = self.directory / f'view-{self.index:03d}'
            self.current_dir.mkdir(exist_ok=True)
            self.scene.sg_masked_status = f'Preparing {self.index+1}/{len(self.ranked)}: {camera.name}'
            mask = self.current_dir / 'remaining-mask.png'
            render_view(context, self.objects, camera, mask, selection=True, remaining=True)
            if np.count_nonzero(mask_pixels(mask) > 8) < 16:
                self.report_rows.append({'camera': camera.name, 'status': 'skipped', 'reason': 'Selection already covered'})
                self.index += 1
                return False
            render = self.current_dir / 'context.png'
            render_view(context, self.objects, camera, render)
            self.write_mask, generation, overlay = prepare_guides(
                render, mask, self.current_dir,
                self.scene.sg_masked_feather, self.scene.sg_masked_expand)
            depth = None
            if self.scene.model_architecture.startswith('qwen') and self.scene.sg_masked_use_depth:
                self.scene.sg_masked_status = f'Preparing depth {self.index+1}/{len(self.ranked)}: {camera.name}'
                depth = render_depth_view(context, camera, self.current_dir)
            self.result = self.current_dir / 'generated.png'
            self.worker = threading.Thread(
                target=self.generate,
                args=(context, render, generation, overlay, depth, self.result),
                daemon=True,
            )
            self._stage = 'Uploading to Server'
            self._progress = 0
            self.worker.start()
            self.phase = 'generate'
            return False
        if self.phase == 'generate':
            self.scene.sg_masked_status = f'{self.index+1}/{len(self.ranked)}: {self._stage} {self._progress:.0f}%'
            if self.worker.is_alive():
                return False
            if self._error:
                raise RuntimeError(self._error)
            from PIL import Image
            with Image.open(self.result) as generated, Image.open(self.current_dir / 'context.png') as original:
                if generated.size != original.size:
                    raise RuntimeError('Generated image dimensions differ from the camera render; refusing misaligned projection')
                # Preserve original RGB outside the mask even in saved camera outputs.
                mask = Image.open(self.write_mask).convert('L')
                Image.composite(generated.convert('RGB'), original.convert('RGB'), mask).save(self.current_dir / 'composited.png')
            # Use raw generated RGB: the surface shader applies the feather once.
            self._stage = 'Projecting Result'
            self._progress = 100
            self.scene['_sg_masked_output'] = str(self.current_dir)
            try:
                project_view(
                    context, self.objects, self.current_camera, self.result,
                    self.write_mask, self._material_id + self.index,
                    self.scene.sg_masked_expand)
            finally:
                del self.scene['_sg_masked_output']
            self.report_rows.append({'camera': self.current_camera.name, 'status': 'edited', 'directory': str(self.current_dir)})
            self.write_report()
            self.index += 1
            self.phase = 'prepare'
            return False

    def modal(self, context, event):
        if event.type == 'ESC':
            self.stop_requested = True
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        now = time.monotonic()
        if now < self.next_tick:
            return {'PASS_THROUGH'}
        self.next_tick = now + 0.2
        if context.scene != self.scene:
            self.window.scene = self.scene
        try:
            done = self.tick(context)
            self.sync_progress()
        except Exception as exc:
            traceback.print_exc()
            self.scene.sg_masked_status = f'Failed: {exc}. Apply or discard any completed views.'
            self.report_rows.append({'status': 'failed', 'error': str(exc)})
            self.report({'ERROR'}, str(exc))
            done = True
        for area in self.window.screen.areas:
            area.tag_redraw()
        if done:
            self.write_report()
            self.cleanup(context)
            return {'FINISHED'}
        return {'PASS_THROUGH'}


class SG_OT_ResolveMaskedEdit(bpy.types.Operator):
    bl_idname = 'stablegen.resolve_masked_edit'
    bl_label = 'Resolve Masked Edit'
    bl_options = {'REGISTER', 'UNDO'}
    apply: bpy.props.BoolProperty(default=True)

    @classmethod
    def poll(cls, context):
        return SG_OT_MaskedEdit._active is None and any(o.sg_mask_original for o in context.scene.objects)

    def execute(self, context):
        resolve_layers(context.scene, self.apply)
        context.scene.sg_masked_status = 'Edit applied' if self.apply else 'Edit discarded; original materials restored'
        return {'FINISHED'}


class SG_OT_StopMaskedEdit(bpy.types.Operator):
    bl_idname = 'stablegen.stop_masked_edit'
    bl_label = 'Stop After Current View'

    def execute(self, context):
        if SG_OT_MaskedEdit._active:
            SG_OT_MaskedEdit._active.stop_requested = True
        return {'FINISHED'}


class SG_OT_ResetMaskedPrompt(bpy.types.Operator):
    bl_idname = 'stablegen.reset_masked_prompt'
    bl_label = 'Reset Masked Edit Prompt'
    bl_description = 'Restore the tested default Masked Edit guidance prompt'

    def execute(self, context):
        context.scene.sg_masked_custom_prompt = DEFAULT_QWEN_MASKED_PROMPT
        return {'FINISHED'}


def draw_masked_edit_controls(layout, context):
    """Draw Masked Edit inside the main panel's Generation Mode Specifics."""
    scene = context.scene
    if SG_OT_MaskedEdit._active:
        layout.label(text=scene.sg_masked_status)
        layout.operator('stablegen.stop_masked_edit')
        return
    if any(o.sg_mask_original for o in scene.objects):
        layout.label(text=scene.sg_masked_status)
        row = layout.row(align=True)
        row.operator('stablegen.resolve_masked_edit', text='Apply', icon='CHECKMARK').apply = True
        row.operator('stablegen.resolve_masked_edit', text='Discard', icon='LOOP_BACK').apply = False
        return
    layout.label(text='Paint a surface selection, then describe the change.')
    if is_painting(scene):
        row = layout.row(align=True)
        row.operator('stablegen.finish_edit_mask', icon='CHECKMARK')
        row.operator('stablegen.clear_edit_mask', icon='X')
        count = next((o.get('_sg_mask_paint_count', 1) for o in scene.objects if o.get('_sg_mask_paint_proxy')), 1)
        layout.label(text=f'Painting across {count} meshes')
        layout.label(text='White selects; black erases (X swaps colours).')
        layout.label(text='Mask view: Finish Painting restores textures.')
        paint = unified_paint(scene)
        brush = scene.tool_settings.image_paint.brush
        if brush:
            layout.prop(paint if paint.use_unified_size else brush, 'size', text='Brush Size')
            layout.prop(paint if paint.use_unified_strength else brush, 'strength', text='Brush Strength')
        row = layout.row(align=True)
        row.prop(paint, 'color', text='Paint')
        row.prop(paint, 'secondary_color', text='Erase')
    else:
        layout.prop(scene, 'texture_objects', text='Paint Targets')
        row = layout.row(align=True)
        row.operator('stablegen.paint_edit_mask', icon='BRUSH_DATA')
        row.operator('stablegen.clear_edit_mask', icon='X')
        layout.prop(scene, 'sg_masked_resolution')
        masks = [o.sg_edit_mask for o in eligible_objects(context) if o.sg_edit_mask]
        if masks:
            sizes = sorted({f'{image.size[0]}×{image.size[1]}' for image in masks})
            layout.label(text='Current mask: ' + ', '.join(sizes), icon='IMAGE_DATA')
            if any(tuple(image.size) != (scene.sg_masked_resolution,) * 2 for image in masks):
                layout.label(text='Masks resize when Paint Selection starts.', icon='INFO')
    layout.prop(scene, 'sg_masked_prompt', text='Edit')
    layout.prop(scene, 'sg_masked_expand')
    layout.prop(scene, 'sg_masked_feather')
    if scene.model_architecture.startswith('qwen'):
        layout.prop(scene, 'sg_masked_overlay')
        layout.prop(scene, 'sg_masked_use_depth')
        layout.prop(scene, 'sg_masked_use_custom_prompt', text='Custom Guidance Prompt', toggle=True, icon='TEXT')
        if scene.sg_masked_use_custom_prompt:
            box = layout.box()
            box.label(text='Use {main_prompt} for the Edit text.')
            row = box.row(align=True)
            row.prop(scene, 'sg_masked_custom_prompt', text='')
            row.operator('stablegen.reset_masked_prompt', text='', icon='FILE_REFRESH')
    elif scene.model_architecture == 'sdxl':
        layout.prop(scene, 'sg_masked_strength')
    else:
        layout.label(text='Masked Edit supports Qwen Image Edit and SDXL.', icon='INFO')
    layout.label(text='Uses the current model and existing cameras.')
    if scene.sg_masked_status:
        layout.label(text=scene.sg_masked_status)


class SG_PT_MaskedEdit(bpy.types.Panel):
    bl_idname = 'SG_PT_masked_edit'
    bl_label = 'Masked Edit'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'StableGen'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        draw_masked_edit_controls(self.layout, context)


MASKED_EDIT_CLASSES = (SG_OT_PaintEditMask, SG_OT_MaskPaintUndo,
                       SG_OT_FinishEditMask, SG_OT_ClearEditMask,
                       SG_OT_MaskedEdit, SG_OT_ResolveMaskedEdit, SG_OT_StopMaskedEdit,
                       SG_OT_ResetMaskedPrompt)


def register_properties():
    bpy.types.Object.sg_edit_mask = bpy.props.PointerProperty(type=bpy.types.Image)
    bpy.types.Object.sg_mask_paint_original = bpy.props.PointerProperty(type=bpy.types.Material)
    bpy.types.Object.sg_mask_original = bpy.props.PointerProperty(type=bpy.types.Material)
    bpy.types.Object.sg_mask_original_mesh = bpy.props.PointerProperty(type=bpy.types.Mesh)
    bpy.types.Scene.sg_mask_previous_canvas = bpy.props.PointerProperty(type=bpy.types.Image)
    bpy.types.Scene.sg_masked_prompt = bpy.props.StringProperty(name='Edit', description='Describe the surface change, without requesting a new viewpoint or crop')
    bpy.types.Scene.sg_masked_overlay = bpy.props.BoolProperty(name='Send Selection Guide to Qwen', default=True, description='Send a second render with a magenta selection overlay; the clean render stays image1')
    bpy.types.Scene.sg_masked_use_depth = bpy.props.BoolProperty(
        name='Use Depth Map', default=False,
        description='Send the current camera depth as Image3 for structural guidance; useful for shape-sensitive edits but may weaken material or additive changes')
    bpy.types.Scene.sg_masked_use_custom_prompt = bpy.props.BoolProperty(
        name='Use Custom Guidance Prompt', default=False,
        description='Override the default Qwen Masked Edit guidance prompt')
    bpy.types.Scene.sg_masked_custom_prompt = bpy.props.StringProperty(
        name='Masked Edit Guidance Prompt', default=DEFAULT_QWEN_MASKED_PROMPT,
        description='Qwen guidance template; use {main_prompt} to insert the text from the Edit field')
    bpy.types.Scene.sg_masked_resolution = bpy.props.IntProperty(
        name='Mask Resolution', default=4096, min=256, max=8192, soft_max=4096,
        description='Resolution of each painted selection mask. Existing masks resize while preserving their contents when Paint Selection starts')
    bpy.types.Scene.sg_masked_expand = bpy.props.IntProperty(
        name='Expand Selection (px)', default=4, min=0, max=64,
        description='Expand the painted selection outward in each camera view before feathering and projection; useful for covering jagged edges or traces of the old texture')
    bpy.types.Scene.sg_masked_feather = bpy.props.IntProperty(
        name='Edge Softness (px)', default=4, min=0, max=32,
        description='Feather inward from the expanded selection boundary')
    bpy.types.Scene.sg_masked_strength = bpy.props.FloatProperty(name='Inpaint Strength', default=0.85, min=0.05, max=1.0)
    bpy.types.Scene.sg_masked_status = bpy.props.StringProperty(default='')
    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig:
        keymap = keyconfig.keymaps.new(name='Image Paint', space_type='EMPTY')
        item = keymap.keymap_items.new('stablegen.mask_paint_undo', 'Z', 'PRESS', ctrl=True)
        _paint_keymaps.append((keymap, item))


def unregister_properties():
    for keymap, item in _paint_keymaps:
        keymap.keymap_items.remove(item)
    _paint_keymaps.clear()
    for owner, names in ((bpy.types.Object, ('sg_edit_mask', 'sg_mask_paint_original', 'sg_mask_original', 'sg_mask_original_mesh')),
                         (bpy.types.Scene, ('sg_mask_previous_canvas', 'sg_masked_prompt', 'sg_masked_overlay', 'sg_masked_use_depth', 'sg_masked_use_custom_prompt', 'sg_masked_custom_prompt', 'sg_masked_resolution', 'sg_masked_expand', 'sg_masked_feather', 'sg_masked_strength', 'sg_masked_status'))):
        for name in names:
            delattr(owner, name)
