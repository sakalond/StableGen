"""GPU draw handlers for camera crop overlays and view labels."""

import math
import bpy  # pylint: disable=import-error
import mathutils  # pylint: disable=import-error
import numpy as np
import blf
import gpu
from gpu_extras.batch import batch_for_shader
from .geometry import _get_camera_resolution

_ADDON_PKG = __package__.rsplit('.', 1)[0]


# ---- Per-camera crop overlay (GPU draw handler) -------------------------

_sg_crop_draw_handle = None


def _sg_camera_overlay_color(camera):
    """Use the solid-projection diagnostic color when assigned to this camera."""
    color = camera.get('_sg_debug_projection_color')
    if color is not None:
        return color
    prefs = bpy.context.preferences.addons.get(_ADDON_PKG)
    return prefs.preferences.overlay_color if prefs else (0.3, 0.5, 1.0)


def _sg_draw_crop_overlays():
    """SpaceView3D draw callback: renders a coloured rectangle inside each
    camera's pyramid to visualise the actual (non-square) crop region."""
    context = bpy.context
    scene = context.scene

    # Respect Blender's overlay toggle
    space = context.space_data
    if space and hasattr(space, 'overlay') and not space.overlay.show_overlays:
        return

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    for obj in scene.objects:
        if obj.type != 'CAMERA' or 'sg_display_crop' not in obj:
            continue
        res_x = obj.get('sg_res_x', 0)
        res_y = obj.get('sg_res_y', 0)
        if res_x <= 0 or res_y <= 0:
            continue

        cam = obj.data
        # Use whichever side of THIS camera is longer as the reference.
        # The scene is set to a square of max_side, so the frustum is square
        # and each side = half_w = half_h.  The crop rectangle occupies
        # res_x/cam_max × res_y/cam_max of that square.
        cam_max = max(res_x, res_y)

        frame = cam.view_frame(scene=scene)
        corner = frame[0]
        half_w = abs(corner[0])
        half_h = abs(corner[1])
        z_depth = corner[2]
        if half_w < 1e-8 or half_h < 1e-8:
            continue

        sx = res_x / cam_max
        sy = res_y / cam_max

        new_hw = half_w * sx
        new_hh = half_h * sy

        crop_local = [
            mathutils.Vector((+new_hw, +new_hh, z_depth)),
            mathutils.Vector((-new_hw, +new_hh, z_depth)),
            mathutils.Vector((-new_hw, -new_hh, z_depth)),
            mathutils.Vector((+new_hw, -new_hh, z_depth)),
        ]

        wm = obj.matrix_world
        world_pts = [wm @ v for v in crop_local]
        coords = [(v.x, v.y, v.z) for v in world_pts]
        indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
        batch = batch_for_shader(shader, 'LINES', {"pos": coords}, indices=indices)
        shader.bind()
        _oc = _sg_camera_overlay_color(obj)
        shader.uniform_float("color", (_oc[0], _oc[1], _oc[2], 0.9))
        gpu.state.line_width_set(2.0)
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(False)
        batch.draw(shader)

    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(True)
    gpu.state.blend_set('NONE')
    gpu.state.line_width_set(1.0)


# ---- Per-camera view-label overlay (GPU text handler) --------------------

_sg_label_draw_handle = None
_sg_labels_user_visible = True


def _sg_draw_view_labels():
    """SpaceView3D POST_PIXEL callback: draws full camera prompt text
    (from ``scene.camera_prompts``) near each camera in the viewport."""
    context = bpy.context
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return

    # Respect Blender's overlay toggle
    space = context.space_data
    if space and hasattr(space, 'overlay') and not space.overlay.show_overlays:
        return

    from bpy_extras.view3d_utils import location_3d_to_region_2d

    font_id = 0
    blf.size(font_id, 13)

    # Build a lookup of camera name -> prompt text
    prompt_lookup = {item.name: item.prompt for item in scene.camera_prompts
                     if item.prompt}

    for obj in scene.objects:
        if obj.type != 'CAMERA':
            continue
        label = prompt_lookup.get(obj.name, '')
        if not label:
            continue
        co_2d = location_3d_to_region_2d(region, rv3d, obj.location)
        if co_2d is None:
            continue
        _oc = _sg_camera_overlay_color(obj)
        blf.color(font_id, _oc[0], _oc[1], _oc[2], 0.95)
        # Offset slightly below the camera marker
        blf.position(font_id, co_2d.x - blf.dimensions(font_id, label)[0] * 0.5,
                     co_2d.y - 20, 0)
        gpu.state.blend_set('ALPHA')
        blf.draw(font_id, label)

    gpu.state.blend_set('NONE')


def _sg_ensure_crop_overlay():
    """Register the crop overlay draw handler if not already active."""
    global _sg_crop_draw_handle
    if _sg_crop_draw_handle is None:
        _sg_crop_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _sg_draw_crop_overlays, (), 'WINDOW', 'POST_VIEW')


def _sg_remove_crop_overlay():
    """Remove the crop overlay draw handler if active."""
    global _sg_crop_draw_handle
    if _sg_crop_draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_sg_crop_draw_handle, 'WINDOW')
        _sg_crop_draw_handle = None


def _sg_ensure_label_overlay():
    """Register the floating prompt-text label overlay if not already active."""
    global _sg_label_draw_handle, _sg_labels_user_visible
    _sg_labels_user_visible = True
    if _sg_label_draw_handle is None:
        _sg_label_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _sg_draw_view_labels, (), 'WINDOW', 'POST_PIXEL')
        # Redraw all viewports so labels appear immediately
        for area in (bpy.context.screen.areas if bpy.context.screen else []):
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _sg_remove_label_overlay():
    """Remove the floating prompt-text label overlay if active."""
    global _sg_label_draw_handle, _sg_labels_user_visible
    _sg_labels_user_visible = False
    if _sg_label_draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_sg_label_draw_handle, 'WINDOW')
        _sg_label_draw_handle = None
        for area in (bpy.context.screen.areas if bpy.context.screen else []):
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _sg_hide_label_overlay():
    """Temporarily hide the label overlay without changing the user's preference."""
    global _sg_label_draw_handle
    if _sg_label_draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_sg_label_draw_handle, 'WINDOW')
        _sg_label_draw_handle = None
        for area in (bpy.context.screen.areas if bpy.context.screen else []):
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _sg_restore_label_overlay():
    """Restore the label overlay only if the user had it enabled."""
    if _sg_labels_user_visible:
        global _sg_label_draw_handle
        if _sg_label_draw_handle is None:
            _sg_label_draw_handle = bpy.types.SpaceView3D.draw_handler_add(
                _sg_draw_view_labels, (), 'WINDOW', 'POST_PIXEL')
            for area in (bpy.context.screen.areas if bpy.context.screen else []):
                if area.type == 'VIEW_3D':
                    area.tag_redraw()


def _sg_restore_square_display(scene):
    """Restore scene resolution to the max-side square and tag viewports
    for redraw so crop overlays reappear correctly."""
    cameras = [o for o in scene.objects if o.type == 'CAMERA' and 'sg_res_x' in o]
    if not cameras:
        return
    max_side = max(
        max(int(c.get('sg_res_x', 0)), int(c.get('sg_res_y', 0)))
        for c in cameras
    )
    if max_side > 0:
        scene.render.resolution_x = max_side
        scene.render.resolution_y = max_side
    # Refresh viewports
    for area in bpy.context.screen.areas if bpy.context.screen else []:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def _setup_square_camera_display(cam_obj, res_x, res_y):
    """Mark a camera for the crop overlay and enable passepartout."""
    cam_data = cam_obj.data
    cam_data.show_passepartout = True
    cam_data.passepartout_alpha = 0.5
    cam_obj["sg_display_crop"] = True
    # Ensure the draw handler is alive
    _sg_ensure_crop_overlay()
