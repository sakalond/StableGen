"""Cached viewport-only camera image planes, crop frames and numbered labels."""
import logging

import bpy
import gpu
import mathutils
from bpy.app.handlers import persistent
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader

from ..texturing.view_plan import generation_cameras, order_cameras
from .geometry import _get_camera_resolution
from . import prompt_overlay

_ADDON_PKG = __package__.rsplit('.', 1)[0]
_sg_crop_draw_handle = None
_sg_label_draw_handle = None
_sg_labels_user_visible = True
_registered = False
_snapshots = {}
_geometry = {}
_textures = {}
_shaders = {}
_log = logging.getLogger(__name__)
_last_error = None


def _sg_camera_overlay_color(camera):
    color = camera.get('_sg_debug_projection_color')
    if color is not None:
        return color
    prefs = bpy.context.preferences.addons.get(_ADDON_PKG)
    return prefs.preferences.overlay_color if prefs else (0.3, 0.5, 1.0)


def _snapshot(context):
    scene = context.scene
    key = (scene.as_pointer(), context.view_layer.name)
    if key not in _snapshots:
        _snapshots[key] = dict(scene=scene, layer=context.view_layer, cameras=(), images={},
                               dirty=True, manifests={})
    return _snapshots[key]


def _source_materials(context):
    state = getattr(context.scene, 'sg_studio_diagnostics', None)
    if state and state.token:
        return {slot.object_material if slot.link == 'OBJECT' else slot.material
                for record in state.meshes for slot in record.slots
                if (slot.object_material if slot.link == 'OBJECT' else slot.material)}
    # Camera selection must not hide the source images of selected-target runs.
    return {slot.material for obj in context.view_layer.objects
            if obj.type == 'MESH' and not obj.hide_get()
            for slot in obj.material_slots if slot.material}


def _generating_camera(context):
    if getattr(context.scene, 'generation_status', '') != 'running':
        return None
    for window in context.window_manager.windows:
        for op in window.modal_operators:
            if getattr(op, 'bl_idname', '') != 'OBJECT_OT_test_stable':
                continue
            cameras = getattr(op, '_cameras', ()) or ()
            if getattr(op, '_pbr_active', False):
                return None  # PBR's subset ordinal is not a camera source index.
            if getattr(context.scene, 'generation_method', '') not in {'separate','sequential','refine','local_edit'}:
                return None
            index = getattr(op, '_current_image', -1)
            if 0 <= index < len(cameras) and cameras[index].name in context.scene.objects:
                return cameras[index]
    return None


def _refresh_snapshot(context):
    """Main-thread timer work only: no filesystem/material scans during drawing."""
    snapshot = _snapshot(context)
    scene = context.scene
    custom = [item.name for item in scene.sg_camera_order] if scene.sg_use_custom_camera_order else ()
    cameras = tuple(order_cameras(generation_cameras(scene), custom))
    # Automatic sequencing and reproject may use a saved order for this run.
    # Read the existing runtime plan; do not recompute coverage or change it.
    if getattr(scene, 'generation_status', '') == 'running':
        for window in context.window_manager.windows:
            op = next((op for op in window.modal_operators
                       if getattr(op, 'bl_idname', '') == 'OBJECT_OT_test_stable'), None)
            plan = getattr(op, '_cameras', None)
            if plan and all(c.name in scene.objects for c in plan):
                cameras = tuple(plan)
                break
    identity = tuple((c.as_pointer(), c.name, c.get('sg_view_uuid')) for c in cameras)
    if identity != snapshot.get('identity'):
        snapshot['dirty'] = True
        snapshot['identity'] = identity
    snapshot['cameras'] = cameras
    snapshot['generating'] = _generating_camera(context)
    if snapshot['dirty'] and scene.sg_camera_display.images != 'NONE':
        from .images import projected_images
        snapshot['images'] = projected_images(context, materials=_source_materials(context),
                                               manifests=snapshot['manifests'])
        snapshot['dirty'] = False
    return snapshot


@persistent
def _invalidate(scene, depsgraph):
    updates = list(depsgraph.updates)
    if any(update.is_updated_shading or isinstance(update.id, (bpy.types.Material, bpy.types.NodeTree, bpy.types.Image,
                                 bpy.types.Collection, bpy.types.Mesh))
           for update in updates):
        for snapshot in _snapshots.values():
            snapshot['dirty'] = True
            snapshot['manifests'].clear()
    # Display settings, selection and material edits do not invalidate textures.
    for update in updates:
        if isinstance(update.id, bpy.types.Image):
            _textures.pop(update.id.original.as_pointer(), None)


def _tick():
    if not _registered:
        return None
    seen = set()
    live_keys = set()
    try:
        prompt_overlay.prune()
        for window in bpy.context.window_manager.windows:
            area = next((a for a in window.screen.areas if a.type == 'VIEW_3D'), None)
            if area is None:
                continue
            key = (window.scene.as_pointer(), window.view_layer.name)
            live_keys.add(key)
            if key in seen:
                continue
            seen.add(key)
            with bpy.context.temp_override(window=window, area=area):
                previous = _snapshots.get(key)
                cameras = previous['cameras'] if previous else ()
                generating = previous.get('generating') if previous else None
                dirty = previous is None or previous['dirty'] and window.scene.sg_camera_display.images != 'NONE'
                current = _refresh_snapshot(bpy.context)
                if dirty or cameras != current['cameras'] or generating != current['generating']:
                    for a in window.screen.areas:
                        if a.type == 'VIEW_3D':
                            a.tag_redraw()
        for key in set(_snapshots)-live_keys:
            del _snapshots[key]
        camera_ids = {c.as_pointer() for snapshot in _snapshots.values() for c in snapshot['cameras']}
        for key in set(_geometry)-camera_ids:
            del _geometry[key]
        image_ids = {image.as_pointer() for snapshot in _snapshots.values() for image in snapshot['images'].values()}
        for key in set(_textures)-image_ids:
            del _textures[key]
    except (ReferenceError, RuntimeError):
        _clear_cache()
    return .6


def _clear_cache():
    _snapshots.clear()
    _geometry.clear()
    _textures.clear()
    _shaders.clear()


@persistent
def _load_pre(_dummy):
    _clear_cache()
    prompt_overlay.clear()


@persistent
def _load_post(_dummy):
    display_settings_changed(bpy.context.scene)


def display_settings_changed(scene):
    global _sg_labels_user_visible
    if bpy.app.background or scene is None:
        return
    settings = getattr(scene, 'sg_camera_display', None)
    if settings:
        _sg_labels_user_visible = settings.labels != 'NONE'
    from ..utils import sg_modal_active
    # Generation deliberately suspends these handlers while preparing renders.
    if sg_modal_active(bpy.context):
        return
    _sg_ensure_crop_overlay()
    if _sg_labels_user_visible:
        _sg_ensure_label_overlay()
    else:
        _sg_hide_label_overlay()
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def register_overlays():
    global _registered
    if bpy.app.background:
        return
    _registered = True
    for handlers, callback in [(bpy.app.handlers.depsgraph_update_post, _invalidate),
                               (bpy.app.handlers.load_pre, _load_pre),
                               (bpy.app.handlers.load_post, _load_post)]:
        if callback not in handlers:
            handlers.append(callback)
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=.1, persistent=True)
    display_settings_changed(bpy.context.scene)


def unregister_overlays():
    global _registered
    _registered = False
    _sg_remove_crop_overlay()
    _sg_hide_label_overlay()
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    for handlers, callback in [(bpy.app.handlers.depsgraph_update_post, _invalidate),
                               (bpy.app.handlers.load_pre, _load_pre),
                               (bpy.app.handlers.load_post, _load_post)]:
        if callback in handlers:
            handlers.remove(callback)
    _clear_cache()


def _draw_context():
    context = bpy.context
    if (context.area is None or context.area.type != 'VIEW_3D' or context.region_data is None
            or not context.space_data.overlay.show_overlays):
        return None
    return context


def _visible(camera, context):
    return camera.visible_get(view_layer=context.view_layer, viewport=context.space_data)


def _in_scope(scope, camera):
    return scope == 'ALL' or scope == 'SELECTED' and camera.select_get()


def _crop_points(camera, scene, depsgraph):
    """Actual render intrinsics, including shift, forced sensor fit and ortho."""
    rx, ry = _get_camera_resolution(camera, scene)
    frame = camera.data.view_frame(scene=scene)
    depth = abs(frame[0].z)
    if camera.data.type != 'ORTHO':
        depth *= camera.data.display_size
    projection = camera.calc_matrix_camera(depsgraph, x=rx, y=ry,
        scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y)
    inverse = projection.inverted_safe()
    points = []
    for x, y in [(-1,-1), (1,-1), (1,1), (-1,1)]:
        v = inverse @ mathutils.Vector((x,y,0,1))
        v = v.xyz / v.w
        if camera.data.type != 'ORTHO':
            v *= depth / abs(v.z)
        else:
            v.z = -depth
        points.append(v)
    return points


def _shader(name):
    if name not in _shaders:
        try:
            _shaders[name] = gpu.shader.from_builtin(name)
        except ValueError:
            if name != 'IMAGE_COLOR_SCENE_LINEAR_TO_REC709_SRGB':
                raise
            _shaders[name] = gpu.shader.from_builtin('IMAGE_COLOR')
    return _shaders[name]


def _camera_geometry(camera, context, image):
    data, render = camera.data, context.scene.render
    signature = (tuple(v for row in camera.matrix_world for v in row),
        data.type, data.lens, data.sensor_fit, data.sensor_width, data.sensor_height,
        data.shift_x, data.shift_y, data.ortho_scale, data.display_size, data.clip_start, data.clip_end,
        _get_camera_resolution(camera, context.scene), render.resolution_x, render.resolution_y,
        render.pixel_aspect_x, render.pixel_aspect_y,
        image.as_pointer() if image else 0, tuple(image.size) if image else ())
    key = camera.as_pointer()
    cached = _geometry.get(key)
    if cached and cached['signature'] == signature:
        return cached
    local = _crop_points(camera, context.scene, context.evaluated_depsgraph_get())
    points = [camera.matrix_world @ v for v in local]
    center = sum(points, mathutils.Vector())/4
    line = _shader('POLYLINE_UNIFORM_COLOR')
    fill = _shader('UNIFORM_COLOR')
    cached = dict(signature=signature, points=points, center=center,
        border=batch_for_shader(line, 'LINES', {'pos':points}, indices=[(0,1),(1,2),(2,3),(3,0)]),
        backing=batch_for_shader(fill, 'TRI_FAN', {'pos':points}))
    if image and all(image.size):
        cx = sum(v.x for v in local)/4
        cy = sum(v.y for v in local)/4
        half_x = abs(local[1].x-local[0].x)/2
        half_y = abs(local[3].y-local[0].y)/2
        aspect = image.size[0]/image.size[1]
        if aspect > half_x/half_y:
            half_y = half_x/aspect
        else:
            half_x = half_y*aspect
        image_points = [camera.matrix_world @ mathutils.Vector((cx+x*half_x, cy+y*half_y, local[0].z))
                        for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        shader = _shader('IMAGE_COLOR_SCENE_LINEAR_TO_REC709_SRGB')
        # Keep pixels anchored to the camera's axes, just like its projection.
        # A reverse-side view naturally reverses the plane on screen; changing
        # UVs to compensate would instead turn the subject away from the model.
        cached['image'] = batch_for_shader(shader, 'TRI_FAN',
            {'pos':image_points, 'texCoord':((0,0),(1,0),(1,1),(0,1))})
    _geometry[key] = cached
    return cached


def _record_error():
    global _last_error
    import sys
    error = repr(sys.exc_info()[1])
    if error != _last_error:
        _last_error = error
        _log.exception('Camera viewport display could not draw')


def _sg_draw_crop_overlays():
    context = _draw_context()
    if context is None:
        return
    settings = context.scene.sg_camera_display
    if not settings.crops and settings.images == 'NONE':
        return
    snapshot = _snapshot(context)
    blend, depth, depth_mask = gpu.state.blend_get(), gpu.state.depth_test_get(), gpu.state.depth_mask_get()
    try:
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(False)
        rv3d = context.region_data
        eye = rv3d.view_matrix.inverted().translation
        cameras = sorted(snapshot['cameras'], key=lambda c:(c.matrix_world.translation-eye).length_squared, reverse=True)
        theme = context.preferences.themes[0].view_3d
        for camera in cameras:
            if not _visible(camera, context) or camera.data.type == 'PANO':
                continue
            image = snapshot['images'].get(camera.name)
            geometry = _camera_geometry(camera, context, image)
            # Previews should not cover the scene when looking through a camera.
            if image and rv3d.view_perspective != 'CAMERA' and _in_scope(settings.images, camera) and 'image' in geometry:
                fill = _shader('UNIFORM_COLOR')
                fill.bind()
                fill.uniform_float('color', (.065,.065,.065,1))
                geometry['backing'].draw(fill)
                pointer = image.as_pointer()
                if pointer not in _textures:
                    _textures[pointer] = gpu.texture.from_image(image)
                shader = _shader('IMAGE_COLOR_SCENE_LINEAR_TO_REC709_SRGB')
                shader.bind()
                shader.uniform_sampler('image', _textures[pointer])
                shader.uniform_float('color', (1,1,1,1))
                geometry['image'].draw(shader)
            if settings.crops:
                shader = _shader('POLYLINE_UNIFORM_COLOR')
                shader.bind()
                diagnostic = camera.get('_sg_debug_projection_color')
                color = diagnostic or (theme.object_active if camera.select_get() else theme.space.text)
                shader.uniform_float('color', tuple(color[:3])+(.95 if camera.select_get() or diagnostic else .5,))
                shader.uniform_float('viewportSize', (context.region.width, context.region.height))
                shader.uniform_float('lineWidth', 1.6 if camera.select_get() else 1.0)
                geometry['border'].draw(shader)
    except (ReferenceError, RuntimeError, ValueError, KeyError):
        _record_error()
    finally:
        gpu.state.blend_set(blend)
        gpu.state.depth_test_set(depth)
        gpu.state.depth_mask_set(depth_mask)


def _label_rect(x, y, width, height, occupied, bounds):
    """Avoid intersecting badges; never pile multiple labels on a viewport edge."""
    region_width, region_height = bounds
    if x+width < 0 or x > region_width or y+height < 0 or y > region_height:
        return None
    x = max(8, min(x, region_width-width-8))
    y = max(8, min(y, region_height-height-8))
    for delta in [0, *[sign*i*(height+5) for i in range(1,9) for sign in (1,-1)]]:
        candidate = (x,y+delta,width,height)
        if candidate[1] < 8 or candidate[1]+height > region_height-8:
            continue
        if not any(x<px+pw+5 and x+width>px-5 and candidate[1]<py+ph+5
                   and candidate[1]+height>py-5 for px,py,pw,ph in occupied):
            return candidate
    return None


def _sg_draw_view_labels():
    context = _draw_context()
    if context is None or context.region_data.view_perspective == 'CAMERA':
        return
    settings = context.scene.sg_camera_display
    if settings.labels == 'NONE':
        return
    from ..ui.studio_widgets import Painter
    scale = context.preferences.system.ui_scale
    p = Painter(context.region, scale, 0)
    p.width = context.region.width/scale
    snapshot = _snapshot(context)
    view = prompt_overlay.begin_draw(context, snapshot['cameras'])
    prompts = {item.name: item for item in context.scene.camera_prompts}
    theme = context.preferences.themes[0].view_3d
    p.colors['camera_selected'] = tuple(theme.object_active)+(1,)
    p.colors['camera_ink'] = (.08,.08,.08,1)
    occupied = []
    try:
        # Prioritize the selected camera if labels need to be omitted in a crowd.
        ordered = sorted(enumerate(snapshot['cameras']), key=lambda item:(
            not (view.edit and view.edit['camera'] == item[1]),
            item[1].as_pointer() not in view.expanded, not item[1].select_get()))
        for index, camera in ordered:
            if not _visible(camera, context) or not _in_scope(settings.labels, camera):
                continue
            center = camera.matrix_world.translation
            clip = context.region_data.perspective_matrix @ center.to_4d()
            if clip.w <= 0 or abs(clip.z) > clip.w:
                continue
            anchor = location_3d_to_region_2d(context.region, context.region_data, center)
            if anchor is None:
                continue
            if not (0<=anchor.x<=context.region.width and 0<=anchor.y<=context.region.height):
                continue
            geometry = (_geometry.get(camera.as_pointer())
                        if camera.data.type != 'PANO' and (settings.crops or _in_scope(settings.images, camera)) else None)
            if geometry and geometry['signature'][0] != tuple(v for row in camera.matrix_world for v in row):
                geometry = None
            if geometry:
                projected = [location_3d_to_region_2d(context.region, context.region_data, v) for v in geometry['points']]
                if all(v is not None for v in projected):
                    anchor = mathutils.Vector((sum(v.x for v in projected)/4, min(v.y for v in projected)))
            x, y = anchor.x/scale, (context.region.height-anchor.y)/scale+6
            title = str(camera.get('sg_view_label') or camera.name)
            if title.endswith(' view'):
                title = title[:-5].capitalize()
            title = p.fit(title, 160)
            number = f'{index+1:02}'
            number_width = max(26,p.measure(number)+12)
            scene_camera = settings.states and camera == context.scene.camera
            generating = settings.states and camera == snapshot.get('generating')
            prompt = prompts.get(camera.name)
            expanded = camera.as_pointer() in view.expanded
            width = number_width+p.measure(title)+16+(23 if scene_camera else 0)+(16 if generating else 0)+25
            value = view.edit['buffer'].text if view.edit and view.edit['camera'] == camera else prompt.prompt if prompt else ''
            field_width, field_height = prompt_overlay.prompt_height(p, value) if expanded else (0,0)
            block_width, block_height = max(width, field_width), 25 + (field_height + 6 if expanded else 0)
            block = _label_rect(x-block_width/2, y, block_width, block_height, occupied, (p.width, context.region.height/scale))
            rect = (block[0]+(block_width-width)/2, block[1], width, 25) if block else None
            if rect is None:
                continue
            occupied.append(block)
            selected = camera.select_get()
            bx,by,bw,bh = rect
            if abs(by-y)>1:
                p.line([(x,y-3),(max(bx+6,min(bx+bw-6,x)),by if by>y else by+bh)], 'muted', .7)
            p.outline(*rect, 'selected' if selected else 'tooltip', 'camera_selected' if selected else 'section_border', 3)
            p.rect(bx+1,by+1,number_width,bh-2,'camera_selected' if selected else 'raised',2,corners='LEFT')
            p.text(number,bx+6,by+6,11,'camera_ink' if selected else 'text')
            p.text(title,bx+number_width+8,by+6,11,'text')
            icon_x = bx+number_width+8+p.measure(title)+5
            if scene_camera:
                p.icon('camera',icon_x,by+5,15,'secondary')
                icon_x += 23
            if generating:
                p.icon('play',icon_x,by+6,12,'warning')
            prompt_overlay.draw_icon(p, view, camera, (bx+bw-25,by,25,bh), bool(value.strip()),
                                     'camera_selected' if selected else 'section_border')
            if expanded:
                prompt_overlay.draw_field(p, view, camera, prompt, (block[0],by+bh+6,block_width,field_height))
    except (ReferenceError, RuntimeError, ValueError, KeyError):
        _record_error()
    finally:
        p.finish()


def _sg_ensure_crop_overlay():
    """Register the crop overlay draw handler if not already active."""
    global _sg_crop_draw_handle
    if _sg_crop_draw_handle is None and not bpy.app.background:
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
    if _sg_label_draw_handle is None and not bpy.app.background:
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
        if _sg_label_draw_handle is None and not bpy.app.background:
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
