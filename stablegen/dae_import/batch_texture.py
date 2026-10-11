"""Batch model import + texture generation pipeline.

Imports each DAE or GLB file, auto-places cameras, generates textures using
the current scene settings, bakes, and exports to GLB — all without
user intervention.  Follows the same timer-based state-machine pattern
as :pymod:`mesh_gen.batch`.
"""

import json
import os
from datetime import datetime
import bpy  # pylint: disable=import-error

from ..core import ADDON_PKG
from ..utils import sg_modal_active
from .batch_manifest import (
    discover_model_files,
    filename_contains_model_stem,
    load_batch_manifest,
)

# Reference image filenames (checked in order)
_REF_NAMES = ('ref.png', 'ref.jpg', 'ref.jpeg')
_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}

# ── Module-level batch state ────────────────────────────────────────────────
_batch_state = {
    'active': False,
    'cancelled': False,
    'files': [],          # validated model-entry dictionaries
    'index': -1,
    'total': 0,
    'top_dir': '',        # root directory (for fallback ref image)
    'out_dir': '',
    'phase': 'idle',      # idle | import | cameras | texturing | bake | export
    'settle_count': 0,
    'retry_count': 0,
    'bake_retry_count': 0,
    'failures': [],
    # Operator settings (copied from the invoking operator)
    'settings': {},
}

_SETTLE_TICKS = 12
_BAKE_SETTLE_TICKS = 4
_BAKE_RETRIES = 2
_TICK_INTERVAL = 0.5


def _redraw():
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:  # noqa: BLE001
        pass


def _sync_wm():
    try:
        wm = bpy.context.window_manager
        wm.sg_batch_tex_running = _batch_state['active']
        wm.sg_batch_tex_index = _batch_state['index'] + 1
        wm.sg_batch_tex_total = _batch_state['total']
        wm.sg_batch_tex_phase = _batch_state['phase']
        _redraw()
    except Exception:  # noqa: BLE001
        pass


def _failure_report_path():
    """Return the persistent report path for skipped batch entries."""
    return os.path.join(_batch_state['out_dir'], 'batch_failures.json')


def _load_failure_report(out_dir):
    """Load failures from an earlier/resumed run, if present."""
    path = os.path.join(out_dir, 'batch_failures.json')
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        failures = data.get('failures', [])
        return failures if isinstance(failures, list) else []
    except FileNotFoundError:
        return []
    except Exception as exc:  # noqa: BLE001
        print(f"[BatchTex] Could not read existing failure report: {exc}")
        return []


def _write_failure_report():
    """Atomically persist failures so an unattended run is auditable."""
    state = _batch_state
    path = _failure_report_path()
    tmp_path = path + '.tmp'
    payload = {
        'schema': 1,
        'updated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'total_models': state['total'],
        'failure_count': len(state['failures']),
        'failures': state['failures'],
    }
    try:
        os.makedirs(state['out_dir'], exist_ok=True)
        with open(tmp_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
        os.replace(tmp_path, path)
    except Exception as exc:  # noqa: BLE001
        print(f"[BatchTex] Could not write failure report: {exc}")


def _record_failure(stage, error, attempts=None):
    """Record the current model's terminal failure once."""
    state = _batch_state
    entry = state['files'][state['index']]
    record = {
        'index': state['index'] + 1,
        'relative_path': entry['relative_path'],
        'model_path': entry['model_path'],
        'reference_path': entry.get('reference_path', ''),
        'output_relative_path': entry['output_relative_path'],
        'stage': stage,
        'error': str(error),
        'attempts': (
            int(attempts)
            if attempts is not None else state['retry_count'] + 1
        ),
        'failed_at': datetime.now().astimezone().isoformat(timespec='seconds'),
    }
    # Replace an older record for the same output when a resumed run fails
    # again, instead of accumulating indistinguishable duplicates.
    state['failures'] = [
        item for item in state['failures']
        if item.get('output_relative_path') != entry['output_relative_path']
    ]
    state['failures'].append(record)
    _write_failure_report()


def _clear_generation_error():
    """Reset per-attempt generation state before retrying a clean import."""
    try:
        scene = bpy.context.scene
        scene.sg_last_gen_error = False
        scene.sg_last_gen_error_message = ''
        scene.generation_status = 'idle'
    except Exception:  # noqa: BLE001
        pass


def _retry_or_skip(stage, error):
    """Retry the current model, or log and skip it after retries expire."""
    state = _batch_state
    max_retries = state['settings'].get('generation_retries', 2)
    current_attempt = state['retry_count'] + 1
    total_attempts = max_retries + 1
    rel = state['files'][state['index']]['relative_path']

    if state['retry_count'] < max_retries:
        state['retry_count'] += 1
        print(f"[BatchTex] {stage} failed for {rel}: {error}")
        print(f"[BatchTex] Retrying current model "
              f"(attempt {state['retry_count'] + 1}/{total_attempts})")
        _clear_generation_error()
        state['phase'] = 'import'
        state['settle_count'] = 0
        _sync_wm()
        return _TICK_INTERVAL

    _record_failure(stage, error)
    print(f"[BatchTex] {stage} failed for {rel} after "
          f"{current_attempt}/{total_attempts} attempt(s): {error}")

    if state['settings'].get('continue_on_error', True):
        print("[BatchTex] Skipping failed model and continuing batch")
        _clear_generation_error()
        state['phase'] = 'idle'
        state['settle_count'] = 0
        state['retry_count'] = 0
        _sync_wm()
        return _TICK_INTERVAL

    print("[BatchTex] Stop-on-error is enabled — stopping batch")
    state['active'] = False
    state['phase'] = 'idle'
    _sync_wm()
    return None


# ── Reference image resolution ──────────────────────────────────────────────

def _find_ref_image(model_path, top_dir, ref_dir='', explicit_ref=''):
    """Return the path to a reference image for this model, or ''.

    Search order:
    An explicit manifest reference always wins. Otherwise, search a separate
    reference tree using a token-safe model-stem match, then ``ref.*``
    fallbacks beside the model and in the configured roots.
    """
    if explicit_ref:
        if os.path.isfile(explicit_ref):
            return explicit_ref
        print(f"[BatchTex] Explicit reference is missing: {explicit_ref}")
        return ''

    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    model_dir = os.path.dirname(model_path)

    # 1. Separate reference image directory (building-specific match)
    if ref_dir and os.path.isdir(ref_dir):
        rel = os.path.relpath(model_dir, top_dir)
        # Try the full relative path first, then progressively shorter
        # suffixes (e.g. "3D_modely/Úzká" → "Úzká") so the ref dir
        # doesn't need to replicate the exact model-tree prefix.
        parts = rel.replace('\\', '/').split('/')
        candidates = [os.path.join(*parts[i:]) for i in range(len(parts))]
        candidates.append('.')  # ref_dir root as final fallback
        for suffix in candidates:
            search_dir = os.path.join(ref_dir, suffix) if suffix != '.' else ref_dir
            if not os.path.isdir(search_dir):
                continue
            for name in sorted(os.listdir(search_dir)):
                ext = os.path.splitext(name)[1].lower()
                if (ext in _IMAGE_EXTS and
                        filename_contains_model_stem(name, model_stem)):
                    return os.path.join(search_dir, name)
    elif ref_dir:
        print(f"[BatchTex] ref_dir={ref_dir!r} isdir={os.path.isdir(ref_dir)}")

    # 2. Same directory as the .dae file (subdir-level fallback)
    for name in _REF_NAMES:
        candidate = os.path.join(model_dir, name)
        if os.path.isfile(candidate):
            return candidate

    # 3. Separate ref dir — subdirectory fallback ref image
    if ref_dir and os.path.isdir(ref_dir):
        rel = os.path.relpath(model_dir, top_dir)
        parts = rel.replace('\\', '/').split('/')
        candidates = [os.path.join(*parts[i:]) for i in range(len(parts))]
        candidates.append('.')
        for suffix in candidates:
            search_dir = os.path.join(ref_dir, suffix) if suffix != '.' else ref_dir
            if not os.path.isdir(search_dir):
                continue
            for name in _REF_NAMES:
                candidate = os.path.join(search_dir, name)
                if os.path.isfile(candidate):
                    return candidate

    # 4. Top-level directory fallback
    for name in _REF_NAMES:
        candidate = os.path.join(top_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return ''


# ── Phase handlers ──────────────────────────────────────────────────────────

def _center_meshes_at_origin(objects):
    """Center mesh bounds in XY and place the lowest point on Z=0."""
    import mathutils  # pylint: disable=import-error

    corners = []
    for obj in objects:
        corners.extend(obj.matrix_world @ mathutils.Vector(corner)
                       for corner in obj.bound_box)
    if not corners:
        return
    center_x = (min(point.x for point in corners) +
                max(point.x for point in corners)) / 2.0
    center_y = (min(point.y for point in corners) +
                max(point.y for point in corners)) / 2.0
    bottom_z = min(point.z for point in corners)
    for obj in objects:
        obj.location.x -= center_x
        obj.location.y -= center_y
        obj.location.z -= bottom_z


def _scale_meshes(objects, target_size):
    """Uniformly scale meshes so their largest world-space extent matches."""
    if target_size <= 0:
        return
    import mathutils  # pylint: disable=import-error

    corners = []
    for obj in objects:
        corners.extend(obj.matrix_world @ mathutils.Vector(corner)
                       for corner in obj.bound_box)
    if not corners:
        return
    extent = max(
        max(point.x for point in corners) - min(point.x for point in corners),
        max(point.y for point in corners) - min(point.y for point in corners),
        max(point.z for point in corners) - min(point.z for point in corners),
    )
    if extent < 1e-6:
        return
    factor = target_size / extent
    for obj in objects:
        obj.scale *= factor


def _import_glb(filepath, settings):
    """Import an already-simplified GLB and prepare it for retexturing.

    Libuše GLBs are output from the project's simplification pipeline. Heavy
    DAE cleanup (interior removal, fan repair, remeshing) would alter that
    verified geometry, so GLB preparation is deliberately light: remove
    non-mesh scene nodes, strip the old baked material, join meshes, normalize
    placement, and apply scale.
    """
    result = bpy.ops.import_scene.gltf(filepath=filepath)
    if 'FINISHED' not in result:
        raise RuntimeError("Blender GLB importer did not finish")

    context = bpy.context
    meshes = [obj for obj in context.scene.objects if obj.type == 'MESH']
    if not meshes:
        raise RuntimeError("GLB contains no mesh objects")

    # Detach meshes while preserving world transforms, then remove imported
    # empties, cameras, and lights so they cannot influence camera setup.
    for obj in meshes:
        world_matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world_matrix
    for obj in list(context.scene.objects):
        if obj.type != 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Apply the imported scene graph before joining separate mesh nodes.
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:
        obj.select_set(True)
    context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    if len(meshes) > 1:
        bpy.ops.object.join()
        meshes = [context.view_layer.objects.active]

    from . import cleanup
    if settings.get('strip_materials', True):
        cleanup.strip_materials_and_apply_clean(meshes[0])

    if settings.get('center_origin', True):
        _center_meshes_at_origin(meshes)
    _scale_meshes(meshes, settings.get('auto_scale', 0.0))

    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:
        obj.select_set(True)
    context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    shading = settings.get('shading_mode', 'FLAT')
    if shading == 'SMOOTH':
        bpy.ops.object.shade_smooth()
    elif shading == 'AUTO':
        cleanup.auto_smooth_shading(meshes[0])
    else:
        bpy.ops.object.shade_flat()

    vertices = sum(len(obj.data.vertices) for obj in meshes)
    faces = sum(len(obj.data.polygons) for obj in meshes)
    print(f"[BatchTex] GLB import complete: {vertices} verts, {faces} faces")


def _do_import():
    """Import the current DAE or GLB file. Returns True on success."""
    state = _batch_state
    filepath = state['files'][state['index']]['model_path']
    settings = state['settings']

    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    try:
        extension = os.path.splitext(filepath)[1].lower()
        if extension == '.dae':
            result = bpy.ops.stablegen.import_dae(
                filepath=filepath,
                merge_threshold=settings.get('merge_threshold', 0.001),
                remove_interior=settings.get('remove_interior', True),
                remove_ground=settings.get('remove_ground', True),
                strip_materials=settings.get('strip_materials', True),
                topology_method=settings.get('topology_method', 'FIX_FANS'),
                edge_ratio=settings.get('edge_ratio', 2.5),
                equalize_iterations=settings.get('equalize_iterations', 4),
                voxel_size=settings.get('voxel_size', 0.0),
                auto_scale=settings.get('auto_scale', 0.0),
                shading_mode=settings.get('shading_mode', 'FLAT'),
                center_origin=settings.get('center_origin', True),
                join_meshes=True,
                triangulate=settings.get('triangulate', True),
            )
            if 'FINISHED' not in result:
                raise RuntimeError("StableGen DAE importer did not finish")
        elif extension == '.glb':
            _import_glb(filepath, settings)
        else:
            raise ValueError(f"Unsupported model format: {extension}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[BatchTex] FAILED to import: {exc}")
        return False


def _do_cameras():
    """Auto-place cameras around the imported mesh."""
    scene = bpy.context.scene
    settings = _batch_state['settings']

    _pm = settings.get('placement_mode', 'normal_weighted')
    _cam_kwargs = {
        'num_cameras': settings.get('camera_count', 8),
        'placement_mode': _pm,
        'auto_prompts': settings.get('auto_prompts', True),
        'review_placement': False,
        'purge_others': True,
        'exclude_bottom': settings.get('exclude_bottom', True),
        'exclude_bottom_angle': settings.get('exclude_bottom_angle', 1.5533),
        'auto_aspect': settings.get('auto_aspect', 'per_camera'),
        'occlusion_mode': settings.get('occlusion_mode', 'none'),
        'consider_existing': settings.get('consider_existing', True),
        'clamp_elevation': settings.get('clamp_elevation', False),
        'max_elevation_angle': settings.get('max_elevation', 1.2217),
        'min_elevation_angle': settings.get('min_elevation', -0.1745),
    }
    if _pm == 'greedy_coverage':
        _cam_kwargs['coverage_target'] = settings.get('coverage_target', 0.95)
        _cam_kwargs['max_auto_cameras'] = settings.get('max_auto_cameras', 12)
    if _pm == 'fan_from_camera':
        _cam_kwargs['fan_angle'] = settings.get('fan_angle', 90.0)

    from .. import utils as _sg_utils

    # Find a 3D viewport for temp_override
    _v3d_area = None
    _v3d_region = None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                _v3d_area = area
                for region in area.regions:
                    if region.type == 'WINDOW':
                        _v3d_region = region
                        break
                break
        if _v3d_area:
            break

    _sg_utils._sg_bypass_modal_check = True
    try:
        if _v3d_area and _v3d_region:
            with bpy.context.temp_override(area=_v3d_area, region=_v3d_region):
                bpy.ops.object.add_cameras(**_cam_kwargs)
        else:
            bpy.ops.object.add_cameras(**_cam_kwargs)
    finally:
        _sg_utils._sg_bypass_modal_check = False


def _do_set_ref_image():
    """Set this model's reference image and never retain the previous one."""
    state = _batch_state
    entry = state['files'][state['index']]
    filepath = entry['model_path']
    scene = bpy.context.scene

    # A valid path from the previous house must never leak into this one when
    # lookup fails. Clear both supported reference mechanisms before looking
    # up the current entry.
    scene.use_ipadapter = False
    scene.ipadapter_image = ""
    scene.qwen_use_external_style_image = False
    scene.qwen_external_style_image = ""

    ref = _find_ref_image(filepath, state['top_dir'],
                          state['settings'].get('ref_image_dir', ''),
                          entry.get('reference_path', ''))
    arch = getattr(scene, 'model_architecture', 'sdxl')

    if ref:
        print(f"[BatchTex] Reference image: {ref}")
        if arch == 'sdxl':
            scene.use_ipadapter = True
            scene.ipadapter_image = ref
        elif arch in ('qwen_image_edit', 'qwen_image_21', 'flux2_klein'):
            scene.qwen_use_external_style_image = True
            scene.qwen_external_style_image = ref
        elif arch == 'flux1':
            scene.use_ipadapter = True
            scene.ipadapter_image = ref
        return True
    else:
        print("[BatchTex] No reference image found")
        return False


def _do_start_texturing():
    """Select all cameras and invoke the texture generation operator."""
    bpy.ops.object.select_all(action='DESELECT')
    scene_cams = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
    for obj in scene_cams:
        obj.select_set(True)

    from .. import utils as _sg_utils
    _sg_utils._sg_bypass_modal_check = True
    try:
        result = bpy.ops.object.test_stable('INVOKE_DEFAULT')
    finally:
        _sg_utils._sg_bypass_modal_check = False
    if not ({'RUNNING_MODAL', 'FINISHED'} & set(result)):
        raise RuntimeError(f"Texture generation operator returned {set(result)}")
    print("[BatchTex] Texture generation started")


def _do_bake():
    """Bake textures for all mesh objects using the utility functions directly."""
    from ..texturing.rendering import prepare_baking, bake_texture
    from ..utils import get_dir_path, get_file_path

    settings = _batch_state['settings']
    tex_res = settings.get('bake_resolution', 2048)

    context = bpy.context
    original_engine = context.scene.render.engine

    baked_objects = []
    try:
        prepare_baking(context)

        mesh_objects = [
            obj for obj in context.scene.objects if obj.type == 'MESH'
        ]
        if not mesh_objects:
            raise RuntimeError("No mesh objects available for baking")

        for obj in mesh_objects:
            # ── Create a fresh BakeUV and unwrap directly ──
            # Remove all non-projection UV maps (DAE originals are per-material
            # SketchUp UVs, not suitable as a bake atlas)
            to_remove = [uv for uv in obj.data.uv_layers
                         if "ProjectionUV" not in uv.name
                         and uv.name != "_SG_ProjectionBuffer"
                         and uv.name != "BakeUV"]
            for uv in to_remove:
                obj.data.uv_layers.remove(uv)

            # Ensure BakeUV exists and is active
            bake_uv = obj.data.uv_layers.get("BakeUV")
            if not bake_uv:
                bake_uv = obj.data.uv_layers.new(name="BakeUV")
            obj.data.uv_layers.active = bake_uv

            # Smart UV Project directly (bypass unwrap() which has its own UV selection logic)
            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
            # UV sync must be ON so mesh face selection drives UV selection
            bpy.context.scene.tool_settings.use_uv_select_sync = True
            bpy.ops.mesh.select_all(action='SELECT')
            # Set BakeUV as the active UV layer in edit mode too
            obj.data.uv_layers.active = obj.data.uv_layers["BakeUV"]
            # UV operators need a proper 3D viewport context — in a timer
            # callback the default context has no area/region, causing
            # smart_project to silently produce garbage UVs.
            _v3d_area = _v3d_region = None
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        _v3d_area = area
                        for region in area.regions:
                            if region.type == 'WINDOW':
                                _v3d_region = region
                                break
                        break
                if _v3d_area:
                    break
            if _v3d_area and _v3d_region:
                with bpy.context.temp_override(area=_v3d_area,
                                               region=_v3d_region):
                    bpy.ops.uv.smart_project(correct_aspect=False)
            else:
                bpy.ops.uv.smart_project(correct_aspect=False)
            bpy.ops.object.mode_set(mode='OBJECT')

            baked = bake_texture(
                context,
                obj,
                texture_resolution=tex_res,
                output_dir=get_dir_path(context, "baked"),
            )
            if not baked:
                raise RuntimeError(f"Bake did not run for {obj.name}")

            baked_path = get_file_path(
                context, "baked", object_name=obj.name)
            if not _baked_file_has_signal(baked_path):
                raise RuntimeError(
                    f"Bake output for {obj.name} is uniformly black; "
                    "keeping the projection material and retrying"
                )
            baked_objects.append(obj)

        # Only replace projection materials after every object has a verified
        # bake. A failed attempt remains recoverable and can be retried.
        for obj in baked_objects:
            _apply_baked_material(context, obj)
        print("[BatchTex] Bake complete")
        return True
    finally:
        if context.scene.render.engine != original_engine:
            context.scene.render.engine = original_engine


def _baked_file_has_signal(filepath):
    """Return whether a saved batch bake contains any non-black RGB value."""
    if not filepath or not os.path.isfile(filepath):
        return False
    try:
        from PIL import Image

        with Image.open(filepath) as image:
            extrema = image.convert('RGB').getextrema()
        return any(high > 0 for _low, high in extrema)
    except Exception as exc:  # noqa: BLE001
        print(f"[BatchTex] Could not validate baked image {filepath}: {exc}")
        return False


def _refresh_projection_for_bake(context):
    """Reload projection inputs and force dependency-graph evaluation."""
    seen_images = set()
    for obj in context.scene.objects:
        if obj.type != 'MESH':
            continue
        obj.data.update()
        for material in obj.data.materials:
            if not material or not material.use_nodes:
                continue
            material.node_tree.update_tag()
            for node in material.node_tree.nodes:
                image = getattr(node, 'image', None)
                if image is None or image.as_pointer() in seen_images:
                    continue
                seen_images.add(image.as_pointer())
                filepath = bpy.path.abspath(image.filepath)
                if filepath and os.path.isfile(filepath):
                    image.reload()
                image.update()
    context.view_layer.update()


def _write_camera_recovery_state(context):
    """Persist exact camera transforms needed to reproject saved views."""
    from ..utils import get_dir_path, get_file_path

    state = _batch_state
    entry = state['files'][state['index']]
    cameras = sorted(
        (obj for obj in context.scene.objects if obj.type == 'CAMERA'),
        key=lambda camera: camera.name,
    )
    payload = {
        'schema': 1,
        'relative_path': entry['relative_path'],
        'model_path': entry['model_path'],
        'output_relative_path': entry['output_relative_path'],
        'output_timestamp': context.scene.output_timestamp,
        'cameras': [],
    }
    for index, camera in enumerate(cameras):
        data = camera.data
        payload['cameras'].append({
            'index': index,
            'name': camera.name,
            'matrix_world': [
                [float(value) for value in row]
                for row in camera.matrix_world
            ],
            'type': data.type,
            'lens': float(data.lens),
            'sensor_width': float(data.sensor_width),
            'sensor_height': float(data.sensor_height),
            'shift_x': float(data.shift_x),
            'shift_y': float(data.shift_y),
            'resolution_x': int(camera.get(
                'sg_res_x', context.scene.render.resolution_x)),
            'resolution_y': int(camera.get(
                'sg_res_y', context.scene.render.resolution_y)),
            'generated_image': get_file_path(
                context, 'generated', camera_id=index, material_id=0),
        })

    revision_dir = get_dir_path(context, 'revision')
    path = os.path.join(revision_dir, 'batch_camera_state.json')
    tmp_path = path + '.tmp'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
        os.replace(tmp_path, path)
        print(f"[BatchTex] Saved camera recovery state: {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[BatchTex] Could not save camera recovery state: {exc}")


def _finish_terminal_bake_failure(error):
    """Log a verified bake failure and either skip or stop the batch."""
    state = _batch_state
    attempts = state['bake_retry_count'] + 1
    _record_failure('bake', error, attempts=attempts)
    rel = state['files'][state['index']]['relative_path']
    print(
        f"[BatchTex] Bake failed for {rel} after {attempts} attempt(s): "
        f"{error}"
    )
    state['bake_retry_count'] = 0
    state['settle_count'] = 0
    if state['settings'].get('continue_on_error', True):
        print("[BatchTex] Skipping failed model and continuing batch")
        state['phase'] = 'idle'
        _sync_wm()
        return _TICK_INTERVAL
    print("[BatchTex] Stop-on-error is enabled — stopping batch")
    state['active'] = False
    state['phase'] = 'idle'
    _sync_wm()
    return None


def _apply_baked_material(context, obj):
    """Replace the projection material with a simple Principled BSDF using the baked texture."""
    from ..utils import get_dir_path, get_file_path
    import os as _os

    output_dir = get_dir_path(context, "baked")
    baked_path = get_file_path(context, "baked", object_name=obj.name)

    if not baked_path or not _os.path.exists(baked_path):
        print(f"[BatchTex] No baked texture found for {obj.name}, skipping material apply")
        return

    mat = bpy.data.materials.new(name=f"{obj.name}_baked")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Output node
    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (600, 0)

    # Principled BSDF — fully matte so baked colour is displayed as-is
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (300, 0)
    principled.inputs["Roughness"].default_value = 1.0
    principled.inputs["Metallic"].default_value = 0.0
    links.new(principled.outputs[0], output_node.inputs["Surface"])

    # UV Map (use BakeUV if available, otherwise first non-projection UV)
    uv_node = nodes.new("ShaderNodeUVMap")
    uv_node.location = (-400, 0)
    if "BakeUV" in [uv.name for uv in obj.data.uv_layers]:
        uv_node.uv_map = "BakeUV"
    else:
        for uv in obj.data.uv_layers:
            if "ProjectionUV" not in uv.name and uv.name != "_SG_ProjectionBuffer":
                uv_node.uv_map = uv.name
                break

    # Base Color texture
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = bpy.data.images.load(baked_path)
    tex_node.location = (-100, 0)
    links.new(uv_node.outputs["UV"], tex_node.inputs["Vector"])
    links.new(tex_node.outputs["Color"], principled.inputs["Base Color"])

    # Clear all existing materials and assign the baked one
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    print(f"[BatchTex] Applied baked material to {obj.name}")


def _do_export():
    """Export the current textured mesh to GLB."""
    state = _batch_state
    entry = state['files'][state['index']]
    out_path = os.path.join(state['out_dir'], entry['output_relative_path'])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Select all mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    if bpy.context.selected_objects:
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]

    try:
        bpy.ops.export_scene.gltf(filepath=out_path)
        print(f"[BatchTex] Exported: {out_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[BatchTex] FAILED to export: {exc}")


# ── State machine tick ──────────────────────────────────────────────────────


def _batch_tex_tick():
    """Timer callback — drives the batch texturing state machine."""
    from ..texturing.generator import ComfyUIGenerate

    state = _batch_state

    if state['cancelled'] or not state['active']:
        state['active'] = False
        state['cancelled'] = False
        state['phase'] = 'idle'
        _sync_wm()
        print("[BatchTex] Batch stopped")
        return None

    phase = state['phase']

    # ── IDLE: advance to next file ──────────────────────────────────────
    if phase == 'idle':
        next_idx = state['index'] + 1
        # Skip files whose output GLB already exists (resume support)
        while next_idx < state['total']:
            entry = state['files'][next_idx]
            rel_path = entry['relative_path']
            out_path = os.path.join(
                state['out_dir'], entry['output_relative_path'])
            if os.path.exists(out_path):
                print(f"[BatchTex] Skipping {next_idx + 1}/{state['total']}: "
                      f"{rel_path} (already exported)")
                next_idx += 1
            else:
                break

        if next_idx >= state['total']:
            state['active'] = False
            state['phase'] = 'idle'
            _sync_wm()
            failed = len(state['failures'])
            print(f"[BatchTex] Complete! {state['total']} file(s) processed, "
                  f"{failed} failed/skipped.")
            if failed:
                print(f"[BatchTex] Failure report: {_failure_report_path()}")
            return None

        state['index'] = next_idx
        state['retry_count'] = 0
        state['bake_retry_count'] = 0
        state['phase'] = 'import'
        _sync_wm()
        rel = state['files'][next_idx]['relative_path']
        print(f"[BatchTex] {next_idx + 1}/{state['total']}: {rel}")
        return _TICK_INTERVAL

    # ── IMPORT ──────────────────────────────────────────────────────────
    if phase == 'import':
        ok = _do_import()
        if not ok:
            return _retry_or_skip('import', 'Model import failed')

        state['phase'] = 'cameras'
        return _TICK_INTERVAL

    # ── CAMERAS ─────────────────────────────────────────────────────────
    if phase == 'cameras':
        try:
            _do_cameras()
        except Exception as exc:  # noqa: BLE001
            return _retry_or_skip('camera placement', exc)

        # Set reference image before texturing. Manifest batches are strict:
        # validation guarantees a reference for every row, so a lookup failure
        # indicates unsafe state and must stop instead of generating with a
        # stale or missing house image.
        has_reference = _do_set_ref_image()
        if (not has_reference and
                state['settings'].get('require_reference', False)):
            rel = state['files'][state['index']]['relative_path']
            print(f"[BatchTex] Stopping: required reference missing for {rel}")
            _record_failure('reference', 'Required manifest reference missing')
            state['cancelled'] = True
            return _TICK_INTERVAL

        state['phase'] = 'texturing'
        state['settle_count'] = 0
        # Small delay so Blender digests the new cameras
        return _TICK_INTERVAL

    # ── TEXTURING (wait for ComfyUIGenerate to finish) ──────────────────
    if phase == 'texturing':
        if state['settle_count'] == 0:
            # First tick: start texturing
            try:
                _do_start_texturing()
            except Exception as exc:  # noqa: BLE001
                return _retry_or_skip('texturing startup', exc)
            state['settle_count'] = 1
            return _TICK_INTERVAL

        if state['settle_count'] < _SETTLE_TICKS:
            state['settle_count'] += 1
            if ComfyUIGenerate._is_running:
                state['settle_count'] = _SETTLE_TICKS  # operator started
            return _TICK_INTERVAL

        # Past settling — wait for completion
        if ComfyUIGenerate._is_running:
            return _TICK_INTERVAL

        # Check if generation was cancelled or failed
        try:
            had_error = bpy.context.scene.sg_last_gen_error
        except Exception:  # noqa: BLE001
            had_error = False
        gen_status = getattr(bpy.context.scene, 'generation_status', 'idle')
        if gen_status == 'waiting':
            print("[BatchTex] Generation was cancelled by user — stopping batch")
            state['active'] = False
            state['phase'] = 'idle'
            _sync_wm()
            return None
        if had_error:
            error = getattr(
                bpy.context.scene,
                'sg_last_gen_error_message',
                '',
            ) or 'Texture generation failed'
            return _retry_or_skip('texturing', error)

        # Preserve the exact camera rig before baking/export. Generated views
        # can then be reprojected without rerunning the image model.
        try:
            _write_camera_recovery_state(bpy.context)
        except Exception as exc:  # noqa: BLE001
            print(f"[BatchTex] Could not save camera recovery state: {exc}")

        # Texturing done — move to bake
        if _batch_state['settings'].get('bake', True):
            state['phase'] = 'bake'
            state['settle_count'] = 0
            state['bake_retry_count'] = 0
        else:
            state['phase'] = 'export'
        return _TICK_INTERVAL

    # ── BAKE (synchronous — no operator popup) ──────────────────────────
    if phase == 'bake':
        # Generation completion and Cycles/image dependency evaluation are
        # separate Blender events. Give the graph time to settle before each
        # attempt, particularly in long unattended runs.
        if state['settle_count'] < _BAKE_SETTLE_TICKS:
            state['settle_count'] += 1
            return _TICK_INTERVAL
        try:
            _refresh_projection_for_bake(bpy.context)
            _do_bake()
        except Exception as exc:  # noqa: BLE001
            if state['bake_retry_count'] < _BAKE_RETRIES:
                state['bake_retry_count'] += 1
                state['settle_count'] = 0
                print(
                    f"[BatchTex] Bake attempt "
                    f"{state['bake_retry_count']}/{_BAKE_RETRIES + 1} "
                    f"failed: {exc}"
                )
                print("[BatchTex] Retrying bake without regenerating images")
                return _TICK_INTERVAL
            return _finish_terminal_bake_failure(exc)

        state['bake_retry_count'] = 0
        state['phase'] = 'export'
        return _TICK_INTERVAL

    # ── EXPORT ──────────────────────────────────────────────────────────
    if phase == 'export':
        try:
            _do_export()
        except Exception as exc:  # noqa: BLE001
            print(f"[BatchTex] Export failed: {exc}")

        state['phase'] = 'idle'
        return _TICK_INTERVAL

    return _TICK_INTERVAL


# ── Operators ───────────────────────────────────────────────────────────────

class BatchTextureDAE(bpy.types.Operator):
    """Batch import DAE/GLB files, auto-place cameras, generate textures,
    bake, and export to .glb — fully automated pipeline."""

    bl_idname = "stablegen.batch_texture_dae"
    bl_label = "Batch Texture DAE / GLB Files"
    bl_options = {'REGISTER'}

    directory: bpy.props.StringProperty(
        name="Directory",
        description="Directory containing DAE/GLB files or a batch manifest",
        subtype='DIR_PATH',
    )  # type: ignore

    filter_glob: bpy.props.StringProperty(
        default="*.dae;*.glb;*.csv",
        options={'HIDDEN'},
    )  # type: ignore

    output_dir: bpy.props.StringProperty(
        name="Output Directory",
        description="Directory for exported .glb files (defaults to 'textured' subfolder)",
        default="",
    )  # type: ignore

    recursive: bpy.props.BoolProperty(
        name="Recursive",
        description="Scan subdirectories recursively",
        default=True,
    )  # type: ignore

    manifest_path: bpy.props.StringProperty(
        name="Batch Manifest",
        description=(
            "Optional CSV with model_path, reference_path, and output_path. "
            "Manifest paths are exact and relative to the CSV location"
        ),
        default="",
        subtype='FILE_PATH',
    )  # type: ignore

    ref_image_dir: bpy.props.StringProperty(
        name="Reference Images",
        description=(
            "Directory with reference images matching the model tree structure. "
            "Images whose filename contains the building number are auto-matched. "
            "Leave empty to use ref.png fallbacks only"
        ),
        default="",
    )  # type: ignore

    bake: bpy.props.BoolProperty(
        name="Bake Textures",
        description="Bake projected textures into an image texture before export",
        default=True,
    )  # type: ignore

    bake_resolution: bpy.props.IntProperty(
        name="Bake Resolution",
        description="Resolution for baked texture images",
        default=2048,
        min=512,
        max=8192,
    )  # type: ignore

    generation_retries: bpy.props.IntProperty(
        name="Generation Retries",
        description=(
            "Number of clean re-import and texturing retries before a model "
            "is skipped"
        ),
        default=2,
        min=0,
        max=10,
    )  # type: ignore

    continue_on_error: bpy.props.BoolProperty(
        name="Continue After Failed Model",
        description=(
            "After retries are exhausted, log and skip the failed model "
            "instead of stopping the entire batch"
        ),
        default=True,
    )  # type: ignore

    # ── Import / cleanup settings (same as BatchImportDAE) ──

    merge_threshold: bpy.props.FloatProperty(
        name="Merge Distance", default=0.001,
        min=0.0, max=0.1, precision=4, unit='LENGTH',
    )  # type: ignore

    remove_interior: bpy.props.BoolProperty(
        name="Remove Interior Faces", default=True,
    )  # type: ignore

    remove_ground: bpy.props.BoolProperty(
        name="Remove Ground Plane",
        description="Remove large horizontal faces at the bottom of the model (SketchUp ground planes)",
        default=True,
    )  # type: ignore

    strip_materials: bpy.props.BoolProperty(
        name="Replace Materials", default=True,
    )  # type: ignore

    triangulate: bpy.props.BoolProperty(
        name="Triangulate", default=True,
    )  # type: ignore

    topology_method: bpy.props.EnumProperty(
        name="Topology",
        items=[
            ('NONE', "None", "Keep original topology"),
            ('FIX_FANS', "Fix Triangle Fans", "Fix fan patterns on flat surfaces"),
            ('SUBDIVIDE_ONLY', "Subdivide Large Faces", "Subdivide faces above threshold"),
            ('PLANAR_SUBDIVIDE', "Dissolve + Subdivide", "Merge coplanar then retriangulate"),
            ('VOXEL', "Voxel Remesh", "Full remesh with uniform voxel size"),
        ],
        default='FIX_FANS',
    )  # type: ignore

    edge_ratio: bpy.props.FloatProperty(
        name="Edge Ratio", default=2.5, min=1.5, max=10.0, precision=1,
    )  # type: ignore

    equalize_iterations: bpy.props.IntProperty(
        name="Equalize Passes", default=4, min=0, max=10,
    )  # type: ignore

    voxel_size: bpy.props.FloatProperty(
        name="Voxel Size", default=0.0, min=0.0, max=1.0, precision=4, unit='LENGTH',
    )  # type: ignore

    auto_scale: bpy.props.FloatProperty(
        name="Target Size (BU)", default=0.0, min=0.0, max=100.0,
        description="Scale so largest dimension equals this value. 0 = keep original",
    )  # type: ignore

    shading_mode: bpy.props.EnumProperty(
        name="Shading",
        items=[
            ('FLAT', "Flat", "Flat shading"),
            ('SMOOTH', "Smooth", "Smooth shading"),
            ('AUTO', "Auto Smooth", "Smooth with auto-smooth angle"),
        ],
        default='FLAT',
    )  # type: ignore

    center_origin: bpy.props.BoolProperty(
        name="Center at Origin", default=True,
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        if _batch_state['active']:
            return False
        if sg_modal_active(context):
            return False
        addon_prefs = context.preferences.addons.get(ADDON_PKG)
        if addon_prefs:
            prefs = addon_prefs.preferences
            if not getattr(prefs, 'server_online', False):
                cls.poll_message_set("ComfyUI server is not connected")
                return False
        return True

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, context):
        layout = self.layout

        layout.label(text="Output", icon='EXPORT')
        box = layout.box()
        box.prop(self, "output_dir")
        box.prop(self, "recursive")
        box.prop(self, "manifest_path")
        box.prop(self, "ref_image_dir")
        box.prop(self, "bake")
        if self.bake:
            box.prop(self, "bake_resolution")

        layout.separator()
        layout.label(text="Import Cleanup", icon='BRUSH_DATA')
        box = layout.box()
        box.prop(self, "merge_threshold")
        box.prop(self, "remove_interior")
        box.prop(self, "remove_ground")
        box.prop(self, "strip_materials")
        box.prop(self, "triangulate")

        layout.separator()
        layout.label(text="Topology", icon='MOD_REMESH')
        box = layout.box()
        box.prop(self, "topology_method")
        if self.topology_method == 'FIX_FANS':
            box.prop(self, "edge_ratio")
            box.prop(self, "equalize_iterations")
        elif self.topology_method == 'VOXEL':
            box.prop(self, "voxel_size")

        layout.separator()
        layout.label(text="Transform", icon='OBJECT_ORIGIN')
        box = layout.box()
        box.prop(self, "auto_scale")
        box.prop(self, "shading_mode")
        box.prop(self, "center_origin")

        layout.separator()
        layout.label(text="Camera & Texturing", icon='CAMERA_DATA')
        box = layout.box()
        box.label(text="Uses current scene generation settings")
        box.label(text="Camera placement uses TRELLIS.2 camera settings")
        box.prop(self, "generation_retries")
        box.prop(self, "continue_on_error")

    def execute(self, context):
        from ..core.server_api import confirm_server
        if not confirm_server(self, context):
            return {'CANCELLED'}
        manifest_path = bpy.path.abspath(self.manifest_path).strip()
        if manifest_path:
            try:
                top_dir, model_files = load_batch_manifest(manifest_path)
            except ValueError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
        else:
            top_dir = bpy.path.abspath(self.directory).rstrip('/\\')
            if not top_dir or not os.path.isdir(top_dir):
                self.report({'ERROR'}, "No valid model directory selected")
                return {'CANCELLED'}

            # The file selector primarily populates ``directory``. When the
            # prepared batch root is selected, automatically use its exact
            # manifest instead of falling back to heuristic recursive lookup.
            default_manifest = os.path.join(top_dir, "batch_manifest.csv")
            if os.path.isfile(default_manifest):
                manifest_path = default_manifest
                try:
                    top_dir, model_files = load_batch_manifest(manifest_path)
                except ValueError as exc:
                    self.report({'ERROR'}, str(exc))
                    return {'CANCELLED'}
                self.report({'INFO'}, "Using batch_manifest.csv from selected directory")
            else:
                model_files = discover_model_files(top_dir, self.recursive)

        if not model_files:
            self.report({'WARNING'}, "No DAE or GLB files found")
            return {'CANCELLED'}

        configured_output = bpy.path.abspath(self.output_dir).strip()
        out_dir = configured_output or os.path.join(top_dir, "textured")
        os.makedirs(out_dir, exist_ok=True)
        existing_failures = _load_failure_report(out_dir)

        scene = context.scene
        _batch_state.update({
            'active': True,
            'cancelled': False,
            'files': model_files,
            'index': -1,
            'total': len(model_files),
            'top_dir': top_dir,
            'out_dir': out_dir,
            'phase': 'idle',
            'settle_count': 0,
            'retry_count': 0,
            'bake_retry_count': 0,
            'failures': existing_failures,
            'settings': {
                # Import settings
                'merge_threshold': self.merge_threshold,
                'remove_interior': self.remove_interior,
                'remove_ground': self.remove_ground,
                'strip_materials': self.strip_materials,
                'topology_method': self.topology_method,
                'edge_ratio': self.edge_ratio,
                'equalize_iterations': self.equalize_iterations,
                'voxel_size': self.voxel_size,
                'auto_scale': self.auto_scale,
                'shading_mode': self.shading_mode,
                'center_origin': self.center_origin,
                'triangulate': self.triangulate,
                # Bake settings
                'bake': self.bake,
                'bake_resolution': self.bake_resolution,
                'generation_retries': self.generation_retries,
                'continue_on_error': self.continue_on_error,
                # Reference image directory
                'ref_image_dir': self.ref_image_dir.rstrip('/\\') if self.ref_image_dir else '',
                'require_reference': bool(manifest_path),
                # Camera settings (read from current scene trellis2_* properties)
                'camera_count': getattr(scene, 'trellis2_camera_count', 8),
                'placement_mode': getattr(scene, 'trellis2_placement_mode', 'normal_weighted'),
                'auto_prompts': getattr(scene, 'trellis2_auto_prompts', True),
                'exclude_bottom': getattr(scene, 'trellis2_exclude_bottom', True),
                'exclude_bottom_angle': getattr(scene, 'trellis2_exclude_bottom_angle', 1.5533),
                'auto_aspect': getattr(scene, 'trellis2_auto_aspect', 'per_camera'),
                'occlusion_mode': getattr(scene, 'trellis2_occlusion_mode', 'none'),
                'consider_existing': getattr(scene, 'trellis2_consider_existing', True),
                'clamp_elevation': getattr(scene, 'trellis2_clamp_elevation', False),
                'max_elevation': getattr(scene, 'trellis2_max_elevation', 1.2217),
                'min_elevation': getattr(scene, 'trellis2_min_elevation', -0.1745),
                'coverage_target': getattr(scene, 'trellis2_coverage_target', 0.95),
                'max_auto_cameras': getattr(scene, 'trellis2_max_auto_cameras', 12),
                'fan_angle': getattr(scene, 'trellis2_fan_angle', 90.0),
            },
        })
        _sync_wm()

        self.report({'INFO'},
                    f"Batch texturing started: {len(model_files)} file(s)")
        source = manifest_path or top_dir
        print(f"[BatchTex] Starting: {len(model_files)} file(s) from '{source}'")
        bpy.app.timers.register(_batch_tex_tick, first_interval=0.1)
        return {'FINISHED'}


class BatchTextureDAECancel(bpy.types.Operator):
    """Cancel the running batch texture pipeline."""

    bl_idname = "stablegen.batch_texture_dae_cancel"
    bl_label = "Cancel Batch Texture"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _batch_state['active']

    def execute(self, context):
        _batch_state['cancelled'] = True
        self.report({'WARNING'}, "Batch texturing cancelling after current model...")
        return {'FINISHED'}


BATCH_TEXTURE_CLASSES = [
    BatchTextureDAE,
    BatchTextureDAECancel,
]


def unregister_batch_texture():
    """Stop any running batch texture timer. Called from addon unregister."""
    _batch_state['cancelled'] = True
    _batch_state['active'] = False
    try:
        if bpy.app.timers.is_registered(_batch_tex_tick):
            bpy.app.timers.unregister(_batch_tex_tick)
    except Exception:  # noqa: BLE001
        pass
