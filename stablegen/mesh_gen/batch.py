"""Batch image-to-3D generation for TRELLIS.2."""

import os
import time
from datetime import datetime
import bpy  # pylint: disable=import-error

# Supported input image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}

BATCH_COLLECTION = "Batch Results"

# ── Module-level batch state ──────────────────────────────────────────────────
_batch_state = {
    'active': False,
    'images': [],
    'index': -1,        # -1 = not yet started; incremented before each gen
    'total': 0,
    'cancelled': False,
    'rename_meshes': True,
    'bake_textures': True,
    'bake_pbr': True,
    'bake_resolution': 2048,
    'bake_try_unwrap': 'smart',
    'bake_overlap_only': False,
    'bake_export_orm': False,
    'bake_normal_convention': 'opengl',
    'bake_add_material': True,
    'bake_flatten': False,
    'pre_objects': set(),       # id() of objects present before current generation
    'isolation_state': {},      # {obj_name: (hide_render, hide_viewport, hide_get)} saved pre-gen
    '_new_obj_ids': set(),      # id() of new objects created during generation

    # Phase state machine:
    #   'idle'       – not yet started (first tick triggers first generation)
    #   'settling'   – waiting for Trellis2Generate._is_running to become True
    #   'generating' – waiting for Trellis2Generate._is_running to become False
    #   'texturing'  – shape done; waiting for Qwen/test_stable via sg_modal_active
    #   'baking'     – BakeTextures running; waiting for it to finish
    'phase': 'idle',
    'settle_count': 0,
    'gen_settle_count': 0,

    # Per-model timestamps
    'model_start': 0.0,
    'thread_done': 0.0,
    'gen_done': 0.0,
    'bake_start': 0.0,

    # Logging
    'batch_start': 0.0,
    'batch_start_str': '',
    'log_path': '',
    'log_entries': [],
    '_current_entry': None,
}

_SETTLE_TICKS     = 12  # max ticks waiting for Trellis2Generate to start
_GEN_SETTLE_TICKS = 3   # consecutive False ticks to confirm generation truly done
_TICK_INTERVAL    = 0.5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scan_images(folder):
    """Return a sorted list of supported image file paths in *folder*."""
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, n)
        for n in sorted(os.listdir(folder))
        if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS
    ]


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
        wm.sg_batch_running = _batch_state['active']
        wm.sg_batch_index   = _batch_state['index'] + 1  # 1-based for display
        wm.sg_batch_total   = _batch_state['total']
        _redraw()
    except Exception:  # noqa: BLE001
        pass


def _fmt_duration(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _sg_modal_active():
    """True if any StableGen heavy modal (including test_stable/Qwen) is running."""
    try:
        from ..utils import sg_modal_active
        return sg_modal_active(bpy.context)
    except Exception:  # noqa: BLE001
        return False


def _is_baking():
    """True if BakeTextures modal operator is currently running."""
    try:
        for window in bpy.context.window_manager.windows:
            for op in window.modal_operators:
                if op.bl_idname == 'OBJECT_OT_bake_textures':
                    return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _get_view3d_override():
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            for region in area.regions:
                                if region.type == 'WINDOW':
                                    return dict(window=window,
                                                screen=window.screen,
                                                area=area,
                                                space_data=space,
                                                region=region)
    except Exception:  # noqa: BLE001
        pass
    return {}


# ── Scene isolation ───────────────────────────────────────────────────────────

def _hide_scene_for_generation(state):
    """Hide all pre-existing objects so Qwen/baker only sees the new model.

    Uses hide_set(True) which affects hide_get() — the only flag checked by
    StableGen's texture generator and BakeTextures when selecting objects.
    """
    saved = {}
    try:
        pre_ids = state['pre_objects']
        for obj in bpy.context.scene.objects:
            if id(obj) in pre_ids:
                saved[obj.name] = (obj.hide_render, obj.hide_viewport, obj.hide_get())
                obj.hide_render   = True
                obj.hide_viewport = True
                obj.hide_set(True)
    except Exception as exc:  # noqa: BLE001
        print(f"[BatchGen] Warning: isolation error: {exc}")
    state['isolation_state'] = saved
    if saved:
        print(f"[BatchGen] Isolated {len(saved)} pre-existing object(s)")


def _restore_scene_visibility(state):
    """Restore visibility of pre-existing objects after generation/baking."""
    saved = state.get('isolation_state', {})
    restored = 0
    try:
        for name, (orig_render, orig_viewport, orig_hidden) in saved.items():
            if name in bpy.data.objects:
                obj = bpy.data.objects[name]
                obj.hide_render   = orig_render
                obj.hide_viewport = orig_viewport
                obj.hide_set(orig_hidden)
                restored += 1
    except Exception as exc:  # noqa: BLE001
        print(f"[BatchGen] Warning: restore error: {exc}")
    state['isolation_state'] = {}
    if restored:
        print(f"[BatchGen] Restored {restored} object(s) visibility")


# ── Scene cleanup ─────────────────────────────────────────────────────────────

def _stash_and_cleanup(new_obj_ids, pre_ids):
    """Move generated meshes to 'Batch Results' collection and delete new cameras."""
    try:
        scene = bpy.context.scene

        # ── 1. Move mesh objects to Batch Results collection ──────────────────
        if BATCH_COLLECTION not in bpy.data.collections:
            batch_coll = bpy.data.collections.new(BATCH_COLLECTION)
            scene.collection.children.link(batch_coll)
        else:
            batch_coll = bpy.data.collections[BATCH_COLLECTION]

        new_objs  = [obj for obj in scene.objects if id(obj) in new_obj_ids]
        mesh_objs = [o for o in new_objs if o.type == 'MESH']
        moved = 0
        for obj in mesh_objs:
            for coll in list(obj.users_collection):
                coll.objects.unlink(obj)
            batch_coll.objects.link(obj)
            obj.hide_viewport = False
            obj.hide_set(False)     # eye icon ON → visible in outliner
            obj.hide_render   = True  # not renderable / not selected by baker
            moved += 1

        # Hide collection in viewport but leave accessible via outliner
        def _find_layer_coll(lc, name):
            if lc.name == name:
                return lc
            for child in lc.children:
                found = _find_layer_coll(child, name)
                if found:
                    return found
            return None

        lc = _find_layer_coll(bpy.context.view_layer.layer_collection, BATCH_COLLECTION)
        if lc:
            lc.hide_viewport = True

        if moved:
            print(f"[BatchGen] Moved {moved} mesh(es) to hidden '{BATCH_COLLECTION}' collection")

        # ── 2. Delete cameras that appeared during this generation ─────────────
        new_cams = [obj for obj in scene.objects
                    if id(obj) not in pre_ids and obj.type == 'CAMERA']
        for cam in new_cams:
            bpy.data.objects.remove(cam, do_unlink=True)
        if new_cams:
            print(f"[BatchGen] Deleted {len(new_cams)} camera(s)")

    except Exception as exc:  # noqa: BLE001
        print(f"[BatchGen] Warning: cleanup error: {exc}")
        import traceback
        traceback.print_exc()


# ── Log helpers ───────────────────────────────────────────────────────────────

def _write_log_header(state):
    try:
        with open(state['log_path'], 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("StableGen Batch Log\n")
            f.write(f"Folder:  {os.path.dirname(state['images'][0])}\n")
            f.write(f"Images:  {state['total']}\n")
            f.write(f"Started: {state['batch_start_str']}\n")
            f.write("=" * 60 + "\n\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[BatchGen] Warning: could not write log header: {exc}")


def _write_log_entry(state, entry):
    try:
        with open(state['log_path'], 'a', encoding='utf-8') as f:
            ts = entry.get('timestamp', '')
            f.write(f"[{entry['index']}/{state['total']}] {entry['name']}"
                    + (f"  ({ts})" if ts else "") + "\n")
            if 'shape_duration'   in entry: f.write(f"  Shape:   {entry['shape_duration']}\n")
            if 'texture_duration' in entry: f.write(f"  Texture: {entry['texture_duration']}\n")
            if 'bake_duration'    in entry: f.write(f"  Bake:    {entry['bake_duration']}\n")
            f.write(f"  Total:   {entry['total_duration']}\n\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[BatchGen] Warning: could not write log entry: {exc}")


def _write_log_footer(state, cancelled=False):
    try:
        total_secs = time.time() - state['batch_start']
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(state['log_path'], 'a', encoding='utf-8') as f:
            f.write("─" * 60 + "\n")
            if cancelled:
                done = len(state['log_entries'])
                f.write(f"Batch CANCELLED after {done}/{state['total']} model(s).\n")
            else:
                f.write(f"Batch complete! {state['total']} model(s) processed.\n")
            f.write(f"Total time: {_fmt_duration(total_secs)}\n")
            f.write(f"Completed:  {now}\n")
        print(f"[BatchGen] Log saved to: {state['log_path']}")
    except Exception as exc:  # noqa: BLE001
        print(f"[BatchGen] Warning: could not write log footer: {exc}")


# ── Core batch logic ──────────────────────────────────────────────────────────

def _rename_new_objects(image_path, pre_ids):
    """Rename new mesh objects to match the input filename stem.

    Returns a set of id() values for the renamed mesh objects.
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    try:
        scene    = bpy.context.scene
        new_objs = [obj for obj in scene.objects if id(obj) not in pre_ids]
        mesh_objs = [o for o in new_objs if o.type == 'MESH'] or new_objs
        if len(mesh_objs) == 1:
            mesh_objs[0].name = stem
            if mesh_objs[0].data:
                mesh_objs[0].data.name = stem
        else:
            for i, obj in enumerate(mesh_objs):
                obj.name = f"{stem}_{i + 1:02d}"
                if obj.data:
                    obj.data.name = f"{stem}_{i + 1:02d}"
        print(f"[BatchGen] Renamed {len(mesh_objs)} object(s) to '{stem}...'")
        return {id(o) for o in mesh_objs}
    except Exception as exc:  # noqa: BLE001
        print(f"[BatchGen] Warning: rename error: {exc}")
        return set()


def _trigger_bake(state):
    """Invoke BakeTextures on the newly generated objects."""
    from ..texturing.rendering import BakeTextures

    try:
        pre_ids  = state['pre_objects']
        new_objs = [o for o in bpy.context.scene.objects
                    if id(o) not in pre_ids and o.type == 'MESH']
        if not new_objs:
            new_objs = [o for o in bpy.context.view_layer.objects
                        if o.type == 'MESH' and not o.hide_get()]
        if not new_objs:
            print("[BatchGen] No mesh objects to bake, skipping.")
            return False

        BakeTextures._objects       = new_objs
        BakeTextures._current_index = 0
        BakeTextures._phase         = 'unwrap'
        BakeTextures._total_objects = len(new_objs)
        BakeTextures._stage         = "Preparing"

        bake_kwargs = dict(
            bake_pbr           = state['bake_pbr'],
            texture_resolution = state['bake_resolution'],
            try_unwrap         = state['bake_try_unwrap'],
            overlap_only       = state['bake_overlap_only'],
            export_orm         = state['bake_export_orm'],
            normal_convention  = state['bake_normal_convention'],
            add_material       = state['bake_add_material'],
            flatten_for_refine = state['bake_flatten'],
        )
        ctx = _get_view3d_override()
        if ctx:
            with bpy.context.temp_override(**ctx):
                result = bpy.ops.object.bake_textures('EXEC_DEFAULT', **bake_kwargs)
        else:
            result = bpy.ops.object.bake_textures('EXEC_DEFAULT', **bake_kwargs)

        if result == {'RUNNING_MODAL'}:
            state['bake_start'] = time.time()
            print(f"[BatchGen] Baking {len(new_objs)} object(s)...")
            return True

        print(f"[BatchGen] Bake returned: {result}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[BatchGen] Bake error: {exc}")
        import traceback
        traceback.print_exc()
        return False


def _trigger_next():
    """Set up isolation, set input image, and invoke Trellis2Generate."""
    state = _batch_state
    image_path = state['images'][state['index']]

    # Snapshot current object ids so we can identify new ones after generation
    try:
        state['pre_objects'] = {id(obj) for obj in bpy.context.scene.objects}
    except Exception:  # noqa: BLE001
        state['pre_objects'] = set()

    # Hide pre-existing objects so Qwen/baker only sees the new model
    _hide_scene_for_generation(state)

    try:
        bpy.context.scene.trellis2_input_image = image_path
        print(f"[BatchGen] [{state['index'] + 1}/{state['total']}] "
              f"{os.path.basename(image_path)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[BatchGen] Error setting input image: {exc}")
        state['cancelled'] = True
        return

    try:
        state['model_start']      = time.time()
        state['thread_done']      = 0.0
        state['gen_done']         = 0.0
        state['bake_start']       = 0.0
        state['gen_settle_count'] = 0
        state['_new_obj_ids']     = set()
        bpy.ops.object.trellis2_generate('EXEC_DEFAULT')
        state['phase']        = 'settling'
        state['settle_count'] = 0
    except Exception as exc:  # noqa: BLE001
        print(f"[BatchGen] Error invoking generate for image "
              f"{state['index'] + 1}: {exc}, skipping")
        state['_new_obj_ids'] = set()
        _restore_scene_visibility(state)


def _stop_batch(cancelled=False):
    """Cleanly stop the batch timer and write log footer."""
    _batch_state['active']    = False
    _batch_state['cancelled'] = False
    _sync_wm()
    if _batch_state['log_path']:
        _write_log_footer(_batch_state, cancelled=cancelled)
    # Unhide Batch Results collection so finished models are immediately visible
    try:
        def _find_lc(lc, name):
            if lc.name == name:
                return lc
            for child in lc.children:
                found = _find_lc(child, name)
                if found:
                    return found
            return None
        lc = _find_lc(bpy.context.view_layer.layer_collection, BATCH_COLLECTION)
        if lc:
            lc.hide_viewport = False
    except Exception:  # noqa: BLE001
        pass
    print("[BatchGen] Batch stopped" + (" (cancelled)" if cancelled else " (complete)"))


def _advance_to_next(state):
    """Stash current model, restore scene, then start next image or finalise."""
    _stash_and_cleanup(state.get('_new_obj_ids', set()), state['pre_objects'])
    _restore_scene_visibility(state)

    next_index = state['index'] + 1
    if next_index >= state['total']:
        _stop_batch(cancelled=False)
        return

    state['index'] = next_index
    _sync_wm()
    _trigger_next()


# ── Timer callback ────────────────────────────────────────────────────────────

def _batch_tick():
    """bpy.app.timers callback – drives the batch state machine."""
    from .trellis2 import Trellis2Generate  # late import to avoid circular dep

    state = _batch_state

    # ── Cancelled / stopped ───────────────────────────────────────────────────
    if state['cancelled'] or not state['active']:
        _restore_scene_visibility(state)
        _stop_batch(cancelled=state['cancelled'])
        return None  # unregister timer

    phase = state['phase']

    # ── Idle: first tick → advance to first generation ────────────────────────
    if phase == 'idle':
        _advance_to_next(state)
        return _TICK_INTERVAL

    # ── Settling: waiting for Trellis2Generate to start running ───────────────
    if phase == 'settling':
        state['settle_count'] += 1
        if Trellis2Generate._is_running:
            state['phase'] = 'generating'
        elif state['settle_count'] >= _SETTLE_TICKS:
            # Timed out — skip this image
            print(f"[BatchGen] Image {state['index'] + 1} failed to start, skipping")
            state['phase'] = 'idle'
            _restore_scene_visibility(state)
        return _TICK_INTERVAL

    # ── Generating: wait for background thread (shape + GLB) to finish ────────
    if phase == 'generating':
        if Trellis2Generate._is_running:
            return _TICK_INTERVAL
        if state['thread_done'] == 0.0:
            state['thread_done'] = time.time()
            print("[BatchGen] Shape/mesh done. Waiting for Qwen texturing...")
        state['phase']            = 'texturing'
        state['gen_settle_count'] = 0
        return _TICK_INTERVAL

    # ── Texturing: wait for Qwen/test_stable to finish ────────────────────────
    # Require _GEN_SETTLE_TICKS consecutive False readings to avoid the brief
    # gap between trellis2_generate finishing and test_stable starting.
    if phase == 'texturing':
        if _sg_modal_active():
            state['gen_settle_count'] = 0
            return _TICK_INTERVAL

        state['gen_settle_count'] += 1
        if state['gen_settle_count'] < _GEN_SETTLE_TICKS:
            return _TICK_INTERVAL

        # Generation (shape + Qwen texture) is truly complete
        state['gen_done'] = time.time()
        img_path = state['images'][state['index']]

        try:
            had_error = bpy.context.scene.sg_last_gen_error
        except Exception:  # noqa: BLE001
            had_error = False

        # Rename new meshes and record their ids for stash
        if not had_error and state['rename_meshes']:
            state['_new_obj_ids'] = _rename_new_objects(img_path, state['pre_objects'])
        else:
            state['_new_obj_ids'] = {
                id(o) for o in bpy.context.scene.objects
                if id(o) not in state['pre_objects'] and o.type == 'MESH'
            }

        shape_secs   = (state['thread_done'] - state['model_start']) if state['thread_done'] else 0
        texture_secs = (state['gen_done']    - state['thread_done']) if state['thread_done'] else 0

        entry = {
            'index':            state['index'] + 1,
            'name':             os.path.basename(img_path),
            'timestamp':        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'shape_duration':   _fmt_duration(shape_secs),
            'texture_duration': _fmt_duration(texture_secs),
        }
        state['_current_entry'] = entry

        if not had_error and state['bake_textures']:
            if _trigger_bake(state):
                state['phase'] = 'baking'
                return _TICK_INTERVAL
            print("[BatchGen] Bake failed to start, skipping bake for this model.")

        # No bake (or bake skipped) – log and advance
        entry['total_duration'] = _fmt_duration(time.time() - state['model_start'])
        state['log_entries'].append(entry)
        _write_log_entry(state, entry)

        if state['cancelled']:
            _restore_scene_visibility(state)
            _stop_batch(cancelled=True)
            return None

        _advance_to_next(state)
        return _TICK_INTERVAL

    # ── Baking: wait for BakeTextures modal to finish ─────────────────────────
    if phase == 'baking':
        if _is_baking():
            return _TICK_INTERVAL

        bake_secs  = time.time() - state['bake_start']
        total_secs = time.time() - state['model_start']
        entry = state.get('_current_entry') or {}
        entry['bake_duration']  = _fmt_duration(bake_secs)
        entry['total_duration'] = _fmt_duration(total_secs)
        state['log_entries'].append(entry)
        _write_log_entry(state, entry)
        print(f"[BatchGen] Bake done ({_fmt_duration(bake_secs)})")

        if state['cancelled']:
            _restore_scene_visibility(state)
            _stop_batch(cancelled=True)
            return None

        _advance_to_next(state)
        return _TICK_INTERVAL

    return _TICK_INTERVAL


# ── Operators ─────────────────────────────────────────────────────────────────

class TRELLIS2_OT_BatchSelectFolder(bpy.types.Operator):
    """Select a folder of images for batch TRELLIS.2 generation"""
    bl_idname  = "object.trellis2_batch_select_folder"
    bl_label   = "Select Image Folder for Batch"
    bl_options = {'REGISTER'}

    directory: bpy.props.StringProperty(subtype='DIR_PATH')  # type: ignore

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        folder = self.directory.rstrip('/\\')
        context.scene.trellis2_batch_folder = folder
        images = _scan_images(folder)
        context.scene.trellis2_batch_count = len(images)
        if images:
            context.scene.trellis2_input_image = images[0]
            self.report({'INFO'}, f"Found {len(images)} image(s) in folder")
        else:
            self.report({'WARNING'}, "No supported images found in folder")
        return {'FINISHED'}


class TRELLIS2_OT_BatchGenerate(bpy.types.Operator):
    """Generate 3D models for all images in the selected batch folder"""
    bl_idname  = "object.trellis2_batch_generate"
    bl_label   = "Generate Batch"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        if _batch_state['active']:
            return False
        from .trellis2 import Trellis2Generate
        if Trellis2Generate._is_running:
            return False
        folder = getattr(context.scene, 'trellis2_batch_folder', '')
        count  = getattr(context.scene, 'trellis2_batch_count', 0)
        return bool(folder) and count > 0

    def execute(self, context):
        folder = context.scene.trellis2_batch_folder
        images = _scan_images(folder)
        if not images:
            self.report({'ERROR'}, "No images found in batch folder")
            return {'CANCELLED'}

        now       = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        log_path  = os.path.join(folder, f"stablegen_batch_{timestamp}.txt")

        _batch_state.update({
            'active':             True,
            'cancelled':          False,
            'images':             images,
            'index':              -1,
            'total':              len(images),
            'rename_meshes':      getattr(context.scene, 'trellis2_batch_rename_meshes', True),
            'bake_textures':      getattr(context.scene, 'trellis2_batch_bake_textures', True),
            'bake_pbr':           getattr(context.scene, 'trellis2_batch_bake_pbr', True),
            'bake_resolution':    getattr(context.scene, 'trellis2_batch_bake_resolution', 2048),
            'bake_try_unwrap':    getattr(context.scene, 'trellis2_batch_bake_try_unwrap', 'smart'),
            'bake_overlap_only':  getattr(context.scene, 'trellis2_batch_bake_overlap_only', False),
            'bake_export_orm':    getattr(context.scene, 'trellis2_batch_bake_export_orm', False),
            'bake_normal_convention': getattr(context.scene, 'trellis2_batch_bake_normal_convention', 'opengl'),
            'bake_add_material':  getattr(context.scene, 'trellis2_batch_bake_add_material', True),
            'bake_flatten':       getattr(context.scene, 'trellis2_batch_bake_flatten', False),
            'pre_objects':        set(),
            'isolation_state':    {},
            'phase':              'idle',
            'settle_count':       0,
            'gen_settle_count':   0,
            'model_start':        0.0,
            'thread_done':        0.0,
            'gen_done':           0.0,
            'bake_start':         0.0,
            'batch_start':        time.time(),
            'batch_start_str':    now.strftime("%Y-%m-%d %H:%M:%S"),
            'log_path':           log_path,
            'log_entries':        [],
            '_current_entry':     None,
            '_new_obj_ids':       set(),
        })
        _write_log_header(_batch_state)
        _sync_wm()
        print(f"[BatchGen] Starting: {len(images)} image(s) from '{folder}'")
        bpy.app.timers.register(_batch_tick, first_interval=0.1)
        return {'FINISHED'}


class TRELLIS2_OT_BatchCancel(bpy.types.Operator):
    """Cancel the entire batch queue immediately"""
    bl_idname  = "object.trellis2_batch_cancel"
    bl_label   = "Cancel Batch"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _batch_state['active']

    def execute(self, context):
        _batch_state['cancelled'] = True
        # Also signal Trellis2Generate to stop immediately
        try:
            from .trellis2 import Trellis2Generate
            Trellis2Generate._cancelled = True
        except Exception:  # noqa: BLE001
            pass
        self.report({'WARNING'}, "Batch cancelled.")
        return {'FINISHED'}


class TRELLIS2_OT_BatchClear(bpy.types.Operator):
    """Clear the batch folder selection"""
    bl_idname  = "object.trellis2_batch_clear"
    bl_label   = "Clear Batch Folder"
    bl_options = {'REGISTER'}

    def execute(self, context):
        context.scene.trellis2_batch_folder = ""
        context.scene.trellis2_batch_count  = 0
        return {'FINISHED'}


class TRELLIS2_OT_BatchBakeSettings(bpy.types.Operator):
    """Configure bake parameters used during batch generation"""
    bl_idname  = "object.trellis2_batch_bake_settings"
    bl_label   = "Batch Bake Settings"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        scene  = context.scene
        layout = self.layout
        layout.prop(scene, "trellis2_batch_bake_resolution")
        layout.prop(scene, "trellis2_batch_bake_try_unwrap")
        layout.prop(scene, "trellis2_batch_bake_overlap_only")
        layout.prop(scene, "trellis2_batch_bake_add_material")
        layout.prop(scene, "trellis2_batch_bake_flatten")
        layout.separator()
        layout.label(text="PBR Export:")
        layout.prop(scene, "trellis2_batch_bake_pbr")
        pbr_col = layout.column()
        pbr_col.enabled = getattr(scene, 'trellis2_batch_bake_pbr', True)
        pbr_col.prop(scene, "trellis2_batch_bake_export_orm")
        pbr_col.prop(scene, "trellis2_batch_bake_normal_convention")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=350)

    def execute(self, context):
        # Parameters stored on scene properties — nothing extra needed
        return {'FINISHED'}


batch_classes = [
    TRELLIS2_OT_BatchSelectFolder,
    TRELLIS2_OT_BatchGenerate,
    TRELLIS2_OT_BatchCancel,
    TRELLIS2_OT_BatchClear,
    TRELLIS2_OT_BatchBakeSettings,
]


def unregister_batch():
    """Stop any running batch timer. Called from addon unregister."""
    _batch_state['cancelled'] = True
    _batch_state['active']    = False
    try:
        if bpy.app.timers.is_registered(_batch_tick):
            bpy.app.timers.unregister(_batch_tick)
    except Exception:  # noqa: BLE001
        pass
