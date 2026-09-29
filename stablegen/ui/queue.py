"""Scene-queue persistence and the SceneQueueItem PropertyGroup.

Contains:
- ``SceneQueueItem`` — PropertyGroup for a single queue entry
- ``_sg_queue_filepath`` / ``_sg_queue_save`` / ``_sg_queue_load``
- ``_sg_queue_load_handler`` — ``@persistent`` load_post handler
"""

import json
import os

import bpy  # pylint: disable=import-error
from bpy.app.handlers import persistent

from ..core import ADDON_PKG
from ..utils import sg_modal_active


# ── Constants ──────────────────────────────────────────────────────────────

_SG_QUEUE_FILENAME = "sg_scene_queue.json"


# ── Filepath helper ────────────────────────────────────────────────────────

def _sg_queue_filepath():
    """Return the path to the queue JSON file.

    Uses the addon output_dir if set; otherwise falls back to a
    ``.stablegen`` folder next to the addon package.
    """
    try:
        prefs = bpy.context.preferences.addons.get(ADDON_PKG)
        if prefs and prefs.preferences.output_dir:
            d = bpy.path.abspath(prefs.preferences.output_dir)
            if os.path.isdir(d):
                return os.path.join(d, _SG_QUEUE_FILENAME)
    except Exception:
        pass
    # Fallback: <addon_dir>/.stablegen/
    fallback = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".stablegen")
    os.makedirs(fallback, exist_ok=True)
    return os.path.join(fallback, _SG_QUEUE_FILENAME)


# ── Save / Load ────────────────────────────────────────────────────────────

def _sg_queue_save(processing=False, current_idx=0, phase='idle'):
    """Serialize the scene queue + processing state to a JSON file on disk."""
    wm = bpy.context.window_manager
    if not hasattr(wm, 'sg_scene_queue'):
        return
    items = [{"label": it.label,
              "scene_name": it.scene_name,
              "blend_file": it.blend_file,
              "prompt": it.prompt,
              "negative_prompt": it.negative_prompt,
              "status": it.status,
              "retries": it.retries,
              "error_reason": it.error_reason}
             for it in wm.sg_scene_queue]

    gif_settings = {}
    _GIF_KEYS = (
        'sg_queue_gif_export', 'sg_queue_gif_duration', 'sg_queue_gif_fps',
        'sg_queue_gif_resolution', 'sg_queue_gif_samples', 'sg_queue_gif_engine',
        'sg_queue_gif_interpolation', 'sg_queue_gif_use_hdri',
        'sg_queue_gif_hdri_path', 'sg_queue_gif_hdri_strength',
        'sg_queue_gif_hdri_rotation', 'sg_queue_gif_env_mode',
        'sg_queue_gif_denoiser', 'sg_queue_gif_use_gpu',
        'sg_queue_gif_also_no_pbr',
    )
    for key in _GIF_KEYS:
        if hasattr(wm, key):
            gif_settings[key] = getattr(wm, key)

    data = {
        "items": items,
        "processing": processing,
        "current_idx": current_idx,
        "phase": phase,
        "gif_settings": gif_settings,
    }
    try:
        fp = _sg_queue_filepath()
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[Queue] save error: {e}")


def _sg_queue_load():
    """Restore the scene queue from the JSON file on disk."""
    wm = bpy.context.window_manager
    if not hasattr(wm, 'sg_scene_queue'):
        return
    fp = _sg_queue_filepath()
    if not os.path.isfile(fp):
        return
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        if isinstance(raw, list):
            items = raw
            meta = {}
        else:
            items = raw.get("items", [])
            meta = raw

        wm.sg_scene_queue.clear()
        for entry in items:
            new_item = wm.sg_scene_queue.add()
            new_item.label = entry.get("label", "")
            new_item.scene_name = entry.get("scene_name", "")
            new_item.blend_file = entry.get("blend_file", "")
            new_item.prompt = entry.get("prompt", "")
            new_item.negative_prompt = entry.get("negative_prompt", "")
            new_item.status = entry.get("status", "pending")
            new_item.retries = entry.get("retries", 0)
            new_item.error_reason = entry.get("error_reason", "")
        wm.sg_scene_queue_index = min(0, len(wm.sg_scene_queue) - 1)

        gif_settings = meta.get("gif_settings", {})
        for key, value in gif_settings.items():
            if hasattr(wm, key):
                try:
                    setattr(wm, key, value)
                except Exception:
                    pass

        if meta.get("processing", False):
            try:
                idx = meta.get("current_idx", 0)
                _resume_queue(idx)
                print(f"[Queue] Auto-resuming from item {idx}")
            except Exception as e:
                print(f"[Queue] Failed to auto-resume: {e}")
    except Exception as e:
        print(f"[Queue] Failed to restore queue: {e}")


# ── Load-post handler ─────────────────────────────────────────────────────

@persistent
def _sg_queue_load_handler(dummy):
    """load_post handler: restore the queue when a .blend is opened."""
    def _deferred():
        _sg_queue_load()
        return None
    bpy.app.timers.register(_deferred, first_interval=0.3)


# ── PropertyGroup ──────────────────────────────────────────────────────────

class SceneQueueItem(bpy.types.PropertyGroup):
    """A single item in the scene generation queue."""
    label: bpy.props.StringProperty(name="Label", default="")  # type: ignore
    scene_name: bpy.props.StringProperty(name="Scene", default="")  # type: ignore
    blend_file: bpy.props.StringProperty(name="Blend File", default="")  # type: ignore
    prompt: bpy.props.StringProperty(name="Prompt", default="")  # type: ignore
    negative_prompt: bpy.props.StringProperty(name="Negative Prompt", default="")  # type: ignore
    status: bpy.props.StringProperty(name="Status", default="pending")  # type: ignore
    retries: bpy.props.IntProperty(name="Retries", default=0)  # type: ignore
    error_reason: bpy.props.StringProperty(name="Error Reason", default="")  # type: ignore


# =====================================================================
# Queue operators and timer (extracted from stablegen.py)
# =====================================================================

_queue_timer = None          # Reference to the active timer callback
_queue_processing = False    # Global flag: queue is being processed
_queue_current_idx = 0       # Index of the currently-processing item
_queue_phase = 'idle'        # 'idle' | 'switching' | 'running' | 'settling' | 'waiting_texturing' | 'exporting_gif'
_queue_settle_deadline = 0.0 # monotonic time after which 'settling' may finish
_queue_force_reload = False  # Force re-open of the .blend on next idle tick (for retries)
_queue_texturing_seen = False  # Set True once generation_status='running' is observed during settling
_queue_refresh_deadline = 0.0  # monotonic deadline while waiting for model-list refresh
_queue_original_pbr = False    # stored pbr_decomposition before no-PBR reproject
_queue_original_gen_method = 'sequential'  # stored generation_method before reproject
_queue_original_overwrite = True  # stored overwrite_material before reproject
_QUEUE_MAX_RETRIES = 3       # how many times a failed item is retried


class SG_UL_SceneQueueList(bpy.types.UIList):
    """UIList for the scene generation queue."""
    bl_idname = "SG_UL_SceneQueueList"

    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)

            # Label (or scene name as fallback)
            display = item.label if item.label else item.scene_name
            row.label(text=display, icon='SCENE_DATA')

            # Status icon + retry indicator
            if item.status == 'done':
                row.label(text="", icon='CHECKMARK')
            elif item.status == 'error':
                retry_txt = f"x{item.retries}" if item.retries else ""
                if item.error_reason:
                    reason_short = item.error_reason[:28] + "…" if len(item.error_reason) > 30 else item.error_reason
                    row.label(text=f"{reason_short} {retry_txt}".strip(), icon='ERROR')
                else:
                    row.label(text=retry_txt, icon='ERROR')
            elif item.status == 'processing':
                retry_txt = f"#{item.retries + 1}" if item.retries else ""
                row.label(text=retry_txt, icon='SORTTIME')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            display = item.label if item.label else item.scene_name
            layout.label(text=display, icon='SCENE_DATA')


def _persist_queue():
    """Write the queue + processing state to a JSON file on disk."""
    try:
        _sg_queue_save(
            processing=_queue_processing,
            current_idx=_queue_current_idx,
            phase=_queue_phase,
        )
    except Exception:
        pass


def _resume_queue(idx):
    """Resume queue processing from *idx* (called after a .blend switch).

    This is invoked by ``_sg_queue_load()`` in ``__init__.py`` when the
    JSON file indicates that processing was active.
    """
    global _queue_processing, _queue_current_idx, _queue_phase, _queue_timer

    _queue_processing = True
    _queue_current_idx = idx
    _queue_phase = 'idle'

    # Avoid duplicate timers
    if _queue_timer is not None:
        try:
            bpy.app.timers.unregister(_queue_tick)
        except Exception:
            pass

    _queue_timer = bpy.app.timers.register(_queue_tick, first_interval=1.5)
    print(f"[Queue] Resumed processing from item {idx}")
    _tag_redraw()


class SceneQueueAdd(bpy.types.Operator):
    """Add the current scene to the generation queue.

    Saves a *copy* of the current .blend into ``<output_dir>/queue_jobs/``.
    The queue item points at that copy so the original file is never
    modified by queue processing.
    """
    bl_idname = "stablegen.queue_add"
    bl_label = "Add to Queue"
    bl_description = "Snapshot the current .blend and add the scene to the batch queue"
    bl_options = {'REGISTER', 'UNDO'}

    item_label: bpy.props.StringProperty(
        name="Name",
        description="A short label for this queue entry",
        default="",
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        return not _queue_processing and not sg_modal_active(context)

    def invoke(self, context, event):
        scene = context.scene
        cur_prompt = getattr(scene, 'comfyui_prompt', '')
        # Pre-fill with a truncated prompt
        self.item_label = (cur_prompt[:40]) if cur_prompt else scene.name
        return context.window_manager.invoke_props_dialog(self, width=350)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "item_label")

    def execute(self, context):
        import time

        wm = context.window_manager
        scene = context.scene
        scene_name = scene.name
        cur_prompt = getattr(scene, 'comfyui_prompt', '')
        cur_neg = getattr(scene, 'comfyui_negative_prompt', '')
        label = self.item_label.strip() or scene_name

        # Determine queue_jobs directory
        prefs = context.preferences.addons.get(ADDON_PKG)
        output_dir = ''
        if prefs:
            output_dir = bpy.path.abspath(prefs.preferences.output_dir or '')
        if not output_dir or not os.path.isdir(output_dir):
            self.report({'ERROR'}, "Set an Output Directory in StableGen preferences first.")
            return {'CANCELLED'}

        jobs_dir = os.path.join(output_dir, "queue_jobs")
        os.makedirs(jobs_dir, exist_ok=True)

        # Build filename from the user-provided label
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_label = label.replace(" ", "_").replace("/", "_").replace("\\", "_")
        # Truncate to keep filenames reasonable
        safe_label = safe_label[:60]
        copy_name = f"{safe_label}_{timestamp}.blend"
        copy_path = os.path.join(jobs_dir, copy_name)

        # Save a copy — current file stays active & untouched
        # Snapshot the model_name into the plain-string backup property
        # BEFORE saving so the copy carries the correct value even if
        # the dynamic Enum resolves wrong on re-open (stale item cache).
        scene.sg_model_name_backup = scene.model_name
        for unit in scene.controlnet_units:
            if unit.model_name and unit.model_name not in ("REFRESH", "NO_ASSIGNED", "NO_PREFS"):
                unit.model_name_backup = unit.model_name
        try:
            bpy.ops.wm.save_as_mainfile(filepath=copy_path, copy=True)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to save queue snapshot: {e}")
            return {'CANCELLED'}

        new_item = wm.sg_scene_queue.add()
        new_item.label = label
        new_item.scene_name = scene_name
        new_item.blend_file = copy_path
        new_item.prompt = cur_prompt
        new_item.negative_prompt = cur_neg
        new_item.status = 'pending'
        wm.sg_scene_queue_index = len(wm.sg_scene_queue) - 1
        _persist_queue()

        self.report({'INFO'}, f"Queued '{label}' ({copy_name})")
        return {'FINISHED'}


class SceneQueueRemove(bpy.types.Operator):
    """Remove the selected scene from the queue"""
    bl_idname = "stablegen.queue_remove"
    bl_label = "Remove from Queue"

    @classmethod
    def poll(cls, context):
        wm = context.window_manager
        return not _queue_processing and len(wm.sg_scene_queue) > 0 and not sg_modal_active(context)

    def invoke(self, context, event):
        wm = context.window_manager
        idx = wm.sg_scene_queue_index
        if 0 <= idx < len(wm.sg_scene_queue):
            item = wm.sg_scene_queue[idx]
            display = item.label or item.scene_name
            # Show a confirmation popup
            return context.window_manager.invoke_confirm(self, event,
                message=f"Remove '{display}' from queue?",
                confirm_text="Remove")
        return {'CANCELLED'}

    def execute(self, context):
        wm = context.window_manager
        idx = wm.sg_scene_queue_index
        if 0 <= idx < len(wm.sg_scene_queue):
            item = wm.sg_scene_queue[idx]
            display = item.label or item.scene_name
            # Delete the snapshot .blend if the item hasn't been processed
            if item.status != 'done' and item.blend_file:
                try:
                    blend_path = item.blend_file
                    if os.path.isfile(blend_path):
                        os.remove(blend_path)
                        print(f"[Queue] Deleted snapshot: {blend_path}")
                except Exception as e:
                    print(f"[Queue] Could not delete snapshot: {e}")
            wm.sg_scene_queue.remove(idx)
            wm.sg_scene_queue_index = min(idx, len(wm.sg_scene_queue) - 1)
            _persist_queue()
            self.report({'INFO'}, f"Removed '{display}' from queue.")
        return {'FINISHED'}


class SceneQueueClear(bpy.types.Operator):
    """Clear the entire queue"""
    bl_idname = "stablegen.queue_clear"
    bl_label = "Clear Queue"

    @classmethod
    def poll(cls, context):
        wm = context.window_manager
        return not _queue_processing and len(wm.sg_scene_queue) > 0 and not sg_modal_active(context)

    def invoke(self, context, event):
        wm = context.window_manager
        count = len(wm.sg_scene_queue)
        return context.window_manager.invoke_confirm(self, event,
            message=f"Clear all {count} item(s) from queue?",
            confirm_text="Clear All")

    def execute(self, context):
        wm = context.window_manager
        # Delete snapshot .blends for unprocessed items
        for item in wm.sg_scene_queue:
            if item.status != 'done' and item.blend_file:
                try:
                    if os.path.isfile(item.blend_file):
                        os.remove(item.blend_file)
                        print(f"[Queue] Deleted snapshot: {item.blend_file}")
                except Exception as e:
                    print(f"[Queue] Could not delete snapshot: {e}")
        wm.sg_scene_queue.clear()
        wm.sg_scene_queue_index = 0
        _persist_queue()
        self.report({'INFO'}, "Queue cleared.")
        return {'FINISHED'}


class SceneQueueMoveUp(bpy.types.Operator):
    """Move the selected queue item up"""
    bl_idname = "stablegen.queue_move_up"
    bl_label = "Move Up"

    @classmethod
    def poll(cls, context):
        wm = context.window_manager
        return not _queue_processing and wm.sg_scene_queue_index > 0 and not sg_modal_active(context)

    def execute(self, context):
        wm = context.window_manager
        idx = wm.sg_scene_queue_index
        wm.sg_scene_queue.move(idx, idx - 1)
        wm.sg_scene_queue_index -= 1
        return {'FINISHED'}


class SceneQueueMoveDown(bpy.types.Operator):
    """Move the selected queue item down"""
    bl_idname = "stablegen.queue_move_down"
    bl_label = "Move Down"

    @classmethod
    def poll(cls, context):
        wm = context.window_manager
        return (not _queue_processing
                and wm.sg_scene_queue_index < len(wm.sg_scene_queue) - 1
                and not sg_modal_active(context))

    def execute(self, context):
        wm = context.window_manager
        idx = wm.sg_scene_queue_index
        wm.sg_scene_queue.move(idx, idx + 1)
        wm.sg_scene_queue_index += 1
        return {'FINISHED'}


class SceneQueueOpenResult(bpy.types.Operator):
    """Open the .blend snapshot for the selected queue item"""
    bl_idname = "stablegen.queue_open_result"
    bl_label = "Open"
    bl_description = "Open the .blend snapshot of the selected queue item"

    @classmethod
    def poll(cls, context):
        if sg_modal_active(context):
            return False
        wm = context.window_manager
        idx = wm.sg_scene_queue_index
        if idx < 0 or idx >= len(wm.sg_scene_queue):
            return False
        item = wm.sg_scene_queue[idx]
        return bool(item.blend_file) and os.path.isfile(item.blend_file)

    def execute(self, context):
        wm = context.window_manager
        item = wm.sg_scene_queue[wm.sg_scene_queue_index]
        filepath = item.blend_file
        try:
            bpy.ops.wm.open_mainfile(filepath=filepath)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to open file: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


class SceneQueueInvalidate(bpy.types.Operator):
    """Reset a processed queue item back to pending and clear its scene"""
    bl_idname = "stablegen.queue_invalidate"
    bl_label = "Reset to Pending"
    bl_description = (
        "Reset the selected item to unprocessed, "
        "removing generated meshes and cameras from its snapshot "
        "so it can be re-processed"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        wm = context.window_manager
        idx = wm.sg_scene_queue_index
        if _queue_processing:
            cls.poll_message_set("Queue is currently processing")
            return False
        if sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        if idx < 0 or idx >= len(wm.sg_scene_queue):
            cls.poll_message_set("No queue item selected")
            return False
        item = wm.sg_scene_queue[idx]
        if item.status not in ('done', 'error'):
            cls.poll_message_set("Only completed or failed items can be reset")
            return False
        if not item.blend_file or not os.path.isfile(item.blend_file):
            cls.poll_message_set("Snapshot .blend file not found")
            return False
        return True

    def invoke(self, context, event):
        wm = context.window_manager
        idx = wm.sg_scene_queue_index
        item = wm.sg_scene_queue[idx]
        display = item.label or item.scene_name
        return context.window_manager.invoke_confirm(
            self, event,
            message=f"Reset '{display}'?\nThis will remove generated meshes and cameras from its snapshot and mark it as unprocessed.",
            confirm_text="Reset",
        )

    def execute(self, context):
        wm = context.window_manager
        idx = wm.sg_scene_queue_index
        item = wm.sg_scene_queue[idx]
        blend_path = item.blend_file
        display = item.label or item.scene_name

        # Open the processed .blend, clean it, save, then re-open current
        current_file = bpy.data.filepath

        try:
            bpy.ops.wm.open_mainfile(filepath=blend_path)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to open snapshot: {e}")
            return {'CANCELLED'}

        # Remove generated objects only if in trellis2 mode; preserve source meshes/cameras in texturing
        scene = bpy.context.scene
        arch_mode = getattr(scene, 'architecture_mode', 'sdxl')
        if arch_mode == 'trellis2':
            to_remove = []
            for obj in scene.objects:
                if obj.get("sg_generated") or obj.name.lower().startswith("trellis2") or obj.type == 'EMPTY':
                    to_remove.append(obj)
            for obj in to_remove:
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except Exception as err:
                    print(f"[Queue] Failed to remove generated object {obj.name}: {err}")

        # Purge orphaned data
        for block_attr in ('meshes', 'cameras', 'materials', 'images'):
            collection = getattr(bpy.data, block_attr, None)
            if collection is None:
                continue
            orphans = [b for b in collection if b.users == 0]
            for orphan in orphans:
                collection.remove(orphan)

        # Save the cleaned .blend in place
        try:
            bpy.ops.wm.save_mainfile()
            print(f"[Queue] Invalidated and saved: {blend_path}")
        except Exception as e:
            print(f"[Queue] Save warning during invalidate: {e}")

        # Mark the item as pending directly in the queue JSON file.
        # After open_mainfile the in-memory queue collection is empty
        # (it's restored 0.3 s later by the deferred load handler), so
        # patching the JSON is the only reliable approach.
        try:
            fp = _sg_queue_filepath()
            if os.path.isfile(fp):
                import json as _json
                with open(fp, 'r', encoding='utf-8') as f:
                    data = _json.load(f)
                items_list = data.get("items", data) if isinstance(data, dict) else data
                if isinstance(items_list, list) and idx < len(items_list):
                    items_list[idx]["status"] = "pending"
                    items_list[idx]["retries"] = 0
                    items_list[idx]["error_reason"] = ""
                with open(fp, 'w', encoding='utf-8') as f:
                    _json.dump(data, f)
                print(f"[Queue] Marked item {idx} as pending in queue JSON")
        except Exception as e:
            print(f"[Queue] Warning: could not update queue JSON: {e}")

        # Re-open the original file (if we had one)
        if current_file and os.path.isfile(current_file):
            try:
                bpy.ops.wm.open_mainfile(filepath=current_file)
            except Exception:
                pass

        self.report({'INFO'}, f"Reset '{display}' — ready for re-processing")
        return {'FINISHED'}


class SceneQueueProcess(bpy.types.Operator):
    """Start or cancel batch processing of the scene queue"""
    bl_idname = "stablegen.queue_process"
    bl_label = "Process Queue"
    bl_description = "Process all queued scenes sequentially"

    @classmethod
    def poll(cls, context):
        wm = context.window_manager
        if _queue_processing:
            return True  # Allow clicking to cancel
        if len(wm.sg_scene_queue) == 0:
            cls.poll_message_set("Queue is empty — add scenes first")
            return False
        if sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        return True

    def execute(self, context):
        global _queue_processing, _queue_timer, _queue_current_idx, _queue_phase

        if _queue_processing:
            # Cancel
            _queue_processing = False
            _queue_phase = 'idle'
            _persist_queue()  # Save stopped state so load_post won't auto-resume
            print("[Queue] Cancelled by user.")
            self.report({'WARNING'}, "Queue processing cancelled.")
            return {'FINISHED'}

        # Reset statuses
        wm = context.window_manager
        for item in wm.sg_scene_queue:
            if item.status != 'done':
                item.status = 'pending'

        _queue_processing = True
        _queue_current_idx = 0
        _queue_phase = 'idle'

        # Find first pending item
        while _queue_current_idx < len(wm.sg_scene_queue):
            if wm.sg_scene_queue[_queue_current_idx].status == 'pending':
                break
            _queue_current_idx += 1

        if _queue_current_idx >= len(wm.sg_scene_queue):
            _queue_processing = False
            self.report({'INFO'}, "No pending items in queue.")
            return {'FINISHED'}

        # Start the timer-based queue driver
        _queue_timer = bpy.app.timers.register(_queue_tick, first_interval=0.5)
        self.report({'INFO'}, f"Queue processing started ({len(wm.sg_scene_queue)} items).")
        return {'FINISHED'}


def _queue_tick():
    """Timer callback that drives queue advancement.

    Returns:
        float: Seconds until next tick, or None to stop.
    """
    import time as _time
    global _queue_processing, _queue_current_idx, _queue_phase, _queue_timer
    global _queue_settle_deadline, _queue_force_reload, _queue_texturing_seen
    global _queue_refresh_deadline

    if not _queue_processing:
        _queue_phase = 'idle'
        _queue_timer = None
        _tag_redraw()
        return None  # Stop timer

    wm = bpy.context.window_manager
    queue = wm.sg_scene_queue

    if _queue_current_idx >= len(queue):
        # All done
        _queue_processing = False
        _queue_phase = 'idle'
        _queue_timer = None
        print("[Queue] All items processed.")
        _tag_redraw()
        return None

    item = queue[_queue_current_idx]

    # ── Phase: IDLE → switch to the next scene and start ──
    if _queue_phase == 'idle':
        # Verify the item belongs to the currently-open .blend
        current_blend = bpy.data.filepath or "<unsaved>"
        need_open = (item.blend_file and item.blend_file != current_blend)

        # On retry, always re-open the .blend to restore clean state
        is_retry_reload = False
        if _queue_force_reload and item.blend_file:
            need_open = True
            is_retry_reload = True
            _queue_force_reload = False

        if need_open:
            # Need to open a different .blend file
            target = item.blend_file
            if not os.path.isfile(target):
                print(f"[Queue] Blend file not found: {target} — skipping.")
                item.status = 'error'
                item.error_reason = "Blend file not found"
                _queue_current_idx += 1
                _persist_queue()
                _tag_redraw()
                return 0.5

            # Save current file first so nothing is lost — BUT skip save
            # on retry reloads; we want to discard dirty state and revert
            # to the original clean copy.
            if bpy.data.filepath and not is_retry_reload:
                try:
                    bpy.ops.wm.save_mainfile()
                except Exception as e:
                    print(f"[Queue] Warning: could not save current file: {e}")

            # Persist state so the load_post handler can resume after
            # the new file is opened.
            _persist_queue()
            print(f"[Queue] Opening '{os.path.basename(target)}' for item "
                  f"'{item.scene_name}'...")

            # Open the target file.  After this call the current WM
            # data (and our timer) are gone — the load_post handler
            # will restore the queue and call _resume_queue().
            try:
                bpy.ops.wm.open_mainfile(filepath=target)
            except Exception as e:
                print(f"[Queue] Failed to open '{target}': {e}")
                # Can't recover gracefully — stop processing
                _queue_processing = False
                _queue_phase = 'idle'
                _queue_timer = None
                _persist_queue()
                _tag_redraw()
            return None  # Stop timer — load_post will restart it

        scene_ref = bpy.data.scenes.get(item.scene_name)
        if not scene_ref:
            print(f"[Queue] Scene '{item.scene_name}' not found — skipping.")
            item.status = 'error'
            item.error_reason = "Scene not found"
            _queue_current_idx += 1
            _tag_redraw()
            return 0.5  # Try next immediately

        # Switch to the target scene
        bpy.context.window.scene = scene_ref
        item.status = 'processing'
        _queue_phase = 'switching'
        print(f"[Queue] Switched to scene '{item.scene_name}'")
        _tag_redraw()
        return 1.0  # Give Blender a moment to digest

    # ── Phase: SWITCHING → invoke the generation operator ──
    if _queue_phase == 'switching':
        scene_ref = bpy.context.scene
        arch_mode = getattr(scene_ref, 'architecture_mode', 'sdxl')

        # Wait for any in-flight checkpoint / LoRA refreshes to finish so
        # ``model_name`` resolves to the correct value.  The load_post
        # handler triggers an async refresh when the cached architecture
        # doesn't match the file that was just opened.
        from ..core import state as _state
        if _state._pending_refreshes > 0:
            if _queue_refresh_deadline == 0.0:
                _queue_refresh_deadline = _time.monotonic() + 15.0
            if _time.monotonic() < _queue_refresh_deadline:
                return 1.0  # Re-check on next tick
            else:
                print("[Queue] Model-list refresh timed out — proceeding anyway")
        _queue_refresh_deadline = 0.0  # Reset for next item

        # Clear the error flag before invoking
        scene_ref.sg_last_gen_error = False

        # Disable interactive elements for unattended batch
        if arch_mode == 'trellis2':
            scene_ref.trellis2_preview_gallery_enabled = False

        try:
            if arch_mode == 'trellis2':
                bpy.ops.object.trellis2_generate('INVOKE_DEFAULT')
                print(f"[Queue] Invoked trellis2_generate for '{item.scene_name}'")
            else:
                # Standard texturing — select all cameras first
                bpy.ops.object.select_all(action='DESELECT')
                for obj in scene_ref.objects:
                    if obj.type == 'CAMERA':
                        obj.select_set(True)
                bpy.ops.object.test_stable('INVOKE_DEFAULT')
                print(f"[Queue] Invoked test_stable for '{item.scene_name}'")
            _queue_phase = 'running'
        except Exception as e:
            print(f"[Queue] Failed to invoke operator for '{item.scene_name}': {e}")
            _queue_handle_failure(item, str(e))
        _tag_redraw()
        return 1.5

    # ── Phase: RUNNING → poll for completion ──
    if _queue_phase == 'running':
        from ..mesh_gen.trellis2 import Trellis2Generate
        from ..texturing.generator import ComfyUIGenerate
        scene_ref = bpy.context.scene
        arch_mode = getattr(scene_ref, 'architecture_mode', 'sdxl')

        if arch_mode == 'trellis2':
            trellis_running = Trellis2Generate._is_running
            gen_err = getattr(scene_ref, 'sg_last_gen_error', False)
            tex_mode = getattr(scene_ref, 'trellis2_texture_mode', 'native')
            pipe_active = getattr(scene_ref, 'trellis2_pipeline_active', False)
            gen_status = getattr(scene_ref, 'generation_status', 'idle')
            # TRELLIS.2: wait for the operator to finish
            if trellis_running:
                return 1.0  # Still running

            # Check if Trellis2 itself failed
            if gen_err:
                print(f"[Queue] TRELLIS.2 generation failed for '{item.scene_name}'")
                _queue_handle_failure(item, "TRELLIS.2 generation failed")
                return 0.5

            # Trellis2Generate finished — enter settling phase to wait for
            # _schedule_texture_generation / _deferred_generate to fire.
            if tex_mode in ('sdxl', 'flux1', 'qwen_image_edit', 'qwen_image_21', 'flux2_klein'):
                _queue_phase = 'settling'
                _queue_settle_deadline = _time.monotonic() + 15.0  # generous: server may need VRAM flush
                _queue_texturing_seen = False
                print(f"[Queue] TRELLIS.2 mesh done, entering settling (tex_mode='{tex_mode}'), "
                      f"waiting for texturing to start (up to 15 s)...")
            else:
                # Native texture mode — mesh only, done
                print(f"[Queue] tex_mode='{tex_mode}' — no diffusion texturing, finishing item")
                _queue_finish_item(item)
            return 1.0
        else:
            # Standard texturing
            gen_status = getattr(scene_ref, 'generation_status', 'idle')
            if gen_status in ('idle', 'waiting'):
                if getattr(scene_ref, 'sg_last_gen_error', False):
                    _queue_handle_failure(item, "Generation failed")
                    return 0.5
                _queue_finish_item(item)
                return 0.5
            return 1.0  # Still running

    # ── Phase: SETTLING → wait for TRELLIS.2 chained texturing to start ──
    if _queue_phase == 'settling':
        scene_ref = bpy.context.scene
        pipeline_active = getattr(scene_ref, 'trellis2_pipeline_active', False)
        gen_status = getattr(scene_ref, 'generation_status', 'idle')
        if pipeline_active or gen_status == 'running':
            _queue_texturing_seen = True

        if _queue_texturing_seen:
            if gen_status == 'running' or pipeline_active:
                # Texturing is actively running — transition
                _queue_phase = 'waiting_texturing'
                print(f"[Queue] Texturing confirmed running, waiting for completion...")
                return 1.0
            else:
                # Texturing already completed (fast finish during settling)
                if getattr(scene_ref, 'sg_last_gen_error', False):
                    _queue_handle_failure(item, "Texturing failed")
                else:
                    _queue_finish_item(item)
                return 0.5

        # Texturing hasn't appeared yet — wait until deadline
        if _time.monotonic() >= _queue_settle_deadline:
            print(f"[Queue] Texturing never started for '{item.scene_name}' (timeout)")
            _queue_handle_failure(item, "Diffusion texturing failed to start")
            return 0.5

        return 1.0  # Still settling

    # ── Phase: WAITING_TEXTURING → poll chained ComfyUIGenerate ──
    if _queue_phase == 'waiting_texturing':
        scene_ref = bpy.context.scene
        gen_status = getattr(scene_ref, 'generation_status', 'idle')
        pipeline_active = getattr(scene_ref, 'trellis2_pipeline_active', False)
        gen_err = getattr(scene_ref, 'sg_last_gen_error', False)

        if gen_status in ('idle', 'waiting') and not pipeline_active:
            if getattr(scene_ref, 'sg_last_gen_error', False):
                _queue_handle_failure(item, "Texturing failed")
                return 0.5
            _queue_finish_item(item)
            return 0.5
        return 1.0  # Still running

    # ── Phase: EXPORTING_GIF → poll ExportOrbitGIF completion ──
    if _queue_phase == 'exporting_gif':
        from ..texturing.orbit_export import ExportOrbitGIF
        if ExportOrbitGIF._rendering:
            return 1.0  # Still rendering frames
        # First GIF export finished
        print("[Queue] GIF export completed")

        # Check whether a second (no-PBR) export is needed
        wm = bpy.context.window_manager
        scene_ref = bpy.context.scene
        if (getattr(wm, 'sg_queue_gif_also_no_pbr', False)
                and getattr(scene_ref, 'pbr_decomposition', False)):
            # Start the no-PBR reproject → second GIF pipeline
            if _queue_start_no_pbr_reproject():
                _queue_phase = 'reprojecting_no_pbr'
                print("[Queue] Starting no-PBR reproject for second GIF")
                _tag_redraw()
                return 1.0
            else:
                print("[Queue] No-PBR reproject failed to start — skipping second GIF")

        _queue_save_and_advance(item)
        return 0.5

    # ── Phase: REPROJECTING_NO_PBR → wait for ComfyUIGenerate to finish ──
    if _queue_phase == 'reprojecting_no_pbr':
        from ..texturing.generator import ComfyUIGenerate
        if ComfyUIGenerate._is_running:
            return 1.0
        # Reproject done — start second GIF with 1 sample + suffix
        print("[Queue] No-PBR reproject finished — starting second GIF export")
        if _queue_start_gif_export(samples_override=1, suffix='_no_pbr'):
            _queue_phase = 'exporting_gif_no_pbr'
            _tag_redraw()
            return 1.0
        else:
            print("[Queue] Second GIF export failed to start — restoring PBR")
            _queue_restore_pbr()
            _queue_save_and_advance(item)
            return 0.5

    # ── Phase: EXPORTING_GIF_NO_PBR → poll second GIF completion ──
    if _queue_phase == 'exporting_gif_no_pbr':
        from ..texturing.orbit_export import ExportOrbitGIF
        if ExportOrbitGIF._rendering:
            return 1.0
        print("[Queue] Second (no-PBR) GIF export completed")
        _queue_restore_pbr()
        _queue_save_and_advance(item)
        return 0.5

    return 1.0


def _queue_handle_failure(item, reason):
    """Handle a failed queue item — retry if under the limit, else mark as error."""
    global _queue_current_idx, _queue_phase, _queue_processing, _queue_timer
    global _queue_force_reload

    item.retries += 1
    if item.retries < _QUEUE_MAX_RETRIES:
        item.status = 'pending'
        _queue_phase = 'idle'  # Will re-attempt on next tick
        _queue_force_reload = True  # Force re-open to restore clean .blend state
        print(f"[Queue] Retrying '{item.label or item.scene_name}' "
              f"(attempt {item.retries + 1}/{_QUEUE_MAX_RETRIES}): {reason}")
    else:
        item.status = 'error'
        item.error_reason = reason
        print(f"[Queue] '{item.label or item.scene_name}' failed after "
              f"{_QUEUE_MAX_RETRIES} attempts: {reason}")
        _queue_current_idx += 1
        # Find next pending
        wm = bpy.context.window_manager
        while _queue_current_idx < len(wm.sg_scene_queue):
            if wm.sg_scene_queue[_queue_current_idx].status == 'pending':
                break
            _queue_current_idx += 1
        if _queue_current_idx >= len(wm.sg_scene_queue):
            _queue_processing = False
            _queue_phase = 'idle'
            _queue_timer = None
            print("[Queue] All queue items processed.")
        else:
            _queue_phase = 'idle'
    _persist_queue()
    _tag_redraw()


def _queue_finish_item(item):
    """Mark the current queue item as done and advance to the next.

    If GIF export is enabled, starts the orbit GIF export first and
    defers the save+advance to the ``exporting_gif`` phase handler.
    """
    global _queue_current_idx, _queue_phase, _queue_processing, _queue_timer

    item.status = 'done'
    print(f"[Queue] Finished '{item.scene_name}'")

    # ── Optional GIF export before saving ──
    wm = bpy.context.window_manager
    if getattr(wm, 'sg_queue_gif_export', False):
        if _queue_start_gif_export():
            _queue_phase = 'exporting_gif'
            print("[Queue] GIF export started — deferring save until export finishes")
            _tag_redraw()
            return  # save + advance will happen in the 'exporting_gif' handler

    # No GIF (or export failed to start) — save and advance immediately
    _queue_save_and_advance(item)


def _queue_start_gif_export(samples_override=None, suffix=''):
    """Try to invoke ExportOrbitGIF with current WM settings.

    Args:
        samples_override: If set, use this sample count instead of the WM setting.
        suffix: Appended to orbit filenames (e.g. '_no_pbr' → orbit_no_pbr.gif).

    Returns True if the operator was successfully started.
    """
    from ..texturing.orbit_export import ExportOrbitGIF

    scene = bpy.context.scene

    # ── Always use the first camera (Camera_0 or lowest-numbered) ──
    cameras = sorted(
        [o for o in scene.objects if o.type == 'CAMERA'],
        key=lambda c: c.name,
    )
    if not cameras:
        print("[Queue] No cameras in scene — skipping GIF export")
        return False
    scene.camera = cameras[0]
    print(f"[Queue] GIF export: using camera '{cameras[0].name}'")

    # ── Ensure an active mesh/empty exists (poll requirement) ──
    vl = bpy.context.view_layer
    active = vl.objects.active
    if not active or active.type not in {'MESH', 'EMPTY'}:
        for obj in scene.objects:
            if obj.type == 'MESH':
                vl.objects.active = obj
                obj.select_set(True)
                print(f"[Queue] GIF export: activated mesh '{obj.name}'")
                break
        else:
            print("[Queue] No mesh in scene — skipping GIF export")
            return False

    wm = bpy.context.window_manager

    # ── Explicitly apply HDRI from the queue's file path (if set) ──
    hdri_path = getattr(wm, 'sg_queue_gif_hdri_path', '')
    hdri_strength = getattr(wm, 'sg_queue_gif_hdri_strength', 1.0)
    use_hdri = getattr(wm, 'sg_queue_gif_use_hdri', False)

    if use_hdri and hdri_path:
        import os as _os
        resolved = bpy.path.abspath(hdri_path)
        if _os.path.isfile(resolved):
            try:
                bpy.ops.object.add_hdri(hdri_path=resolved,
                                        strength=hdri_strength)
                print(f"[Queue] Applied HDRI: {resolved}  "
                      f"(strength={hdri_strength:.2f})")
            except Exception as e:
                print(f"[Queue] AddHDRI failed: {e}  — continuing anyway")
        else:
            print(f"[Queue] HDRI file not found: {resolved}  — skipping HDRI")

    kwargs = {
        'duration':                getattr(wm, 'sg_queue_gif_duration', 5.0),
        'frame_rate':              getattr(wm, 'sg_queue_gif_fps', 24),
        'resolution_percentage':   getattr(wm, 'sg_queue_gif_resolution', 50),
        'samples':                 samples_override if samples_override is not None else getattr(wm, 'sg_queue_gif_samples', 32),
        'engine':                  getattr(wm, 'sg_queue_gif_engine', 'CYCLES'),
        'interpolation':           getattr(wm, 'sg_queue_gif_interpolation', 'LINEAR'),
        'use_hdri':                use_hdri,
        'hdri_rotation':           getattr(wm, 'sg_queue_gif_hdri_rotation', 0.0),
        'env_mode':                getattr(wm, 'sg_queue_gif_env_mode', 'FIXED'),
        'use_denoiser':            getattr(wm, 'sg_queue_gif_denoiser', True),
        'use_gpu':                 getattr(wm, 'sg_queue_gif_use_gpu', True),
        'filename_suffix':         suffix,
    }

    # Build a temp_override so the modal operator has a proper window
    window = bpy.context.window
    if not window:
        for w in wm.windows:
            window = w
            break
    if not window:
        print("[Queue] No window available — skipping GIF export")
        return False

    try:
        with bpy.context.temp_override(window=window):
            result = bpy.ops.object.export_orbit_gif('EXEC_DEFAULT', **kwargs)
        if 'RUNNING_MODAL' in result:
            print(f"[Queue] GIF export started successfully")
            return True
        print(f"[Queue] GIF export returned {result} — skipping")
        return False
    except Exception as e:
        print(f"[Queue] GIF export failed to start: {e}")
        import traceback; traceback.print_exc()
        return False


def _queue_start_no_pbr_reproject():
    """Disable PBR, then invoke a project-only reproject.

    Saves the original PBR / generation settings so they can be restored
    by ``_queue_restore_pbr()`` after the second GIF is done.

    Returns True if the reproject operator was successfully started.
    """
    global _queue_original_pbr, _queue_original_gen_method, _queue_original_overwrite

    scene = bpy.context.scene
    _queue_original_pbr = scene.pbr_decomposition
    _queue_original_gen_method = scene.generation_method
    _queue_original_overwrite = scene.overwrite_material

    scene.pbr_decomposition = False
    scene.generation_mode = 'project_only'
    scene.generation_method = 'separate'
    scene.overwrite_material = True

    # Select all cameras (required by ComfyUIGenerate)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in scene.objects:
        if obj.type == 'CAMERA':
            obj.select_set(True)

    window = bpy.context.window
    wm = bpy.context.window_manager
    if not window:
        for w in wm.windows:
            window = w
            break
    if not window:
        print("[Queue] No window available — cannot start reproject")
        return False

    try:
        with bpy.context.temp_override(window=window):
            bpy.ops.object.test_stable('INVOKE_DEFAULT')
        print("[Queue] No-PBR reproject invoked")
        return True
    except Exception as e:
        print(f"[Queue] Failed to invoke reproject: {e}")
        import traceback; traceback.print_exc()
        # Restore settings on failure
        scene.pbr_decomposition = _queue_original_pbr
        scene.generation_mode = 'standard'
        scene.generation_method = _queue_original_gen_method
        scene.overwrite_material = _queue_original_overwrite
        return False


def _queue_restore_pbr():
    """Restore PBR and generation settings saved by ``_queue_start_no_pbr_reproject``."""
    scene = bpy.context.scene
    scene.pbr_decomposition = _queue_original_pbr
    scene.generation_mode = 'standard'
    scene.generation_method = _queue_original_gen_method
    scene.overwrite_material = _queue_original_overwrite
    print(f"[Queue] Restored pbr_decomposition={_queue_original_pbr}")


def _queue_save_and_advance(item):
    """Save the current .blend and advance to the next queue item."""
    global _queue_current_idx, _queue_phase, _queue_processing, _queue_timer

    # Save the result in-place on the queue copy .blend
    if bpy.data.filepath:
        try:
            bpy.ops.wm.save_mainfile()
            print(f"[Queue] Saved result: {bpy.data.filepath}")
        except Exception as e:
            print(f"[Queue] Save warning: {e}")

    # Advance to next pending item
    _queue_current_idx += 1
    wm = bpy.context.window_manager
    while _queue_current_idx < len(wm.sg_scene_queue):
        if wm.sg_scene_queue[_queue_current_idx].status == 'pending':
            break
        _queue_current_idx += 1

    if _queue_current_idx >= len(wm.sg_scene_queue):
        _queue_processing = False
        _queue_phase = 'idle'
        _queue_timer = None
        print("[Queue] All queue items processed.")
    else:
        _queue_phase = 'idle'  # Will switch to next scene on next tick

    _persist_queue()
    _tag_redraw()


def _tag_redraw():
    """Redraw all 3D viewports so the queue UI updates."""
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    except Exception:
        pass
