"""Main StableGen UI panel and preset diff helpers."""

import os
import bpy  # pylint: disable=import-error
import mathutils  # pylint: disable=import-error
import math
from ..utils import sg_modal_active
from .presets import PRESETS, GEN_PARAMETERS
from . import queue as _queue_mod

_ADDON_PKG = __package__.rsplit('.', 1)[0]

def _is_refreshing():
    """Return True while async model-list refreshes are in-flight.

    If the counter has been stuck for longer than ``_REFRESH_TIMEOUT``
    seconds (e.g. due to a lost timer), force-reset it to 0 so the UI
    is not permanently blocked.
    """
    import time as _time
    from ..core import state as _state
    count = _state._pending_refreshes
    if count <= 0:
        return False
    started = _state._refresh_started_at
    if started > 0 and (_time.monotonic() - started) > _state._REFRESH_TIMEOUT:
        # Safety net: force-clear a stuck counter
        print(f"[StableGen] Refreshing model lists stuck for >{_state._REFRESH_TIMEOUT:.0f}s – resetting.")
        _state._pending_refreshes = 0
        _state._refresh_started_at = 0.0
        return False
    return True

# Stock presets

_PRESET_DIFF_CORE = [
    'architecture_mode', 'model_architecture', 'steps', 'cfg',
    'sampler', 'scheduler', 'generation_method', 'denoise',
    'generation_mode', 'pbr_decomposition',
]

# Known enum values that need specific casing in diff labels.
_DISPLAY_NAMES = {
    'sdxl': 'SDXL',
    'flux1': 'FLUX.1',
    'qwen_image_edit': 'Qwen',
    'flux2_klein': 'FLUX.2 Klein',
    'trellis2': 'TRELLIS.2',
    'standard': 'Standard',
}

def _fmt_diff_val(v):
    """Format a property value for compact display in the diff preview."""
    if isinstance(v, bool):
        return 'On' if v else 'Off'
    if isinstance(v, float):
        # Strip trailing zeros: 1.50 → "1.5", 8.00 → "8"
        return f'{v:.2f}'.rstrip('0').rstrip('.')
    if isinstance(v, str):
        if v in _DISPLAY_NAMES:
            return _DISPLAY_NAMES[v]
        s = v.replace('_', ' ').title()
        return s[:20] + '…' if len(s) > 22 else s
    return str(v)


def _preset_diff(context):
    """Return list of (param, current_formatted, preset_formatted) for
    parameters that differ between the scene and the selected preset.
    Core parameters are listed first."""
    scene = context.scene
    preset_key = scene.stablegen_preset
    if preset_key not in PRESETS:
        return []
    preset = PRESETS[preset_key]
    core_diffs = []
    other_diffs = []
    core_set = set(_PRESET_DIFF_CORE)
    for key in GEN_PARAMETERS:
        if key not in preset or not hasattr(scene, key):
            continue
        current = getattr(scene, key)
        target = preset[key]
        try:
            if isinstance(current, (int, float)) and isinstance(target, (int, float)):
                if math.isclose(float(current), float(target), rel_tol=1e-7, abs_tol=0.0):
                    continue
            elif current == target:
                continue
        except Exception:
            continue
        entry = (key, _fmt_diff_val(current), _fmt_diff_val(target))
        if key in core_set:
            core_diffs.append(entry)
        else:
            other_diffs.append(entry)
    # Sort core diffs in the defined display order
    core_order = {k: i for i, k in enumerate(_PRESET_DIFF_CORE)}
    core_diffs.sort(key=lambda x: core_order.get(x[0], 999))
    return core_diffs + other_diffs



class StableGenPanel(bpy.types.Panel):
    """     
    Creates the main UI panel for the StableGen addon.     
    """
    bl_label = "StableGen"
    bl_idname = "OBJECT_PT_stablegen"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "StableGen"
    bl_context = "objectmode"
    bl_ui_units_x = 600

    def draw_header(self, _):
        """     
        Draws the header of the panel.         
        :param _: Unused parameter.         
        :return: None     
        """
        self.layout.label(icon="WORLD_DATA")

    def draw(self, context):
        """     
        Draws the panel with reorganized Advanced Parameters.         
        :param context: Blender context.         
        :return: None     
        """
        layout = self.layout
        scene = context.scene # Get the scene for easier access

        # Detect the current width of the panel
        region = context.region
        width = region.width
        width_mode = 'narrow' if width < 420 else 'wide'

        # Compute properties that differ from the pending (unapplied) preset.
        # Used to highlight affected fields inline with alert coloring.
        # Dict maps param name → formatted target value for "→ X" labels.
        _diff_props = {}
        if (hasattr(scene, 'active_preset')
                and scene.stablegen_preset != 'CUSTOM'
                and getattr(scene, 'active_preset', '') != scene.stablegen_preset):
            _diff_props = {d[0]: d[2] for d in _preset_diff(context)}

         # --- Action Buttons & Progress ---
        cam_tools_row = layout.row()
        cam_tools_row.operator("object.add_cameras", text="Add Cameras", icon="CAMERA_DATA")
        if width_mode == 'narrow':
            cam_tools_row = layout.row() 
        cam_tools_row.operator("object.collect_camera_prompts", text="Collect Camera Prompts", icon="FILE_TEXT")

        cam_extra_row = layout.row(align=True)
        cam_extra_row.operator("object.clone_camera", text="Clone Camera", icon="DUPLICATE")
        cam_extra_row.operator("object.mirror_camera", text="Mirror", icon="MOD_MIRROR")
        if hasattr(bpy.ops.object, 'apply_auto_aspect'):
            cam_extra_row.operator("object.apply_auto_aspect", text="Auto Aspect", icon="FULLSCREEN_ENTER")
        cam_extra_row.operator("object.toggle_camera_labels", text="Labels", icon="FONT_DATA")
        

        addon_prefs = context.preferences.addons[_ADDON_PKG].preferences
        config_error_message = None

        if not os.path.exists(addon_prefs.output_dir):
            config_error_message = "Output Path Invalid"
        elif not addon_prefs.server_address:
            config_error_message = "Server Address Missing"
        elif not addon_prefs.server_online:
            config_error_message = "Cannot reach server"

        # Determine if we are in TRELLIS.2 mode (used by generate button & later sections)
        _arch_mode = getattr(scene, 'architecture_mode', 'sdxl')
        _is_trellis2_mode = (_arch_mode == 'trellis2')

        # ── Prerequisite warnings (shown inline before the Generate button) ──
        if not _is_trellis2_mode and not config_error_message:
            has_cameras = any(obj.type == 'CAMERA' for obj in scene.objects)
            has_meshes = any(obj.type == 'MESH' for obj in context.view_layer.objects if not obj.hide_get())
            if not has_meshes:
                warn_box = layout.box()
                warn_row = warn_box.row()
                warn_row.alert = True
                warn_row.label(text="No visible mesh objects", icon='ERROR')
                warn_row.operator("stablegen.switch_to_mesh_generation",
                                  text="Mesh Generation", icon='MESH_DATA')
            if not has_cameras:
                warn_box = layout.box()
                warn_row = warn_box.row()
                warn_row.alert = True
                warn_row.label(text="No cameras in the scene", icon='ERROR')
                warn_row.operator("object.add_cameras", text="Add Cameras", icon='CAMERA_DATA')

        action_row = layout.row()
        action_row.scale_y = 2.0 # Scale the row vertically

        # Show a "Refreshing…" indicator while async model fetches are in-flight
        if _is_refreshing():
            refresh_row = layout.row()
            refresh_row.alignment = 'CENTER'
            refresh_row.label(text="Refreshing model lists...", icon="SORTTIME")

        if _is_trellis2_mode:
            # --- TRELLIS.2 Generate Button ---
            trellis2_op = next(
                (op for win in context.window_manager.windows
                 for op in win.modal_operators
                 if op.bl_idname == 'OBJECT_OT_trellis2_generate'),
                None
            )
            # Also look for ComfyUIGenerate running as the texturing phase
            comfy_tex_op = next(
                (op for win in context.window_manager.windows
                 for op in win.modal_operators
                 if op.bl_idname == 'OBJECT_OT_test_stable'),
                None
            ) if not trellis2_op else None

            if config_error_message:
                if config_error_message == "Cannot reach server":
                    split = action_row.split(factor=0.85, align=True)
                    split.operator("object.trellis2_generate", text="Cannot generate: " + config_error_message, icon="ERROR")
                    split.operator("stablegen.check_server_status", text="", icon='FILE_REFRESH')
                else:
                    action_row.operator("object.trellis2_generate", text="Cannot generate: " + config_error_message, icon="ERROR")
                    action_row.enabled = False

            elif trellis2_op:
                # ── Phases 1 & 2: Trellis2Generate is alive ──
                sg_batch_running = getattr(context.window_manager, 'sg_batch_running', False)
                if sg_batch_running:
                    action_row.operator("object.trellis2_batch_cancel", text="Cancel Batch", icon="CANCEL")
                else:
                    action_row.operator("object.trellis2_generate", text="Cancel TRELLIS.2", icon="CANCEL")
                progress_col = layout.column()

                # Bar 0 — Batch model counter (only during batch)
                _bi = getattr(context.window_manager, 'sg_batch_index', 0)
                _bt = getattr(context.window_manager, 'sg_batch_total', 0)
                if getattr(context.window_manager, 'sg_batch_running', False) and _bt > 0:
                    progress_col.progress(
                        text=f"Batch: Model {_bi}/{_bt}",
                        factor=max(0.0, min(_bi / _bt, 1.0))
                    )

                # Bar 1 — Overall
                overall_pct = getattr(trellis2_op, '_overall_progress', 0)
                overall_label = getattr(trellis2_op, '_overall_stage', 'Initializing')
                progress_col.progress(
                    text=f"{overall_label} ({overall_pct:.0f}%)",
                    factor=max(0.0, min(overall_pct / 100.0, 1.0))
                )

                # Bar 2 — Phase
                phase_pct = getattr(trellis2_op, '_phase_progress', 0)
                phase_label = getattr(trellis2_op, '_phase_stage', '')
                if phase_label:
                    progress_col.progress(
                        text=f"{phase_label} ({phase_pct:.0f}%)",
                        factor=max(0.0, min(phase_pct / 100.0, 1.0))
                    )

                # Bar 3 — Detail (only when there's actual sampler progress)
                detail_pct = getattr(trellis2_op, '_detail_progress', 0)
                detail_label = getattr(trellis2_op, '_detail_stage', '')
                if detail_label and detail_pct > 0:
                    progress_col.progress(
                        text=f"{detail_label} ({detail_pct:.0f}%)",
                        factor=max(0.0, min(detail_pct / 100.0, 1.0))
                    )

            elif comfy_tex_op and getattr(scene, 'trellis2_pipeline_active', False):
                # ── Phase 3: Texturing via ComfyUIGenerate ──
                sg_batch_running = getattr(context.window_manager, 'sg_batch_running', False)
                if sg_batch_running:
                    action_row.operator("object.trellis2_batch_cancel", text="Cancel Batch", icon="CANCEL")
                else:
                    action_row.operator("object.test_stable", text="Cancel Texturing", icon="CANCEL")
                progress_col = layout.column()

                # Batch overall bar (above StableGen bars)
                _bi = getattr(context.window_manager, 'sg_batch_index', 0)
                _bt = getattr(context.window_manager, 'sg_batch_total', 0)
                if sg_batch_running and _bt > 0:
                    progress_col.progress(
                        text=f"Batch: Model {_bi}/{_bt}",
                        factor=max(0.0, min(_bi / _bt, 1.0))
                    )

                pbr_active = getattr(comfy_tex_op, '_pbr_active', False)
                if pbr_active:
                    # PBR decomposition sub-phase — 3 bars:
                    # Bar 1 (top): overall pipeline progress
                    # Bar 2: current PBR model (camera progress)
                    # Bar 3: PBR step N/M
                    pbr_step = getattr(comfy_tex_op, '_pbr_step', 0)
                    pbr_total = max(getattr(comfy_tex_op, '_pbr_total_steps', 1), 1)
                    pbr_cam = getattr(comfy_tex_op, '_pbr_cam', 0)
                    pbr_cam_total = max(getattr(comfy_tex_op, '_pbr_cam_total', 1), 1)
                    cam_frac = pbr_cam / pbr_cam_total
                    stage_text = getattr(comfy_tex_op, '_stage', 'PBR Decomposition')

                    # Overall: texturing is done, PBR owns the upper slice of Phase 3
                    phase_start = getattr(scene, 'trellis2_pipeline_phase_start_pct', 65.0)
                    phase_weight = 100.0 - phase_start
                    tex_portion = phase_weight * 0.6  # first 60% of phase = texturing
                    pbr_portion = phase_weight * 0.4  # last 40% = PBR
                    pbr_frac = max(0.0, min(((pbr_step - 1) + cam_frac) / pbr_total, 1.0))
                    overall_pct = phase_start + tex_portion + pbr_portion * pbr_frac
                    overall_pct = max(0.0, min(overall_pct, 100.0))
                    total_phases = getattr(scene, 'trellis2_pipeline_total_phases', 3)

                    # Bar 1 — Overall pipeline
                    progress_col.progress(
                        text=f"Phase {total_phases}/{total_phases}: PBR Decomposition ({overall_pct:.0f}%)",
                        factor=max(0.0, min(overall_pct / 100.0, 1.0))
                    )
                    # Bar 2 — Current model (camera progress)
                    progress_col.progress(text=stage_text, factor=max(0.0, min(cam_frac, 1.0)))
                    # Bar 3 — PBR step N/M
                    progress_col.progress(text=f"PBR: Step {pbr_step}/{pbr_total}", factor=pbr_frac)
                else:
                    # Normal texturing progress
                    # Compute overall from scene pipeline props + ComfyUI progress
                    phase_start = getattr(scene, 'trellis2_pipeline_phase_start_pct', 65.0)
                    phase_weight = 100.0 - phase_start
                    comfy_progress = getattr(comfy_tex_op, '_progress', 0) / 100.0
                    total_imgs = getattr(comfy_tex_op, '_total_images', 0)
                    cur_img_idx = getattr(comfy_tex_op, '_current_image', 0)
                    if total_imgs > 1:
                        comfy_overall = (cur_img_idx + comfy_progress) / total_imgs
                    else:
                        comfy_overall = comfy_progress
                    overall_pct = phase_start + comfy_overall * phase_weight
                    overall_pct = max(0.0, min(overall_pct, 100.0))

                    total_phases = getattr(scene, 'trellis2_pipeline_total_phases', 3)
                    # Bar 1 — Overall
                    progress_col.progress(
                        text=f"Phase {total_phases}/{total_phases}: Texturing ({overall_pct:.0f}%)",
                        factor=max(0.0, min(overall_pct / 100.0, 1.0))
                    )

                    # Bar 2 — Per-image (same as normal ComfyUI bar)
                    stage = getattr(comfy_tex_op, '_stage', 'Generating')
                    img_pct = getattr(comfy_tex_op, '_progress', 0)
                    progress_col.progress(
                        text=f"{stage} ({img_pct:.0f}%)",
                        factor=max(0.0, min(img_pct / 100.0, 1.0))
                    )

                    # Bar 3 — Image N/M (same as normal ComfyUI overall bar)
                    if total_imgs > 1:
                        cur_img = min(cur_img_idx + 1, total_imgs)
                        img_overall = max(0.0, min(comfy_overall, 1.0))
                        progress_col.progress(
                            text=f"Overall: Image {cur_img}/{total_imgs}",
                            factor=img_overall
                        )

            elif comfy_tex_op and scene.generation_status == 'running':
                # Standalone ComfyUIGenerate (e.g. Reproject with PBR)
                # outside the TRELLIS.2 pipeline.
                action_row.operator("object.test_stable", text="Cancel Generation", icon="CANCEL")
                progress_col = layout.column()
                raw_progress = getattr(comfy_tex_op, '_progress', 0) / 100.0
                pbr_active = getattr(comfy_tex_op, '_pbr_active', False)

                if pbr_active:
                    pbr_step = getattr(comfy_tex_op, '_pbr_step', 0)
                    pbr_total = max(getattr(comfy_tex_op, '_pbr_total_steps', 1), 1)
                    pbr_cam = getattr(comfy_tex_op, '_pbr_cam', 0)
                    pbr_cam_total = max(getattr(comfy_tex_op, '_pbr_cam_total', 1), 1)
                    cam_frac = pbr_cam / pbr_cam_total
                    stage_text = getattr(comfy_tex_op, '_stage', 'PBR Decomposition')
                    progress_col.progress(text=stage_text, factor=max(0.0, min(cam_frac, 1.0)))
                    pbr_factor = max(0.0, min(((pbr_step - 1) + cam_frac) / pbr_total, 1.0))
                    progress_col.progress(text=f"PBR: Step {pbr_step}/{pbr_total}", factor=pbr_factor)
                else:
                    progress_text = f"{getattr(comfy_tex_op, '_stage', 'Generating')} ({getattr(comfy_tex_op, '_progress', 0):.0f}%)"
                    progress_col.progress(text=progress_text, factor=max(0.0, min(raw_progress, 1.0)))
                    total_images = getattr(comfy_tex_op, '_total_images', 0)
                    if total_images > 1:
                        current_image_idx = getattr(comfy_tex_op, '_current_image', 0)
                        overall_progress = (current_image_idx + max(0.0, min(raw_progress, 1.0))) / total_images if total_images > 0 else 0
                        cur_img = min(current_image_idx + 1, total_images)
                        progress_col.progress(text=f"Overall: Image {cur_img}/{total_images}", factor=max(0.0, min(overall_progress, 1.0)))

            elif scene.trellis2_generate_from == 'image' and not scene.trellis2_input_image:
                action_row.operator("object.trellis2_generate", text="Select an image first", icon="ERROR")
                action_row.enabled = False
            elif not getattr(scene, 'trellis2_available', False):
                split = action_row.split(factor=0.8)
                err_sub = split.row()
                err_sub.operator("object.trellis2_generate", text="TRELLIS.2 nodes not found", icon="ERROR")
                err_sub.enabled = False
                split.operator("stablegen.check_server_status", text="", icon="FILE_REFRESH")
            else:
                _batch_folder = getattr(scene, 'trellis2_batch_folder', '')
                _batch_count = getattr(scene, 'trellis2_batch_count', 0)
                _batch_running = getattr(context.window_manager, 'sg_batch_running', False)
                if _batch_running:
                    _bi = getattr(context.window_manager, 'sg_batch_index', 0)
                    _bt = getattr(context.window_manager, 'sg_batch_total', 0)
                    action_row.operator("object.trellis2_batch_cancel",
                                       text=f"Cancel Batch ({_bi}/{_bt})",
                                       icon="CANCEL")
                elif _batch_folder and _batch_count > 0:
                    action_row.operator("object.trellis2_generate",
                                        text="Generate Single", icon="MESH_ICOSPHERE")
                    action_row.operator("object.trellis2_batch_generate",
                                        text=f"Generate Batch ({_batch_count})",
                                        icon="IMGDISPLAY")
                else:
                    action_row.operator("object.trellis2_generate",
                                        text="Generate 3D Mesh", icon="MESH_ICOSPHERE")
        else:
            # --- Standard Diffusion Generate Button ---
            if config_error_message:
                # Split the row to have the error message/disabled button and the refresh button
                if config_error_message == "Cannot reach server":
                    split = action_row.split(factor=0.85, align=True) # Adjust factor as needed
                    split.operator("object.test_stable", text="Cannot generate: " + config_error_message, icon="ERROR") # Use ERROR icon
                    # Use the operator from __init__.py
                    split.operator("stablegen.check_server_status", text="", icon='FILE_REFRESH')
                else:
                    action_row.operator("object.test_stable", text="Cannot generate: " + config_error_message, icon="ERROR")
                    action_row.enabled = False
            else:
                action_row.enabled = True
                if not bpy.app.online_access:
                    action_row.operator("object.test_stable", text="Enable online access in preferences", icon="ERROR")
                    action_row.enabled = False
                elif not scene.model_name or scene.model_name == "NONE_FOUND":
                    action_row.operator("object.test_stable", text="Cannot generate: Model Directory Empty", icon="ERROR")
                    action_row.enabled = False
                elif scene.generation_status == 'idle':
                    # Check if any cameras are selected and if there is existing output
                    selected_cameras = [obj for obj in context.selected_objects if obj.type == 'CAMERA']
                    if not selected_cameras or scene.get("output_timestamp") == "":
                        action_row.operator("object.test_stable", text="Generate", icon="PLAY")
                    else:
                        # Use the regenerate operator
                        action_row.operator("object.stablegen_regenerate", text="Regenerate Selected Views", icon="PLAY")
                elif scene.generation_status == 'running':
                    action_row.operator("object.test_stable", text="Cancel Generation", icon="CANCEL")

                    operator_instance = next((op for win in context.window_manager.windows for op in win.modal_operators if op.bl_idname == 'OBJECT_OT_test_stable'), None)
                    if operator_instance:
                        progress_col = layout.column()
                        raw_progress = getattr(operator_instance, '_progress', 0) / 100.0
                        pbr_active = getattr(operator_instance, '_pbr_active', False)

                        if pbr_active:
                            # During PBR: top bar shows camera progress within
                            # the current model step (no raw ComfyUI jitter).
                            pbr_step = getattr(operator_instance, '_pbr_step', 0)
                            pbr_total = max(getattr(operator_instance, '_pbr_total_steps', 1), 1)
                            pbr_cam = getattr(operator_instance, '_pbr_cam', 0)
                            pbr_cam_total = max(getattr(operator_instance, '_pbr_cam_total', 1), 1)

                            # Top bar: camera X out of N within this step
                            cam_frac = pbr_cam / pbr_cam_total
                            stage_text = getattr(operator_instance, '_stage', 'PBR Decomposition')
                            progress_col.progress(
                                text=stage_text,
                                factor=max(0.0, min(cam_frac, 1.0))
                            )
                            # Bottom bar: overall PBR progress
                            pbr_factor = max(0.0, min(
                                ((pbr_step - 1) + cam_frac) / pbr_total, 1.0))
                            progress_col.progress(
                                text=f"PBR: Step {pbr_step}/{pbr_total}",
                                factor=pbr_factor
                            )
                        else:
                            # Normal generation progress
                            progress_text = f"{getattr(operator_instance, '_stage', 'Generating')} ({getattr(operator_instance, '_progress', 0):.0f}%)"
                            progress_col.progress(text=progress_text, factor=max(0.0, min(raw_progress, 1.0)))

                            total_images = getattr(operator_instance, '_total_images', 0)
                            if total_images > 1:
                                current_image_idx = getattr(operator_instance, '_current_image', 0)
                                current_image_decimal_progress = max(0.0, min(raw_progress, 1.0))
                                
                                # Ensure total_images is not zero to prevent division by zero
                                overall_progress_factor = (current_image_idx + current_image_decimal_progress) / total_images if total_images > 0 else 0
                                overall_progress_factor_clamped = max(0.0, min(overall_progress_factor, 1.0))

                                current_img = min(current_image_idx + 1, total_images)  # Clamp to total_images

                                progress_col.progress(
                                    text=f"Overall: Image {current_img}/{total_images}",
                                    factor=overall_progress_factor_clamped # Ensure factor is <= 1.0 (logic maintained)
                                )
                            
                elif context.scene.generation_status == 'waiting':
                    action_row.operator("object.test_stable", text="Waiting for Cancellation", icon="TIME")
                else:
                    action_row.operator("object.test_stable", text="Fix Issues to Generate", icon="ERROR")
                    action_row.enabled = False
        
        bake_row = layout.row()
        if config_error_message:
            bake_row.operator("object.bake_textures", text="Cannot Bake: " + config_error_message, icon="ERROR")
            bake_row.enabled = False
        else:
            bake_row.operator("object.bake_textures", text="Bake Textures", icon="RENDER_STILL")
            bake_row.enabled = True
        bake_operator = next((op for win in context.window_manager.windows for op in win.modal_operators if op.bl_idname == 'OBJECT_OT_bake_textures'), None)
        if bake_operator:
            bake_progress_col = layout.column()
            # Batch overall bar during bake phase
            _bi = getattr(context.window_manager, 'sg_batch_index', 0)
            _bt = getattr(context.window_manager, 'sg_batch_total', 0)
            if getattr(context.window_manager, 'sg_batch_running', False) and _bt > 0:
                bake_progress_col.progress(
                    text=f"Batch: Model {_bi}/{_bt}",
                    factor=max(0.0, min(_bi / _bt, 1.0))
                )
            bake_stage = getattr(bake_operator, '_stage', 'Baking')
            bake_progress = getattr(bake_operator, '_progress', 0) / 100.0
            bake_progress_col.progress(text=bake_stage, factor=bake_progress if bake_progress <=1.0 else 1.0) # Ensure factor is <= 1.0
            
            total_objects = getattr(bake_operator, '_total_objects', 0)
            if total_objects > 1:
                current_object = getattr(bake_operator, '_current_object', 0)
                # Ensure total_objects is not zero
                overall_bake_progress = ((current_object + bake_progress) / total_objects) if total_objects > 0 else 0
                bake_progress_col.progress(
                    text=f"{bake_stage}: Object {current_object + 1}/{total_objects}",
                    factor=overall_bake_progress if overall_bake_progress <=1.0 else 1.0 # Ensure factor is <= 1.0
                )

        export_row = layout.row()
        export_row.operator("object.export_game_engine",
                            text="Export for Game Engine",
                            icon="EXPORT")

        # --- Preset Management ---
        preset_box = layout.box()
        row = preset_box.row(align=True)
        row.prop(scene, "stablegen_preset", text="Preset")
        
        # Conditional button: Apply for stock presets, Save for custom preset
        if not hasattr(scene, 'active_preset'):
            scene.active_preset = scene.stablegen_preset

        if scene.stablegen_preset == "CUSTOM":
            row.operator("stablegen.save_preset", text="Save Preset", icon="PLUS")
        else:
            if scene.active_preset != scene.stablegen_preset:
                row.operator("stablegen.apply_preset", text="Apply Preset", icon="CHECKMARK")
            
            is_stock_preset = PRESETS.get(scene.stablegen_preset, {}).get("custom", False) is False
            if not is_stock_preset and scene.stablegen_preset != "DEFAULT": 
                 row.operator("stablegen.delete_preset", text="Delete", icon="TRASH")

        # --- Scene Queue ---
        queue_box = layout.box()
        queue_col = queue_box.column()
        wm = context.window_manager
        show_queue = getattr(wm, 'sg_show_queue', False)
        queue_header = queue_col.row()
        queue_header.prop(wm, "sg_show_queue",
                          text=f"Scene Queue ({len(wm.sg_scene_queue)})",
                          icon="TRIA_DOWN" if show_queue else "TRIA_RIGHT",
                          emboss=False)
        if _queue_mod._queue_processing:
            status_text = "Exporting GIF..." if _queue_mod._queue_phase == 'exporting_gif' else "Processing..."
            queue_header.label(text=status_text, icon="SORTTIME")

        if show_queue:
            queue_content = queue_col.box()
            row = queue_content.row()
            row.template_list("SG_UL_SceneQueueList", "",
                              wm, "sg_scene_queue",
                              wm, "sg_scene_queue_index",
                              rows=3)
            col = row.column(align=True)
            col.operator("stablegen.queue_move_up", text="", icon="TRIA_UP")
            col.operator("stablegen.queue_move_down", text="", icon="TRIA_DOWN")
            col.separator()
            col.operator("stablegen.queue_remove", text="", icon="REMOVE")

            btn_row = queue_content.row(align=True)
            btn_row.operator("stablegen.queue_add", text="Add Scene", icon="ADD")
            btn_row.operator("stablegen.queue_open_result", text="Open", icon="FILE_BLEND")
            btn_row2 = queue_content.row(align=True)
            btn_row2.operator("stablegen.queue_invalidate", text="Reset", icon="LOOP_BACK")
            btn_row2.operator("stablegen.queue_clear", text="Clear", icon="TRASH")

            process_row = queue_content.row()
            if _queue_mod._queue_processing:
                process_row.alert = True
                process_row.operator("stablegen.queue_process", text="Cancel Queue", icon="CANCEL")
            else:
                process_row.operator("stablegen.queue_process", text="Process Queue", icon="PLAY")
                process_row.enabled = len(wm.sg_scene_queue) > 0

            # ── GIF Export settings ──
            gif_box = queue_content.box()
            gif_row = gif_box.row()
            gif_row.prop(wm, "sg_queue_gif_export", text="Export Orbit GIF/MP4")
            if getattr(wm, 'sg_queue_gif_export', False):
                gif_col = gif_box.column(align=True)
                row = gif_col.row(align=True)
                row.prop(wm, "sg_queue_gif_duration")
                row.prop(wm, "sg_queue_gif_fps")
                row = gif_col.row(align=True)
                row.prop(wm, "sg_queue_gif_resolution")
                row.prop(wm, "sg_queue_gif_samples")
                gif_col.prop(wm, "sg_queue_gif_engine")
                gif_col.prop(wm, "sg_queue_gif_interpolation")
                gif_col.separator()
                gif_col.prop(wm, "sg_queue_gif_use_hdri")
                if getattr(wm, 'sg_queue_gif_use_hdri', False):
                    gif_col.prop(wm, "sg_queue_gif_hdri_path")
                    gif_col.prop(wm, "sg_queue_gif_hdri_strength")
                    gif_col.prop(wm, "sg_queue_gif_hdri_rotation")
                    gif_col.prop(wm, "sg_queue_gif_env_mode")
                gif_col.prop(wm, "sg_queue_gif_denoiser")
                if bpy.app.version >= (5, 1, 0) and getattr(wm, 'sg_queue_gif_engine', 'CYCLES') == 'CYCLES':
                    gif_col.prop(wm, "sg_queue_gif_use_gpu")
                gif_col.prop(wm, "sg_queue_gif_also_no_pbr")

        # --- Main Parameters section ---
        if not hasattr(scene, 'show_generation_params'): 
            scene.show_generation_params = True

        is_trellis2 = getattr(scene, 'architecture_mode', 'sdxl') == 'trellis2'
        trellis2_tex_mode = getattr(scene, 'trellis2_texture_mode', 'native')
        trellis2_diffusion_texturing = is_trellis2 and trellis2_tex_mode in ('sdxl', 'flux1', 'qwen_image_edit', 'flux2_klein')
            
        main_params_box = layout.box()
        main_params_col = main_params_box.column()
        main_params_col.prop(scene, "show_generation_params", text="Main Parameters", icon="TRIA_DOWN" if scene.show_generation_params else "TRIA_RIGHT", emboss=False)
        if scene.show_generation_params:
            params_container = main_params_col.box()
            # Split for prompt
            split = params_container.split(factor=0.25)
            split.label(text="Prompt:")
            prompt_row = split.row(align=True)
            prompt_row.prop(scene, "comfyui_prompt", text="")
            # Hide texture prompt toggle when texture mode is none/native (no separate texture pipeline)
            if not (is_trellis2 and trellis2_tex_mode in ('none', 'native')):
                prompt_row.prop(scene, "use_separate_texture_prompt", text="",
                               icon='BRUSH_DATA',
                               icon_only=True)
            if scene.use_separate_texture_prompt and not (is_trellis2 and trellis2_tex_mode in ('none', 'native')):
                split = params_container.split(factor=0.25)
                split.label(text="Texture Prompt:")
                split.prop(scene, "texture_prompt", text="")

            # Architecture selector (architecture_mode — includes TRELLIS.2)
            # Alert when either architecture_mode or model_architecture changes.
            _arch_diff = 'architecture_mode' in _diff_props or 'model_architecture' in _diff_props
            split = params_container.split(factor=0.5)
            if _arch_diff:
                split.alert = True
            split.label(text="Architecture:")
            split.prop(scene, "architecture_mode", text="")
            if _arch_diff:
                _has_mode = 'architecture_mode' in _diff_props
                _has_model = 'model_architecture' in _diff_props
                if _has_mode:
                    # architecture_mode differs — always show model in parens.
                    # Use the preset's model_architecture even if it didn't change.
                    _preset_key = scene.stablegen_preset
                    _preset_data = PRESETS.get(_preset_key, {})
                    _model_display = _fmt_diff_val(_preset_data.get(
                        'model_architecture', scene.model_architecture))
                    split.label(text="→ " + _diff_props['architecture_mode']
                                + " (" + _model_display + ")")
                else:
                    # Only model_architecture differs (e.g. SDXL → Qwen).
                    split.label(text="→ " + _diff_props['model_architecture'])

            if is_trellis2:
                # --- TRELLIS.2 specific layout ---
                # Warning if TRELLIS.2 nodes not detected
                if not getattr(scene, 'trellis2_available', False):
                    warn_row = params_container.row()
                    warn_row.alert = True
                    warn_split = warn_row.split(factor=0.9)
                    warn_split.label(text="TRELLIS.2 nodes not detected on server", icon="ERROR")
                    warn_split.operator("stablegen.check_server_status", text="", icon="FILE_REFRESH")

                # Generate From toggle (Image / Prompt)
                split = params_container.split(factor=0.5)
                split.label(text="Generate From:")
                split.prop(scene, "trellis2_generate_from", text="")

                # Input image picker (only when generate_from = image)
                if scene.trellis2_generate_from == 'image':
                    split = params_container.split(factor=0.25)
                    split.label(text="Input Image:")
                    img_row = split.row(align=True)
                    img_row.prop(scene, "trellis2_input_image", text="")
                    img_row.operator("object.trellis2_batch_select_folder",
                                     text="", icon="OUTLINER_COLLECTION")

                    # Batch folder status row
                    batch_folder = getattr(scene, 'trellis2_batch_folder', '')
                    if batch_folder:
                        batch_count = getattr(scene, 'trellis2_batch_count', 0)
                        folder_name = os.path.basename(
                            batch_folder.rstrip('/\\')) or batch_folder
                        b_row = params_container.row(align=True)
                        b_split = b_row.split(factor=0.78, align=True)
                        b_split.label(
                            text=f"Batch: {batch_count} image(s) – {folder_name}",
                            icon="IMAGE_DATA")
                        b_split.operator("object.trellis2_batch_clear",
                                         text="", icon="X")
                        rn_row = params_container.row()
                        rn_row.prop(scene, "trellis2_batch_rename_meshes")
                        bake_row = params_container.row(align=True)
                        bake_row.prop(scene, "trellis2_batch_bake_textures")
                        if getattr(scene, 'trellis2_batch_bake_textures', True):
                            bake_row.operator("object.trellis2_batch_bake_settings",
                                              text="Bake Settings")

                # Preview gallery (only when generate_from = prompt)
                if scene.trellis2_generate_from == 'prompt':
                    row = params_container.row(align=True)
                    row.prop(scene, "trellis2_preview_gallery_enabled", text="Preview Gallery", toggle=True, icon="IMAGE_REFERENCE")
                    sub = row.row(align=True)
                    sub.enabled = scene.trellis2_preview_gallery_enabled
                    sub.prop(scene, "trellis2_preview_gallery_count", text="Count")

                # Texture Generation Mode
                split = params_container.split(factor=0.5)
                split.label(text="Texture Mode:")
                split.prop(scene, "trellis2_texture_mode", text="")

                # Prompt + native/none: show initial-image architecture & checkpoint
                _prompt_needs_initial = (
                    scene.trellis2_generate_from == 'prompt'
                    and trellis2_tex_mode in ('native', 'none')
                )
                if _prompt_needs_initial:
                    split = params_container.split(factor=0.5)
                    split.label(text="Initial Image Arch:")
                    split.prop(scene, "trellis2_initial_image_arch", text="")

                    split = params_container.split(factor=0.25)
                    split.label(text="Checkpoint:")
                    row = split.row(align=True)
                    row.prop(scene, "model_name", text="")
                    row.operator("stablegen.refresh_checkpoint_list", text="", icon='FILE_REFRESH')

                # When diffusion texturing: show checkpoint, generation mode, camera count
                if trellis2_diffusion_texturing:
                    split = params_container.split(factor=0.25)
                    split.label(text="Checkpoint:")
                    row = split.row(align=True)
                    row.prop(scene, "model_name", text="")
                    row.operator("stablegen.refresh_checkpoint_list", text="", icon='FILE_REFRESH')

                    split = params_container.split(factor=0.5)
                    if 'generation_method' in _diff_props or 'qwen_generation_method' in _diff_props:
                        split.alert = True
                    split.label(text="Generation Mode:")
                    if scene.model_architecture.startswith("qwen"):
                        split.prop(scene, "qwen_generation_method", text="")
                    else:
                        split.prop(scene, "generation_method", text="")
                    _lbl = ''.join(f'→{_diff_props[k]} ' for k in ('generation_method', 'qwen_generation_method') if k in _diff_props)
                    if _lbl:
                        split.label(text=_lbl.strip())
            else:
                # --- Standard diffusion layout ---
                # Split for model name
                split = params_container.split(factor=0.25)
                split.label(text="Checkpoint:")
                row = split.row(align=True)
                row.prop(scene, "model_name", text="")
                row.operator("stablegen.refresh_checkpoint_list", text="", icon='FILE_REFRESH')

                # Split for generation method
                split = params_container.split(factor=0.5)
                if 'generation_method' in _diff_props or 'qwen_generation_method' in _diff_props:
                    split.alert = True
                split.label(text="Generation Mode:")
                if scene.model_architecture.startswith("qwen"):
                    split.prop(scene, "qwen_generation_method", text="")
                else:
                    split.prop(scene, "generation_method", text="")
                _lbl = ''.join(f'→{_diff_props[k]} ' for k in ('generation_method', 'qwen_generation_method') if k in _diff_props)
                if _lbl:
                    split.label(text=_lbl.strip())

                # Split for object selection
                split = params_container.split(factor=0.5)
                split.label(text="Target Objects:")
                split.prop(scene, "texture_objects", text="")

        # Helper to create collapsible sections
        def draw_collapsible_section(parent_layout, toggle_prop_name, title, icon="NONE"):
            if not hasattr(scene, toggle_prop_name):
                setattr(bpy.types.Scene, toggle_prop_name, bpy.props.BoolProperty(name=title, default=False))

            box = parent_layout.box()
            col = box.column()
            is_expanded = getattr(scene, toggle_prop_name, False)
            col.prop(scene, toggle_prop_name, text=title, icon="TRIA_DOWN" if is_expanded else "TRIA_RIGHT", emboss=False)
            if is_expanded:
                return col.box() # Return a new box for content if expanded
            return None

        core_settings_props = [
            "show_core_settings", "show_lora_settings", "show_camera_options",
            "show_scene_understanding_settings", 
            "show_output_material_settings", "show_image_guidance_settings",
            "show_masking_inpainting_settings", "show_mode_specific_settings"
        ]
        for prop_name in core_settings_props:
            if not hasattr(scene, prop_name):
                setattr(bpy.types.Scene, prop_name, bpy.props.BoolProperty(name=prop_name.replace("_", " ").title(), default=False))

        # --- ADVANCED PARAMETERS ---
        advanced_params_box = layout.box()
        advanced_params_box = advanced_params_box.column()
        advanced_params_box.prop(scene, "show_advanced_params", text="Advanced Parameters", icon="TRIA_DOWN" if scene.show_advanced_params else "TRIA_RIGHT", emboss=False)
        if context.scene.show_advanced_params:

            # --- TRELLIS.2: Mesh Generation Settings ---
            if is_trellis2:
                content_box = draw_collapsible_section(advanced_params_box, "show_trellis2_mesh_settings", "Mesh Generation Settings", icon="MESH_DATA")
                if content_box:
                    # Core mesh params
                    row = content_box.row()
                    row.prop(scene, "trellis2_seed", text="Seed")

                    content_box.separator()

                    # Post-processing (ComfyUI-side decimation + remesh)
                    content_box.label(text="Post-Processing:", icon="OUTLINER_OB_MESH")
                    row = content_box.row()
                    row.prop(scene, "trellis2_post_processing_enabled", text="Enable Post-Processing", toggle=True, icon="MOD_DECIM")

                    if scene.trellis2_post_processing_enabled:
                        row = content_box.row()
                        row.prop(scene, "trellis2_decimation", text="Decimation Target")
                        row = content_box.row()
                        row.prop(scene, "trellis2_remesh", text="Remesh", toggle=True, icon="MOD_REMESH")

                    content_box.separator()

                    # Model settings
                    content_box.label(text="Model:", icon="SETTINGS")
                    split = content_box.split(factor=0.5)
                    split.label(text="Resolution:")
                    split.prop(scene, "trellis2_resolution", text="")

                    split = content_box.split(factor=0.5)
                    split.label(text="VRAM Mode:")
                    split.prop(scene, "trellis2_vram_mode", text="")

                    split = content_box.split(factor=0.5)
                    split.label(text="Attention:")
                    split.prop(scene, "trellis2_attn_backend", text="")

                    content_box.separator()

                    # Shape generation
                    content_box.label(text="Shape Generation:", icon="MESH_ICOSPHERE")
                    row = content_box.row(align=True)
                    row.prop(scene, "trellis2_ss_guidance", text="SS Guidance")
                    row.prop(scene, "trellis2_ss_steps", text="SS Steps")
                    row = content_box.row(align=True)
                    row.prop(scene, "trellis2_shape_guidance", text="Shape Guidance")
                    row.prop(scene, "trellis2_shape_steps", text="Shape Steps")
                    row = content_box.row()
                    row.prop(scene, "trellis2_max_tokens", text="Max Tokens (VRAM)")

                    content_box.separator()

                    # Conditioning
                    content_box.label(text="Conditioning:", icon="IMAGE_DATA")
                    split = content_box.split(factor=0.5)
                    split.label(text="Background:")
                    split.prop(scene, "trellis2_background_color", text="")
                    content_box.separator()

                    # Misc
                    content_box.label(text="Misc:", icon="PREFERENCES")
                    split = content_box.split(factor=0.5)
                    split.label(text="BG Removal:")
                    split.prop(scene, "trellis2_bg_removal", text="")

                    split = content_box.split(factor=0.5)
                    split.label(text="Shading:")
                    split.prop(scene, "trellis2_shade_mode", text="")
                    content_box.separator()

            # --- TRELLIS.2: Native Texture Settings ---
            if is_trellis2 and trellis2_tex_mode == 'native':
                content_box = draw_collapsible_section(advanced_params_box, "show_trellis2_texture_settings", "Texture Settings (TRELLIS.2 Native)", icon="TEXTURE")
                if content_box:
                    row = content_box.row(align=True)
                    row.prop(scene, "trellis2_tex_guidance", text="Tex Guidance")
                    row.prop(scene, "trellis2_tex_steps", text="Tex Steps")
                    row = content_box.row()
                    row.prop(scene, "trellis2_texture_size", text="Texture Size")

                    content_box.separator()
                    row = content_box.row()
                    row.prop(scene, "trellis2_auto_lighting", text="Studio Lighting", icon="LIGHT_AREA")

            # --- TRELLIS.2: Camera Placement Settings (diffusion texturing) ---
            if is_trellis2 and trellis2_diffusion_texturing:
                content_box = draw_collapsible_section(advanced_params_box, "show_trellis2_camera_settings", "Camera Placement (TRELLIS.2)", icon="CAMERA_DATA")
                if content_box:
                    _t2_pm = getattr(scene, 'trellis2_placement_mode', 'normal_weighted')

                    row = content_box.row()
                    row.prop(scene, "trellis2_import_scale", text="Import Scale (BU)")

                    content_box.separator()

                    split = content_box.split(factor=0.4)
                    split.label(text="Placement:")
                    split.prop(scene, "trellis2_placement_mode", text="")

                    # Camera count (not used by greedy)
                    if _t2_pm != 'greedy_coverage':
                        row = content_box.row()
                        row.prop(scene, "trellis2_camera_count", text="Camera Count")

                    # Greedy-specific
                    if _t2_pm == 'greedy_coverage':
                        row = content_box.row(align=True)
                        row.prop(scene, "trellis2_coverage_target", text="Coverage Target")
                        row.prop(scene, "trellis2_max_auto_cameras", text="Max Cameras")

                    # Fan-specific
                    if _t2_pm == 'fan_from_camera':
                        row = content_box.row()
                        row.prop(scene, "trellis2_fan_angle", text="Fan Angle")

                    content_box.separator()

                    row = content_box.row()
                    row.prop(scene, "trellis2_auto_prompts", text="Auto View Prompts", toggle=True, icon="OUTLINER_OB_CAMERA")

                    split = content_box.split(factor=0.4)
                    split.label(text="Auto Aspect:")
                    split.prop(scene, "trellis2_auto_aspect", text="")

                    split = content_box.split(factor=0.4)
                    split.label(text="Occlusion:")
                    split.prop(scene, "trellis2_occlusion_mode", text="")

                    row = content_box.row()
                    row.prop(scene, "trellis2_exclude_bottom", text="Exclude Bottom Faces", toggle=True, icon="TRIA_DOWN_BAR")
                    if scene.trellis2_exclude_bottom:
                        row = content_box.row()
                        row.prop(scene, "trellis2_exclude_bottom_angle", text="Bottom Angle")

                    row = content_box.row()
                    row.prop(scene, "trellis2_consider_existing", text="Consider Existing Cameras", toggle=True)

                    row = content_box.row()
                    row.prop(scene, "trellis2_delete_cameras", text="Delete Cameras After", toggle=True, icon="TRASH")

                    content_box.separator()

                    row = content_box.row()
                    row.prop(scene, "trellis2_clamp_elevation", text="Clamp Elevation", toggle=True, icon="CON_ROTLIMIT")
                    if scene.trellis2_clamp_elevation:
                        row = content_box.row(align=True)
                        row.prop(scene, "trellis2_min_elevation", text="Min")
                        row.prop(scene, "trellis2_max_elevation", text="Max")

            # --- Diffusion-based advanced sections ---
            # Each is individually guarded: shown for standard arches or TRELLIS.2 with diffusion texturing
            _show_diffusion_sections = not is_trellis2 or trellis2_diffusion_texturing
            # Mesh-only + prompt mode still needs initial-image pipeline settings
            _show_initial_image_settings = (
                is_trellis2 and trellis2_tex_mode == 'none'
                and getattr(scene, 'trellis2_generate_from', 'image') == 'prompt'
            )

            # --- Core Generation Settings ---
            
            if _show_diffusion_sections or _show_initial_image_settings:
                content_box = draw_collapsible_section(advanced_params_box, "show_core_settings", "Core Generation Settings", icon="SETTINGS")
            else:
                content_box = None
            if content_box:
                row = content_box.row()
                row.prop(scene, "seed", text="Seed")
                if width_mode == 'narrow':
                    row = content_box.row()
                sub = row.row()
                if 'steps' in _diff_props:
                    sub.alert = True
                sub.prop(scene, "steps", text="Steps")
                if 'steps' in _diff_props:
                    sub.label(text="→" + _diff_props['steps'])
                if width_mode == 'narrow':
                    row = content_box.row()
                sub = row.row()
                if 'cfg' in _diff_props:
                    sub.alert = True
                sub.prop(scene, "cfg", text="CFG")
                if 'cfg' in _diff_props:
                    sub.label(text="→" + _diff_props['cfg'])

                split = content_box.split(factor=0.5)
                split.label(text="Negative Prompt:")
                split.prop(scene, "comfyui_negative_prompt", text="")
                
                split = content_box.split(factor=0.5)
                split.label(text="Control After Generate:")
                split.prop(scene, "control_after_generate", text="")

                split = content_box.split(factor=0.5)
                if 'sampler' in _diff_props:
                    split.alert = True
                split.label(text="Sampler:")
                split.prop(scene, "sampler", text="")
                if 'sampler' in _diff_props:
                    split.label(text="→ " + _diff_props['sampler'])

                split = content_box.split(factor=0.5)
                if 'scheduler' in _diff_props:
                    split.alert = True
                split.label(text="Scheduler:")
                split.prop(scene, "scheduler", text="")
                if 'scheduler' in _diff_props:
                    split.label(text="→ " + _diff_props['scheduler'])
                
                row = content_box.row()
                row.prop(scene, "clip_skip", text="Clip Skip")

           # --- LoRA Settings ---
            if _show_diffusion_sections or _show_initial_image_settings:
                content_box = draw_collapsible_section(advanced_params_box, "show_lora_settings", "LoRA Management", icon="MODIFIER")
            else:
                content_box = None
            if content_box:
                row = content_box.row()
                row.alignment = 'CENTER'
                row.label(text="LoRA Units", icon="BRUSHES_ALL") # Using decimate icon for LoRA

                if scene.lora_units:
                    for i, lora_unit in enumerate(scene.lora_units):
                        is_selected_lora = (scene.lora_units_index == i)
                        unit_box = content_box.box()
                        row = unit_box.row()
                        row.prop(lora_unit, "model_name", text=f"LoRA {i+1}") # Shows selected model
                        
                        sub_row = unit_box.row(align=True)
                        sub_row.prop(lora_unit, "model_strength", text="Model Strength")
                        if not scene.model_architecture.startswith("qwen") and scene.model_architecture != 'flux2_klein': # Qwen/Klein use model only loras
                            sub_row.prop(lora_unit, "clip_strength", text="CLIP Strength")

                        # Icon to indicate selection more clearly alongside the alert state
                        select_icon = 'CHECKBOX_HLT' if is_selected_lora else 'CHECKBOX_DEHLT'
                        
                        # Selection button (now more like a radio button)
                        op_select_lora = row.operator("wm.context_set_int", text="", icon=select_icon, emboss=True) # Keep emboss for the button itself
                        op_select_lora.data_path = "scene.lora_units_index"
                        op_select_lora.value = i

                btn_row_lora = content_box.row(align=True)

                if not scene.lora_units:
                    # Only one button if no LoRA units are present
                    button_text = "Add LoRA Unit" # Default text
                    
                    # Draw the operator with the dynamically determined text
                    btn_row_lora.operator("stablegen.add_lora_unit", text=button_text, icon="ADD")
                    # The enabled state (greying out) will be handled by AddLoRAUnit.poll()
                else:
                    # Multiple buttons if LoRA units exist
                    btn_row_lora.operator("stablegen.add_lora_unit", text="Add Another LoRA", icon="ADD")
                    btn_row_lora.operator("stablegen.remove_lora_unit", text="Remove Selected", icon="REMOVE")

            # --- Camera Options ---
            if _show_diffusion_sections:
                content_box = draw_collapsible_section(advanced_params_box, "show_camera_options", "Camera Settings", icon="CAMERA_DATA")
            else:
                content_box = None
            if content_box:
                row = content_box.row(align=True)
                row.prop(scene, "use_camera_prompts", text="Use Camera Prompts", toggle=True, icon="OUTLINER_OB_CAMERA")

                # ── Camera Generation Order ──
                row = content_box.row(align=True)
                row.prop(scene, "sg_use_custom_camera_order",
                         text="Custom Generation Order", toggle=True, icon="SORTALPHA")
                if scene.sg_use_custom_camera_order:
                    order_box = content_box.box()
                    # Preset strategy selector + sync
                    row = order_box.row(align=True)
                    row.operator_menu_enum(
                        "stablegen.apply_camera_order_preset", "strategy",
                        text="Sort Preset", icon="PRESET")
                    row.operator("stablegen.sync_camera_order",
                                 text="", icon="FILE_REFRESH")

                    # UIList with move buttons
                    row = order_box.row()
                    row.template_list(
                        "SG_UL_CameraOrderList", "",
                        scene, "sg_camera_order",
                        scene, "sg_camera_order_index",
                        rows=4, maxrows=8)
                    col = row.column(align=True)
                    op = col.operator("stablegen.move_camera_order",
                                      text="", icon="TRIA_UP")
                    op.direction = 'UP'
                    op = col.operator("stablegen.move_camera_order",
                                      text="", icon="TRIA_DOWN")
                    op.direction = 'DOWN'

            # --- Viewpoint Blending Settings ---
            if _show_diffusion_sections:
                content_box = draw_collapsible_section(advanced_params_box, "show_scene_understanding_settings", "Viewpoint Blending Settings", icon="ZOOM_IN")
            else:
                content_box = None
            if content_box:
                # Row 1: Discard-Over Angle | Weight Exponent
                split = content_box.split(factor=0.5, align=True)
                sub = split.row()
                if 'discard_factor' in _diff_props:
                    sub.alert = True
                sub.prop(scene, "discard_factor", text="Discard Angle")
                if 'discard_factor' in _diff_props:
                    sub.label(text="→" + _diff_props['discard_factor'])
                sub = split.row()
                if 'weight_exponent' in _diff_props:
                    sub.alert = True
                sub.prop(scene, "weight_exponent", text="Exponent")
                if 'weight_exponent' in _diff_props:
                    sub.label(text="→" + _diff_props['weight_exponent'])

                # Row 2 & 3: Reset toggles and values (only for sequential / qwen)
                if scene.generation_method == 'sequential' or scene.model_architecture in ('qwen_image_edit', 'flux2_klein'):
                    split = content_box.split(factor=0.5, align=True)
                    split.prop(scene, "discard_factor_generation_only", text="Reset Angle", toggle=True)
                    split.prop(scene, "weight_exponent_generation_only", text="Reset Exponent", toggle=True)

                    if scene.discard_factor_generation_only or scene.weight_exponent_generation_only:
                        split = content_box.split(factor=0.5, align=True)
                        if scene.discard_factor_generation_only:
                            split.prop(scene, "discard_factor_after_generation", text="Angle After")
                        else:
                            split.label(text="")
                        if scene.weight_exponent_generation_only:
                            split.prop(scene, "weight_exponent_after_generation", text="Exp. After")
                        else:
                            split.label(text="")

                row = content_box.row()
                row.prop(scene, "early_priority", text="Prioritize Initial Views", toggle=True, icon="REW")
                if scene.early_priority:
                    row = content_box.row()
                    row.prop(scene, "early_priority_strength", text="Priority Strength")

                row = content_box.row()
                row.prop(scene, "bake_visibility_weights", text="Bake Visibility (Transform-Stable)", toggle=True, icon="MESH_DATA")

                row = content_box.row()
                row.prop(scene, "sg_silhouette_margin", text="Silhouette Margin (px)")

                row = content_box.row()
                row.prop(scene, "sg_silhouette_depth", text="Silhouette Depth Threshold")

                row = content_box.row()
                row.prop(scene, "sg_silhouette_rays", text="Silhouette Rays")

            # --- Output & Material Settings ---
            if _show_diffusion_sections:
                content_box = draw_collapsible_section(advanced_params_box, "show_output_material_settings", "Output & Material Settings", icon="MATERIAL")
            else:
                content_box = None
            if content_box:
                # ── PBR Decomposition ──
                row = content_box.row()
                if 'pbr_decomposition' in _diff_props:
                    row.alert = True
                row.prop(scene, "pbr_decomposition", text="PBR Decomposition", toggle=True, icon="NODE_MATERIAL")
                if 'pbr_decomposition' in _diff_props:
                    row.label(text="→ " + _diff_props['pbr_decomposition'])
                if scene.pbr_decomposition:
                    # Warning if PBR decomposition nodes not detected
                    if not getattr(scene, 'pbr_nodes_available', False):
                        warn_row = content_box.row()
                        warn_row.alert = True
                        warn_split = warn_row.split(factor=0.9)
                        warn_split.label(text="PBR nodes not detected on server", icon="ERROR")
                        warn_split.operator("stablegen.check_server_status", text="", icon="FILE_REFRESH")
                    sub = content_box.box()
                    sub.label(text="PBR Decomposition", icon="NODE_MATERIAL")
                    # ── Map toggles ──
                    sub.label(text="Maps to Extract:", icon="IMAGE_DATA")
                    grid = sub.grid_flow(row_major=True, columns=3, even_columns=True, align=True)
                    grid.prop(scene, "pbr_map_albedo", toggle=True, icon="SHADING_SOLID")
                    grid.prop(scene, "pbr_map_roughness", toggle=True, icon="MATFLUID")
                    grid.prop(scene, "pbr_map_metallic", toggle=True, icon="META_BALL")
                    grid.prop(scene, "pbr_map_normal", toggle=True, icon="NORMALS_FACE")
                    grid.prop(scene, "pbr_map_height", toggle=True, icon="MOD_DISPLACE")
                    grid.prop(scene, "pbr_map_ao", toggle=True, icon="SHADING_RENDERED")
                    grid.prop(scene, "pbr_map_emission", toggle=True, icon="LIGHT_POINT")
                    # ── Per-map adjustment controls ──
                    if scene.pbr_map_normal:
                        adj_row = sub.row()
                        adj_row.prop(scene, "pbr_normal_strength", text="Normal Strength", slider=True)
                    if scene.pbr_map_height:
                        adj_row = sub.row()
                        adj_row.prop(scene, "pbr_height_scale", text="Height Scale", slider=True)
                    if scene.pbr_map_ao:
                        ao_row = sub.row(align=True)
                        ao_row.prop(scene, "pbr_ao_samples", text="AO Samples")
                        ao_row.prop(scene, "pbr_ao_distance", text="AO Distance")
                    if scene.pbr_map_emission:
                        adj_row = sub.row()
                        adj_row.prop(scene, "pbr_emission_method", text="Method")
                        adj_row = sub.row()
                        adj_row.prop(scene, "pbr_emission_threshold", text="Threshold", slider=True)
                        adj_row = sub.row()
                        adj_row.prop(scene, "pbr_emission_strength", text="Emission Strength", slider=True)
                        if scene.pbr_emission_method == 'hsv':
                            hsv_row = sub.row(align=True)
                            hsv_row.prop(scene, "pbr_emission_saturation_min", text="Sat Min", slider=True)
                            hsv_row.prop(scene, "pbr_emission_value_min", text="Val Min", slider=True)
                            hsv_row = sub.row()
                            hsv_row.prop(scene, "pbr_emission_bloom", text="Bloom Radius", slider=True)
                    # ── Albedo source selector (only when albedo enabled) ──
                    if scene.pbr_map_albedo:
                        sub.separator()
                        row = sub.row()
                        row.prop(scene, "pbr_albedo_source", text="Albedo Source")
                        if scene.pbr_albedo_source == 'delight':
                            adj_row = sub.row()
                            adj_row.prop(scene, "pbr_delight_strength", text="Delight Strength", slider=True)
                        adj_row = sub.row(align=True)
                        adj_row.prop(scene, "pbr_albedo_auto_saturation", text="Correct Albedo Saturation", toggle=True, icon="BRUSHES_ALL")
                        if scene.pbr_albedo_auto_saturation:
                            adj_row.prop(scene, "pbr_albedo_saturation_mode", text="")
                    sub.separator()
                    # ── Quality settings ──
                    row = sub.row(align=True)
                    row.prop(scene, "pbr_use_native_resolution", text="Native Resolution", toggle=True, icon="FULLSCREEN_ENTER")
                    row.prop(scene, "pbr_tiling", text="")
                    if scene.pbr_tiling != 'off':
                        row = sub.row(align=True)
                        row.prop(scene, "pbr_tile_grid", text="Tile Grid (N\u00d7N)")
                        row.prop(scene, "pbr_tile_superres", text="Super Res", toggle=True, icon="IMAGE_PLANE")
                    if scene.pbr_tiling == 'custom':
                        row = sub.row(align=True)
                        row.prop(scene, "pbr_tile_albedo", text="Albedo", toggle=True)
                        row.prop(scene, "pbr_tile_material", text="Material", toggle=True)
                        row.prop(scene, "pbr_tile_normal", text="Normal", toggle=True)
                        row.prop(scene, "pbr_tile_height", text="Height", toggle=True)
                        row.prop(scene, "pbr_tile_emission", text="Emission", toggle=True)
                    if not scene.pbr_use_native_resolution:
                        row = sub.row()
                        row.prop(scene, "pbr_processing_resolution", text="Processing Resolution")
                    row = sub.row(align=True)
                    row.prop(scene, "pbr_denoise_steps", text="Denoise Steps")
                    row.prop(scene, "pbr_ensemble_size", text="Ensemble Size")
                    sub.separator()
                    row = sub.row()
                    row.prop(scene, "pbr_replace_color_with_albedo", text="Use Albedo as Base Color", toggle=True, icon="SHADING_SOLID")
                    row = sub.row()
                    row.prop(scene, "pbr_auto_lighting", text="Studio Lighting", toggle=True, icon="LIGHT_AREA")

                content_box.separator()
                split = content_box.split(factor=0.5)
                split.label(text="Fallback Color:")
                split.prop(scene, "fallback_color", text="")

                row = content_box.row()
                row.prop(scene, "auto_rescale", text="Auto Rescale Resolution", toggle=True, icon="ARROW_LEFTRIGHT")
                if scene.auto_rescale:
                    sub_box_rescale = content_box.box()
                    row = sub_box_rescale.row()
                    row.prop(scene, "auto_rescale_target_mp", text="Target Megapixels")
                    if scene.model_architecture.startswith('qwen'):
                        row = sub_box_rescale.row()
                        row.prop(scene, "qwen_rescale_alignment", text="Qwen VL-Aligned Rescale (112px)", toggle=True, icon="SNAP_INCREMENT")
                row = content_box.row()
                if 'overwrite_material' in _diff_props:
                    row.alert = True
                row.prop(scene, "overwrite_material", text="Overwrite Material", toggle=True, icon="FILE_REFRESH")
                if 'overwrite_material' in _diff_props:
                    row.label(text="→ " + _diff_props['overwrite_material'])

            # --- Image Guidance (IPAdapter & ControlNet) ---
            if _show_diffusion_sections:
                if scene.model_architecture in ['sdxl', 'flux1']:
                    content_box = draw_collapsible_section(advanced_params_box, "show_image_guidance_settings", "Image Guidance (IPAdapter & ControlNet)", icon="MODIFIER")
                elif scene.model_architecture == 'flux2_klein':
                    content_box = draw_collapsible_section(advanced_params_box, "show_image_guidance_settings", "FLUX.2 Klein Guidance", icon="MODIFIER")
                else: # Qwen Image Edit
                    content_box = draw_collapsible_section(advanced_params_box, "show_image_guidance_settings", "Qwen-Image-Edit Guidance", icon="MODIFIER")
            else:
                content_box = None
            if content_box:
                if scene.model_architecture in ('qwen_image_edit', 'flux2_klein'):
                    if scene.model_architecture == 'flux2_klein' or scene.qwen_generation_method == 'generate':
                        split = content_box.split(factor=0.5)
                        split.label(text="Guidance Map:")
                        split.prop(scene, "qwen_guidance_map_type", text="")

                        row = content_box.row()
                        row.prop(scene, "qwen_use_external_style_image", text="Use External Image as Style", toggle=True, icon="FILE_IMAGE")

                        # TRELLIS.2 input as style (only in trellis2 architecture mode)
                        if getattr(scene, 'architecture_mode', '') == 'trellis2':
                            row = content_box.row()
                            row.prop(scene, "qwen_use_trellis2_style", text="Use TRELLIS.2 Input as Style", toggle=True, icon="MESH_MONKEY")

                        if scene.qwen_use_external_style_image:
                            style_box = content_box.box()
                            row = style_box.row()
                            row.prop(scene, "qwen_external_style_image", text="Style Image")
                            row = style_box.row()
                            row.prop(scene, "qwen_external_style_initial_only", text="External for Initial Only", toggle=True)

                        if scene.qwen_use_external_style_image and scene.qwen_external_style_initial_only:
                            subsequent_box = style_box.box()
                            split = subsequent_box.split(factor=0.5)
                            split.label(text="Subsequent mode:")
                            split.prop(scene, "sequential_ipadapter_mode", text="")
                            if scene.sequential_ipadapter_mode == 'recent':
                                subsequent_box.prop(scene, "sequential_desaturate_factor", text="Desaturate")
                                subsequent_box.prop(scene, "sequential_contrast_factor", text="Reduce Contrast")

                        if getattr(scene, 'architecture_mode', '') == 'trellis2' and scene.qwen_use_trellis2_style:
                            t2_style_box = content_box.box()
                            row = t2_style_box.row()
                            row.prop(scene, "qwen_trellis2_style_initial_only", text="TRELLIS.2 Style for Initial Only", toggle=True)
                            if scene.qwen_trellis2_style_initial_only:
                                subsequent_box = t2_style_box.box()
                                split = subsequent_box.split(factor=0.5)
                                split.label(text="Subsequent mode:")
                                split.prop(scene, "sequential_ipadapter_mode", text="")
                                if scene.sequential_ipadapter_mode == 'recent':
                                    subsequent_box.prop(scene, "sequential_desaturate_factor", text="Desaturate")
                                    subsequent_box.prop(scene, "sequential_contrast_factor", text="Reduce Contrast")

                        if not scene.qwen_use_external_style_image and scene.generation_method in ['sequential', 'separate']:
                            row = content_box.row()
                            row.prop(scene, "sequential_ipadapter", text="Use Previous Image as Style", toggle=True, icon="MODIFIER")
                            if scene.sequential_ipadapter:
                                sub_ip_box = content_box.box()
                                split = sub_ip_box.split(factor=0.5)
                                split.label(text="Mode:")
                                split.prop(scene, "sequential_ipadapter_mode", text="")
                                if scene.sequential_ipadapter_mode == 'recent':
                                    sub_ip_box.prop(scene, "sequential_desaturate_factor", text="Desaturate")
                                    sub_ip_box.prop(scene, "sequential_contrast_factor", text="Reduce Contrast")

                        if scene.generation_method == 'sequential':
                            split = content_box.split(factor=0.5)
                            split.label(text="Context Render:")
                            split.prop(scene, "qwen_context_render_mode", text="")

                            row = content_box.row()
                            row.prop(scene, "qwen_voronoi_mode", text="Voronoi Projection", toggle=True, icon="MESH_GRID")
                    
                    elif scene.qwen_generation_method in ('refine', 'local_edit'):
                        row = content_box.row()
                        row.prop(scene, "qwen_refine_use_prev_ref", text="Use Previous Refined View", toggle=True)
                        
                        row = content_box.row()
                        row.prop(scene, "qwen_refine_use_depth", text="Use Depth Map", toggle=True, icon="MODIFIER")
                        
                        row = content_box.row()
                        row.prop(scene, "qwen_use_external_style_image", text="Use External Image as Style", toggle=True, icon="FILE_IMAGE")

                        # TRELLIS.2 input as style (only in trellis2 architecture mode)
                        if getattr(scene, 'architecture_mode', '') == 'trellis2':
                            row = content_box.row()
                            row.prop(scene, "qwen_use_trellis2_style", text="Use TRELLIS.2 Input as Style", toggle=True, icon="MESH_MONKEY")

                        if scene.qwen_use_external_style_image:
                            style_box = content_box.box()
                            row = style_box.row()
                            row.prop(scene, "qwen_external_style_image", text="Style Image")
                    
                    # Timestep-zero reference method — shared across all Qwen modes
                    row = content_box.row()
                    row.prop(scene, "qwen_timestep_zero_ref", text="Timestep-Zero References (color shift fix)", toggle=True, icon="COLORSET_08_VEC")

                    row = content_box.row()
                    row.prop(scene, "qwen_use_custom_prompts", text="Custom Guidance Prompts", toggle=True, icon="TEXT")
                    if scene.qwen_use_custom_prompts:
                        custom_prompt_box = content_box.box()
                        
                        # Initial Image Prompt
                        col = custom_prompt_box.column()
                        col.label(text="Initial Image Prompt:")
                        row = col.row(align=True)
                        row.prop(scene, "qwen_custom_prompt_initial", text="")
                        op = row.operator("stablegen.reset_qwen_prompt", text="", icon='FILE_REFRESH')
                        op.prompt_type = 'initial'

                        # Subsequent Images Prompt (conditional)
                        if scene.generation_method == 'sequential' or (scene.qwen_generation_method in ('refine', 'local_edit') and scene.qwen_refine_mode == 'sequential'):
                            col = custom_prompt_box.column()
                            col.label(text="Subsequent Images Prompt:")
                            row = col.row(align=True)
                            
                            if scene.qwen_context_render_mode == 'NONE' and scene.qwen_generation_method == 'generate':
                                row.prop(scene, "qwen_custom_prompt_seq_none", text="")
                                op_prop = 'seq_none'
                            elif scene.qwen_context_render_mode == 'REPLACE_STYLE' and scene.qwen_generation_method == 'generate':
                                row.prop(scene, "qwen_custom_prompt_seq_replace", text="")
                                op_prop = 'seq_replace'
                            elif scene.qwen_context_render_mode == 'ADDITIONAL' and scene.qwen_generation_method == 'generate':
                                row.prop(scene, "qwen_custom_prompt_seq_additional", text="")
                                op_prop = 'seq_additional'
                            else: # Refine mode or other
                                row.prop(scene, "qwen_custom_prompt_seq_none", text="")
                                op_prop = 'seq_none'
                            
                            op = row.operator("stablegen.reset_qwen_prompt", text="", icon='FILE_REFRESH')
                            op.prompt_type = op_prop

                    if (scene.generation_method == 'sequential' and scene.qwen_generation_method == 'generate' and
                            scene.qwen_context_render_mode in {'REPLACE_STYLE', 'ADDITIONAL'}):
                        context_box = content_box.box()
                        context_box.label(text="Context Render Options")

                        row = context_box.row()
                        row.prop(scene, "qwen_prompt_gray_background", text="Prompt: Gray Background", toggle=True)

                        if scene.qwen_use_custom_prompts:
                            colors_box = context_box.box()
                            colors_box.label(text="Context Render Colors")
                            row = colors_box.row()
                            row.prop(scene, "qwen_guidance_background_color", text="Background")
                            row = colors_box.row()
                            row.prop(scene, "qwen_guidance_fallback_color", text="Fallback")

                        dilation_row = context_box.row()
                        dilation_row.prop(scene, "qwen_context_fallback_dilation", text="Fallback Dilate (px)")

                        cleanup_row = context_box.row()
                        cleanup_row.prop(scene, "qwen_context_cleanup", text="Apply Cleanup", toggle=True, icon="BRUSH_DATA")
                        if scene.qwen_context_cleanup:
                            row = context_box.row()
                            row.prop(scene, "qwen_context_cleanup_hue_tolerance", text="Hue Tol (°)")
                            row = context_box.row()
                            row.prop(scene, "qwen_context_cleanup_value_adjust", text="Value Adjust")

                elif scene.model_architecture == 'sdxl' or scene.model_architecture == 'flux1':
                    # IPAdapter Parameters
                    if not scene.generation_method == 'uv_inpaint':
                        ipadapter_main_box = content_box.box() # Group IPAdapter settings together
                        if scene.model_architecture == 'flux1':
                            row = ipadapter_main_box.row()
                            row.prop(scene, "use_flux_lora", text="Use Flux Depth LoRA", toggle=True, icon="MODIFIER")
                        row = ipadapter_main_box.row()
                        row.prop(scene, "use_ipadapter", text="Use IPAdapter (External image)", toggle=True, icon="MOD_MULTIRES")
                        if scene.use_ipadapter:
                            sub_ip_box = ipadapter_main_box.box() 
                            row = sub_ip_box.row()
                            row.prop(scene, "ipadapter_image", text="Image")
                            row = sub_ip_box.row()
                            row.prop(scene, "ipadapter_strength", text="Strength")
                            if width_mode == 'narrow':
                                row = sub_ip_box.row()
                            row.prop(scene, "ipadapter_start", text="Start")
                            if width_mode == 'narrow':
                                row = sub_ip_box.row()
                            row.prop(scene, "ipadapter_end", text="End")
                            split = sub_ip_box.split(factor=0.5)
                            if context.scene.model_architecture == 'sdxl':
                                split.label(text="Weight Type:")
                                split.prop(scene, "ipadapter_weight_type", text="")
                    
                    content_box.separator() # Separator between IPAdapter and ControlNet if both are shown
                    # ControlNet Parameters
                    if not (scene.model_architecture == 'flux1' and scene.use_flux_lora):
                        cn_box = content_box.box()
                        row = cn_box.row()
                        row.alignment = 'CENTER'
                        row.label(text="ControlNet Units", icon="NODETREE")
                        for i, unit in enumerate(scene.controlnet_units): 
                            sub_unit_box = cn_box.box() # Each unit gets its own box
                            row = sub_unit_box.row()
                            row.label(text=f"Unit: {unit.unit_type.replace('_', ' ').title()}", icon="DOT") 
                            row.alignment = 'LEFT' 
                            
                            if width_mode == 'narrow':
                                split = sub_unit_box.split(factor=0.35, align=True) 
                            else:
                                split = sub_unit_box.split(factor=0.2, align=True) 
                            split.label(text="Model:")
                            split.prop(unit, "model_name", text="")
                            
                            row = sub_unit_box.row()
                            row.prop(unit, "strength", text="Strength")
                            if width_mode == 'narrow':
                                row = sub_unit_box.row()
                            row.prop(unit, "start_percent", text="Start")
                            if width_mode == 'narrow':
                                row = sub_unit_box.row()
                            row.prop(unit, "end_percent", text="End")
                            
                            if unit.unit_type == 'canny':
                                row = sub_unit_box.row()
                                row.prop(scene, "canny_threshold_low", text="Canny Low")
                                if width_mode == 'narrow':
                                    row = sub_unit_box.row()
                                row.prop(scene, "canny_threshold_high", text="Canny High")
                            if hasattr(unit, 'is_union') and unit.is_union: 
                                row = sub_unit_box.row()
                                row.prop(unit, "use_union_type", text="Set Union Type", toggle=True, icon="MOD_BOOLEAN")
                        
                        btn_row = cn_box.row(align=True) 
                        if width_mode == 'wide':
                            btn_row.operator("stablegen.add_controlnet_unit", text="Add Unit", icon="ADD")
                            btn_row.operator("stablegen.remove_controlnet_unit", text="Remove Unit", icon="REMOVE")
                        else:
                            cn_box.operator("stablegen.add_controlnet_unit", text="Add ControlNet Unit", icon="ADD")
                            cn_box.operator("stablegen.remove_controlnet_unit", text="Remove Last ControlNet Unit", icon="REMOVE")

            if _show_diffusion_sections and scene.model_architecture not in ('qwen_image_edit', 'flux2_klein'):
                # --- Inpainting Options (Conditional) ---
                if scene.generation_method == 'uv_inpaint' or scene.generation_method == 'sequential':
                    content_box = draw_collapsible_section(advanced_params_box, "show_masking_inpainting_settings", "Inpainting Options", icon="MOD_MASK")
                    if content_box: # content_box is the container for these settings
                        row = content_box.row()
                        if 'differential_diffusion' in _diff_props:
                            row.alert = True
                        row.prop(scene, "differential_diffusion", text="Use Differential Diffusion", toggle=True, icon="SMOOTHCURVE")
                        if 'differential_diffusion' in _diff_props:
                            row.label(text="→ " + _diff_props['differential_diffusion'])
                        
                        if scene.differential_diffusion:
                            row = content_box.row()
                            row.prop(scene, "differential_noise", text="Add Latent Noise Mask", toggle=True, icon="MOD_NOISE")

                        if not (scene.differential_diffusion and not scene.differential_noise): 
                            row = content_box.row()
                            row.prop(scene, "mask_blocky", text="Use Blocky Mask", icon="MOD_MASK") 
                            
                            if width_mode == 'narrow':
                                row = content_box.row()
                                
                            row.prop(scene, "blur_mask", text="Blur Mask", toggle=True, icon="SURFACE_NSPHERE")

                            if scene.blur_mask:
                                row = content_box.row()
                                row.prop(scene, "blur_mask_radius", text="Blur Radius")
                                if width_mode == 'narrow':
                                    row = content_box.row()
                                row.prop(scene, "blur_mask_sigma", text="Blur Sigma")

                            row = content_box.row() # Draw directly in content_box
                            row.prop(scene, "grow_mask_by", text="Grow Mask By")


            # --- Generation Mode Specifics ---
            if _show_diffusion_sections:
                mode_specific_outer_box = draw_collapsible_section(advanced_params_box, "show_mode_specific_settings", "Generation Mode Specifics", icon="OPTIONS")
            else:
                mode_specific_outer_box = None
            if mode_specific_outer_box: # This is the box where all mode-specific UIs should go
                
                # Qwen Local Edit Mode Parameters
                if scene.model_architecture.startswith('qwen') and scene.qwen_generation_method == 'local_edit':
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="Qwen Local Edit Parameters", icon='BRUSH_DATA')
                    row = mode_specific_outer_box.row()
                    if 'denoise' in _diff_props:
                        row.alert = True
                    row.prop(scene, "denoise", text="Denoise")
                    if 'denoise' in _diff_props:
                        row.label(text="→ " + _diff_props['denoise'])
                    
                    # Angle Ramp Controls
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "refine_angle_ramp_active", text="Use Angle-Based Blending", icon="DRIVER")
                    if scene.refine_angle_ramp_active:
                        row = box.row()
                        row.prop(scene, "refine_angle_ramp_pos_0", text="Black Point")
                        row.prop(scene, "refine_angle_ramp_pos_1", text="White Point")
                    
                    # Feather Ramp Controls
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "visibility_vignette", text="Use Vignette Blending", icon="DRIVER")
                    if scene.visibility_vignette:
                        row = box.row()
                        row.prop(scene, "refine_feather_ramp_pos_0", text="Black Point")
                        row.prop(scene, "refine_feather_ramp_pos_1", text="White Point")
                        row = box.row()
                        row.prop(scene, "visibility_vignette_width", text="Feather Width")
                        if width_mode == 'narrow':
                            row = box.row()
                        row.prop(scene, "visibility_vignette_softness", text="Feather Softness")
                        row = box.row()
                        row.prop(scene, "visibility_vignette_blur", text="Blur Mask", icon="SURFACE_NSPHERE")

                    # Edge Feather Projection Controls
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "refine_edge_feather_projection", text="Edge Feather (Projection)", icon="MOD_EDGESPLIT")
                    if scene.refine_edge_feather_projection:
                        row = box.row()
                        row.prop(scene, "refine_edge_feather_width", text="Feather Width (px)")
                        row = box.row()
                        row.prop(scene, "refine_edge_feather_softness", text="Feather Softness")

                    # Color Matching
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "view_blend_use_color_match", text="Match Colors to Viewport", toggle=True, icon="COLOR")
                    if scene.view_blend_use_color_match:
                        row = box.row(align=True)
                        row.prop(scene, "view_blend_color_match_method", text="Method")
                        row = box.row()
                        row.prop(scene, "view_blend_color_match_strength", text="Strength")

                # Qwen Refine Mode Parameters
                elif scene.model_architecture.startswith('qwen') and scene.qwen_generation_method == 'refine':
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="Qwen Refine Parameters", icon='SHADERFX')
                    row = mode_specific_outer_box.row()
                    if 'denoise' in _diff_props:
                        row.alert = True
                    row.prop(scene, "denoise", text="Denoise")
                    if 'denoise' in _diff_props:
                        row.label(text="→ " + _diff_props['denoise'])

                # Grid Mode Parameters
                elif scene.generation_method == 'grid':
                    # Draw Grid parameters directly into mode_specific_outer_box
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="Grid Mode Parameters", icon="MESH_GRID")
                    
                    row = mode_specific_outer_box.row()
                    row.prop(scene, "refine_images", text="Refine Images", toggle=True, icon="SHADERFX")
                    if scene.refine_images:
                        split = mode_specific_outer_box.split(factor=0.5)
                        split.label(text="Refine Sampler:")
                        split.prop(scene, "refine_sampler", text="")
                        
                        split = mode_specific_outer_box.split(factor=0.5)
                        split.label(text="Refine Scheduler:")
                        split.prop(scene, "refine_scheduler", text="")
                        
                        row = mode_specific_outer_box.row()
                        if 'denoise' in _diff_props:
                            row.alert = True
                        row.prop(scene, "denoise", text="Denoise")
                        if 'denoise' in _diff_props:
                            row.label(text="→ " + _diff_props['denoise'])
                        if width_mode == 'narrow':
                            row = mode_specific_outer_box.row()
                        row.prop(scene, "refine_cfg", text="Refine CFG")
                        if width_mode == 'narrow':
                            row = mode_specific_outer_box.row()
                        row.prop(scene, "refine_steps", text="Refine Steps")

                        row = mode_specific_outer_box.row() 
                        split = mode_specific_outer_box.split(factor=0.25)
                        split.label(text="Refine Prompt:")
                        split.prop(scene, "refine_prompt", text="")
                        
                        split = mode_specific_outer_box.split(factor=0.5) 
                        split.label(text="Refine Upscale:") 
                        split.prop(scene, "refine_upscale_method", text="")

                # Separate Mode Parameters
                elif scene.generation_method == 'separate':
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="Separate Mode Parameters", icon='FORCE_FORCE')
                    
                    row = mode_specific_outer_box.row() 
                    row.prop(scene, "sequential_ipadapter", text="Use IPAdapter for Separate Mode", toggle=True, icon="MODIFIER")
                    if scene.sequential_ipadapter: 
                        sub_ip_box_separate = mode_specific_outer_box.box()
                        
                        split = sub_ip_box_separate.split(factor=0.5) 
                        split.label(text="Mode:")
                        split.prop(scene, "sequential_ipadapter_mode", text="") 

                        if scene.sequential_ipadapter_mode == 'recent':
                            sub_ip_box_separate.prop(scene, "sequential_desaturate_factor", text="Desaturate")
                            sub_ip_box_separate.prop(scene, "sequential_contrast_factor", text="Reduce Contrast")

                        if context.scene.model_architecture not in ('qwen_image_edit', 'flux2_klein'):
                            split = sub_ip_box_separate.split(factor=0.5) 
                            if context.scene.model_architecture == 'sdxl':
                                split.label(text="Weight Type:")
                                split.prop(scene, "ipadapter_weight_type", text="")
                        
                        row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_strength", text="Strength")
                        if width_mode == 'narrow':
                            row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_start", text="Start")
                        if width_mode == 'narrow':
                            row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_end", text="End")    
                        
                        if context.scene.sequential_ipadapter_mode == 'first':
                            row = sub_ip_box_separate.row()
                            row.prop(scene, "sequential_ipadapter_regenerate", text="Regenerate First Image", toggle=True, icon="FILE_REFRESH")
                            if context.scene.sequential_ipadapter_regenerate:
                                row = sub_ip_box_separate.row()
                                row.prop(scene, "sequential_ipadapter_regenerate_wo_controlnet", text="Generate reference without ControlNet", toggle=True, icon="HIDE_OFF")

                # Refine Mode Parameters
                elif scene.generation_method == 'refine':
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="Refine Mode Parameters", icon='SHADERFX')
                    row = mode_specific_outer_box.row()
                    if 'denoise' in _diff_props:
                        row.alert = True
                    row.prop(scene, "denoise", text="Denoise")
                    if 'denoise' in _diff_props:
                        row.label(text="→ " + _diff_props['denoise'])
                    row = mode_specific_outer_box.row() 
                    row.prop(scene, "sequential_ipadapter", text="Use IPAdapter for Refine Mode", toggle=True, icon="MODIFIER")
                    if scene.sequential_ipadapter: 
                        sub_ip_box_separate = mode_specific_outer_box.box()
                        
                        split = sub_ip_box_separate.split(factor=0.5) 
                        split.label(text="Mode:")
                        split.prop(scene, "sequential_ipadapter_mode", text="") 

                        split = sub_ip_box_separate.split(factor=0.5) 
                        if context.scene.model_architecture == 'sdxl':
                            split.label(text="Weight Type:")
                            split.prop(scene, "ipadapter_weight_type", text="")
                        
                        row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_strength", text="Strength")
                        if width_mode == 'narrow':
                            row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_start", text="Start")
                        if width_mode == 'narrow':
                            row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_end", text="End")    
                        
                        if context.scene.sequential_ipadapter_mode == 'first':
                            row = sub_ip_box_separate.row()
                            row.prop(scene, "sequential_ipadapter_regenerate", text="Regenerate First Image", toggle=True, icon="FILE_REFRESH")
                            if context.scene.sequential_ipadapter_regenerate:
                                row = sub_ip_box_separate.row()
                                row.prop(scene, "sequential_ipadapter_regenerate_wo_controlnet", text="Generate reference without ControlNet", toggle=True, icon="HIDE_OFF")

                # Local Edit Mode Parameters
                elif scene.generation_method == 'local_edit':
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="Local Edit Parameters", icon='BRUSH_DATA')
                    row = mode_specific_outer_box.row()
                    if 'denoise' in _diff_props:
                        row.alert = True
                    row.prop(scene, "denoise", text="Denoise")
                    if 'denoise' in _diff_props:
                        row.label(text="→ " + _diff_props['denoise'])
                    
                    # Angle Ramp Controls
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "refine_angle_ramp_active", text="Use Angle-Based Blending", icon="DRIVER")
                    if scene.refine_angle_ramp_active:
                        row = box.row()
                        row.prop(scene, "refine_angle_ramp_pos_0", text="Black Point")
                        row.prop(scene, "refine_angle_ramp_pos_1", text="White Point")
                    
                    # Feather Ramp Controls
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "visibility_vignette", text="Use Vignette Blending", icon="DRIVER")
                    if scene.visibility_vignette:
                        row = box.row()
                        row.prop(scene, "refine_feather_ramp_pos_0", text="Black Point")
                        row.prop(scene, "refine_feather_ramp_pos_1", text="White Point")
                        row = box.row()
                        row.prop(scene, "visibility_vignette_width", text="Feather Width")
                        if width_mode == 'narrow':
                            row = box.row()
                        row.prop(scene, "visibility_vignette_softness", text="Feather Softness")
                        row = box.row()
                        row.prop(scene, "visibility_vignette_blur", text="Blur Mask", icon="SURFACE_NSPHERE")

                    # Edge Feather Projection Controls
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "refine_edge_feather_projection", text="Edge Feather (Projection)", icon="MOD_EDGESPLIT")
                    if scene.refine_edge_feather_projection:
                        row = box.row()
                        row.prop(scene, "refine_edge_feather_width", text="Feather Width (px)")
                        row = box.row()
                        row.prop(scene, "refine_edge_feather_softness", text="Feather Softness")

                    # Color Matching
                    box = mode_specific_outer_box.box()
                    row = box.row()
                    row.prop(scene, "view_blend_use_color_match", text="Match Colors to Viewport", toggle=True, icon="COLOR")
                    if scene.view_blend_use_color_match:
                        row = box.row(align=True)
                        row.prop(scene, "view_blend_color_match_method", text="Method")
                        row = box.row()
                        row.prop(scene, "view_blend_color_match_strength", text="Strength")

                    row = mode_specific_outer_box.row() 
                    row.prop(scene, "sequential_ipadapter", text="Use IPAdapter for Local Edit", toggle=True, icon="MODIFIER")
                    if scene.sequential_ipadapter: 
                        sub_ip_box_separate = mode_specific_outer_box.box()
                        
                        split = sub_ip_box_separate.split(factor=0.5) 
                        split.label(text="Mode:")
                        split.prop(scene, "sequential_ipadapter_mode", text="") 

                        split = sub_ip_box_separate.split(factor=0.5) 
                        if context.scene.model_architecture == 'sdxl':
                            split.label(text="Weight Type:")
                            split.prop(scene, "ipadapter_weight_type", text="")
                        
                        row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_strength", text="Strength")
                        if width_mode == 'narrow':
                            row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_start", text="Start")
                        if width_mode == 'narrow':
                            row = sub_ip_box_separate.row()
                        row.prop(scene, "ipadapter_end", text="End")    
                        
                        if context.scene.sequential_ipadapter_mode == 'first':
                            row = sub_ip_box_separate.row()
                            row.prop(scene, "sequential_ipadapter_regenerate", text="Regenerate First Image", toggle=True, icon="FILE_REFRESH")
                            if context.scene.sequential_ipadapter_regenerate:
                                row = sub_ip_box_separate.row()
                                row.prop(scene, "sequential_ipadapter_regenerate_wo_controlnet", text="Generate reference without ControlNet", toggle=True, icon="HIDE_OFF")
                
                # UV Inpainting Parameters
                elif scene.generation_method == 'uv_inpaint':
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="UV Inpainting Parameters", icon="IMAGE_PLANE")
                    row = mode_specific_outer_box.row()
                    row.prop(scene, "allow_modify_existing_textures", text="Allow Modifying Existing Textures", toggle=True, icon="TEXTURE")
                    row = mode_specific_outer_box.row()
                    row.prop(scene, "ask_object_prompts", text="Ask for Object Specific Prompts", toggle=True, icon="QUESTION")

                # Sequential Mode Parameters
                elif scene.generation_method == 'sequential':
                    row = mode_specific_outer_box.row()
                    row.alignment = 'CENTER'
                    row.label(text="Sequential Mode Parameters", icon="SEQUENCE")
                    
                    if not (scene.differential_diffusion and not scene.differential_noise): 
                        row = mode_specific_outer_box.row()
                        if 'sequential_smooth' in _diff_props:
                            row.alert = True
                        row.prop(scene, "sequential_smooth", text="Use Smooth Visibility Map", toggle=True, icon="MOD_SMOOTH")
                        if 'sequential_smooth' in _diff_props:
                            row.label(text="→ " + _diff_props['sequential_smooth'])
                        if width_mode == 'narrow':
                            row = mode_specific_outer_box.row()
                        row.prop(scene, "weight_exponent_mask", text="Exponent for Visibility Map", toggle=True, icon="IPO_EXPO") 
                        
                        if not scene.sequential_smooth:
                            row = mode_specific_outer_box.row()
                            row.prop(scene, "sequential_factor", text="Visibility Threshold") 
                        else:
                            row = mode_specific_outer_box.row()
                            row.prop(scene, "sequential_factor_smooth", text="Smooth Visibility Black Point")
                            if width_mode == 'narrow':
                                row = mode_specific_outer_box.row()
                            row.prop(scene, "sequential_factor_smooth_2", text="Smooth Visibility White Point")
                    
                    row = mode_specific_outer_box.row()
                    row.prop(scene, "sequential_ipadapter", text="Use IPAdapter for Sequential Mode", toggle=True, icon="MODIFIER")
                    if scene.sequential_ipadapter:
                        sub_ip_seq_box = mode_specific_outer_box.box()
                        
                        split = sub_ip_seq_box.split(factor=0.5)
                        split.label(text="Mode:")
                        split.prop(scene, "sequential_ipadapter_mode", text="")

                        if scene.sequential_ipadapter_mode == 'recent':
                            sub_ip_seq_box.prop(scene, "sequential_desaturate_factor", text="Desaturate")
                            sub_ip_seq_box.prop(scene, "sequential_contrast_factor", text="Reduce Contrast")

                        if context.scene.model_architecture not in ('qwen_image_edit', 'flux2_klein'):
                            split = sub_ip_seq_box.split(factor=0.5)
                            if context.scene.model_architecture == 'sdxl':
                                split.label(text="Weight Type:")
                                split.prop(scene, "ipadapter_weight_type", text="")
                        
                        row = sub_ip_seq_box.row()
                        row.prop(scene, "ipadapter_strength", text="Strength")
                        if width_mode == 'narrow':
                            row = sub_ip_seq_box.row()
                        row.prop(scene, "ipadapter_start", text="Start")
                        if width_mode == 'narrow':  
                            row = sub_ip_seq_box.row()
                        row.prop(scene, "ipadapter_end", text="End")     
                        
                        if context.scene.sequential_ipadapter_mode == 'first':
                            row = sub_ip_seq_box.row()
                            row.prop(scene, "sequential_ipadapter_regenerate", text="Regenerate First Image", toggle=True, icon="FILE_REFRESH")
                            if context.scene.sequential_ipadapter_regenerate:
                                row = sub_ip_seq_box.row()
                                row.prop(scene, "sequential_ipadapter_regenerate_wo_controlnet", text="Generate reference without ControlNet", toggle=True, icon="HIDE_OFF")   

        # --- Tools ---
        layout.separator()
        tools_box = layout.box()
        row = tools_box.row()
        row.alignment = 'CENTER'
        row.label(text="Tools", icon="TOOL_SETTINGS")
        
        row = tools_box.row() 
        row.operator("object.switch_material", text="Switch Material", icon="MATERIAL_DATA")
        if width_mode == 'narrow':
            row = tools_box.row()
        row.operator("object.add_hdri", text="Add HDRI Light", icon="WORLD")
        
        row = tools_box.row()
        row.operator("object.apply_all_mesh_modifiers", text="Apply All Modifiers", icon="MODIFIER_DATA") 
        if width_mode == 'narrow':
            row = tools_box.row()
        row.operator("object.curves_to_mesh", text="Convert Curves to Mesh", icon="CURVE_DATA")

        if hasattr(bpy.ops.stablegen, 'import_dae'):
            row = tools_box.row()
            row.operator("stablegen.import_dae", text="Import DAE", icon="IMPORT")
            if width_mode == 'narrow':
                row = tools_box.row()
            row.operator("stablegen.batch_import_dae", text="Batch Import DAE", icon="FILE_FOLDER")
        
        row = tools_box.row()
        if config_error_message:
            row.enabled = False
            row.operator("object.export_orbit_gif", text=f"Cannot Export: {config_error_message}", icon="ERROR")
        else:
            row.enabled = True
            row.operator("object.export_orbit_gif", text="Export Orbit GIF/MP4", icon="RENDER_ANIMATION")

        if width_mode == 'narrow':
            row = tools_box.row()
        row.operator("object.stablegen_reproject", text="Reproject Textures", icon="FILE_REFRESH")

        row = tools_box.row()
        row.operator("object.stablegen_mirror_reproject", text="Mirror Last Projection", icon="MOD_MIRROR")

        # --- Debug Tools ---
        prefs = context.preferences.addons.get(_ADDON_PKG)
        if prefs and prefs.preferences.enable_debug:
            layout.separator()
            debug_box = layout.box()
            row = debug_box.row()
            row.alignment = 'CENTER'
            row.label(text="Debug / Diagnostics", icon="GHOST_ENABLED")

            row = debug_box.row()
            row.operator("stablegen.debug_solid_colors", text="Draw Solid Colors", icon="COLOR")
            if width_mode == 'narrow':
                row = debug_box.row()
            row.operator("stablegen.debug_grid_pattern", text="Grid Pattern", icon="MESH_GRID")

            row = debug_box.row()
            row.operator("stablegen.debug_coverage_heatmap", text="Coverage Heatmap", icon="AREA_SWAP")
            if width_mode == 'narrow':
                row = debug_box.row()
            row.operator("stablegen.debug_visibility_material", text="Visibility Material", icon="HIDE_OFF")

            row = debug_box.row()
            row.operator("stablegen.debug_uv_seam_viz", text="UV Seam Visualizer", icon="UV")
            if width_mode == 'narrow':
                row = debug_box.row()
            row.operator("stablegen.debug_restore_material", text="Remove Debug Mats", icon="TRASH")

            row = debug_box.row()
            op = row.operator("stablegen.debug_per_camera_weight", text="Per-Camera Weight", icon="CAMERA_DATA")
            if width_mode == 'narrow':
                row = debug_box.row()
            op2 = row.operator("stablegen.debug_feather_preview", text="Feather Preview", icon="MOD_SMOOTH")

        # --- Narrow panel hint ---
        if width_mode == 'narrow':
            hint_row = layout.row()
            hint_row.alignment = 'CENTER'
            hint_row.label(text="Widen panel for side-by-side layout", icon="INFO")

        layout.separator()
          
