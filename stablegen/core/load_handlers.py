"""Persistent load_post handler for scene defaults and cache sync.

The ``load_handler`` function runs on every .blend file load to:
- set default ControlNet / LoRA units
- re-register crop overlays
- trigger a checkpoint cache refresh if the architecture changed
- re-check TRELLIS.2 and PBR node availability (async)
"""

import bpy  # pylint: disable=import-error
from bpy.app.handlers import persistent

from . import ADDON_PKG
from .architectures import QWEN_ARCHITECTURES


@persistent
def load_handler(dummy, _second=None, file_loaded=True):
    """Set default ControlNet/LoRA units and sync checkpoint cache on file load.

    Server checks reuse it with file_loaded=False: a generation may be
    running then, so its status must not be reset. Blender 5.1 calls load_post
    handlers with two positional arguments (the file path and None), which
    must not land in *file_loaded*.
    """
    from ..ui.model_units import get_lora_models
    from . import state as _state
    from ..texturing.print_export import prepopulate_palette

    if not bpy.context.scene:
        return

    scene = bpy.context.scene

    # A file saved while a Masked Edit ran kept the edit's own output folder as
    # the run timestamp; Reproject and Regenerate then looked for the run there.
    # No edit survives a load, so put the run back.
    if file_loaded:
        for each in bpy.data.scenes:
            if '_sg_mask_run_timestamp' in each:
                each.output_timestamp = each['_sg_mask_run_timestamp']
                del each['_sg_mask_run_timestamp']
        # Bakes made before their normal maps named a UV map showed a triangle mosaic.
        from ..texturing.rendering import repair_baked_normal_maps
        repaired = repair_baked_normal_maps(bpy.data.materials)
        if repaired:
            print(f"[StableGen] Baked normal maps now decode with their bake UV map: {repaired}")
        # Masked Edits on a bake mixed world-space normals into its tangent-space map (black edits).
        from ..texturing.masked_edit import repair_edit_normal_layers
        repaired = repair_edit_normal_layers(bpy.data.materials)
        if repaired:
            print(f"[StableGen] Masked Edit normal layers now mix decoded normals: {repaired}")

    # Builds saved while Qwen-Image 2.1 shared the Qwen-Image-Edit enum need a
    # one-time architecture migration.  The filename is used only here for
    # compatibility; new generations dispatch exclusively by architecture.
    from ..texturing.qwen_models import (
        is_qwen_image_21_model, is_qwen_image_21_turbo_model,
    )
    saved_model_name = (
        getattr(scene, "sg_model_name_backup", "")
        or getattr(scene, "model_name", "")
    )
    if not getattr(scene, "sg_qwen_21_architecture_migrated", False):
        if (getattr(scene, "model_architecture", "") == 'qwen_image_edit'
                and is_qwen_image_21_model(saved_model_name)):
            target = ('qwen_image_21_turbo'
                      if is_qwen_image_21_turbo_model(saved_model_name)
                      else 'qwen_image_21')
            scene.model_architecture = target
            if getattr(scene, "architecture_mode", "") == 'qwen_image_edit':
                from ..ui.presets import preset_carry_held
                with preset_carry_held():           # the file keeps its own settings
                    scene.architecture_mode = target
            elif getattr(scene, "architecture_mode", "") == 'trellis2':
                if getattr(scene, "trellis2_texture_mode", "") == 'qwen_image_edit':
                    scene.trellis2_texture_mode = target
                if getattr(scene, "trellis2_initial_image_arch", "") == 'qwen_image_edit':
                    scene.trellis2_initial_image_arch = target
            print("[StableGen] Migrated saved Qwen-Image 2.1 model to its dedicated architecture.")
        scene.sg_qwen_21_architecture_migrated = True

    # The old Qwen UI stored custom/default as a boolean.  Preserve that
    # choice once when opening an existing file, then let the enum be the
    # single source of truth for both Qwen architectures.  Klein continues to
    # use the legacy boolean and is intentionally excluded.
    for loaded_scene in bpy.data.scenes:
        if getattr(loaded_scene, "sg_qwen_prompt_template_migrated", False):
            continue
        template_was_saved = loaded_scene.is_property_set(
            "qwen_prompt_template"
        )
        architecture = getattr(loaded_scene, "model_architecture", "")
        current_template = getattr(
            loaded_scene, "qwen_prompt_template", ""
        )
        if architecture in QWEN_ARCHITECTURES:
            valid_templates = {
                'QWEN_EDIT_LEGACY', 'QWEN_21_DEFAULT', 'CANVAS_EDIT',
                'REFERENCE_RECONSTRUCTION', 'CONCISE_TRANSFER',
                'RESULT_DESCRIPTION', 'CANVAS_AUTHORITY', 'MINIMAL_EDIT',
                'CUSTOM',
            }
            needs_legacy_migration = (
                not template_was_saved or current_template not in valid_templates
            )
            if (needs_legacy_migration
                    and getattr(
                        loaded_scene, "qwen_use_custom_prompts", False
                    )):
                loaded_scene.qwen_prompt_template = 'CUSTOM'
            elif needs_legacy_migration:
                loaded_scene.qwen_prompt_template = (
                    'QWEN_EDIT_LEGACY'
                    if architecture == 'qwen_image_edit'
                    else 'QWEN_21_DEFAULT'
                )
        loaded_scene.sg_qwen_prompt_template_migrated = True

    # Modal operators and worker sockets cannot survive loading a .blend.
    # Older files may nevertheless contain the serialized transient value
    # "running" or "waiting", which leaves the Generate button permanently
    # stuck after reopening. Runtime status is SKIP_SAVE now, but normalize
    # legacy files as well.
    for loaded_scene in (bpy.data.scenes if file_loaded else ()):
        if getattr(loaded_scene, "generation_status", 'idle') != 'idle':
            loaded_scene.generation_status = 'idle'
        if hasattr(loaded_scene, "generation_progress"):
            loaded_scene.generation_progress = 0.0

    # Pre-populate multi-color printing palette
    try:
        prepopulate_palette(scene)
    except Exception as e:
        print(f"[StableGen] Load Handler: failed to prepopulate palette: {e}")
    addon_prefs = bpy.context.preferences.addons[ADDON_PKG].preferences

    # Re-register aspect-ratio crop overlays
    try:
        from ..cameras.overlays import _sg_ensure_crop_overlay
        for obj in scene.objects:
            if obj.type == 'CAMERA' and 'sg_display_crop' in obj:
                _sg_ensure_crop_overlay()
                break
    except Exception:
        pass

    # Default ControlNet unit
    if hasattr(scene, "controlnet_units") and not scene.controlnet_units:
        default_unit = scene.controlnet_units.add()
        default_unit.unit_type = 'depth'

    # Restore ControlNet unit selections from backup
    if hasattr(scene, "controlnet_units"):
        for unit in scene.controlnet_units:
            backup = getattr(unit, "model_name_backup", "")
            if backup and backup not in ("REFRESH", "NO_ASSIGNED", "NO_PREFS"):
                try:
                    unit.model_name = backup
                except TypeError:
                    # Model not in enum items yet, which is fine
                    pass

    # Default LoRA unit
    if hasattr(scene, "lora_units") and not scene.lora_units:
        default_lora_filename_to_find = None
        model_strength = 1.0
        clip_strength = 1.0

        if scene.model_architecture == 'sdxl':
            default_lora_filename_to_find = 'sdxl_lightning_8step_lora.safetensors'
        elif scene.model_architecture == 'qwen_image_edit':
            default_lora_filename_to_find = 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors'
            clip_strength = 0.0

        if not default_lora_filename_to_find:
            pass  # No default LoRA for this architecture; skip LoRA setup
        else:
            all_available_loras_enums = get_lora_models(scene, bpy.context)

            found_lora_identifier_to_load = None
            for identifier, name, description in all_available_loras_enums:
                if identifier.endswith(default_lora_filename_to_find):
                    if identifier not in ("NONE_AVAILABLE", "NO_COMFYUI_DIR_LORA", "NO_LORAS_SUBDIR",
                                          "PERM_ERROR", "SCAN_ERROR", "NONE_FOUND"):
                        found_lora_identifier_to_load = identifier
                        break

            if found_lora_identifier_to_load:
                new_lora_unit = None
                try:
                    new_lora_unit = scene.lora_units.add()
                    new_lora_unit.model_name = found_lora_identifier_to_load
                    new_lora_unit.model_strength = model_strength
                    new_lora_unit.clip_strength = clip_strength
                except TypeError:
                    print(f"[StableGen] StableGen Load Handler: TypeError setting default LoRA "
                          f"'{found_lora_identifier_to_load}'. Enum items might not be fully ready.")
                    if new_lora_unit and scene.lora_units and new_lora_unit == scene.lora_units[-1]:
                        scene.lora_units.remove(len(scene.lora_units) - 1)
                except Exception as e:
                    print(f"[StableGen] StableGen Load Handler: Unexpected error setting default LoRA "
                          f"'{found_lora_identifier_to_load}': {e}")
                    if new_lora_unit and scene.lora_units and new_lora_unit == scene.lora_units[-1]:
                        scene.lora_units.remove(len(scene.lora_units) - 1)

    # Ensure checkpoint cache matches the scene architecture that just loaded
    current_architecture = getattr(scene, "model_architecture", None)
    prefs_wrapper = bpy.context.preferences.addons.get(ADDON_PKG)
    if current_architecture and prefs_wrapper:
        prefs = prefs_wrapper.preferences
        if (prefs.server_address
                and current_architecture != _state._cached_checkpoint_architecture
                and _state._pending_checkpoint_refresh_architecture != current_architecture):

            def _refresh_checkpoint_for_architecture():
                try:
                    bpy.ops.stablegen.refresh_checkpoint_list('INVOKE_DEFAULT')
                except Exception as timer_error:
                    print(f"[StableGen] StableGen Load Handler: Failed to refresh checkpoints for "
                          f"'{current_architecture}': {timer_error}")
                finally:
                    _state._pending_checkpoint_refresh_architecture = None
                return None

            _state._pending_checkpoint_refresh_architecture = current_architecture
            bpy.app.timers.register(_refresh_checkpoint_for_architecture, first_interval=0.2)

    # Re-check TRELLIS.2 and PBR node availability asynchronously.
    # Scene properties reset to False on file load; this restores them.
    if prefs_wrapper:
        server_address = prefs_wrapper.preferences.server_address
        if server_address:
            from .server_api import (
                check_pbr_available,
                check_skintokens_autorig_available,
                check_trellis2_available,
            )
            from ..timeout_config import get_timeout

            _timeout = get_timeout('api')

            def _check_nodes():
                result = {}
                result['trellis2'] = check_trellis2_available(server_address, timeout=_timeout)
                result['pbr'] = check_pbr_available(server_address, timeout=_timeout)
                result['skintokens'] = check_skintokens_autorig_available(server_address, timeout=_timeout)
                return result

            def _apply_node_result(result):
                if result is None:
                    return
                if hasattr(bpy.context, 'scene') and bpy.context.scene:
                    bpy.context.scene.trellis2_available = result.get('trellis2', False)
                    bpy.context.scene.pbr_nodes_available = result.get('pbr', False)
                    bpy.context.scene.skintokens_autorig_available = result.get('skintokens', False)

            _state._run_async(_check_nodes, _apply_node_result)
