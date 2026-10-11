"""ControlNet / LoRA PropertyGroups and their CRUD + refresh operators.

Contains:
- ``ControlNetUnit``, ``LoRAUnit`` — CollectionProperty item types
- ``get_controlnet_models``, ``get_lora_models`` — EnumProperty callbacks
- ``RefreshCheckpointList``, ``RefreshLoRAList`` — async refresh operators
- ``AddControlNetUnit``, ``RemoveControlNetUnit``
- ``AddLoRAUnit``, ``RemoveLoRAUnit``
- ``update_model_list``, ``update_union``, ``update_controlnet``
"""

import bpy  # pylint: disable=import-error

from .presets import update_parameters
from ..utils import sg_modal_active
from ..core import ADDON_PKG
from ..core.server_api import _fetch_api_list, fetch_diffusion_models
from ..texturing.qwen_models import is_qwen_image_21_model
from ..core.state import (
    _cached_checkpoint_list,
    _cached_checkpoint_architecture,
    _cached_lora_list,
    _dec_pending_refreshes,
    _inc_pending_refreshes,
    _run_async,
)


# ── Enum callbacks for model dropdowns ─────────────────────────────────────

def update_model_list(self, context):
    """Returns the cached list of checkpoint/unet models."""
    from ..core import state as _state
    if not _state._cached_checkpoint_list:
        return [("NONE_AVAILABLE", "None available", "Fetch models from server")]
    return _state._cached_checkpoint_list


def update_union(self, context):
    if "union" in self.model_name.lower() or "promax" in self.model_name.lower():
        self.is_union = True
    else:
        self.is_union = False


def update_controlnet(self, context):
    if hasattr(self, "model_name") and self.model_name and self.model_name not in ("REFRESH", "NO_ASSIGNED", "NO_PREFS"):
        self.model_name_backup = self.model_name
    update_parameters(self, context)
    update_union(self, context)
    return None


def get_controlnet_models(context, unit_type):
    """Get available ControlNet models suitable for *unit_type*."""
    items = []
    prefs = context.preferences.addons.get(ADDON_PKG)
    if not prefs:
        return [("NO_PREFS", "Addon Error", "Could not access preferences")]

    mappings = prefs.preferences.controlnet_model_mappings

    if not mappings:
        return [("REFRESH", "Refresh List in Prefs", "Fetch models via Preferences")]

    prop_name = f"supports_{unit_type}"

    found_count = 0
    for item in mappings:
        if hasattr(item, prop_name):
            if getattr(item, prop_name):
                items.append((item.name, item.name, f"ControlNet: {item.name}"))
                found_count += 1

    if found_count == 0:
        return [("NO_ASSIGNED", f"No models assigned to '{unit_type}'",
                 "Assign types in Addon Preferences or Refresh")]

    # Sort compatible models first based on the current architecture mode
    scene = context.scene if hasattr(context, "scene") else None
    arch = getattr(scene, "architecture_mode", "sdxl") if scene else "sdxl"
    is_flux = "flux" in arch.lower()

    def sort_key(item_tuple):
        name_lower = item_tuple[0].lower()
        if is_flux:
            is_compat = ("flux" in name_lower)
        else:
            is_compat = ("flux" not in name_lower)
        return (-int(is_compat), item_tuple[1].lower())

    items.sort(key=sort_key)
    return items


def get_lora_models(self, context):
    """Returns the cached list of LoRA models."""
    from ..core import state as _state
    if not _state._cached_lora_list:
        return [("NONE_AVAILABLE", "None available", "Fetch models from server")]
    return _state._cached_lora_list


# ── PropertyGroups ─────────────────────────────────────────────────────────

class ControlNetUnit(bpy.types.PropertyGroup):
    unit_type: bpy.props.StringProperty(
        name="Type",
        description="ControlNet type (e.g. 'depth', 'canny')",
        default="",
        update=update_parameters
    )  # type: ignore
    model_name: bpy.props.EnumProperty(
        name="Model",
        description="Select the ControlNet model",
        items=lambda self, context: get_controlnet_models(context, self.unit_type),
        update=update_controlnet
    )  # type: ignore
    model_name_backup: bpy.props.StringProperty(
        name="Model Name Backup",
        description="Backup of the selected ControlNet model name",
        default=""
    )  # type: ignore
    strength: bpy.props.FloatProperty(
        name="Strength",
        description="Strength of the ControlNet effect",
        default=0.5, min=0.0, max=3.0,
        update=update_parameters
    )  # type: ignore
    start_percent: bpy.props.FloatProperty(
        name="Start",
        description="Start percentage (/100)",
        default=0.0, min=0.0, max=1.0,
        update=update_parameters
    )  # type: ignore
    end_percent: bpy.props.FloatProperty(
        name="End",
        description="End percentage (/100)",
        default=1.0, min=0.0, max=1.0,
        update=update_parameters
    )  # type: ignore
    is_union: bpy.props.BoolProperty(
        name="Is Union Type",
        description="Is this a union ControlNet?",
        default=False,
        update=update_parameters
    )  # type: ignore
    use_union_type: bpy.props.BoolProperty(
        name="Use Union Type",
        description="Use union type for ControlNet",
        default=True,
        update=update_parameters
    )  # type: ignore


class LoRAUnit(bpy.types.PropertyGroup):
    model_name: bpy.props.EnumProperty(
        name="LoRA Model",
        description="Select the LoRA model file",
        items=lambda self, context: get_lora_models(self, context),
        update=update_parameters
    )  # type: ignore
    model_strength: bpy.props.FloatProperty(
        name="Model Strength",
        description="Strength of the LoRA's effect on the model's weights",
        default=1.0, min=0.0, max=100.0,
        update=update_parameters
    )  # type: ignore
    clip_strength: bpy.props.FloatProperty(
        name="CLIP Strength",
        description="Strength of the LoRA's effect on the CLIP/text conditioning",
        default=1.0, min=0.0, max=100.0,
        update=update_parameters
    )  # type: ignore


# ── Refresh operators ──────────────────────────────────────────────────────

class RefreshCheckpointList(bpy.types.Operator):
    """Fetches Checkpoint/UNET models from ComfyUI API and updates the cache."""
    bl_idname = "stablegen.refresh_checkpoint_list"
    bl_label = "Refresh Checkpoint/UNET List"
    bl_description = "Connect to ComfyUI server to get available Checkpoint/UNET models"

    @classmethod
    def poll(cls, context):
        prefs = context.preferences.addons.get(ADDON_PKG)
        if not prefs or not prefs.preferences.server_address:
            cls.poll_message_set("Server address not configured (check addon preferences)")
            return False
        if sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        return True

    def execute(self, context):
        from ..core import state as _state

        prefs = context.preferences.addons.get(ADDON_PKG)
        if not prefs:
            self.report({'ERROR'}, "Cannot access addon preferences.")
            return {'CANCELLED'}

        server_address = prefs.preferences.server_address
        architecture = getattr(context.scene, "model_architecture", "sdxl")

        def _bg_work():
            model_list = None
            if architecture == 'sdxl':
                model_list = _fetch_api_list(server_address, "/models/checkpoints")
                model_type_desc = "Checkpoint"
            elif architecture in ('flux1', 'qwen_image_edit', 'qwen_image_21'):
                model_list = fetch_diffusion_models(server_address)
                if model_list is not None and architecture == 'qwen_image_21':
                    model_list = [name for name in model_list
                                  if is_qwen_image_21_model(name)]
                elif model_list is not None and architecture == 'qwen_image_edit':
                    model_list = [name for name in model_list
                                  if not is_qwen_image_21_model(name)]
                model_type_desc = (
                    "UNET" if architecture == 'flux1'
                    else "Qwen-Image 2.1 model" if architecture == 'qwen_image_21'
                    else "Qwen-Image-Edit model"
                )
            elif architecture == 'flux2_klein':
                model_list = _fetch_api_list(server_address, "/models/diffusion_models")
                model_type_desc = "Diffusion Model"
            else:
                model_type_desc = "Model"
            return {'model_list': model_list, 'architecture': architecture,
                    'model_type_desc': model_type_desc}

        def _on_done(result):
            _state._dec_pending_refreshes()
            if result is None:
                return

            model_list = result.get('model_list')
            arch = result.get('architecture')
            desc = result.get('model_type_desc', 'Model')

            if model_list is None:
                _state._cached_checkpoint_list = [_state.no_server_item(server_address)]
                _state._cached_checkpoint_architecture = None
                print("[StableGen] Checkpoint refresh: cannot reach server.")
            elif not model_list:
                _state._cached_checkpoint_list = [("NONE_FOUND", f"No {desc}s Found", "Server list is empty")]
                _state._cached_checkpoint_architecture = arch
                print(f"[StableGen] Checkpoint refresh: no {desc} models found.")
            else:
                items = []
                for name in sorted(model_list):
                    items.append((name, name, f"{desc}: {name}"))
                _state._cached_checkpoint_list = items
                _state._cached_checkpoint_architecture = arch
                print(f"[StableGen] Checkpoint refresh: {len(items)} {desc}(s) found.")

            scene = bpy.context.scene if hasattr(bpy.context, 'scene') else None
            if scene:
                valid_ids = {it[0] for it in _state._cached_checkpoint_list}
                backup = getattr(scene, 'sg_model_name_backup', '')
                if backup and backup in valid_ids:
                    if scene.model_name != backup:
                        scene.model_name = backup
                elif scene.model_name not in valid_ids:
                    placeholder = next((it[0] for it in _state._cached_checkpoint_list
                                        if it[0].startswith("NO_") or it[0] == "NONE_FOUND"), None)
                    if placeholder:
                        scene.model_name = placeholder
                    elif _state._cached_checkpoint_list:
                        scene.model_name = _state._cached_checkpoint_list[0][0]

            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()

        _run_async(_bg_work, _on_done)
        _inc_pending_refreshes()
        self.report({'INFO'}, "Fetching checkpoint list...")
        return {'FINISHED'}


class RefreshLoRAList(bpy.types.Operator):
    """Fetches LoRA models from ComfyUI API and updates the cache."""
    bl_idname = "stablegen.refresh_lora_list"
    bl_label = "Refresh LoRA List"
    bl_description = "Connect to ComfyUI server to get available LoRA models"

    @classmethod
    def poll(cls, context):
        prefs = context.preferences.addons.get(ADDON_PKG)
        if not prefs or not prefs.preferences.server_address:
            cls.poll_message_set("Server address not configured (check addon preferences)")
            return False
        if sg_modal_active(context):
            cls.poll_message_set("Another operation is in progress")
            return False
        return True

    def execute(self, context):
        from ..core import state as _state

        prefs = context.preferences.addons.get(ADDON_PKG)
        if not prefs:
            self.report({'ERROR'}, "Cannot access addon preferences.")
            return {'CANCELLED'}

        server_address = prefs.preferences.server_address

        def _bg_work():
            return {'lora_list': _fetch_api_list(server_address, "/models/loras")}

        def _on_done(result):
            _state._dec_pending_refreshes()
            if result is None:
                return

            lora_list = result.get('lora_list')

            if lora_list is None:
                _state._cached_lora_list = [_state.no_server_item(server_address)]
                print("[StableGen] LoRA refresh: cannot reach server.")
            elif not lora_list:
                _state._cached_lora_list = [("NONE_FOUND", "No LoRAs Found", "Server list is empty")]
                print("[StableGen] LoRA refresh: no models found.")
            else:
                items = []
                for name in sorted(lora_list):
                    items.append((name, name, f"LoRA: {name}"))
                _state._cached_lora_list = items
                print(f"[StableGen] LoRA refresh: {len(items)} model(s) found.")

            scene = bpy.context.scene if hasattr(bpy.context, 'scene') else None
            if scene and hasattr(scene, 'lora_units'):
                valid_ids = {it[0] for it in _state._cached_lora_list}
                placeholder = next((it[0] for it in _state._cached_lora_list
                                    if it[0].startswith("NO_") or it[0] == "NONE_FOUND"), None)
                indices_to_remove = []
                for i, unit in enumerate(scene.lora_units):
                    if unit.model_name not in valid_ids or unit.model_name == placeholder:
                        indices_to_remove.append(i)
                for i in sorted(indices_to_remove, reverse=True):
                    scene.lora_units.remove(i)

                num_loras = len(scene.lora_units)
                if scene.lora_units_index >= num_loras:
                    scene.lora_units_index = max(0, num_loras - 1)
                elif num_loras == 0:
                    scene.lora_units_index = 0

            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()

        _run_async(_bg_work, _on_done)
        _inc_pending_refreshes()
        self.report({'INFO'}, "Fetching LoRA list...")
        return {'FINISHED'}


# ── Add / Remove operators ─────────────────────────────────────────────────

class AddControlNetUnit(bpy.types.Operator):
    bl_idname = "stablegen.add_controlnet_unit"
    bl_label = "Add ControlNet Unit"
    bl_description = "Add a ControlNet Unit. Only one unit per type is allowed."

    unit_type: bpy.props.EnumProperty(
        name="Type",
        items=[('depth', 'Depth', ''), ('canny', 'Canny', ''), ('normal', 'Normal', '')],
        default='depth',
        update=update_parameters
    )  # type: ignore

    model_name: bpy.props.EnumProperty(
        name="Model",
        description="Select the ControlNet model",
        items=lambda self, context: get_controlnet_models(context, self.unit_type),
        update=update_parameters
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        return not sg_modal_active(context)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "unit_type")
        models = get_controlnet_models(context, self.unit_type)
        if len(models) > 1:
            layout.prop(self, "model_name")

    def execute(self, context):
        units = context.scene.controlnet_units
        for unit in units:
            if unit.unit_type == self.unit_type:
                self.report({'WARNING'}, f"Unit '{self.unit_type}' already exists.")
                return {'CANCELLED'}
        new_unit = units.add()
        new_unit.unit_type = self.unit_type
        new_unit.model_name = self.model_name
        new_unit.strength = 0.5
        new_unit.start_percent = 0.0
        new_unit.end_percent = 1.0
        if "union" in new_unit.model_name.lower() or "promax" in new_unit.model_name.lower():
            new_unit.is_union = True
        context.scene.controlnet_units_index = len(units) - 1
        for area in context.screen.areas:
            area.tag_redraw()
        return {'FINISHED'}


class RemoveControlNetUnit(bpy.types.Operator):
    bl_idname = "stablegen.remove_controlnet_unit"
    bl_label = "Remove ControlNet Unit"
    bl_description = "Remove the selected ControlNet Unit"

    unit_type: bpy.props.EnumProperty(
        name="Type",
        items=[('depth', 'Depth', ''), ('canny', 'Canny', ''), ('normal', 'Normal', '')],
        default='depth',
        update=update_parameters
    )  # type: ignore

    @classmethod
    def poll(cls, context):
        return not sg_modal_active(context)

    def invoke(self, context, event):
        units = context.scene.controlnet_units
        if len(units) == 1:
            self.unit_type = units[0].unit_type
            return self.execute(context)
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "unit_type")

    def execute(self, context):
        units = context.scene.controlnet_units
        for index, unit in enumerate(units):
            if unit.unit_type == self.unit_type:
                units.remove(index)
                context.scene.controlnet_units_index = min(max(0, index - 1), len(units) - 1)
                update_parameters(self, context)
                for area in context.screen.areas:
                    area.tag_redraw()
                return {'FINISHED'}
        self.report({'WARNING'}, f"No unit of type '{self.unit_type}' found.")
        return {'CANCELLED'}


class AddLoRAUnit(bpy.types.Operator):
    bl_idname = "stablegen.add_lora_unit"
    bl_label = "Add LoRA Unit"
    bl_description = "Add a LoRA to the chain. Disabled if no LoRAs are available or all available LoRAs have been added."

    @classmethod
    def poll(cls, context):
        scene = context.scene
        addon_prefs = context.preferences.addons.get(ADDON_PKG)

        if not addon_prefs:
            return False
        addon_prefs = addon_prefs.preferences

        lora_enum_items = get_lora_models(scene, context)

        placeholder_ids = {"NONE_AVAILABLE", "NO_COMFYUI_DIR_LORA", "NO_LORAS_SUBDIR",
                          "PERM_ERROR", "SCAN_ERROR", "NONE_FOUND"}

        available_lora_files_count = sum(1 for item in lora_enum_items if item[0] not in placeholder_ids)

        if available_lora_files_count == 0:
            cls.poll_message_set("No LoRA model files found in any specified directory (including subdirectories).")
            return False

        num_current_lora_units = len(scene.lora_units)
        if num_current_lora_units >= available_lora_files_count:
            cls.poll_message_set("All available distinct LoRA models appear to have corresponding units.")
            return False

        return not sg_modal_active(context)

    def execute(self, context):
        loras = context.scene.lora_units
        new_lora = loras.add()

        all_lora_enum_items = get_lora_models(context.scene, context)

        placeholder_ids = {"NONE_AVAILABLE", "NO_COMFYUI_DIR_LORA", "NO_LORAS_SUBDIR",
                          "PERM_ERROR", "SCAN_ERROR", "NONE_FOUND"}
        available_lora_identifiers = [item[0] for item in all_lora_enum_items if item[0] not in placeholder_ids]

        if available_lora_identifiers:
            current_lora_model_identifiers_in_use = {unit.model_name for unit in loras
                                                     if unit.model_name and unit.model_name not in placeholder_ids}

            assigned_model = None
            for lora_id in available_lora_identifiers:
                if lora_id not in current_lora_model_identifiers_in_use:
                    assigned_model = lora_id
                    break

            if not assigned_model:
                assigned_model = available_lora_identifiers[0]

            if assigned_model:
                try:
                    new_lora.model_name = assigned_model
                except TypeError:
                    print(f"[StableGen] AddLoRAUnit Execute: TypeError assigning model '{assigned_model}'. Enum might not be ready.")

        new_lora.model_strength = 1.0
        new_lora.clip_strength = 1.0
        context.scene.lora_units_index = len(loras) - 1

        update_parameters(self, context)

        for area in context.screen.areas:
            if area.type in ('VIEW_3D', 'PROPERTIES'):
                area.tag_redraw()

        return {'FINISHED'}


class RemoveLoRAUnit(bpy.types.Operator):
    bl_idname = "stablegen.remove_lora_unit"
    bl_label = "Remove Selected LoRA Unit"
    bl_description = "Remove the selected LoRA from the chain"

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return (len(scene.lora_units) > 0 and
                0 <= scene.lora_units_index < len(scene.lora_units)
                and not sg_modal_active(context))

    def execute(self, context):
        loras = context.scene.lora_units
        index = context.scene.lora_units_index
        if 0 <= index < len(loras):
            loras.remove(index)
            context.scene.lora_units_index = min(max(0, index - 1), len(loras) - 1)
            update_parameters(self, context)
            for area in context.screen.areas:
                area.tag_redraw()
            return {'FINISHED'}
        self.report({'WARNING'}, "No LoRA unit selected or list is empty.")
        return {'CANCELLED'}
