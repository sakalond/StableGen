"""Addon preferences, server-status operator, and ControlNet mapping UI.

Contains:
- ``ControlNetModelMappingItem`` — PropertyGroup for CN model ↔ type mapping
- ``StableGenAddonPreferences`` — the addon's Preferences panel
- ``CheckServerStatus`` — operator that pings ComfyUI
- ``STABLEGEN_UL_ControlNetMappingList`` — UIList for CN model table
- ``RefreshControlNetMappings`` — operator to refresh CN models from server
"""

import bpy  # pylint: disable=import-error

from ..ui.presets import update_parameters
from ..utils import sg_modal_active
from . import ADDON_PKG
from .callbacks import update_combined
from .server_api import (
    _fetch_api_list,
    check_pbr_available,
    check_server_availability,
    check_trellis2_available,
)
from .state import (
    _dec_pending_refreshes,
    _inc_pending_refreshes,
    _run_async,
)
from ..timeout_config import get_timeout


# ── PropertyGroup for a single ControlNet model mapping ────────────────────

class ControlNetModelMappingItem(bpy.types.PropertyGroup):
    """Stores info about a detected ControlNet model and its supported types."""
    name: bpy.props.StringProperty(name="Model Filename")  # type: ignore

    supports_depth: bpy.props.BoolProperty(
        name="Depth",
        description="Check if this model supports Depth guidance",
        default=False
    )  # type: ignore
    supports_canny: bpy.props.BoolProperty(
        name="Canny",
        description="Check if this model supports Canny/Edge guidance",
        default=False
    )  # type: ignore
    supports_normal: bpy.props.BoolProperty(
        name="Normal",
        description="Check if this model supports Normal map guidance",
        default=False
    )  # type: ignore


# ── Addon Preferences ─────────────────────────────────────────────────────

class StableGenAddonPreferences(bpy.types.AddonPreferences):
    """Preferences for the StableGen addon."""
    bl_idname = ADDON_PKG

    server_address: bpy.props.StringProperty(
        name="Server Address",
        description="Address of the ComfyUI server",
        default="127.0.0.1:8188",
        update=update_combined
    )  # type: ignore

    output_dir: bpy.props.StringProperty(
        name="Output Directory",
        description="Directory to save generated outputs",
        default="",
        subtype='DIR_PATH',
        update=update_parameters
    )  # type: ignore

    controlnet_model_mappings: bpy.props.CollectionProperty(
        type=ControlNetModelMappingItem,
        name="ControlNet Model Mappings"
    )  # type: ignore

    save_blend_file: bpy.props.BoolProperty(
        name="Save Blend File",
        description="Save the current Blender file with packed textures",
        default=False,
        update=update_parameters
    )  # type: ignore

    enable_print_tab: bpy.props.BoolProperty(
        name="Enable 3D Print Tab",
        description="Show 3D Printing Palette and Exporters in a separate Sidebar tab",
        default=False,
        update=update_parameters
    )  # type: ignore

    controlnet_mapping_index: bpy.props.IntProperty(
        default=0, name="Active ControlNet Mapping Index"
    )  # type: ignore

    server_online: bpy.props.BoolProperty(
        name="Server Online",
        description="Indicates if the ComfyUI server is reachable",
        default=False
    )  # type: ignore

    overlay_color: bpy.props.FloatVectorProperty(
        name="Overlay Color",
        description="Color used for the camera aspect-ratio crop rectangle and floating view labels in the viewport",
        subtype='COLOR',
        size=3,
        min=0.0, max=1.0,
        default=(0.3, 0.5, 1.0),
        update=lambda self, ctx: [a.tag_redraw() for a in ctx.screen.areas if a.type == 'VIEW_3D'] if ctx.screen else None
    )  # type: ignore

    enable_debug: bpy.props.BoolProperty(
        name="Enable Debug Settings",
        description="Show diagnostic tools in the main panel for visualising projection weights, blending and coverage",
        default=False
    )  # type: ignore

    show_advanced: bpy.props.BoolProperty(
        name="Advanced Preferences",
        description="Show advanced preferences",
        default=False
    )  # type: ignore

    timeout_ping: bpy.props.FloatProperty(
        name="Ping Timeout (s)",
        description="Timeout for quick server availability checks (connectivity pings)",
        default=1.0, min=0.1, max=30.0, step=10, precision=1,
    )  # type: ignore

    timeout_api: bpy.props.FloatProperty(
        name="API Timeout (s)",
        description="Timeout for standard API requests (VRAM flush, queue prompt, server info)",
        default=10.0, min=1.0, max=120.0, step=100, precision=0,
    )  # type: ignore

    timeout_transfer: bpy.props.FloatProperty(
        name="Transfer Timeout (s)",
        description="Timeout for large data transfers (image upload, file download)",
        default=120.0, min=10.0, max=600.0, step=100, precision=0,
    )  # type: ignore

    timeout_reboot: bpy.props.FloatProperty(
        name="Reboot Timeout (s)",
        description="How long to wait for ComfyUI server to come back after a reboot",
        default=120.0, min=10.0, max=600.0, step=100, precision=0,
    )  # type: ignore

    timeout_mesh_gen: bpy.props.FloatProperty(
        name="Mesh Generation Timeout (s)",
        description="Timeout for TRELLIS.2 mesh generation WebSocket. "
                    "Mesh simplification / post-processing can take several "
                    "minutes without sending progress messages",
        default=600.0, min=60.0, max=3600.0, step=100, precision=0,
    )  # type: ignore

    timeout_scan: bpy.props.FloatProperty(
        name="Scan Timeout (s)",
        description="Timeout per request when guessing GLB filenames",
        default=10.0, min=1.0, max=120.0, step=100, precision=0,
    )  # type: ignore

    enable_scene_queue: bpy.props.BoolProperty(
        name="Enable Scene Queue",
        description="Show the Scene Queue panel in the sidebar",
        default=False,
        update=update_parameters
    )  # type: ignore

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "output_dir")
        row = layout.row(align=True)
        row.prop(self, "server_address")
        row.operator("stablegen.check_server_status", text="", icon='FILE_REFRESH')

        layout.prop(self, "save_blend_file")
        layout.prop(self, "enable_print_tab")
        layout.prop(self, "enable_scene_queue")

        layout.separator()

        box = layout.box()
        row = box.row()
        row.label(text="ControlNet Model Assignments:")
        row.operator("stablegen.refresh_controlnet_mappings", text="", icon='FILE_REFRESH')

        if not self.controlnet_model_mappings:
             box.label(text="No models found or list not refreshed.", icon='INFO')
        else:
             rows = max(1, min(len(self.controlnet_model_mappings), 5))
             box.template_list(
                  "STABLEGEN_UL_ControlNetMappingList",
                  "",
                  self,
                  "controlnet_model_mappings",
                  self,
                  "controlnet_mapping_index",
                  rows=rows
             )

        layout.separator()
        adv_box = layout.box()
        row = adv_box.row()
        row.prop(self, "show_advanced",
                 icon='TRIA_DOWN' if self.show_advanced else 'TRIA_RIGHT',
                 emboss=False, text="Advanced Preferences")
        if self.show_advanced:
            adv_box.prop(self, "overlay_color")
            adv_box.prop(self, "enable_debug")

            adv_box.separator()
            adv_box.label(text="Timeouts:", icon="TIME")
            col = adv_box.column(align=True)
            col.prop(self, "timeout_ping")
            col.prop(self, "timeout_api")
            col.prop(self, "timeout_transfer")
            col.prop(self, "timeout_reboot")
            col.prop(self, "timeout_mesh_gen")
            col.prop(self, "timeout_scan")


# ── Server-status check operator ──────────────────────────────────────────

class CheckServerStatus(bpy.types.Operator):
    """Checks if the ComfyUI server is reachable."""
    bl_idname = "stablegen.check_server_status"
    bl_label = "Check Server Status"
    bl_description = "Ping the ComfyUI server to check connectivity"

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
        from .load_handlers import load_handler

        prefs = context.preferences.addons[ADDON_PKG].preferences
        server_addr = prefs.server_address

        print(f"[StableGen] Checking server status at {server_addr}...")

        def _bg_work():
            result = {}
            result['online'] = check_server_availability(server_addr, timeout=get_timeout('ping'))
            if result['online']:
                result['trellis2'] = check_trellis2_available(server_addr, timeout=get_timeout('api'))
                result['pbr'] = check_pbr_available(server_addr, timeout=get_timeout('api'))
            else:
                result['trellis2'] = False
                result['pbr'] = False
            return result

        def _on_done(result):
            if result is None:
                return
            _prefs = bpy.context.preferences.addons[ADDON_PKG].preferences
            _prefs.server_online = result.get('online', False)

            if hasattr(bpy.context, 'scene') and bpy.context.scene:
                bpy.context.scene.trellis2_available = result.get('trellis2', False)
                bpy.context.scene.pbr_nodes_available = result.get('pbr', False)

            if result.get('online', False):
                print(f"[StableGen] ComfyUI server is online at {server_addr}.")
                load_handler(None)
                def _deferred_refresh():
                    try:
                        bpy.ops.stablegen.refresh_checkpoint_list('INVOKE_DEFAULT')
                        bpy.ops.stablegen.refresh_lora_list('INVOKE_DEFAULT')
                        bpy.ops.stablegen.refresh_controlnet_mappings('INVOKE_DEFAULT')
                    except Exception as e:
                        print(f"[StableGen] Error during deferred refresh: {e}")
                    return None
                bpy.app.timers.register(_deferred_refresh, first_interval=0.1)
            else:
                print(f"[StableGen] ComfyUI server unreachable or timed out at {server_addr}.")

        _run_async(_bg_work, _on_done, track_generation=True)
        self.report({'INFO'}, f"Checking server at {server_addr}...")

        return {'FINISHED'}


# ── ControlNet mapping UIList ──────────────────────────────────────────────

class STABLEGEN_UL_ControlNetMappingList(bpy.types.UIList):
    """UIList for displaying ControlNet model mappings."""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            split = layout.split(factor=0.65)
            col_name = split.column(align=True)
            col_checks = split.column(align=True)

            col_name.prop(item, "name", text="", emboss=False)

            row = col_checks.row(align=True)
            row.prop(item, "supports_depth", text="Depth", toggle=True)
            row.prop(item, "supports_canny", text="Canny", toggle=True)
            row.prop(item, "supports_normal", text="Normal", toggle=True)

        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon_value=icon)


# ── Refresh ControlNet mappings operator ───────────────────────────────────

class RefreshControlNetMappings(bpy.types.Operator):
    """Fetches ControlNet models from ComfyUI API and updates the mapping list."""
    bl_idname = "stablegen.refresh_controlnet_mappings"
    bl_label = "Refresh ControlNet Model List"
    bl_description = "Connect to ComfyUI server to get ControlNet models and update assignments"

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
        prefs = context.preferences.addons.get(ADDON_PKG)
        if not prefs:
            self.report({'ERROR'}, "Cannot access addon preferences.")
            return {'CANCELLED'}

        server_address = prefs.preferences.server_address

        existing_model_names = set()
        for item in prefs.preferences.controlnet_model_mappings:
            existing_model_names.add(item.name)

        def _bg_work():
            server_models = _fetch_api_list(server_address, "/models/controlnet")
            return {'server_models': server_models,
                    'existing_names': existing_model_names}

        def _on_done(result):
            _dec_pending_refreshes()
            if result is None:
                return
            server_models = result.get('server_models')
            existing_names = result.get('existing_names', set())

            _prefs = bpy.context.preferences.addons.get(ADDON_PKG)
            if not _prefs:
                return
            mappings = _prefs.preferences.controlnet_model_mappings

            if server_models is None:
                print("[StableGen] ControlNet refresh: cannot reach server.")
                return

            if not server_models:
                print("[StableGen] ControlNet refresh: no models found, clearing list.")
                mappings.clear()
            else:
                server_set = set(server_models)
                current_set = set(item.name for item in mappings)

                models_to_remove = current_set - server_set
                indices_to_remove = []
                for i, item in enumerate(mappings):
                    if item.name in models_to_remove:
                        indices_to_remove.append(i)
                for i in sorted(indices_to_remove, reverse=True):
                    mappings.remove(i)
                    if _prefs.preferences.controlnet_mapping_index >= len(mappings):
                        _prefs.preferences.controlnet_mapping_index = max(0, len(mappings) - 1)

                models_to_add = server_set - current_set
                for model_name in sorted(models_to_add):
                    new_item = mappings.add()
                    new_item.name = model_name
                    name_lower = model_name.lower()
                    is_union = 'union' in name_lower or 'promax' in name_lower
                    if is_union:
                        new_item.supports_depth = True
                        new_item.supports_canny = True
                        new_item.supports_normal = True
                        print(f"[StableGen]   Guessed '{model_name}' as Union (Depth, Canny, Normal).")
                    else:
                        if 'depth' in name_lower:
                            new_item.supports_depth = True
                            print(f"[StableGen]   Guessed '{model_name}' as Depth.")
                        if 'canny' in name_lower or 'lineart' in name_lower or 'scribble' in name_lower:
                            new_item.supports_canny = True
                            print(f"[StableGen]   Guessed '{model_name}' as Canny.")
                        if 'normal' in name_lower:
                            new_item.supports_normal = True
                            print(f"[StableGen]   Guessed '{model_name}' as Normal.")
                        if not (new_item.supports_depth or new_item.supports_canny or new_item.supports_normal):
                            print(f"[StableGen]   Could not guess type for '{model_name}'. Please assign manually.")

                print(f"[StableGen] ControlNet refresh: {len(models_to_add)} added, {len(models_to_remove)} removed.")

            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()

        _run_async(_bg_work, _on_done)
        _inc_pending_refreshes()
        self.report({'INFO'}, "Fetching ControlNet models...")
        return {'FINISHED'}
