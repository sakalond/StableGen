""" This script registers the addon. """
import bpy # pylint: disable=import-error
from .stablegen import StableGenPanel, ApplyPreset, SavePreset, DeletePreset, get_preset_items, update_parameters, ResetQwenPrompt
from .render_tools import BakeTextures, AddCameras, CloneCamera, MirrorCamera, ToggleCameraLabels, SwitchMaterial, ExportOrbitGIF, CollectCameraPrompts, CameraPromptItem, CameraOrderItem, SG_UL_CameraOrderList, SyncCameraOrder, MoveCameraOrder, ApplyCameraOrderPreset
from .debug_tools import debug_classes as _debug_classes
from .utils import AddHDRI, ApplyModifiers, CurvesToMesh
from .generator import ComfyUIGenerate, Reproject, Regenerate, MirrorReproject, Trellis2Generate
import os
import requests
import json
from bpy.app.handlers import persistent
from urllib.parse import urlparse

bl_info = {
    "name": "StableGen",
    "category": "Object",
    "author": "Ondrej Sakala",
    "version": (0, 2, 0),
    'blender': (4, 2, 0)
}

classes = [
    StableGenPanel,
    ApplyPreset,
    SavePreset,
    DeletePreset,
    ResetQwenPrompt,
    BakeTextures,
    AddCameras,
    CloneCamera,
    MirrorCamera,
    ToggleCameraLabels,
    SwitchMaterial,
    ExportOrbitGIF,
    CollectCameraPrompts,
    CameraPromptItem,
    CameraOrderItem,
    SG_UL_CameraOrderList,
    SyncCameraOrder,
    MoveCameraOrder,
    ApplyCameraOrderPreset,
    AddHDRI,
    ApplyModifiers,
    CurvesToMesh,
    ComfyUIGenerate,
    Reproject,
    Regenerate,
    MirrorReproject,
    Trellis2Generate,
]

# Global caches for model lists fetched via API
_cached_checkpoint_list = [("NONE_AVAILABLE", "None available", "Fetch models from server")]
_cached_lora_list = [("NONE_AVAILABLE", "None available", "Fetch models from server")]
_cached_checkpoint_architecture = None
_pending_checkpoint_refresh_architecture = None

# Counter for in-flight async refresh operations (checkpoint, LoRA, controlnet).
# The UI checks this to display a "Refreshing…" indicator.
_pending_refreshes = 0

# ---------------------------------------------------------------------------
#  Async network helper — run blocking I/O off the main thread
# ---------------------------------------------------------------------------
import threading as _threading
import traceback as _traceback

# Monotonically incrementing token so we can discard stale results when the
# server address changes while a request is still in-flight.
_async_generation = 0

def _run_async(work_fn, done_fn, poll_interval=0.25, track_generation=False):
    """Run *work_fn* in a background thread; call *done_fn(result)* on the
    main thread via ``bpy.app.timers`` when finished.

    *work_fn* receives no arguments and should return a result dict/object.
    *done_fn* receives that result.  Both run without any lock — *done_fn*
    is guaranteed to execute on the main Blender thread.

    If *track_generation* is True the call increments the global
    ``_async_generation`` counter and the result is silently discarded if
    a newer tracked call was started before this one finishes.  Use this
    only for server-address-change callbacks where stale results must be
    dropped; refresh operators should leave it False so their results are
    never accidentally discarded.
    """
    global _async_generation
    if track_generation:
        _async_generation += 1
    gen = _async_generation

    container = {}  # mutable box for the thread to deposit its result

    def _worker():
        try:
            container['result'] = work_fn()
        except Exception:
            _traceback.print_exc()
            container['result'] = None

    t = _threading.Thread(target=_worker, daemon=True)
    t.start()

    def _poll():
        # Discard result if the server address changed after we started.
        if gen != _async_generation:
            return None  # stop polling — stale
        if t.is_alive():
            return poll_interval  # keep polling
        try:
            done_fn(container.get('result'))
        except Exception:
            _traceback.print_exc()
        return None  # done
    bpy.app.timers.register(_poll, first_interval=poll_interval)


def update_architecture_mode(self, context):
    """Called when the user changes the architecture_mode dropdown.

    For standard diffusion modes (sdxl / flux1 / qwen_image_edit) the hidden
    ``model_architecture`` property is synced which in turn triggers the
    existing ``update_combined`` callback (server check, checkpoint refresh,
    etc.).

    For trellis2 mode the diffusion backbone is only synced when the
    active ``trellis2_texture_mode`` is itself a diffusion architecture.
    """
    scene = context.scene
    mode = scene.architecture_mode

    if mode != 'trellis2':
        # Standard diffusion — sync backbone and let its callback refresh
        if scene.model_architecture != mode:
            scene.model_architecture = mode          # triggers update_combined
    else:
        # Entering TRELLIS.2 mode — sync backbone for any diffusion need
        _sync_trellis2_backbone(scene)

    # Trigger preset detection
    update_parameters(self, context)


def _sync_trellis2_backbone(scene):
    """Pick the right diffusion backbone for the current TRELLIS.2 state.

    If the texture-mode is a diffusion arch, use that.  Otherwise fall
    back to the initial-image arch (relevant when generate_from == prompt
    and texture_mode is native/none).
    """
    tex_mode = getattr(scene, 'trellis2_texture_mode', 'native')
    if tex_mode in ('sdxl', 'flux1', 'qwen_image_edit'):
        target = tex_mode
    elif getattr(scene, 'trellis2_generate_from', 'image') == 'prompt':
        target = getattr(scene, 'trellis2_initial_image_arch', 'sdxl')
    else:
        return  # no diffusion backbone needed
    if scene.model_architecture != target:
        scene.model_architecture = target  # triggers update_combined


def update_trellis2_texture_mode(self, context):
    """Called when the user changes the texture generation mode inside TRELLIS.2.

    * Syncs ``trellis2_skip_texture`` (True when *not* native).
    * If the chosen mode is a diffusion architecture and differs from the
      current backbone, syncs ``model_architecture`` (which triggers a
      checkpoint refresh).
    * When ``generate_from`` is *prompt* and texture_mode is non-diffusion,
      syncs the backbone from ``trellis2_initial_image_arch`` instead.
    """
    scene = context.scene
    if getattr(scene, 'architecture_mode', '') != 'trellis2':
        return

    tex_mode = scene.trellis2_texture_mode

    # Sync skip_texture flag used by the workflow code
    scene.trellis2_skip_texture = (tex_mode != 'native')

    _sync_trellis2_backbone(scene)

    # Trigger preset detection
    update_parameters(self, context)


def update_trellis2_initial_image_arch(self, context):
    """Called when the user changes the initial-image architecture for TRELLIS.2.

    Syncs the diffusion backbone when texture_mode is native/none so the
    checkpoint list refreshes to match the chosen architecture.
    """
    scene = context.scene
    if getattr(scene, 'architecture_mode', '') != 'trellis2':
        return
    _sync_trellis2_backbone(scene)

    # Trigger preset detection
    update_parameters(self, context)


def update_trellis2_generate_from(self, context):
    """Called when the user switches between Image and Prompt input mode.

    Syncs the diffusion backbone via ``_sync_trellis2_backbone`` so a
    checkpoint is available for the initial-image generation step.
    """
    scene = context.scene
    if getattr(scene, 'architecture_mode', '') != 'trellis2':
        return

    if scene.trellis2_generate_from == 'prompt':
        _sync_trellis2_backbone(scene)

    # Trigger preset detection
    update_parameters(self, context)


def update_combined(self, context):
    # This now primarily updates the preset status and might trigger Enum updates implicitly
    prefs = context.preferences.addons[__package__].preferences
    raw_address = prefs.server_address

    if raw_address:
        # Ensure we have a scheme for correct parsing
        if not raw_address.startswith(('http://', 'https://')):
            # Prepend http scheme if it's missing
            parsed_url = urlparse(f"http://{raw_address}")
        else:
            parsed_url = urlparse(raw_address)
        
        clean_address = parsed_url.netloc

        # If parsing resulted in a change, update the property.
        # This will re-trigger the update function, so we return early.
        if clean_address and raw_address != clean_address:
            prefs.server_address = clean_address
            return None

    server_address = prefs.server_address

    if not server_address:
        prefs.server_online = False
        global _cached_checkpoint_list, _cached_lora_list
        _cached_checkpoint_list = [("NO_SERVER", "Set Server Address", "...")]
        _cached_lora_list = [("NO_SERVER", "Set Server Address", "...")]
        return None

    # ── Async: run ping + TRELLIS check + model list refresh in background ──
    # Results are applied on the main thread via _run_async's done callback.
    print("Server address changed, checking asynchronously...")

    def _bg_work():
        """Background thread – only network I/O, no bpy access."""
        result = {}
        result['online'] = check_server_availability(server_address, timeout=get_timeout('ping'))
        if result['online']:
            result['trellis2'] = check_trellis2_available(server_address, timeout=get_timeout('api'))
        else:
            result['trellis2'] = False
        return result

    def _on_done(result):
        """Main-thread callback – apply results to bpy properties."""
        if result is None:
            return
        prefs = bpy.context.preferences.addons[__package__].preferences
        prefs.server_online = result.get('online', False)

        if hasattr(bpy.context, 'scene') and bpy.context.scene:
            bpy.context.scene.trellis2_available = result.get('trellis2', False)

        if not result.get('online', False):
            print("ComfyUI server is not reachable.")
            return

        # Server is online — trigger model list refresh + load_handler
        # on the main thread.  The refresh operators themselves will
        # be made async individually.
        update_parameters(None, bpy.context)
        load_handler(None)

        def _deferred_refresh():
            try:
                bpy.ops.stablegen.refresh_checkpoint_list('INVOKE_DEFAULT')
                bpy.ops.stablegen.refresh_lora_list('INVOKE_DEFAULT')
                bpy.ops.stablegen.refresh_controlnet_mappings('INVOKE_DEFAULT')
            except Exception as e:
                print(f"Error during deferred refresh: {e}")
            return None
        bpy.app.timers.register(_deferred_refresh, first_interval=0.1)

    _run_async(_bg_work, _on_done, track_generation=True)

    # Synchronous parts that don't block (no network) can stay here:
    update_parameters(self, context)
    load_handler(None)

    return None


class ControlNetModelMappingItem(bpy.types.PropertyGroup):
    """Stores info about a detected ControlNet model and its supported types."""
    name: bpy.props.StringProperty(name="Model Filename") # Read-only, set by refresh op

    # Use Booleans for each supported type
    supports_depth: bpy.props.BoolProperty(
        name="Depth",
        description="Check if this model supports Depth guidance",
        default=False
    ) # type: ignore
    supports_canny: bpy.props.BoolProperty(
        name="Canny",
        description="Check if this model supports Canny/Edge guidance",
        default=False
    ) # type: ignore
    supports_normal: bpy.props.BoolProperty(
        name="Normal",
        description="Check if this model supports Normal map guidance",
        default=False
    ) # type: ignore

class StableGenAddonPreferences(bpy.types.AddonPreferences):
    """     
    Preferences for the StableGen addon.     
    """
    bl_idname = __package__

    server_address: bpy.props.StringProperty(
        name="Server Address",
        description="Address of the ComfyUI server",
        default="127.0.0.1:8188",
        update=update_combined
    ) # type: ignore

    output_dir: bpy.props.StringProperty(
        name="Output Directory",
        description="Directory to save generated outputs",
        default="",
        subtype='DIR_PATH',
        update=update_parameters
    ) # type: ignore

    controlnet_model_mappings: bpy.props.CollectionProperty(
        type=ControlNetModelMappingItem,
        name="ControlNet Model Mappings"
    ) # type: ignore
    
    save_blend_file: bpy.props.BoolProperty(
        name="Save Blend File",
        description="Save the current Blender file with packed textures",
        default=False,
        update=update_parameters
    ) # type: ignore

    controlnet_mapping_index: bpy.props.IntProperty(default=0, name="Active ControlNet Mapping Index") # type: ignore

    server_online: bpy.props.BoolProperty(
        name="Server Online",
        description="Indicates if the ComfyUI server is reachable",
        default=False
    ) # type: ignore

    overlay_color: bpy.props.FloatVectorProperty(
        name="Overlay Color",
        description="Color used for the camera aspect-ratio crop rectangle and floating view labels in the viewport",
        subtype='COLOR',
        size=3,
        min=0.0, max=1.0,
        default=(0.3, 0.5, 1.0),
        update=lambda self, ctx: [a.tag_redraw() for a in ctx.screen.areas if a.type == 'VIEW_3D'] if ctx.screen else None
    ) # type: ignore

    enable_debug: bpy.props.BoolProperty(
        name="Enable Debug Settings",
        description="Show diagnostic tools in the main panel for visualising projection weights, blending and coverage",
        default=False
    ) # type: ignore

    show_advanced: bpy.props.BoolProperty(
        name="Advanced Preferences",
        description="Show advanced preferences",
        default=False
    ) # type: ignore

    timeout_ping: bpy.props.FloatProperty(
        name="Ping Timeout (s)",
        description="Timeout for quick server availability checks (connectivity pings)",
        default=1.0,
        min=0.1,
        max=30.0,
        step=10,
        precision=1,
    ) # type: ignore

    timeout_api: bpy.props.FloatProperty(
        name="API Timeout (s)",
        description="Timeout for standard API requests (VRAM flush, queue prompt, server info)",
        default=10.0,
        min=1.0,
        max=120.0,
        step=100,
        precision=0,
    ) # type: ignore

    timeout_transfer: bpy.props.FloatProperty(
        name="Transfer Timeout (s)",
        description="Timeout for large data transfers (image upload, file download)",
        default=120.0,
        min=10.0,
        max=600.0,
        step=100,
        precision=0,
    ) # type: ignore

    timeout_reboot: bpy.props.FloatProperty(
        name="Reboot Timeout (s)",
        description="How long to wait for ComfyUI server to come back after a reboot",
        default=120.0,
        min=10.0,
        max=600.0,
        step=100,
        precision=0,
    ) # type: ignore

    timeout_mesh_gen: bpy.props.FloatProperty(
        name="Mesh Generation Timeout (s)",
        description="Timeout for TRELLIS.2 mesh generation WebSocket. "
                    "Mesh simplification / post-processing can take several "
                    "minutes without sending progress messages",
        default=600.0,
        min=60.0,
        max=3600.0,
        step=100,
        precision=0,
    ) # type: ignore

    def draw(self, context):
        """     
        Draws the preferences panel.         
        :param context: Blender context.         
        :return: None     
        """
        layout = self.layout
        layout.prop(self, "output_dir")
        row = layout.row(align=True)
        row.prop(self, "server_address")

        # Add the check button
        row.operator("stablegen.check_server_status", text="", icon='FILE_REFRESH')

        layout.prop(self, "save_blend_file")

        layout.separator()

        box = layout.box()
        row = box.row()
        row.label(text="ControlNet Model Assignments:")
        row.operator("stablegen.refresh_controlnet_mappings", text="", icon='FILE_REFRESH')

        # Use a template_list for a cleaner UI if many models
        # Requires creating a UIList class, more complex.
        # Simple loop for now:
        if not self.controlnet_model_mappings:
             box.label(text="No models found or list not refreshed.", icon='INFO')
        else:
             rows = max(1, min(len(self.controlnet_model_mappings), 5)) # Show up to 5 rows
             box.template_list(
                  "STABLEGEN_UL_ControlNetMappingList", # Custom UIList identifier
                  "",                  # Data List ID (unused here)
                  self,                # Data source (the preferences instance)
                  "controlnet_model_mappings", # Property name of the collection
                  self,                # Active item index source
                  "controlnet_mapping_index", # Property name for active index
                  rows=rows
             )

        # --- Advanced Preferences ---
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

class CheckServerStatus(bpy.types.Operator):
    """Checks if the ComfyUI server is reachable."""
    bl_idname = "stablegen.check_server_status"
    bl_label = "Check Server Status"
    bl_description = "Ping the ComfyUI server to check connectivity"

    @classmethod
    def poll(cls, context):
        # Can run if server address is set
        prefs = context.preferences.addons.get(__package__)
        # Also check that another check isn't running if using threading later
        # global _is_refreshing
        # return prefs and prefs.preferences.server_address and not _is_refreshing
        return prefs and prefs.preferences.server_address # Simplified for sync check

    def execute(self, context):
        prefs = context.preferences.addons[__package__].preferences
        server_addr = prefs.server_address

        print(f"Checking server status at {server_addr}...")

        def _bg_work():
            result = {}
            result['online'] = check_server_availability(server_addr, timeout=get_timeout('ping'))
            if result['online']:
                result['trellis2'] = check_trellis2_available(server_addr, timeout=get_timeout('api'))
            else:
                result['trellis2'] = False
            return result

        def _on_done(result):
            if result is None:
                return
            _prefs = bpy.context.preferences.addons[__package__].preferences
            _prefs.server_online = result.get('online', False)

            if hasattr(bpy.context, 'scene') and bpy.context.scene:
                bpy.context.scene.trellis2_available = result.get('trellis2', False)

            if result.get('online', False):
                print(f"ComfyUI server is online at {server_addr}.")
                load_handler(None)
                def _deferred_refresh():
                    try:
                        bpy.ops.stablegen.refresh_checkpoint_list('INVOKE_DEFAULT')
                        bpy.ops.stablegen.refresh_lora_list('INVOKE_DEFAULT')
                        bpy.ops.stablegen.refresh_controlnet_mappings('INVOKE_DEFAULT')
                    except Exception as e:
                        print(f"Error during deferred refresh: {e}")
                    return None
                bpy.app.timers.register(_deferred_refresh, first_interval=0.1)
            else:
                print(f"ComfyUI server unreachable or timed out at {server_addr}.")

        _run_async(_bg_work, _on_done, track_generation=True)
        self.report({'INFO'}, f"Checking server at {server_addr}...")

        return {'FINISHED'}

class STABLEGEN_UL_ControlNetMappingList(bpy.types.UIList):
    """UIList for displaying ControlNet model mappings."""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        prefs = data # 'data' is the AddonPreferences instance
        # 'item' is the ControlNetModelMappingItem instance

        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            # Use a split so the filename gets more space than the checkboxes
            split = layout.split(factor=0.65)  # adjust factor to give filename more room
            col_name = split.column(align=True)
            col_checks = split.column(align=True)

            # Layout: Filename | [x] Depth | [x] Canny | [x] Normal
            col_name.prop(item, "name", text="", emboss=False) # Show filename read-only

            # Add checkboxes for each type using icons
            row = col_checks.row(align=True)
            row.prop(item, "supports_depth", text="Depth", toggle=True)
            row.prop(item, "supports_canny", text="Canny", toggle=True)
            row.prop(item, "supports_normal", text="Normal", toggle=True)
            # Add more props here if you add types

        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon_value=icon)

class RefreshControlNetMappings(bpy.types.Operator):
    """Fetches ControlNet models from ComfyUI API and updates the mapping list."""
    bl_idname = "stablegen.refresh_controlnet_mappings"
    bl_label = "Refresh ControlNet Model List"
    bl_description = "Connect to ComfyUI server to get ControlNet models and update assignments"

    @classmethod
    def poll(cls, context):
        # Can run if server address is set
        prefs = context.preferences.addons.get(__package__)
        return prefs and prefs.preferences.server_address

    def execute(self, context):
        prefs = context.preferences.addons.get(__package__)
        if not prefs:
            self.report({'ERROR'}, "Cannot access addon preferences.")
            return {'CANCELLED'}

        server_address = prefs.preferences.server_address

        # Snapshot existing mapping names so the bg thread can compute
        # which models to add/remove without touching bpy.
        existing_model_names = set()
        for item in prefs.preferences.controlnet_model_mappings:
            existing_model_names.add(item.name)

        def _bg_work():
            server_models = _fetch_api_list(server_address, "/models/controlnet")
            return {'server_models': server_models,
                    'existing_names': existing_model_names}

        def _on_done(result):
            global _pending_refreshes
            _pending_refreshes = max(0, _pending_refreshes - 1)
            if result is None:
                return
            server_models = result.get('server_models')
            existing_names = result.get('existing_names', set())

            _prefs = bpy.context.preferences.addons.get(__package__)
            if not _prefs:
                return
            mappings = _prefs.preferences.controlnet_model_mappings

            if server_models is None:
                print("ControlNet refresh: cannot reach server.")
                return

            if not server_models:
                print("ControlNet refresh: no models found, clearing list.")
                mappings.clear()
            else:
                server_set = set(server_models)
                current_set = set(item.name for item in mappings)

                # Remove stale entries
                models_to_remove = current_set - server_set
                indices_to_remove = []
                for i, item in enumerate(mappings):
                    if item.name in models_to_remove:
                        indices_to_remove.append(i)
                for i in sorted(indices_to_remove, reverse=True):
                    mappings.remove(i)
                    if _prefs.preferences.controlnet_mapping_index >= len(mappings):
                        _prefs.preferences.controlnet_mapping_index = max(0, len(mappings) - 1)

                # Add new entries with type guessing
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
                        print(f"  Guessed '{model_name}' as Union (Depth, Canny, Normal).")
                    else:
                        if 'depth' in name_lower:
                            new_item.supports_depth = True
                            print(f"  Guessed '{model_name}' as Depth.")
                        if 'canny' in name_lower or 'lineart' in name_lower or 'scribble' in name_lower:
                            new_item.supports_canny = True
                            print(f"  Guessed '{model_name}' as Canny.")
                        if 'normal' in name_lower:
                            new_item.supports_normal = True
                            print(f"  Guessed '{model_name}' as Normal.")
                        if not (new_item.supports_depth or new_item.supports_canny or new_item.supports_normal):
                            print(f"  Could not guess type for '{model_name}'. Please assign manually.")

                print(f"ControlNet refresh: {len(models_to_add)} added, {len(models_to_remove)} removed.")

            # Redraw UI
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()

        _run_async(_bg_work, _on_done)
        global _pending_refreshes
        _pending_refreshes += 1
        self.report({'INFO'}, "Fetching ControlNet models...")
        return {'FINISHED'}
    
from .timeout_config import get_timeout  # noqa: E402 – placed here to avoid circular imports


def check_server_availability(server_address, timeout=0.5):
    """
    Quickly checks if the ComfyUI server is responding.

    Args:
        server_address (str): The address:port of the ComfyUI server.
        timeout (float): Strict timeout in seconds for this check.

    Returns:
        bool: True if the server responds quickly, False otherwise.
    """
    if not server_address:
        return False

    # Use a lightweight endpoint like /queue or root '/'
    # /system_stats might be slightly heavier
    url = f"http://{server_address}/queue" # Or just f"http://{server_address}/"
    # print(f"Pinging server at: {url} (Timeout: {timeout}s)") # Debug
    try:
        # HEAD request is faster as it doesn't download the body
        response = requests.head(url, timeout=timeout)
        # Check for successful status codes (2xx, maybe 404 if hitting root)
        # /queue typically gives 200 OK even on GET/HEAD
        response.raise_for_status()
        # print("  Server responded.") # Debug
        return True
    except requests.exceptions.Timeout:
        print(f"  Initial server check failed: Timeout ({timeout}s).")
        return False
    except requests.exceptions.ConnectionError:
        print("  Initial server check failed: Connection Error.")
        return False


def check_trellis2_available(server_address, timeout=1.0):
    """
    Checks if TRELLIS.2 custom nodes are installed in ComfyUI by querying /object_info.

    Args:
        server_address (str): The address:port of the ComfyUI server.
        timeout (float): Timeout in seconds.

    Returns:
        bool: True if TRELLIS.2 nodes are detected, False otherwise.
    """
    if not server_address:
        return False
    try:
        url = f"http://{server_address}/object_info/Trellis2ImageToShape"
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            if 'Trellis2ImageToShape' in data:
                print("[TRELLIS2] Auto-detect: TRELLIS.2 nodes found in ComfyUI.")
                return True
        print("[TRELLIS2] Auto-detect: TRELLIS.2 nodes NOT found in ComfyUI.")
        return False
    except Exception as e:
        print(f"[TRELLIS2] Auto-detect failed: {e}")
        return False
    except requests.exceptions.RequestException as e:
        # Other errors (like 404 on root) might still mean the server is *running*
        # but depends on the chosen endpoint. /queue should be reliable.
        # Let's consider most request exceptions here as a failure to connect quickly.
        print(f"  Initial server check failed: Request Error ({e}).")
        return False
    except Exception as e:
        print(f"  Initial server check failed: Unexpected Error ({e}).")
        return False

def fetch_from_comfyui_api(context, endpoint):
    """
    Fetches data from a specified ComfyUI API endpoint.

    Args:
        context: Blender context to access addon preferences.
        endpoint (str): The API endpoint path (e.g., "/models/checkpoints").

    Returns:
        list: A list of items returned by the API (usually filenames),
              or an empty list if the request fails or returns invalid data.
              Returns None if the server address is not set.
    """
    addon_prefs = context.preferences.addons.get(__package__)
    if not addon_prefs:
        print("Error: Could not access StableGen addon preferences.")
        return None # Indicate config error

    server_address = addon_prefs.preferences.server_address
    if not server_address:
        print("Error: ComfyUI Server Address is not set in preferences.")
        # Return None to signify a configuration issue preventing the API call
        return None
    
    if not check_server_availability(server_address, timeout=get_timeout('ping')): # Use a strict timeout here
         # Error message printed by check_server_availability
         return None # Server unreachable or timed out on initial check
    
    # Ensure endpoint starts with a slash
    if not endpoint.startswith('/'):
        endpoint = '/' + endpoint

    url = f"http://{server_address}{endpoint}"

    try:
        response = requests.get(url, timeout=get_timeout('api')) # Add a timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Basic validation: Check if the response is a list (expected for model lists)
        if isinstance(data, list):
            # Further check if list items look like filenames (simple check)
            if all(isinstance(item, str) for item in data):
                return data # Return the list of filenames
            elif data: # List exists but contains non-strings
                 print(f"  Warning: API endpoint {endpoint} returned a list, but it contains non-string items: {data[:5]}...") # Show first few
                 # Decide how to handle: return empty, or try to filter strings?
                 # For now, let's filter assuming filenames are strings:
                 string_items = [item for item in data if isinstance(item, str)]
                 if string_items:
                      return string_items
                 else:
                      print(f"  Error: No valid string filenames found in list from {endpoint}.")
                      return [] # Return empty list if no strings found
            else:
                 # API returned an empty list, which is valid
                 return []
        else:
            print(f"  Error: API endpoint {endpoint} did not return a JSON list. Received: {type(data)}")
            return [] # Return empty list on unexpected type

    except requests.exceptions.Timeout:
        print(f"  Error: Timeout connecting to {url}.")
    except requests.exceptions.ConnectionError:
        print(f"  Error: Connection failed to {url}. Is ComfyUI running and accessible?")
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching from {url}: {e}")
    except json.JSONDecodeError:
        print(f"  Error: Could not decode JSON response from {url}. Response text: {response.text}")
    except Exception as e:
        print(f"  An unexpected error occurred fetching from {url}: {e}")

    return [] # Return empty list on any failure


def _fetch_api_list(server_address, endpoint):
    """Thread-safe variant of *fetch_from_comfyui_api* — takes an explicit
    *server_address* instead of reading from bpy context, so it can run in a
    background thread.

    Returns a list of strings on success, ``None`` on connection/config error,
    or ``[]`` when the server returns an empty or invalid list.
    """
    if not server_address:
        return None

    if not check_server_availability(server_address, timeout=get_timeout('ping')):
        return None

    if not endpoint.startswith('/'):
        endpoint = '/' + endpoint

    url = f"http://{server_address}{endpoint}"
    try:
        response = requests.get(url, timeout=get_timeout('api'))
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            string_items = [item for item in data if isinstance(item, str)]
            return string_items
        else:
            print(f"  Error: API endpoint {endpoint} did not return a JSON list.")
            return []
    except requests.exceptions.Timeout:
        print(f"  Error: Timeout connecting to {url}.")
    except requests.exceptions.ConnectionError:
        print(f"  Error: Connection failed to {url}.")
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching from {url}: {e}")
    except json.JSONDecodeError:
        print(f"  Error: Could not decode JSON from {url}.")
    except Exception as e:
        print(f"  Unexpected error fetching from {url}: {e}")

    return []


def get_models_from_directory(scan_root_path: str, valid_extensions: tuple, type_for_description: str, path_prefix_for_id: str = ""):
    """
    Scans a given root directory (and its subdirectories) for model files.
    Returns paths relative to scan_root_path, optionally prefixed.

    Args:
        scan_root_path (str): The absolute root path to start scanning from.
        valid_extensions (tuple): Tuple of valid lowercase file extensions.
        type_for_description (str): String like "Checkpoint" or "LoRA" for UI descriptions.
        path_prefix_for_id (str): A prefix to add to the identifier if needed to distinguish sources 
    """
    items = []
    if not (scan_root_path and os.path.isdir(scan_root_path)):
        # Don't add error items here, let the caller handle empty results
        return items

    try:
        for root, _, files in os.walk(scan_root_path):
            for f_name in files:
                if f_name.lower().endswith(valid_extensions):
                    full_path = os.path.join(root, f_name)
                    # Path relative to the specific scan_root_path (ComfyUI or external)
                    relative_path = os.path.relpath(full_path, scan_root_path)
                    
                    # The identifier sent to ComfyUI should be this relative_path
                    # if scan_root_path is a path ComfyUI recognizes.
                    identifier = path_prefix_for_id + relative_path 
                    display_name = identifier # Show the full "prefixed" path if prefix is used

                    items.append((identifier, display_name, f"{type_for_description}: {display_name}"))
    except PermissionError:
        print(f"Permission Denied for {scan_root_path}") # Log it
    except Exception as e:
        print(f"Error Scanning {scan_root_path}: {e}") # Log it
    
    return items

def merge_and_deduplicate_models(model_lists: list):
    """
    Merges multiple lists of model items and de-duplicates based on the identifier.
    Keeps the first encountered entry in case of duplicate identifiers.
    """
    merged_items = []
    seen_identifiers = set()
    for model_list in model_lists:
        for identifier, name, description in model_list:
            # Filter out placeholder/error items from get_models_from_directory if they existed
            if identifier.startswith("NO_") or identifier.startswith("PERM_") or identifier.startswith("SCAN_") or identifier == "NONE_FOUND":
                continue
            if identifier not in seen_identifiers:
                merged_items.append((identifier, name, description))
                seen_identifiers.add(identifier)
    
    if not merged_items: # If after all scans and merges, still nothing
        merged_items.append(("NONE_AVAILABLE", "No Models Found", "Check ComfyUI and External Directories in Preferences"))
    
    merged_items.sort(key=lambda x: x[1]) # Sort by display name
    return merged_items

def update_model_list(self, context):
    """Returns the cached list of checkpoint/unet models."""
    global _cached_checkpoint_list
    # Basic check in case cache hasn't been populated correctly
    if not _cached_checkpoint_list:
         return [("NONE_AVAILABLE", "None available", "Fetch models from server")]
    return _cached_checkpoint_list

def update_union(self, context):
    if "union" in self.model_name.lower() or "promax" in self.model_name.lower():
        self.is_union = True
    else:
        self.is_union = False

def update_controlnet(self, context):
    update_parameters(self, context)
    update_union(self, context)
    return None

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
    ) # type: ignore
    strength: bpy.props.FloatProperty(
        name="Strength",
        description="Strength of the ControlNet effect",
        default=0.5,
        min=0.0,
        max=3.0,
        update=update_parameters
    )  # type: ignore
    start_percent: bpy.props.FloatProperty(
        name="Start",
        description="Start percentage (/100)",
        default=0.0,
        min=0.0,
        max=1.0,
        update=update_parameters
    )  # type: ignore
    end_percent: bpy.props.FloatProperty(
        name="End",
        description="End percentage (/100)",
        default=1.0,
        min=0.0,
        max=1.0,
        update=update_parameters
    )  # type: ignore
    is_union: bpy.props.BoolProperty(
        name="Is Union Type",
        description="Is this a union ControlNet?",
        default=False,
        update=update_parameters
    ) # type: ignore
    use_union_type: bpy.props.BoolProperty(
        name="Use Union Type",
        description="Use union type for ControlNet",
        default=True,
        update=update_parameters
    ) # type: ignore

class LoRAUnit(bpy.types.PropertyGroup):
    model_name: bpy.props.EnumProperty(
        name="LoRA Model",
        description="Select the LoRA model file",
        items=lambda self, context: get_lora_models(self, context),
        update=update_parameters
    ) # type: ignore
    model_strength: bpy.props.FloatProperty(
        name="Model Strength",
        description="Strength of the LoRA's effect on the model's weights",
        default=1.0,
        min=0.0,
        max=100.0, # Adjusted max based on typical LoRA usage
        update=update_parameters
    )  # type: ignore
    clip_strength: bpy.props.FloatProperty(
        name="CLIP Strength",
        description="Strength of the LoRA's effect on the CLIP/text conditioning",
        default=1.0,
        min=0.0,
        max=100.0, # Adjusted max
        update=update_parameters
    )  # type: ignore

def get_controlnet_models(context, unit_type):
    """
    Get available ControlNet models suitable for a specific 'unit_type'
    based on user assignments in addon preferences.

    Args:
        context: Blender context.
        unit_type (str): The type required (e.g., 'depth', 'canny').

    Returns:
        list: A list of (identifier, name, description) tuples for EnumProperty.
    """
    items = []
    prefs = context.preferences.addons.get(__package__)
    if not prefs:
        return [("NO_PREFS", "Addon Error", "Could not access preferences")]

    mappings = prefs.preferences.controlnet_model_mappings

    if not mappings:
         return [("REFRESH", "Refresh List in Prefs", "Fetch models via Preferences")]

    # Determine which boolean property corresponds to the requested unit_type
    prop_name = f"supports_{unit_type}"

    found_count = 0
    for item in mappings:
        # Check if the item object actually has the property (safety check)
        if hasattr(item, prop_name):
            # Check if the boolean flag for the required type is True
            if getattr(item, prop_name):
                # Identifier and Name are the filename
                items.append((item.name, item.name, f"ControlNet: {item.name}"))
                found_count += 1

    if found_count == 0:
         return [("NO_ASSIGNED", f"No models assigned to '{unit_type}'", f"Assign types in Addon Preferences or Refresh")]

    # Sort alphabetically
    items.sort(key=lambda x: x[1])

    return items

def get_lora_models(self, context):
    """Returns the cached list of LoRA models."""
    global _cached_lora_list
    # Basic check
    if not _cached_lora_list:
        return [("NONE_AVAILABLE", "None available", "Fetch models from server")]
    return _cached_lora_list

class RefreshCheckpointList(bpy.types.Operator):
    """Fetches Checkpoint/UNET models from ComfyUI API and updates the cache."""
    bl_idname = "stablegen.refresh_checkpoint_list"
    bl_label = "Refresh Checkpoint/UNET List"
    bl_description = "Connect to ComfyUI server to get available Checkpoint/UNET models"

    @classmethod
    def poll(cls, context):
        prefs = context.preferences.addons.get(__package__)
        return prefs and prefs.preferences.server_address

    def execute(self, context):
        prefs = context.preferences.addons.get(__package__)
        if not prefs:
            self.report({'ERROR'}, "Cannot access addon preferences.")
            return {'CANCELLED'}

        server_address = prefs.preferences.server_address
        architecture = getattr(context.scene, "model_architecture", "sdxl")

        def _bg_work():
            """Background thread — network I/O only, no bpy access."""
            model_list = None
            if architecture == 'sdxl':
                model_list = _fetch_api_list(server_address, "/models/checkpoints")
                model_type_desc = "Checkpoint"
            elif architecture in ('flux1', 'qwen_image_edit'):
                model_list = _fetch_api_list(server_address, "/models/unet_gguf")
                if model_list is not None:
                    extra = _fetch_api_list(server_address, "/models/diffusion_models")
                    if extra:
                        model_list.extend(extra)
                model_type_desc = "UNET" if architecture == 'flux1' else "UNET (GGUF/Safetensors)"
            else:
                model_type_desc = "Model"
            return {'model_list': model_list, 'architecture': architecture,
                    'model_type_desc': model_type_desc}

        def _on_done(result):
            """Main-thread callback — update caches and UI."""
            global _cached_checkpoint_list, _cached_checkpoint_architecture, _pending_refreshes
            _pending_refreshes = max(0, _pending_refreshes - 1)
            if result is None:
                return

            model_list = result.get('model_list')
            arch = result.get('architecture')
            desc = result.get('model_type_desc', 'Model')

            if model_list is None:
                _cached_checkpoint_list = [("NO_SERVER", "Set Server Address", "Cannot fetch")]
                _cached_checkpoint_architecture = None
                print(f"Checkpoint refresh: cannot reach server.")
            elif not model_list:
                _cached_checkpoint_list = [("NONE_FOUND", f"No {desc}s Found", "Server list is empty")]
                _cached_checkpoint_architecture = arch
                print(f"Checkpoint refresh: no {desc} models found.")
            else:
                items = []
                for name in sorted(model_list):
                    items.append((name, name, f"{desc}: {name}"))
                _cached_checkpoint_list = items
                _cached_checkpoint_architecture = arch
                print(f"Checkpoint refresh: {len(items)} {desc}(s) found.")

            # Reset model_name if current selection is no longer valid
            scene = bpy.context.scene if hasattr(bpy.context, 'scene') else None
            if scene:
                current = scene.model_name
                valid_ids = {it[0] for it in _cached_checkpoint_list}
                if current not in valid_ids:
                    placeholder = next((it[0] for it in _cached_checkpoint_list
                                        if it[0].startswith("NO_") or it[0] == "NONE_FOUND"), None)
                    if placeholder:
                        scene.model_name = placeholder
                    elif _cached_checkpoint_list:
                        scene.model_name = _cached_checkpoint_list[0][0]

            # Redraw UI
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()

        _run_async(_bg_work, _on_done)
        global _pending_refreshes
        _pending_refreshes += 1
        self.report({'INFO'}, "Fetching checkpoint list...")
        return {'FINISHED'}
    
class RefreshLoRAList(bpy.types.Operator):
    """Fetches LoRA models from ComfyUI API and updates the cache."""
    bl_idname = "stablegen.refresh_lora_list"
    bl_label = "Refresh LoRA List"
    bl_description = "Connect to ComfyUI server to get available LoRA models"

    @classmethod
    def poll(cls, context):
        prefs = context.preferences.addons.get(__package__)
        return prefs and prefs.preferences.server_address

    def execute(self, context):
        prefs = context.preferences.addons.get(__package__)
        if not prefs:
            self.report({'ERROR'}, "Cannot access addon preferences.")
            return {'CANCELLED'}

        server_address = prefs.preferences.server_address

        def _bg_work():
            return {'lora_list': _fetch_api_list(server_address, "/models/loras")}

        def _on_done(result):
            global _cached_lora_list, _pending_refreshes
            _pending_refreshes = max(0, _pending_refreshes - 1)
            if result is None:
                return

            lora_list = result.get('lora_list')

            if lora_list is None:
                _cached_lora_list = [("NO_SERVER", "Set Server Address", "Cannot fetch")]
                print("LoRA refresh: cannot reach server.")
            elif not lora_list:
                _cached_lora_list = [("NONE_FOUND", "No LoRAs Found", "Server list is empty")]
                print("LoRA refresh: no models found.")
            else:
                items = []
                for name in sorted(lora_list):
                    items.append((name, name, f"LoRA: {name}"))
                _cached_lora_list = items
                print(f"LoRA refresh: {len(items)} model(s) found.")

            # Clean up invalid lora_units selections
            scene = bpy.context.scene if hasattr(bpy.context, 'scene') else None
            if scene and hasattr(scene, 'lora_units'):
                valid_ids = {it[0] for it in _cached_lora_list}
                placeholder = next((it[0] for it in _cached_lora_list
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

            # Redraw UI
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()

        _run_async(_bg_work, _on_done)
        global _pending_refreshes
        _pending_refreshes += 1
        self.report({'INFO'}, "Fetching LoRA list...")
        return {'FINISHED'}

class AddControlNetUnit(bpy.types.Operator):
    bl_idname = "stablegen.add_controlnet_unit"
    bl_label = "Add ControlNet Unit"
    bl_description = "Add a ControlNet Unit. Only one unit per type is allowed."

    unit_type: bpy.props.EnumProperty(
        name="Type",
        items=[('depth', 'Depth', ''), ('canny', 'Canny', ''), ('normal', 'Normal', '')],
        default='depth',
        update=update_parameters
    ) # type: ignore

    model_name: bpy.props.EnumProperty(
        name="Model",
        description="Select the ControlNet model",
        items=lambda self, context: get_controlnet_models(context, self.unit_type),
        update=update_parameters
    ) # type: ignore

    def invoke(self, context, event):
        # Always prompt for unit type and model selection
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "unit_type")
        models = get_controlnet_models(context, self.unit_type)
        if len(models) > 1:
            layout.prop(self, "model_name")

    def execute(self, context):
        units = context.scene.controlnet_units
        # Only add if not already present
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
        # Force redraw of the UI
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
                # Force redraw of the UI
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
        addon_prefs = context.preferences.addons.get(__package__)

        if not addon_prefs: # Should not happen if addon is enabled
            return False
        addon_prefs = addon_prefs.preferences

        # Get the merged list of LoRAs.
        # Assuming get_lora_models is robust and returns placeholders if dirs are bad.
        lora_enum_items = get_lora_models(scene, context) 
        
        # Count actual available LoRAs, excluding placeholders/errors
        # Placeholders used in get_models_from_directory and merge_and_deduplicate_models
        placeholder_ids = {"NONE_AVAILABLE", "NO_COMFYUI_DIR_LORA", "NO_LORAS_SUBDIR", "PERM_ERROR", "SCAN_ERROR", "NONE_FOUND"} # Add any others used by your helpers
        
        available_lora_files_count = sum(1 for item in lora_enum_items if item[0] not in placeholder_ids)

        if available_lora_files_count == 0:
            cls.poll_message_set("No LoRA model files found in any specified directory (including subdirectories).")
            return False

        num_current_lora_units = len(scene.lora_units)
        # Prevent adding more units than distinct available LoRA files
        if num_current_lora_units >= available_lora_files_count:
            cls.poll_message_set("All available distinct LoRA models appear to have corresponding units.")
            return False
            
        return True

    def execute(self, context):
        loras = context.scene.lora_units
        new_lora = loras.add()
        
        # Get available LoRAs (these are (identifier, name, description) tuples)
        all_lora_enum_items = get_lora_models(context.scene, context)
        
        placeholder_ids = {"NONE_AVAILABLE", "NO_COMFYUI_DIR_LORA", "NO_LORAS_SUBDIR", "PERM_ERROR", "SCAN_ERROR", "NONE_FOUND"}
        available_lora_identifiers = [item[0] for item in all_lora_enum_items if item[0] not in placeholder_ids]
        
        if available_lora_identifiers:
            current_lora_model_identifiers_in_use = {unit.model_name for unit in loras if unit.model_name and unit.model_name not in placeholder_ids}
            
            assigned_model = None
            # Try to assign a LoRA that isn't already in use by another unit
            for lora_id in available_lora_identifiers:
                if lora_id not in current_lora_model_identifiers_in_use:
                    assigned_model = lora_id
                    break
            
            # If all available LoRAs are "in use" or no unused one was found, assign the first available one
            if not assigned_model:
                assigned_model = available_lora_identifiers[0]

            if assigned_model:
                try:
                    new_lora.model_name = assigned_model
                except TypeError: 
                    # This might happen if the enum items list isn't perfectly in sync
                    print(f"AddLoRAUnit Execute: TypeError assigning model '{assigned_model}'. Enum might not be ready.")
                    pass 
        
        new_lora.model_strength = 1.0
        new_lora.clip_strength = 1.0
        context.scene.lora_units_index = len(loras) - 1 # Select the newly added unit
        
        # Ensure parameters are updated which might affect preset status
        update_parameters(self, context) 
        
        # Force UI redraw
        for area in context.screen.areas: 
            if area.type == 'VIEW_3D': # Redraw 3D views, common place for the panel
                area.tag_redraw()
            elif area.type == 'PROPERTIES': # Redraw properties editor if panel is there
                 area.tag_redraw()

        return {'FINISHED'}
    
class RemoveLoRAUnit(bpy.types.Operator):
    bl_idname = "stablegen.remove_lora_unit"
    bl_label = "Remove Selected LoRA Unit"
    bl_description = "Remove the selected LoRA from the chain"

    @classmethod
    def poll(cls, context):
        scene = context.scene
        # Operator can run if there are LoRA units AND the current index is valid
        return len(scene.lora_units) > 0 and \
               0 <= scene.lora_units_index < len(scene.lora_units)

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

# load handler to set default ControlNet and LoRA units on first load
@persistent
def load_handler(dummy):
    global _cached_checkpoint_architecture, _pending_checkpoint_refresh_architecture
    if bpy.context.scene:
        scene = bpy.context.scene
        addon_prefs = bpy.context.preferences.addons[__package__].preferences

        # Re-register aspect-ratio crop overlays if any cameras have
        # per-camera resolutions stored.  GPU draw handlers do not
        # survive file loads so they must be re-created here.
        try:
            from .render_tools import _sg_ensure_crop_overlay
            for obj in scene.objects:
                if obj.type == 'CAMERA' and 'sg_display_crop' in obj:
                    _sg_ensure_crop_overlay()
                    break
        except Exception:
            pass

        if hasattr(scene, "controlnet_units") and not scene.controlnet_units:
            default_unit = scene.controlnet_units.add()
            default_unit.unit_type = 'depth'
        # Default LoRA Unit
        if hasattr(scene, "lora_units") and not scene.lora_units:
            default_lora_filename_to_find = None
            model_strength = 1.0
            clip_strength = 1.0

            if scene.model_architecture == 'sdxl':
                default_lora_filename_to_find = 'sdxl_lightning_8step_lora.safetensors'
            elif scene.model_architecture == 'qwen_image_edit':
                default_lora_filename_to_find = 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors'
                clip_strength = 0.0 # Qwen uses model-only LoRA

            if not default_lora_filename_to_find:
                return # No default LoRA for this architecture

            all_available_loras_enums = get_lora_models(scene, bpy.context) 
            
            found_lora_identifier_to_load = None
            for identifier, name, description in all_available_loras_enums:
                # Identifiers are relative paths like "subdir/model.safetensors" or "model.safetensors"
                # Check if the identifier (which is the relative path) ends with the desired filename
                if identifier.endswith(default_lora_filename_to_find):
                    # Ensure it's not a placeholder/error identifier
                    if identifier not in ["NONE_AVAILABLE", "NO_COMFYUI_DIR_LORA", "NO_LORAS_SUBDIR", "PERM_ERROR", "SCAN_ERROR", "NONE_FOUND"]:
                        found_lora_identifier_to_load = identifier
                        break 
            
            if found_lora_identifier_to_load:
                new_lora_unit = None 
                try:
                    new_lora_unit = scene.lora_units.add()
                    new_lora_unit.model_name = found_lora_identifier_to_load
                    new_lora_unit.model_strength = model_strength
                    new_lora_unit.clip_strength = clip_strength
                    # print(f"StableGen Load Handler: Default LoRA '{found_lora_identifier_to_load}' added.")
                except TypeError:
                    # This can happen if Enum items are not fully synchronized at this early stage of loading.
                    print(f"StableGen Load Handler: TypeError setting default LoRA '{found_lora_identifier_to_load}'. Enum items might not be fully ready.")
                    if new_lora_unit and scene.lora_units and new_lora_unit == scene.lora_units[-1]:
                        scene.lora_units.remove(len(scene.lora_units)-1) # Attempt to remove partially added unit
                except Exception as e:
                    print(f"StableGen Load Handler: Unexpected error setting default LoRA '{found_lora_identifier_to_load}': {e}")
                    if new_lora_unit and scene.lora_units and new_lora_unit == scene.lora_units[-1]:
                        scene.lora_units.remove(len(scene.lora_units)-1)

        # Ensure checkpoint cache matches the scene architecture that just loaded
        current_architecture = getattr(scene, "model_architecture", None)
        prefs_wrapper = bpy.context.preferences.addons.get(__package__)
        if current_architecture and prefs_wrapper:
            prefs = prefs_wrapper.preferences
            if prefs.server_address and current_architecture != _cached_checkpoint_architecture and _pending_checkpoint_refresh_architecture != current_architecture:

                def _refresh_checkpoint_for_architecture():
                    global _pending_checkpoint_refresh_architecture
                    try:
                        bpy.ops.stablegen.refresh_checkpoint_list('INVOKE_DEFAULT')
                    except Exception as timer_error:
                        print(f"StableGen Load Handler: Failed to refresh checkpoints for '{current_architecture}': {timer_error}")
                    finally:
                        _pending_checkpoint_refresh_architecture = None
                    return None

                _pending_checkpoint_refresh_architecture = current_architecture
                bpy.app.timers.register(_refresh_checkpoint_for_architecture, first_interval=0.2)

classes_to_append = [CheckServerStatus, RefreshCheckpointList, RefreshLoRAList, STABLEGEN_UL_ControlNetMappingList, ControlNetModelMappingItem, RefreshControlNetMappings, StableGenAddonPreferences, ControlNetUnit, LoRAUnit, AddControlNetUnit, RemoveControlNetUnit, AddLoRAUnit, RemoveLoRAUnit]
for cls in classes_to_append:
    classes.append(cls)
for cls in _debug_classes:
    classes.append(cls)

def register():
    """     
    Registers the addon.         
    :return: None     
    """
    for cls in classes:
        bpy.utils.register_class(cls)

    def initial_refresh():
        print("StableGen: Performing initial model list refresh...")
        try:
            bpy.ops.stablegen.check_server_status('INVOKE_DEFAULT')
            if not bpy.context.preferences.addons.get(__package__).preferences.server_online:
                print("StableGen: Server not reachable during initial refresh.")
                return None
            # Check if server address is set before attempting
            prefs = bpy.context.preferences.addons.get(__package__)
            if prefs and prefs.preferences.server_address:
                 bpy.ops.stablegen.refresh_checkpoint_list('INVOKE_DEFAULT')
                 bpy.ops.stablegen.refresh_lora_list('INVOKE_DEFAULT')
                 bpy.ops.stablegen.refresh_controlnet_mappings('INVOKE_DEFAULT')
            else:
                 print("StableGen: Server address not set, skipping initial refresh.")
            # Run load handler to set defaults
            load_handler(None)
        except Exception as e:
            # Catch potential errors during startup refresh
            print(f"StableGen: Error during initial refresh: {e}")

        return None # Timer runs only once
    
    bpy.app.timers.register(initial_refresh, first_interval=1.0) # Delay slightly

    bpy.types.Scene.comfyui_prompt = bpy.props.StringProperty(
        name="ComfyUI Prompt",
        description="Enter the text prompt for ComfyUI generation",
        default="gold cube",
        update=update_parameters
    )
    bpy.types.Scene.comfyui_negative_prompt = bpy.props.StringProperty(
        name="ComfyUI Negative Prompt",
        description="Enter the negative text prompt for ComfyUI generation",
        default="",
        update=update_parameters
    )
    bpy.types.Scene.model_name = bpy.props.EnumProperty(
        name="Model Name",
        description="Select the SDXL checkpoint",
        items=update_model_list,
        update=update_parameters
    )
    bpy.types.Scene.seed = bpy.props.IntProperty(
        name="Seed",
        description="Seed for image generation",
        default=42,
        min=0,
        max=1000000,
        update=update_parameters
    )
    bpy.types.Scene.control_after_generate = bpy.props.EnumProperty(
        name="Control After Generate",
        description="Control behavior after generation",
        items=[
            ('fixed', 'Fixed', ''),
            ('increment', 'Increment', ''),
            ('decrement', 'Decrement', ''),
            ('randomize', 'Randomize', '')
        ],
        default='fixed',
        update=update_parameters
    )
    bpy.types.Scene.steps = bpy.props.IntProperty(
        name="Steps",
        description="Number of steps for generation",
        default=8,
        min=0,
        max=200,
        update=update_parameters
    )
    bpy.types.Scene.cfg = bpy.props.FloatProperty(
        name="CFG",
        description="Classifier-Free Guidance scale",
        default=1.5,
        min=0.0,
        max=100.0,
        update=update_parameters
    )
    bpy.types.Scene.sampler = bpy.props.EnumProperty(
        name="Sampler",
        description="Sampler for generation",
        items=[
            ('euler', 'Euler', ''),
            ('euler_ancestral', 'Euler A', ''),
            ('dpmpp_sde', 'DPM++ SDE', ''),
            ('dpmpp_2m', 'DPM++ 2M', ''),
            ('dpmpp_2s_ancestral', 'DPM++ 2S Ancestral', ''),
        ],
        default='dpmpp_2s_ancestral',
        update=update_parameters
    )
    bpy.types.Scene.scheduler = bpy.props.EnumProperty(
        name="Scheduler",
        description="Scheduler for generation",
        items=[
            ('sgm_uniform', 'SGM Uniform', ''),
            ('karras', 'Karras', ''),
            ('beta', 'Beta', ''),
            ('normal', 'Normal', ''),
            ('simple', 'Simple', ''),
        ],
        default='sgm_uniform',
        update=update_parameters
    )
    bpy.types.Scene.show_advanced_params = bpy.props.BoolProperty(
        name="Show Advanced Parameters",
        description="Show or hide advanced parameters",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.show_generation_params = bpy.props.BoolProperty(
        name="Show Generation Parameters",
        description="Most important parameters",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.auto_rescale = bpy.props.BoolProperty(
        name="Auto Rescale Resolution",
        description="Automatically rescale resolution to appropriate size for the selected model",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.qwen_rescale_alignment = bpy.props.BoolProperty(
        name="Qwen VL-Aligned Rescale",
        description="Round resolution to multiples of 112 instead of 8 when auto-rescaling. "
                    "112 is the window size used by the Qwen2.5-VL vision encoder "
                    "(LCM of VAE factor 8, ViT patch 14, spatial merge 28). "
                    "The official diffusers pipeline rounds to 32; using 112 is stricter "
                    "and may reduce subtle zoom / pixel-shift artifacts in some cases",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.auto_rescale_target_mp = bpy.props.FloatProperty(
        name="Target Megapixels",
        description=(
            "Target total megapixels when auto-rescaling the render resolution.\n\n"
            "Default: 1.0 MP (~1024\u00d71024). This is the native resolution for SDXL "
            "and the recommended default for Qwen Image Edit.\n\n"
            "For SDXL / Flux models 1.0 MP is a safe default that matches "
            "the training resolution. Higher values are possible if your "
            "model and hardware support it, but may reduce quality or cause "
            "out-of-memory issues.\n\n"
            "For Qwen Image Edit the vision encoder becomes unreliable above "
            "~1.4 MP and the model often fails to follow the prompt above "
            "~1.2 MP. Values below ~0.3 MP also cause failures.\n\n"
            "Adjust with care \u2013 staying close to 1.0 MP works best for most setups"
        ),
        default=1.0,
        min=0.1,
        max=4.0,
        step=10,
        precision=2,
        update=update_parameters
    )
    bpy.types.Scene.use_ipadapter = bpy.props.BoolProperty(
        name="Use IPAdapter",
        description="""Use IPAdapter for image generation. Requires an external reference image. Can improve consistency, can be useful for generating images with similar styles.\n\n - Has priority over mode specific IPAdapter.""",
        default=False,
        update=update_parameters
    )
    #IPAdapter image
    bpy.types.Scene.ipadapter_image = bpy.props.StringProperty(
        name="Reference Image",
        description="Path to the reference image",
        default="",
        subtype='FILE_PATH',
        update=update_parameters
    )
    bpy.types.Scene.ipadapter_strength = bpy.props.FloatProperty(
        name="IPAdapter Strength",
        description="Strength for IPAdapter",
        default=1.0,
        min=-1.0,
        max=3.0,
        update=update_parameters
    )
    bpy.types.Scene.ipadapter_start = bpy.props.FloatProperty(
        name="IPAdapter Start",
        description="Start percentage for IPAdapter (/100)",
        default=0.0,
        min=0.0,
        max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.ipadapter_end = bpy.props.FloatProperty(
        name="IPAdapter End",
        description="End percentage for IPAdapter (/100)",
        default=1.0,
        min=0.0,
        max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.ipadapter_weight_type = bpy.props.EnumProperty(
        name="IPAdapter Weight Type",
        description="Weight type for IPAdapter",
        items=[
            ('standard', 'Standard', ''),
            ('prompt', 'Prompt is more important', ''),
            ('style', 'Style transfer', ''),
        ],
        default='style',
        update=update_parameters
    )
    bpy.types.Scene.sequential_ipadapter = bpy.props.BoolProperty(
        name="Use IPAdapter",
        description="""Uses IPAdapter to improve consistency between images.\n\n - Applicable for Separate, Sequential and Refine modes.\n - Uses either the first generated image or the most recent one as a reference for the rest of the images.\n - If 'Regenerate IPAdapter' is enabled, the first viewpoint will be regenerated with IPAdapter to match the rest of the images.\n - If 'Use IPAdapter (External Image)' is enabled, this setting is effectively overriden.""",
        default=False,
        update=update_parameters
    )
    def _get_ipadapter_mode_items(self, context):
        items = [
            ('first', 'Use first generated image', '', 0),
            ('recent', 'Use most recent generated image', '', 1),
        ]
        if context:
            is_local_edit = (
                context.scene.generation_method == 'local_edit'
                or (context.scene.model_architecture.startswith('qwen')
                    and context.scene.qwen_generation_method == 'local_edit')
            )
            if is_local_edit:
                items.append(('original_render', 'Use original render', 'Uses the existing texture render from each camera viewpoint as IPAdapter reference', 2))
            # TRELLIS.2 input image option — only when architecture is trellis2
            if getattr(context.scene, 'architecture_mode', '') == 'trellis2':
                items.append(('trellis2_input', 'Use TRELLIS.2 input image',
                              'Uses the input image from TRELLIS.2 mesh generation as IPAdapter reference', 3))
        return items
    bpy.types.Scene.sequential_ipadapter_mode = bpy.props.EnumProperty(
        name="IPAdapter Mode",
        description="Mode for IPAdapter in sequential generation",
        items=_get_ipadapter_mode_items,
        update=update_parameters
    )
    bpy.types.Scene.sequential_desaturate_factor = bpy.props.FloatProperty(
        name="Desaturate Recent Image",
        description="Desaturation factor for the 'most recent' image to prevent color stacking. 0.0 is no change, 1.0 is fully desaturated",
        default=0.0,
        min=0.0,
        max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.sequential_contrast_factor = bpy.props.FloatProperty(
        name="Reduce Contrast of Recent Image",
        description="Contrast reduction factor for the 'most recent' image to prevent contrast stacking. 0.0 is no change, 1.0 is maximum reduction (grey)",
        default=0.0,
        min=0.0,
        max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.sequential_ipadapter_regenerate = bpy.props.BoolProperty(
        name="Regenerate IPAdapter",
        description="IPAdapter generations may differ from the original image. This option regenerates the first viewpoint with IPAdapter to match the rest of the images.",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.sequential_ipadapter_regenerate_wo_controlnet = bpy.props.BoolProperty(
        name="Generate IPAdapter reference without ControlNet",
        description="Generate the first viewpoint with IPAdapter without ControlNet. This is useful for generating a reference image that is not affected by ControlNet. Can possibly generate higher quality reference.",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.generation_method = bpy.props.EnumProperty(
        name="Generation Mode",
        description="Choose the mode for generating images",
        items=[
            ('separate', 'Generate Separately', 'Generates images one by one for each viewpoint. Each image is generated independently using only its own control signals (e.g., depth map) without context from other views. All images are applied at the end.'),
            ('sequential', 'Generate Sequentially', 'Generates images viewpoint by viewpoint. After the first view, each subsequent view is generated using inpainting, guided by a visibility mask and an RGB render of the texture projected from previous viewpoints to maintain consistency.'),
            ('grid', 'Generate Using Grid', 'Combines control signals from all viewpoints into a single grid, generates a single image, then splits it back into individual viewpoint textures. Faster but lower resolution per view. Includes an optional second pass to refine each split image individually at full resolution for improved quality.'),
            ('refine', 'Refine/Restyle Texture (Img2Img)', 'Uses the current rendered texture appearance as input for an img2img generation pass. Replaces the previous material with the new result. Good for restyling or globally changing the look of an existing texture. Works on any existing material setup.'),
            ('local_edit', 'Local Edit', 'Make localized changes to an existing texture. Point cameras at areas you want to modify — the new generation blends over the original using angle and vignette-based feathering, preserving untouched areas. Works only with StableGen generated textures.'),
            ('uv_inpaint', 'UV Inpaint Missing Areas', 'Identifies untextured areas on a standard UV map using a visibility calculation. Performs baking if not baked already. Performs diffusion inpainting directly on the UV texture map to fill only these missing regions, using the surrounding texture as context.'),
        ],
        default='sequential',
        update=update_parameters
    )
    bpy.types.Scene.qwen_generation_method = bpy.props.EnumProperty(
        name="Generation Mode",
        description="Choose the mode for generating images with Qwen",
        items=[
            ('generate', 'Generate', 'Standard generation mode'),
            ('refine', 'Refine', 'Refine/restyle the entire texture using Qwen Image Edit. Replaces the existing material with the new result. Describe the desired look in the prompt — you can completely change the style, color scheme, or overall appearance.'),
            ('local_edit', 'Local Edit', 'Make targeted changes to specific areas of the texture. Point cameras at what you want to change and describe the edit — you can change colors, add details, sharpen, alter text, or restyle selected parts. Untouched areas are preserved.'),
        ],
        default='generate',
        update=update_parameters
    )

    bpy.types.Scene.qwen_refine_use_prev_ref = bpy.props.BoolProperty(
        name="Use Previous Refined View",
        description="Use the previous modified/refined view output as a second image for reference (Image 2)",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.qwen_refine_use_depth = bpy.props.BoolProperty(
        name="Use Depth Map",
        description="Include depth map as an additional reference image (Image 3)",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.qwen_timestep_zero_ref = bpy.props.BoolProperty(
        name="Timestep-Zero References",
        description="Tell the model that reference images are fully clean (timestep 0) instead of noisy. "
                    "Reduces color shift and over-saturation, especially with Qwen 2511. "
                    "Requires the FluxKontextMultiReferenceLatentMethod node in ComfyUI",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.refine_images = bpy.props.BoolProperty(
        name="Refine Images",
        description="Refine images after generation",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.refine_steps = bpy.props.IntProperty(
        name="Refine Steps",
        description="Number of steps for refining",
        default=8,
        min=0,
        max=200,
        update=update_parameters
    )
    bpy.types.Scene.refine_sampler = bpy.props.EnumProperty(
        name="Refine Sampler",
        description="Sampler for refining",
        items=[
            ('euler', 'Euler', ''),
            ('euler_ancestral', 'Euler A', ''),
            ('dpmpp_sde', 'DPM++ SDE', ''),
            ('dpmpp_2m', 'DPM++ 2M', ''),
            ('dpmpp_2s_ancestral', 'DPM++ 2S Ancestral', ''),
        ],
        default='dpmpp_2s_ancestral',
        update=update_parameters
    )
    bpy.types.Scene.refine_scheduler = bpy.props.EnumProperty(
        name="Refine Scheduler",
        description="Scheduler for refining",
        items=[
            ('sgm_uniform', 'SGM Uniform', ''),
            ('karras', 'Karras', ''),
            ('beta', 'Beta', ''),
            ('normal', 'Normal', ''),
            ('simple', 'Simple', ''),
        ],
        default='sgm_uniform',
        update=update_parameters
    )
    bpy.types.Scene.denoise = bpy.props.FloatProperty(
        name="Denoise",
        description="Denoise level for refining",
        default=0.8,
        min=0.0,
        max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.refine_cfg = bpy.props.FloatProperty(
        name="Refine CFG",
        description="Classifier-Free Guidance scale for refining",
        default=1.5,
        min=0.0,
        max=100.0,
        update=update_parameters
    )
    bpy.types.Scene.refine_prompt = bpy.props.StringProperty(
        name="Refine Prompt",
        description="Prompt for refining (leave empty to use same prompt as generation)",
        default="",
        update=update_parameters
    )
    bpy.types.Scene.refine_upscale_method = bpy.props.EnumProperty(
        name="Refine Upscale Method",
        description="Upscale method for refining",
        items=[
            ('nearest-exact', 'Nearest Exact', ''),
            ('bilinear', 'Bilinear', ''),
            ('bicubic', 'Bicubic', ''),
            ('lanczos', 'Lanczos', ''),
        ],
        default='lanczos',
        update=update_parameters
    )
    bpy.types.Scene.generation_status = bpy.props.EnumProperty(
        name="Generation Status",
        description="Status of the generation process",
        items=[
            ('idle', 'Idle', ''),
            ('running', 'Running', ''),
            ('waiting', 'Waiting for cancel', ''),
            ('error', 'Error', '')
        ],
        default='idle',
        update=update_parameters
    )
    bpy.types.Scene.generation_progress = bpy.props.FloatProperty(
        name="Generation Progress",
        description="Current progress of image generation",
        default=0.0,
        min=0.0,
        max=100.0,
        update=update_parameters
    )
    bpy.types.Scene.overwrite_material = bpy.props.BoolProperty(
        name="Overwrite Material",
        description="Overwrite existing material",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.discard_factor = bpy.props.FloatProperty(
        name="Discard Factor",
        description="If the texture is facing the camera at an angle greater than this value, it will be discarded. This is useful for preventing artifacts from the very edge of the generated texture appearing when keeping high discard factor (use ~65 for best results when generating textures around an object)",
        default=90.0,
        min=0.0,
        max=180.0,
        update=update_parameters
    )

    bpy.types.Scene.discard_factor_generation_only = bpy.props.BoolProperty(
        name="Reset Discard Angle After Generation",
        description="If enabled, the 'Discard Factor' will be reset to a specified value after generation completes. Useful for sequential/Qwen modes where a low discard angle is needed during generation but not for final blending",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.discard_factor_after_generation = bpy.props.FloatProperty(
        name="Discard Factor After Generation",
        description="The value to set the 'Discard Factor' to after generation is complete",
        default=90.0,
        min=0.0,
        max=180.0,
        update=update_parameters
    )
    bpy.types.Scene.weight_exponent_generation_only = bpy.props.BoolProperty(
        name="Reset Exponent After Generation",
        description="If enabled, the Weight Exponent will be reset to a specified value after generation completes. Useful for Voronoi projection mode where a high exponent produces sharp segmentation during generation but softer blending is preferred for the final result",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.weight_exponent_after_generation = bpy.props.FloatProperty(
        name="Exponent After Generation",
        description="The value to set the Weight Exponent to after generation is complete",
        default=15.0,
        min=0.01,
        max=1000.0,
        update=update_parameters
    )
    bpy.types.Scene.view_blend_use_color_match = bpy.props.BoolProperty(
        name="Match Colors to Viewport",
        description="Match each generated view’s colors to the current viewport texture before blending",
        default=False,
        update=update_parameters,
    )
    bpy.types.Scene.view_blend_color_match_method = bpy.props.EnumProperty(
        name="Color Match Method",
        description="Algorithm used when matching view colors to the viewport texture",
        items=[
            ("mkl",        "MKL",           ""),
            ("hm",         "Histogram",     ""),
            ("reinhard",   "Reinhard",      ""),
            ("mvgd",       "MVGD",          ""),
            ("hm-mvgd-hm", "HM–MVGD–HM",    ""),
            ("hm-mkl-hm",  "HM–MKL–HM",     ""),
        ],
        default="reinhard",
        update=update_parameters,
    )
    bpy.types.Scene.view_blend_color_match_strength = bpy.props.FloatProperty(
        name="Match Strength",
        description="Blend between original and viewport-matched colors",
        default=1.0,
        min=0.0,
        max=2.0,
        update=update_parameters,
    )
    bpy.types.Scene.weight_exponent = bpy.props.FloatProperty(
        name="Weight Exponent",
        description="Controls the falloff curve for viewpoint weighting based on the angle to the surface normal (θ). "
                     "Weight = |cos(θ)|^Exponent. Higher values prioritize straight-on views more strongly, creating sharper transitions. "
                     "1.0 = standard |cos(θ)| weighting..",
        default=3.0,
        min=0.1,
        max=1000.0,
        update=update_parameters
    )
    bpy.types.Scene.allow_modify_existing_textures = bpy.props.BoolProperty(
        name="Allow modifying existing textures",
        description="Disconnect compare node in export_visibility so that smooth output is not pure 1 areas",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.ask_object_prompts = bpy.props.BoolProperty(
        name="Ask for object prompts",
        description="Use object-specific prompts; if disabled, the normal prompt is used for all objects",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.fallback_color = bpy.props.FloatVectorProperty(
        name="Fallback Color",
        description="Color to use as fallback in texture generation",
        subtype='COLOR',
        default=(0.0, 0.0, 0.0),
        min=0.0, max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.sequential_smooth = bpy.props.BoolProperty(
        name="Sequential Smooth",
        description="""Use smooth visibility map for sequential generation mode. Disabling this uses a binary visibility map and may need more mask blurring to reduce artifacts.
        
 - Visibility map is a mask that indicates which pixels have textures already projected from previous viewpoints.
 - Both methods are using weights which are calculated based on the angle between the surface normal and the camera view direction.
 - 'Smooth' uses these calculated weights directly (0.0-1.0 range, giving gradual transitions). The transition point can be further tuned by the 'Smooth Factor' parameters.
 - Disabling 'Smooth' thresholds these weights to create a hard-edged binary mask (0.0 or 1.0).""",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.weight_exponent_mask = bpy.props.BoolProperty(
        name="Weight Exponent Mask",
        description="Use weight exponent for visibility map generation. Uses 1.0 if disabled.",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.canny_threshold_low = bpy.props.IntProperty(
        name="Canny Threshold Low",
        description="Low threshold for Canny edge detection",
        default=0,
        min=0,
        max=255,
        update=update_parameters
    )
    bpy.types.Scene.canny_threshold_high = bpy.props.IntProperty(
        name="Canny Threshold High",
        description="High threshold for Canny edge detection",
        default=80,
        min=0,
        max=255,
        update=update_parameters
    )
    bpy.types.Scene.sequential_factor_smooth = bpy.props.FloatProperty(
        name="Smooth Visibility Black Point",
        description="Controls the black point (start) of the Color Ramp used for the smooth visibility mask in sequential mode. Defines the weight threshold below which areas are considered fully invisible/untextured from previous views. Higher values create a sharper transition start.",
        default=0.15,
        min=0.0,
        max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.sequential_factor_smooth_2 = bpy.props.FloatProperty(
        name="Smooth Visibility White Point",
        description="Controls the white point (end) of the Color Ramp used for the smooth visibility mask in sequential mode. Defines the weight threshold above which areas are considered fully visible/textured from previous views. Lower values create a sharper transition end.",
        default=1.0,
        min=0.0,
        max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.sequential_factor = bpy.props.FloatProperty(
        name="Binary Visibility Threshold",
        description="Threshold value used when 'Sequential Smooth' is OFF. Calculated visibility weights below this value are treated as 0 (invisible), and those above as 1 (visible), creating a hard-edged binary mask.",
        default=0.7,
        min=0.0,
        max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.differential_noise = bpy.props.BoolProperty(
        name="Differential Noise",
        description="Adds latent noise mask to the image before inpainting. This must be used with low factor smooth mask or with a high blur mask radius. Disabling this effectively discrads the mask and only uses the inapaint conditioning.",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.grow_mask_by = bpy.props.IntProperty(
        name="Grow Mask By",
        description="Grow mask by this amount (ComfyUI)",
        default=3,
        min=0,
        update=update_parameters
    )
    bpy.types.Scene.mask_blocky = bpy.props.BoolProperty(
        name="Blocky Visibility Map",
        description="Uses a blocky visibility map. This will downscale the visibility map according to the 8x8 grid which Stable Diffusion uses in latent space. Highly experimental.",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.visibility_vignette = bpy.props.BoolProperty(
        name="Feather Visibility Edges",
        description="Blend refinement edges using a vignette mask to reduce seams",
        default=True,
        update=update_parameters,
    )

    bpy.types.Scene.visibility_vignette_width = bpy.props.FloatProperty(
        name="Vignette Width",
        description="Fraction of the image radius used as a feather band (0 = no feather, 0.5 = very soft edges)",
        default=0.15,
        min=0.0,
        max=0.5,
        update=update_parameters,
    )
    bpy.types.Scene.visibility_vignette_softness = bpy.props.FloatProperty(
        name="Vignette Softness",
        description="Exponent shaping feather falloff (<1 = sharper transition, >1 = softer/wider transition)",
        default=1.0,
        min=0.1,
        max=5.0,
        update=update_parameters,
    )
    bpy.types.Scene.visibility_vignette_blur = bpy.props.BoolProperty(
        name="Blur Vignette Mask",
        description="Apply a Gaussian blur to the vignette mask to soften edges",
        default=False,
        update=update_parameters,
    )

    # Refine Mode Ramp Controls
    bpy.types.Scene.refine_angle_ramp_active = bpy.props.BoolProperty(
        name="Use Angle Ramp",
        description="Blend refinement based on surface angle",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.refine_angle_ramp_pos_0 = bpy.props.FloatProperty(
        name="Angle Ramp Black Point",
        description="Position of the black point (Invisible) for the angle ramp",
        default=0.0,
        min=0.0,
        max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.refine_angle_ramp_pos_1 = bpy.props.FloatProperty(
        name="Angle Ramp White Point",
        description="Position of the white point (Visible) for the angle ramp",
        default=0.05,
        min=0.0,
        max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.refine_feather_ramp_pos_0 = bpy.props.FloatProperty(
        name="Feather Ramp Black Point",
        description="Position of the black point (Invisible) for the feather ramp",
        default=0.0,
        min=0.0,
        max=1.0,
        update=update_parameters
    )
    bpy.types.Scene.refine_feather_ramp_pos_1 = bpy.props.FloatProperty(
        name="Feather Ramp White Point",
        description="Position of the white point (Visible) for the feather ramp",
        default=0.6,
        min=0.0,
        max=1.0,
        update=update_parameters
    )

    bpy.types.Scene.refine_edge_feather_projection = bpy.props.BoolProperty(
        name="Edge Feather (Projection)",
        description="Add a screen-space distance-transform ramp at the projection "
                    "silhouette boundary as an additional multiplier on top of the "
                    "angle×feather weight. Interior surfaces keep full strength; "
                    "only the geometric edge is softened",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.refine_edge_feather_width = bpy.props.IntProperty(
        name="Edge Feather Width",
        description="Width in pixels of the transition band at projection boundaries. "
                    "Larger values produce a wider blend zone",
        default=30,
        min=1,
        max=200,
        update=update_parameters
    )
    bpy.types.Scene.refine_edge_feather_softness = bpy.props.FloatProperty(
        name="Edge Feather Softness",
        description="Rounds off the sharp corners at both ends of the linear feather ramp "
                    "using a Gaussian blur.  0 = raw linear ramp (hard kinks at edge and "
                    "interior boundary).  1 = moderate smoothing.  Higher values give "
                    "progressively gentler transitions without shifting the ramp position",
        default=1.0,
        min=0.0,
        max=5.0,
        step=10,
        precision=2,
        update=update_parameters
    )

    bpy.types.Scene.differential_diffusion = bpy.props.BoolProperty(
        name="Differential Diffusion",
        description="Replace standard inpainting with a differential diffusion based workflow\n\n - Generally works better and reduces artifacts.\n - Using a Smooth Visibilty Map is recommended for Sequential Mode.",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.blur_mask = bpy.props.BoolProperty(
        name="Blur Mask",
        description="Blur mask before inpainting (ComfyUI)",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.blur_mask_radius = bpy.props.IntProperty(
        name="Blur Mask Radius",
        description="Radius for mask blurring (ComfyUI)",
        default=1,
        min=1,
        max=31,
        update=update_parameters
    )
    bpy.types.Scene.blur_mask_sigma = bpy.props.FloatProperty(
        name="Blur Mask Sigma",
        description="Sigma for mask blurring (ComfyUI)",
        default=1.0,
        min=0.1,
        update=update_parameters
    )
    bpy.types.Scene.sequential_custom_camera_order = bpy.props.StringProperty(
        name="Custom Camera Order",
        description="""Custom camera order for Sequential Mode. Format: 'index1,index2,index3,...'
        
 - This will permanently change the order of the cameras in the scene.""",
        default="",
        update=update_parameters
    )
    bpy.types.Scene.clip_skip = bpy.props.IntProperty(
        name="CLIP Skip",
        description="CLIP skip value for generation",
        default=1,
        min=1,
        update=update_parameters
    )
    bpy.types.Scene.stablegen_preset = bpy.props.EnumProperty(
        name="Preset",
        description="Select a preset for easy mode",
        items=get_preset_items,
        default=0
    )

    bpy.types.Scene.active_preset = bpy.props.StringProperty(
    name="Active Preset",
    default="DEFAULT"
    )

    bpy.types.Scene.model_architecture = bpy.props.EnumProperty(
        name="Model Architecture",
        description="Select the model architecture to use for generation",
        items=[
            ('sdxl', 'SDXL', ''),
            ('flux1', 'Flux 1', ''),
            ('qwen_image_edit', 'Qwen Image Edit', '')
        ],
        default='sdxl',
        update=update_combined
    )

    bpy.types.Scene.architecture_mode = bpy.props.EnumProperty(
        name="Architecture",
        description="Select the overall generation architecture",
        items=[
            ('sdxl', 'SDXL', 'Stable Diffusion XL'),
            ('flux1', 'Flux 1', 'Flux 1 architecture'),
            ('qwen_image_edit', 'Qwen Image Edit', 'Qwen Image Edit architecture'),
            ('trellis2', 'TRELLIS.2', 'Image to 3D mesh generation with TRELLIS.2'),
        ],
        default='sdxl',
        update=update_architecture_mode
    )

    bpy.types.Scene.trellis2_generate_from = bpy.props.EnumProperty(
        name="Generate From",
        description="Source for the TRELLIS.2 input",
        items=[
            ('image', 'Image', 'Use an existing image as input'),
            ('prompt', 'Prompt', 'Generate an image from the prompt first, then feed to TRELLIS.2'),
        ],
        default='image',
        update=update_trellis2_generate_from
    )

    bpy.types.Scene.trellis2_texture_mode = bpy.props.EnumProperty(
        name="Texture Mode",
        description="How to generate textures for the 3D mesh",
        items=[
            ('none', 'None', 'Shape only, no texture generation'),
            ('native', 'Native (TRELLIS.2)', 'Use TRELLIS.2 built-in texture generation'),
            ('sdxl', 'SDXL', 'Use SDXL for camera-based texture projection'),
            ('flux1', 'Flux 1', 'Use Flux 1 for camera-based texture projection'),
            ('qwen_image_edit', 'Qwen Image Edit', 'Use Qwen for camera-based texture projection'),
        ],
        default='native',
        update=update_trellis2_texture_mode
    )

    bpy.types.Scene.trellis2_initial_image_arch = bpy.props.EnumProperty(
        name="Initial Image Architecture",
        description="Diffusion architecture used to generate the initial image when Generate From is set to Prompt and Texture Mode is Native or None",
        items=[
            ('sdxl', 'SDXL', 'Stable Diffusion XL'),
            ('flux1', 'Flux 1', 'Flux 1 architecture'),
            ('qwen_image_edit', 'Qwen Image Edit', 'Qwen Image Edit architecture'),
        ],
        default='sdxl',
        update=update_trellis2_initial_image_arch
    )

    bpy.types.Scene.trellis2_camera_count = bpy.props.IntProperty(
        name="Camera Count",
        description="Number of cameras to place around the generated mesh for texture projection",
        default=8,
        min=2,
        max=32,
        update=update_parameters
    )

    bpy.types.Scene.trellis2_placement_mode = bpy.props.EnumProperty(
        name="Placement Mode",
        description="Strategy for placing cameras around the generated mesh",
        items=[
            ('orbit_ring', "Orbit Ring", "Place cameras in a circle around the center"),
            ('hemisphere', "Sphere Coverage", "Distribute cameras evenly across a sphere using a Fibonacci spiral"),
            ('normal_weighted', "Normal-Weighted", "Automatically place cameras to cover the most surface area, using K-means on face normals weighted by area"),
            ('pca_axes', "PCA Axes", "Place cameras along the mesh's principal axes of variation"),
            ('greedy_coverage', "Greedy Coverage", "Iteratively add cameras that maximise new visible surface. Automatically determines the number of cameras needed"),
            ('fan_from_camera', "Fan from Camera", "Spread cameras in an arc around the active camera's orbit position"),
        ],
        default='normal_weighted',
        update=update_parameters
    )

    bpy.types.Scene.trellis2_auto_prompts = bpy.props.BoolProperty(
        name="Auto View Prompts",
        description="Automatically generate view-direction prompts for each camera (e.g. 'front view', 'rear view, from above')",
        default=True,
        update=update_parameters
    )

    bpy.types.Scene.trellis2_exclude_bottom = bpy.props.BoolProperty(
        name="Exclude Bottom Faces",
        description="Ignore downward-facing geometry when placing cameras",
        default=True,
        update=update_parameters
    )

    bpy.types.Scene.trellis2_exclude_bottom_angle = bpy.props.FloatProperty(
        name="Bottom Angle Threshold",
        description="Faces whose normal points more than this many degrees below the horizon are excluded",
        default=1.5533,  # 89 degrees in radians
        min=0.1745,
        max=1.5708,
        subtype='ANGLE',
        unit='ROTATION',
        update=update_parameters
    )

    bpy.types.Scene.trellis2_auto_aspect = bpy.props.EnumProperty(
        name="Auto Aspect Ratio",
        description="Automatically adjust render aspect ratio to match the mesh silhouette",
        items=[
            ('off', "Off", "Use current scene resolution for all cameras"),
            ('shared', "Shared", "Average silhouette aspect across all cameras and set a single scene resolution"),
            ('per_camera', "Per Camera", "Each camera gets its own optimal aspect ratio"),
        ],
        default='per_camera',
        update=update_parameters
    )

    bpy.types.Scene.trellis2_occlusion_mode = bpy.props.EnumProperty(
        name="Occlusion Handling",
        description="How to account for self-occlusion when choosing camera directions",
        items=[
            ('none', "None (Fast)", "Back-face culling only — ignores self-occlusion"),
            ('full_matrix', "Full Occlusion Matrix", "Build a complete BVH-validated visibility matrix. Most accurate but slower"),
            ('two_pass', "Two-Pass Refinement", "Fast back-face pass, then targeted BVH refinement"),
            ('vis_weighted', "Visibility-Weighted", "Weight faces by visibility fraction; mostly-occluded faces have reduced influence"),
        ],
        default='none',
        update=update_parameters
    )

    bpy.types.Scene.trellis2_consider_existing = bpy.props.BoolProperty(
        name="Consider Existing Cameras",
        description="Treat existing cameras as already-placed directions so auto modes avoid duplicating their coverage",
        default=True,
        update=update_parameters
    )

    bpy.types.Scene.trellis2_delete_cameras = bpy.props.BoolProperty(
        name="Delete Cameras After",
        description="Automatically delete cameras placed during the TRELLIS.2 pipeline once generation and texturing are complete",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.trellis2_coverage_target = bpy.props.FloatProperty(
        name="Coverage Target",
        description="Stop adding cameras when this fraction of surface area is visible (Greedy mode)",
        default=0.95,
        min=0.5,
        max=1.0,
        subtype='FACTOR',
        update=update_parameters
    )

    bpy.types.Scene.trellis2_max_auto_cameras = bpy.props.IntProperty(
        name="Max Cameras (Greedy)",
        description="Upper limit on cameras for Greedy Coverage mode",
        default=12,
        min=2,
        max=50,
        update=update_parameters
    )

    bpy.types.Scene.trellis2_fan_angle = bpy.props.FloatProperty(
        name="Fan Angle",
        description="Total angular spread of the fan in degrees",
        default=90.0,
        min=10.0,
        max=350.0,
        update=update_parameters
    )

    bpy.types.Scene.trellis2_import_scale = bpy.props.FloatProperty(
        name="Import Scale (BU)",
        description="Scale imported TRELLIS.2 meshes so the longest axis equals this many Blender units. Set to 0 to keep the original size",
        default=2.0,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=10.0,
        update=update_parameters
    )

    bpy.types.Scene.trellis2_clamp_elevation = bpy.props.BoolProperty(
        name="Clamp Elevation",
        description="Restrict camera elevation to avoid extreme top-down or bottom-up views that diffusion models often struggle with",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.trellis2_max_elevation = bpy.props.FloatProperty(
        name="Max Elevation",
        description="Maximum upward elevation angle. Cameras looking further up will be clamped",
        default=1.2217,  # 70 degrees
        min=0.0,
        max=1.5708,      # 90 degrees
        subtype='ANGLE',
        unit='ROTATION',
        update=update_parameters
    )

    bpy.types.Scene.trellis2_min_elevation = bpy.props.FloatProperty(
        name="Min Elevation",
        description="Minimum downward elevation angle. Cameras looking further down will be clamped",
        default=-0.1745,  # -10 degrees
        min=-1.5708,      # -90 degrees
        max=0.0,
        subtype='ANGLE',
        unit='ROTATION',
        update=update_parameters
    )

    bpy.types.Scene.trellis2_preview_gallery_enabled = bpy.props.BoolProperty(
        name="Preview Gallery",
        description="When in prompt mode, generate multiple images with different seeds and let you pick the best one before proceeding to 3D generation",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.trellis2_preview_gallery_count = bpy.props.IntProperty(
        name="Gallery Count",
        description="Number of images to generate per batch in the preview gallery",
        default=4,
        min=1,
        max=16,
        update=update_parameters
    )

    bpy.types.Scene.qwen_guidance_map_type = bpy.props.EnumProperty(
        name="Guidance Map",
        description="The type of guidance map to use for Qwen Image Edit",
        items=[
            ('depth', 'Depth Map', 'Use depth map for structural guidance'),
            ('normal', 'Normal Map', 'Use normal map for structural guidance'),
            ('workbench', 'Workbench Render', 'Use workbench render for structural guidance'),
            ('viewport', 'Viewport Render', 'Use viewport render (OpenGL) for structural guidance')
        ],
        default='depth',
        update=update_parameters
    )

    bpy.types.Scene.qwen_voronoi_mode = bpy.props.BoolProperty(
        name="Voronoi Projection",
        description=(
            "Instead of zeroing weights of non-generated cameras, keep natural "
            "angle-based weights and project magenta from cameras that have not "
            "been generated yet. Use a high Weight Exponent to achieve Voronoi-"
            "like segmentation where each surface point is dominated by its "
            "closest camera. This eliminates the need for low discard-over-angle "
            "thresholds"
        ),
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.qwen_context_render_mode = bpy.props.EnumProperty(
        name="Context Render",
        description="How to use the RGB context render in sequential mode for Qwen",
        items=[
            ('NONE', 'Disabled', 'Do not use the RGB context render'),
            ('REPLACE_STYLE', 'Replace Style Image', 'Use context render instead of the previous generated image as the style reference'),
            ('ADDITIONAL', 'Additional Context', 'Use context render as an additional image input (image 3) for context')
        ],
        default='NONE',
        update=update_parameters
    )

    bpy.types.Scene.qwen_use_external_style_image = bpy.props.BoolProperty(
        name="Use External Style Image",
        description="Use a separate, external image as the style reference for all viewpoints",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.qwen_external_style_image = bpy.props.StringProperty(
        name="Style Reference Image",
        description="Path to the external style reference image",
        default="",
        subtype='FILE_PATH',
        update=update_parameters
    )

    bpy.types.Scene.qwen_external_style_initial_only = bpy.props.BoolProperty(
        name="External for Initial Only",
        description="Use the external style image for the first image, then use the previously generated image for subsequent images",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.qwen_use_custom_prompts = bpy.props.BoolProperty(
        name="Use Custom Qwen Prompts",
        description="Enable to override the default guidance prompts for the Qwen Image Edit workflow",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.qwen_custom_prompt_initial = bpy.props.StringProperty(
        name="Initial Image Prompt",
        description="Custom prompt for the first generated image. Use {main_prompt} to insert the main prompt text.",
        default="Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2",
        update=update_parameters
    )

    bpy.types.Scene.qwen_custom_prompt_seq_none = bpy.props.StringProperty(
        name="Sequential Prompt (No Context)",
        description="Custom prompt for subsequent images when Context Render is 'Disabled'. Use {main_prompt} to insert the main prompt text.",
       
        default="Change and transfer the format of '{main_prompt}' in image 1 to the style from image 2",
        update=update_parameters
    )

    bpy.types.Scene.qwen_custom_prompt_seq_replace = bpy.props.StringProperty(
        name="Sequential Prompt (Replace Style)",
        description="Custom prompt for subsequent images when Context Render is 'Replace Style'. Use {main_prompt} to insert the main prompt text.",
        default="Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas.",
        update=update_parameters
    )

    bpy.types.Scene.qwen_custom_prompt_seq_additional = bpy.props.StringProperty(
        name="Sequential Prompt (Additional Context)",
        description="Custom prompt for subsequent images when Context Render is 'Additional Context'. Use {main_prompt} to insert the main prompt text.",
        default="Change and transfer the format of image 1 to '{main_prompt}'. Replace all solid magenta areas in image 2. Replace the background with solid gray. The style from image 2 should smoothly continue into the previously magenta areas. Image 3 represents the overall style of the object.",
        update=update_parameters
    )

    bpy.types.Scene.qwen_guidance_fallback_color = bpy.props.FloatVectorProperty(
        name="Guidance Fallback Color",
        description="Color used for fallback regions in the Qwen context render",
        subtype='COLOR',
        default=(1.0, 0.0, 1.0),
        min=0.0,
        max=1.0,
        update=update_parameters
    )

    bpy.types.Scene.qwen_guidance_background_color = bpy.props.FloatVectorProperty(
        name="Guidance Background Color",
        description="Background color for the Qwen context render",
        subtype='COLOR',
        default=(1.0, 0.0, 1.0),
        min=0.0,
        max=1.0,
        update=update_parameters
    )

    bpy.types.Scene.qwen_context_cleanup = bpy.props.BoolProperty(
        name="Use Context Cleanup",
        description="Replace fallback color in subsequent Qwen renders before projection",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.qwen_context_cleanup_hue_tolerance = bpy.props.FloatProperty(
        name="Cleanup Hue Tolerance",
        description="Hue tolerance in degrees for identifying fallback regions during cleanup",
        default=5.0,
        min=0.0,
        max=180.0,
        update=update_parameters
    )

    bpy.types.Scene.qwen_context_cleanup_value_adjust = bpy.props.FloatProperty(
        name="Cleanup Value Adjustment",
        description="Adjust value (brightness) for cleaned pixels. -1 darkens to black, 1 brightens to white.",
        default=0.0,
        min=-1.0,
        max=1.0,
        update=update_parameters
    )

    bpy.types.Scene.qwen_context_fallback_dilation = bpy.props.IntProperty(
        name="Fallback Dilation",
        description="Dilate fallback color regions in the context render before sending to Qwen.",
        default=1,
        min=0,
        max=64,
        update=update_parameters
    )

    bpy.types.Scene.output_timestamp = bpy.props.StringProperty(
        name="Output Timestamp",
        description="Timestamp for generation output directory",
        default=""
    )
    
    bpy.types.Scene.camera_prompts = bpy.props.CollectionProperty(
        type=CameraPromptItem,
        name="Camera Prompts",
        description="Stores viewpoint descriptions for each camera"
    ) # type: ignore
    
    bpy.types.Scene.use_camera_prompts = bpy.props.BoolProperty(
        name="Use Camera Prompts",
        description="Use camera prompts for generating images",
        default=True,
        update=update_parameters
    )

    bpy.types.Scene.sg_camera_order = bpy.props.CollectionProperty(
        type=CameraOrderItem,
        name="Camera Generation Order",
        description="Defines the order in which cameras are processed during generation"
    ) # type: ignore
    bpy.types.Scene.sg_camera_order_index = bpy.props.IntProperty(
        name="Active Camera Order Index",
        default=0
    )
    bpy.types.Scene.sg_use_custom_camera_order = bpy.props.BoolProperty(
        name="Use Custom Camera Order",
        description="When enabled, generation uses the custom camera order list instead of alphabetical sorting",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.show_core_settings = bpy.props.BoolProperty(
        name="Core Generation Settings",
        description="Parameters used for the image generation process. Also includes LoRAs for faster generation.",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.show_lora_settings = bpy.props.BoolProperty(
        name="LoRA Settings",
        description="Settings for custom LoRA management.",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.show_camera_options = bpy.props.BoolProperty(
        name="Camera Settings",
        description="Camera prompt and generation order settings.",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.show_scene_understanding_settings = bpy.props.BoolProperty(
        name="Viewpoint Blending Settings",
        description="Settings for how the addon blends different viewpoints together.",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.show_output_material_settings = bpy.props.BoolProperty(
        name="Output & Material Settings",
        description="Settings for output characteristics and material handling, including texture processing and final image resolution.",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.show_image_guidance_settings = bpy.props.BoolProperty(
        name="Image Guidance (IPAdapter & ControlNet)",
        description="Configuration for advanced image guidance techniques, allowing more precise control via reference images or structural inputs.",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.show_masking_inpainting_settings = bpy.props.BoolProperty(
        name="Inpainting Options",
        description="Parameters for inpainting and mask manipulation to refine specific image areas. (Visible for UV Inpaint & Sequential modes).",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.show_mode_specific_settings = bpy.props.BoolProperty(
        name="Generation Mode Specifics",
        description="Parameters exclusively available for the selected Generation Mode, allowing tailored control over mode-dependent behaviors.",
        default=False,
        update=update_parameters
    )
    
    bpy.types.Scene.apply_bsdf = bpy.props.BoolProperty(
        name ="Apply BSDF",
        description="""Apply the BSDF shader to the material
    - when set to FALSE, the material will be emissive and will not be affected by the scene lighting
    - when set to TRUE, the material will be affected by the scene lighting""",
        default=False,
        update=update_parameters
    )
    
    bpy.types.Scene.generation_mode = bpy.props.EnumProperty(
        name="Generation Mode",
        description="Controls the generation behavior",
        items=[
            ('standard', 'Standard', 'Standard generation process'),
            ('regenerate_selected', 'Regenerate Selected', 'Regenerate only specific viewpoints, keeping the rest from the previous run'),
            ('project_only', 'Project Only', 'Only project existing textures onto the model without generating new ones')
        ],
        default='standard',
        update=update_parameters
    )

    bpy.types.Scene.early_priority_strength = bpy.props.FloatProperty(
        name="Prioritize Initial Views",
        description="""Strength of the priority applied to initial views. Higher values will make the earlier cameras more important than the later ones. Every view will be prioritized over the next one.
    - Very high values may cause various artifacts.""",
        default=0.5,
        min=0.0,
        max=1.0,
        update=update_parameters
    )

    bpy.types.Scene.early_priority = bpy.props.BoolProperty(
        name="Priority Strength",
        description="""Enable blending priority for earlier cameras.
    - This may prevent artifacts caused by later cameras overwriting earlier ones.
    - You will have to place the important cameras first.""",
        default=False,
        update=update_parameters
    )

    bpy.types.Scene.texture_objects = bpy.props.EnumProperty(
        name="Objects to Texture",
        description="Select the objects to texture",
        items=[
            ('all', 'All Visible', 'Texture all visible objects in the scene'),
            ('selected', 'Selected', 'Texture only selected objects'),
        ],
        default='all',
        update=update_parameters
    )

    bpy.types.Scene.use_flux_lora = bpy.props.BoolProperty(
        name="Use FLUX Depth LoRA",
        description="Use FLUX.1-Depth-dev LoRA for depth conditioning instead of ControlNet. This disables all ControlNet units.",
        default=True,
        update=update_parameters
    )


    # IPADAPTER parameters

    bpy.types.Scene.controlnet_units = bpy.props.CollectionProperty(type=ControlNetUnit)
    bpy.types.Scene.lora_units = bpy.props.CollectionProperty(type=LoRAUnit)
    bpy.types.Scene.controlnet_units_index = bpy.props.IntProperty(default=0)
    bpy.types.Scene.lora_units_index = bpy.props.IntProperty(default=0)
    bpy.app.handlers.load_post.append(load_handler)

    # --- TRELLIS.2 Properties ---
    bpy.types.Scene.trellis2_available = bpy.props.BoolProperty(
        name="TRELLIS.2 Available",
        description="Whether TRELLIS.2 nodes are available on the ComfyUI server (auto-detected)",
        default=False
    )
    bpy.types.Scene.show_trellis2_params = bpy.props.BoolProperty(
        name="Show TRELLIS.2 Section",
        description="Toggle TRELLIS.2 Image-to-3D section",
        default=False
    )
    bpy.types.Scene.show_trellis2_advanced = bpy.props.BoolProperty(
        name="Show TRELLIS.2 Settings",
        description="Toggle TRELLIS.2 advanced settings",
        default=False
    )
    bpy.types.Scene.show_trellis2_mesh_settings = bpy.props.BoolProperty(
        name="Show Mesh Generation Settings",
        description="Toggle TRELLIS.2 mesh generation advanced settings",
        default=False
    )
    bpy.types.Scene.show_trellis2_texture_settings = bpy.props.BoolProperty(
        name="Show Texture Settings",
        description="Toggle TRELLIS.2 native texture settings",
        default=False
    )
    bpy.types.Scene.show_trellis2_camera_settings = bpy.props.BoolProperty(
        name="Show Camera Placement Settings",
        description="Toggle TRELLIS.2 camera placement settings",
        default=False
    )
    bpy.types.Scene.show_trellis2_artifact_filter = bpy.props.BoolProperty(
        name="Show Artifact Filtering",
        description="Toggle TRELLIS.2 artifact filtering settings",
        default=False
    )
    bpy.types.Scene.trellis2_last_input_image = bpy.props.StringProperty(
        name="TRELLIS.2 Last Input Image",
        description="Path to the image most recently used as TRELLIS.2 input (set automatically after generation)",
        subtype='FILE_PATH',
        default=""
    )
    bpy.types.Scene.qwen_use_trellis2_style = bpy.props.BoolProperty(
        name="Use TRELLIS.2 Input as Style",
        description="Use the TRELLIS.2 input image as the Qwen style reference",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.qwen_trellis2_style_initial_only = bpy.props.BoolProperty(
        name="TRELLIS.2 Style for Initial Only",
        description="Use the TRELLIS.2 input image only for the first image, then fall back to sequential style for subsequent images",
        default=False,
        update=update_parameters
    )
    # 3-tier progress pipeline flags (written by Trellis2Generate, read by UI)
    bpy.types.Scene.trellis2_pipeline_active = bpy.props.BoolProperty(
        name="TRELLIS.2 Pipeline Active",
        description="True while the TRELLIS.2 pipeline (incl. texturing) is running",
        default=False
    )
    bpy.types.Scene.trellis2_pipeline_phase_start_pct = bpy.props.FloatProperty(
        name="Phase Start %",
        description="Overall-% at which the current texturing phase begins",
        default=0.0, min=0.0, max=100.0
    )
    bpy.types.Scene.trellis2_pipeline_total_phases = bpy.props.IntProperty(
        name="Total Phases",
        description="Total number of phases in the TRELLIS.2 pipeline",
        default=3, min=1, max=3
    )
    bpy.types.Scene.trellis2_input_image = bpy.props.StringProperty(
        name="Input Image",
        description="Path to the reference image for 3D mesh generation",
        subtype='FILE_PATH',
        default=""
    )
    bpy.types.Scene.trellis2_resolution = bpy.props.EnumProperty(
        name="Resolution",
        description="Model resolution for generation. Higher values use more VRAM",
        items=[
            ('512', '512', 'Low resolution, fast, less VRAM'),
            ('1024', '1024', 'Direct 1024 generation with higher sparse structure resolution'),
            ('1024_cascade', '1024 Cascade', 'Medium resolution with cascade (recommended)'),
            ('1536_cascade', '1536 Cascade', 'High resolution with cascade, most VRAM'),
        ],
        default='1024_cascade',
        update=update_parameters
    )
    bpy.types.Scene.trellis2_vram_mode = bpy.props.EnumProperty(
        name="VRAM Mode",
        description="Controls model offloading strategy to manage VRAM usage",
        items=[
            ('keep_loaded', 'Keep Loaded', 'Keep all models in VRAM (fastest, ~12GB VRAM)'),
            ('disk_offload', 'Disk Offload', 'Load models on demand from disk (recommended for <=16GB VRAM)'),
        ],
        default='disk_offload',
        update=update_parameters
    )
    bpy.types.Scene.trellis2_attn_backend = bpy.props.EnumProperty(
        name="Attention Backend",
        description="Attention implementation to use. flash_attn is fastest but requires CUDA",
        items=[
            ('flash_attn', 'Flash Attention', 'Fastest (requires flash-attn package)'),
            ('xformers', 'xFormers', 'Fast (requires xformers package)'),
            ('sdpa', 'SDPA', 'PyTorch native, always available'),
            ('sageattn', 'SageAttention', 'SageAttention backend (requires sageattn package)'),
        ],
        default='flash_attn',
        update=update_parameters
    )
    bpy.types.Scene.trellis2_seed = bpy.props.IntProperty(
        name="Seed",
        description="Random seed for generation (0 = random)",
        default=0,
        min=0,
        max=2147483647
    )
    bpy.types.Scene.trellis2_ss_guidance = bpy.props.FloatProperty(
        name="SS Guidance",
        description="Sparse structure CFG scale. Higher = stronger adherence to input image",
        default=7.5,
        min=1.0,
        max=20.0,
        step=10,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_ss_steps = bpy.props.IntProperty(
        name="SS Steps",
        description="Sparse structure sampling steps. More steps = better quality but slower",
        default=12,
        min=1,
        max=50,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_shape_guidance = bpy.props.FloatProperty(
        name="Shape Guidance",
        description="Shape CFG scale. Higher = stronger adherence to input image",
        default=7.5,
        min=1.0,
        max=20.0,
        step=10,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_shape_steps = bpy.props.IntProperty(
        name="Shape Steps",
        description="Shape sampling steps. More steps = better quality but slower",
        default=12,
        min=1,
        max=50,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_tex_guidance = bpy.props.FloatProperty(
        name="Texture Guidance",
        description="Texture CFG scale. Higher = stronger adherence to input image",
        default=7.5,
        min=1.0,
        max=20.0,
        step=10,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_tex_steps = bpy.props.IntProperty(
        name="Texture Steps",
        description="Texture sampling steps. More steps = better quality but slower",
        default=12,
        min=1,
        max=50,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_max_tokens = bpy.props.IntProperty(
        name="Max Tokens",
        description="Max sparse-voxel tokens during cascade upsampling (only affects cascade modes). "
                    "Higher = more detail but more VRAM. "
                    "Try 32768 or 24576 if running out of VRAM",
        default=49152,
        min=16384,
        max=65536,
        step=4096,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_texture_size = bpy.props.IntProperty(
        name="Texture Size",
        description="Resolution of the UV-baked texture in pixels. Higher values reduce aliasing at UV seams. "
                    "The native voxel grid is 1024, so values above 1024 interpolate existing data",
        default=4096,
        min=512,
        max=8192,
        step=512,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_decimation = bpy.props.IntProperty(
        name="Decimation Target",
        description="Target polygon count for mesh simplification. Lower = simpler mesh",
        default=100000,
        min=1000,
        max=5000000,
        step=10000,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_remesh = bpy.props.BoolProperty(
        name="Remesh",
        description="Apply remeshing for cleaner topology",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_post_processing_enabled = bpy.props.BoolProperty(
        name="Post-Processing",
        description="Run ComfyUI-side mesh post-processing (decimation + remeshing). "
                    "Disable to import the raw mesh for manual retopology",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_auto_lighting = bpy.props.BoolProperty(
        name="Auto Studio Lighting",
        description="Create a three-point studio lighting rig (key, fill, rim) after import "
                    "to showcase PBR materials",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_skip_texture = bpy.props.BoolProperty(
        name="Skip Texture",
        description="Export shape-only mesh (no PBR textures). Much faster and uses less VRAM",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_low_vram = bpy.props.BoolProperty(
        name="Low VRAM BG Removal",
        description="Use low VRAM mode for background removal (BiRefNet)",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_bg_removal = bpy.props.EnumProperty(
        name="Background Removal",
        description="Method for removing the image background before TRELLIS.2 processing",
        items=[
            ('auto', 'Auto (BiRefNet)', 'Automatically remove background using BiRefNet model'),
            ('skip', 'Skip (Use Alpha)', 'Skip background removal — use the input image\'s alpha channel as mask. '
             'Use this when your image already has a transparent background'),
        ],
        default='auto',
        update=update_parameters
    )
    bpy.types.Scene.trellis2_background_color = bpy.props.EnumProperty(
        name="Background Color",
        description="Background color for image conditioning",
        items=[
            ('black', 'Black', 'Black background (default)'),
            ('gray', 'Gray', 'Gray background'),
            ('white', 'White', 'White background'),
        ],
        default='black',
        update=update_parameters
    )
    bpy.types.Scene.trellis2_fill_holes = bpy.props.BoolProperty(
        name="Fill Holes",
        description="Fill holes in the mesh during simplification (shape-only mode)",
        default=True,
        update=update_parameters
    )

    # ── PBR Decomposition (Marigold IID) ──────────────────────────────
    bpy.types.Scene.pbr_decomposition = bpy.props.BoolProperty(
        name="PBR Decomposition",
        description="Run Marigold decomposition on each generated image to "
                    "produce PBR material maps (albedo, roughness, metallic, normal, depth)",
        default=False,
        update=update_parameters
    )

    # ── Albedo source ─────────────────────────────────────────────────
    bpy.types.Scene.pbr_albedo_source = bpy.props.EnumProperty(
        name="Albedo Source",
        description="Model to use for extracting the Base Color / albedo map",
        items=[
            ('marigold', "Marigold IID (Flat Albedo)",
             "True albedo from Marigold IID-Appearance — removes all lighting "
             "but may lose texture detail"),
            ('delight', "StableDelight (Delighted)",
             "Specular-removed image via StableDelight — preserves diffuse "
             "shading and texture detail, only strips highlights"),
        ],
        default='marigold',
        update=update_parameters,
    )

    # ── Per-map enable toggles ────────────────────────────────────────
    bpy.types.Scene.pbr_map_albedo = bpy.props.BoolProperty(
        name="Albedo",
        description="Extract albedo (diffuse colour without lighting) and use it as Base Color",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.pbr_map_roughness = bpy.props.BoolProperty(
        name="Roughness",
        description="Extract roughness map (0 = mirror, 1 = rough)",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.pbr_map_metallic = bpy.props.BoolProperty(
        name="Metallic",
        description="Extract metallic map (0 = dielectric, 1 = metal)",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.pbr_map_normal = bpy.props.BoolProperty(
        name="Normal",
        description="Extract surface normal map for detail and bump. "
                    "Warning: may cause triangle artifacts on voxel-remeshed geometry",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.pbr_normal_mode = bpy.props.EnumProperty(
        name="Normal Mode",
        description="How to apply the predicted normal map to the mesh",
        items=[
            ('world', "World Space",
             "Camera-space normals are converted to world space. "
             "Works on any geometry (no tangent frame needed). "
             "Best general-purpose mode"),
            ('bump', "Bump from Normal",
             "Convert the normal map to a bump/height map. "
             "Works on any geometry including voxel remesh, "
             "but loses directional detail"),
            ('tangent', "Tangent Space",
             "Standard tangent-space normal map. "
             "Best quality on properly UV-unwrapped meshes, "
             "may show triangle artifacts on voxel-remeshed geometry"),
        ],
        default='world',
        update=update_parameters,
    )
    bpy.types.Scene.pbr_normal_strength = bpy.props.FloatProperty(
        name="Normal Strength",
        description="How strongly the normal map perturbs the surface shading. "
                    "Lower values reduce triangle artifacts on poor topology",
        default=1.0,
        min=0.0,
        max=2.0,
        step=0.05,
        update=update_parameters
    )
    bpy.types.Scene.pbr_delight_strength = bpy.props.FloatProperty(
        name="Delight Strength",
        description="How strongly StableDelight removes specular reflections. "
                    "Lower values preserve more original texture detail. "
                    "1.0 = full delighting, 0.5 = subtle",
        default=1.0,
        min=0.01,
        max=5.0,
        step=0.1,
        update=update_parameters
    )
    bpy.types.Scene.pbr_map_depth = bpy.props.BoolProperty(
        name="Depth / Height",
        description="Extract depth map for displacement/height",
        default=False,
        update=update_parameters
    )
    bpy.types.Scene.pbr_use_native_resolution = bpy.props.BoolProperty(
        name="Use Native Resolution",
        description="Process PBR maps at the image's native resolution (longest "
                    "edge, rounded to 64px). Produces sharper results but uses "
                    "more VRAM. When disabled, the fixed Processing Resolution is used",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.pbr_tiling = bpy.props.EnumProperty(
        name="Tiling",
        description="Tile-based super-resolution for PBR maps.  Each tile "
                    "is upscaled to the full image resolution before "
                    "processing, producing N\u00b2\u00d7 the effective detail",
        items=[
            ('off',       "Off",       "No tiling — process the full image in one pass"),
            ('selective', "Selective", "Tile albedo only (StableDelight and/or "
                                       "IID albedo).  Normals, roughness, metallic "
                                       "and depth are processed normally"),
            ('all',       "All",       "Tile every PBR model including normals and depth"),
        ],
        default='selective',
        update=update_parameters
    )
    bpy.types.Scene.pbr_tile_grid = bpy.props.IntProperty(
        name="Tile Grid",
        description="N\u00d7N grid size for tiling.  2 = 4 tiles (4\u00d7 detail), "
                    "3 = 9 tiles (9\u00d7 detail), 4 = 16 tiles (16\u00d7 detail).  "
                    "Processing time scales with N\u00b2",
        default=2,
        min=2,
        max=4,
        update=update_parameters
    )
    bpy.types.Scene.pbr_tile_superres = bpy.props.BoolProperty(
        name="Super Resolution",
        description="When enabled, the stitched PBR maps are kept at the "
                    "upscaled tile resolution (~N\u00d7 the original image size).  "
                    "When disabled (default), tiles are scaled back to the "
                    "original image resolution — still higher detail from "
                    "tiled processing, but matching the source texture size",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.pbr_processing_resolution = bpy.props.IntProperty(
        name="Processing Resolution",
        description="Internal processing resolution for the Marigold model. "
                    "Output is upscaled back to the original resolution. "
                    "768 is the default for the model's training resolution",
        default=768,
        min=256,
        max=2048,
        step=64,
        update=update_parameters
    )
    bpy.types.Scene.pbr_denoise_steps = bpy.props.IntProperty(
        name="Denoise Steps",
        description="Number of denoising steps. "
                    "More steps = better quality but slower. 4 is a good balance",
        default=4,
        min=1,
        max=50,
        update=update_parameters
    )
    bpy.types.Scene.pbr_ensemble_size = bpy.props.IntProperty(
        name="Ensemble Size",
        description="Number of ensemble predictions to average. "
                    "Higher = more stable results but linearly slower. "
                    "1 is usually sufficient",
        default=1,
        min=1,
        max=10,
        update=update_parameters
    )
    bpy.types.Scene.pbr_replace_color_with_albedo = bpy.props.BoolProperty(
        name="Use Albedo as Base Color",
        description="Replace the projected colour texture with the albedo map. "
                    "This effectively delights the texture",
        default=True,
        update=update_parameters
    )
    bpy.types.Scene.pbr_auto_lighting = bpy.props.BoolProperty(
        name="Studio Lighting",
        description="Create a three-point studio lighting rig (key, fill, rim) "
                    "after PBR projection to showcase PBR materials",
        default=False,
        update=update_parameters
    )

    # Artifact Filtering
    bpy.types.Scene.trellis2_artifact_laplacian_sigma = bpy.props.FloatProperty(
        name="Laplacian Sigma",
        description="Laplacian displacement outlier threshold in standard deviations. "
                    "Vertices displaced more than this many σ from the mesh-wide mean "
                    "are flagged as spikes. Lower = stricter filtering",
        default=8.0,
        min=1.0,
        max=30.0,
        step=10,
        precision=1,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_artifact_spike_abs_max = bpy.props.IntProperty(
        name="Max Spike Count",
        description="Maximum number of Laplacian-outlier vertices allowed before "
                    "the mesh is considered corrupt and triggers a retry",
        default=25,
        min=1,
        max=500,
        update=update_parameters
    )
    bpy.types.Scene.trellis2_artifact_max_retries = bpy.props.IntProperty(
        name="Max Retries",
        description="Maximum number of retry attempts when artifact filtering "
                    "detects a corrupt mesh (each retry clears Triton cache and reboots ComfyUI)",
        default=1,
        min=0,
        max=10,
        update=update_parameters
    )

def unregister():   
    """     
    Unregisters the addon.         
    :return: None     
    """
    # Ensure properties added to preferences are deleted
    if hasattr(bpy.types.Scene, 'controlnet_model_mappings'): # Check if added to Scene mistakenly
         del bpy.types.Scene.controlnet_model_mappings
    if hasattr(bpy.types.Scene, 'controlnet_mapping_index'):
         del bpy.types.Scene.controlnet_mapping_index

    del bpy.types.Scene.use_flux_lora
    del bpy.types.Scene.comfyui_prompt
    del bpy.types.Scene.comfyui_negative_prompt
    del bpy.types.Scene.model_name
    del bpy.types.Scene.seed
    del bpy.types.Scene.control_after_generate
    del bpy.types.Scene.steps
    del bpy.types.Scene.cfg
    del bpy.types.Scene.sampler
    del bpy.types.Scene.scheduler
    del bpy.types.Scene.show_advanced_params
    del bpy.types.Scene.show_generation_params
    del bpy.types.Scene.auto_rescale
    del bpy.types.Scene.qwen_rescale_alignment
    del bpy.types.Scene.auto_rescale_target_mp
    del bpy.types.Scene.generation_method
    del bpy.types.Scene.use_ipadapter
    del bpy.types.Scene.refine_images
    del bpy.types.Scene.refine_steps
    del bpy.types.Scene.refine_sampler
    del bpy.types.Scene.refine_scheduler
    del bpy.types.Scene.denoise
    del bpy.types.Scene.refine_cfg
    del bpy.types.Scene.refine_prompt
    del bpy.types.Scene.refine_upscale_method
    del bpy.types.Scene.generation_status
    del bpy.types.Scene.generation_progress
    del bpy.types.Scene.overwrite_material
    del bpy.types.Scene.discard_factor
    del bpy.types.Scene.discard_factor_generation_only
    del bpy.types.Scene.discard_factor_after_generation
    del bpy.types.Scene.weight_exponent
    del bpy.types.Scene.weight_exponent_generation_only
    del bpy.types.Scene.weight_exponent_after_generation
    del bpy.types.Scene.allow_modify_existing_textures
    del bpy.types.Scene.ask_object_prompts
    del bpy.types.Scene.fallback_color
    del bpy.types.Scene.controlnet_units
    del bpy.types.Scene.controlnet_units_index
    del bpy.types.Scene.lora_units
    del bpy.types.Scene.lora_units_index
    del bpy.types.Scene.weight_exponent_mask
    del bpy.types.Scene.sequential_smooth
    del bpy.types.Scene.canny_threshold_low
    del bpy.types.Scene.canny_threshold_high
    del bpy.types.Scene.sequential_factor_smooth
    del bpy.types.Scene.sequential_factor_smooth_2
    del bpy.types.Scene.sequential_factor
    del bpy.types.Scene.grow_mask_by
    del bpy.types.Scene.mask_blocky
    del bpy.types.Scene.visibility_vignette
    del bpy.types.Scene.visibility_vignette_width
    del bpy.types.Scene.visibility_vignette_softness
    del bpy.types.Scene.refine_edge_feather_projection
    del bpy.types.Scene.refine_edge_feather_width
    del bpy.types.Scene.refine_edge_feather_softness
    del bpy.types.Scene.differential_diffusion
    del bpy.types.Scene.differential_noise
    del bpy.types.Scene.blur_mask
    del bpy.types.Scene.blur_mask_radius
    del bpy.types.Scene.blur_mask_sigma
    del bpy.types.Scene.sequential_custom_camera_order
    del bpy.types.Scene.ipadapter_strength
    del bpy.types.Scene.ipadapter_start
    del bpy.types.Scene.ipadapter_end
    del bpy.types.Scene.sequential_ipadapter
    del bpy.types.Scene.sequential_ipadapter_mode
    del bpy.types.Scene.sequential_desaturate_factor
    del bpy.types.Scene.sequential_contrast_factor
    del bpy.types.Scene.sequential_ipadapter_regenerate
    del bpy.types.Scene.ipadapter_weight_type
    del bpy.types.Scene.clip_skip
    del bpy.types.Scene.stablegen_preset
    del bpy.types.Scene.model_architecture
    del bpy.types.Scene.architecture_mode
    del bpy.types.Scene.output_timestamp
    del bpy.types.Scene.camera_prompts
    del bpy.types.Scene.use_camera_prompts
    del bpy.types.Scene.sg_camera_order
    del bpy.types.Scene.sg_camera_order_index
    del bpy.types.Scene.sg_use_custom_camera_order
    del bpy.types.Scene.show_core_settings
    del bpy.types.Scene.show_lora_settings
    del bpy.types.Scene.show_camera_options
    del bpy.types.Scene.show_scene_understanding_settings
    del bpy.types.Scene.show_output_material_settings
    del bpy.types.Scene.show_image_guidance_settings
    del bpy.types.Scene.show_masking_inpainting_settings
    del bpy.types.Scene.show_mode_specific_settings
    del bpy.types.Scene.generation_mode
    del bpy.types.Scene.early_priority_strength
    del bpy.types.Scene.early_priority
    del bpy.types.Scene.texture_objects
    del bpy.types.Scene.qwen_guidance_map_type
    del bpy.types.Scene.qwen_voronoi_mode
    del bpy.types.Scene.qwen_context_render_mode
    del bpy.types.Scene.qwen_use_external_style_image
    del bpy.types.Scene.qwen_external_style_image
    del bpy.types.Scene.qwen_external_style_initial_only
    del bpy.types.Scene.qwen_use_custom_prompts
    del bpy.types.Scene.qwen_custom_prompt_initial
    del bpy.types.Scene.qwen_custom_prompt_seq_none
    del bpy.types.Scene.qwen_custom_prompt_seq_replace
    del bpy.types.Scene.qwen_custom_prompt_seq_additional
    del bpy.types.Scene.qwen_guidance_fallback_color
    del bpy.types.Scene.qwen_guidance_background_color
    del bpy.types.Scene.qwen_context_cleanup
    del bpy.types.Scene.qwen_context_cleanup_hue_tolerance
    del bpy.types.Scene.qwen_context_cleanup_value_adjust
    del bpy.types.Scene.qwen_context_fallback_dilation
    del bpy.types.Scene.qwen_timestep_zero_ref

    # --- PBR Decomposition Properties ---
    pbr_props = [
        'pbr_decomposition', 'pbr_albedo_source',
        'pbr_map_albedo', 'pbr_map_roughness', 'pbr_map_metallic',
        'pbr_map_normal', 'pbr_map_depth',
        'pbr_normal_mode', 'pbr_normal_strength',
        'pbr_delight_strength',
        'pbr_use_native_resolution', 'pbr_tiling', 'pbr_tile_grid', 'pbr_tile_superres',
        'pbr_processing_resolution', 'pbr_denoise_steps', 'pbr_ensemble_size',
        'pbr_replace_color_with_albedo', 'pbr_auto_lighting',
        'pbr_model_variant',  # legacy, kept for compat
    ]
    for prop in pbr_props:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    # --- TRELLIS.2 Properties ---
    trellis2_props = [
        'trellis2_available', 'show_trellis2_params', 'show_trellis2_advanced',
        'show_trellis2_mesh_settings', 'show_trellis2_texture_settings',
        'show_trellis2_camera_settings',
        'trellis2_last_input_image', 'qwen_use_trellis2_style', 'qwen_trellis2_style_initial_only',
        'trellis2_pipeline_active', 'trellis2_pipeline_phase_start_pct',
        'trellis2_pipeline_total_phases',
        'trellis2_generate_from', 'trellis2_texture_mode', 'trellis2_initial_image_arch',
        'trellis2_camera_count',
        'trellis2_placement_mode', 'trellis2_auto_prompts',
        'trellis2_exclude_bottom', 'trellis2_exclude_bottom_angle',
        'trellis2_auto_aspect', 'trellis2_occlusion_mode',
        'trellis2_consider_existing', 'trellis2_delete_cameras',
        'trellis2_coverage_target',
        'trellis2_max_auto_cameras', 'trellis2_fan_angle',
        'trellis2_import_scale',
        'trellis2_clamp_elevation', 'trellis2_max_elevation', 'trellis2_min_elevation',
        'trellis2_preview_gallery_enabled', 'trellis2_preview_gallery_count',
        'trellis2_input_image', 'trellis2_resolution', 'trellis2_vram_mode',
        'trellis2_attn_backend', 'trellis2_seed', 'trellis2_ss_guidance',
        'trellis2_ss_steps', 'trellis2_shape_guidance', 'trellis2_shape_steps',
        'trellis2_tex_guidance', 'trellis2_tex_steps', 'trellis2_max_tokens',
        'trellis2_texture_size', 'trellis2_decimation', 'trellis2_remesh',
        'trellis2_post_processing_enabled',
        'trellis2_auto_lighting',
        'trellis2_skip_texture', 'trellis2_low_vram', 'trellis2_bg_removal', 'trellis2_background_color',
        'trellis2_fill_holes',
        'trellis2_artifact_laplacian_sigma', 'trellis2_artifact_spike_abs_max',
        'trellis2_artifact_max_retries', 'show_trellis2_artifact_filter',
    ]
    for prop in trellis2_props:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
        
    # Remove the load handler for default controlnet unit
    if load_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(load_handler)
   

if __name__ == "__main__":
    register()
