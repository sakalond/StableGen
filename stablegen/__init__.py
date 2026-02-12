""" This script registers the addon. """
import bpy # pylint: disable=import-error
from .stablegen import StableGenPanel, ApplyPreset, SavePreset, DeletePreset, get_preset_items, update_parameters, ResetQwenPrompt
from .render_tools import BakeTextures, AddCameras, SwitchMaterial, ExportOrbitGIF, CollectCameraPrompts, CameraPromptItem 
from .utils import AddHDRI, ApplyModifiers, CurvesToMesh
from .generator import ComfyUIGenerate, Reproject, Regenerate, MirrorReproject
import os
import requests
import json
from bpy.app.handlers import persistent
from urllib.parse import urlparse

bl_info = {
    "name": "StableGen",
    "category": "Object",
    "author": "Ondrej Sakala",
    "version": (0, 1, 1),
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
    SwitchMaterial,
    ExportOrbitGIF,
    CollectCameraPrompts,
    CameraPromptItem,
    AddHDRI,
    ApplyModifiers,
    CurvesToMesh,
    ComfyUIGenerate,
    Reproject,
    Regenerate,
    MirrorReproject,
]

# Global caches for model lists fetched via API
_cached_checkpoint_list = [("NONE_AVAILABLE", "None available", "Fetch models from server")]
_cached_lora_list = [("NONE_AVAILABLE", "None available", "Fetch models from server")]
_cached_checkpoint_architecture = None
_pending_checkpoint_refresh_architecture = None

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

    # Check if server is reachable
    if not check_server_availability(context.preferences.addons[__package__].preferences.server_address, timeout=0.5):
        context.preferences.addons[__package__].preferences.server_online = False
        print("ComfyUI server is not reachable.")
        return None
    else:
        context.preferences.addons[__package__].preferences.server_online = True

    update_parameters(self, context)
    load_handler(None)

    # Automatically Refresh Lists on Server Change
    # Check if server address is valid before trying to refresh
    if context.preferences.addons[__package__].preferences.server_address:
         print("Server address changed, attempting to refresh model lists...")
         def deferred_refresh():
             try:
                  bpy.ops.stablegen.refresh_checkpoint_list('INVOKE_DEFAULT')
                  bpy.ops.stablegen.refresh_lora_list('INVOKE_DEFAULT')
                  bpy.ops.stablegen.refresh_controlnet_mappings('INVOKE_DEFAULT') # Refresh CN too
             except Exception as e:
                  print(f"Error during deferred refresh: {e}")
             return None # Timer runs only once
         bpy.app.timers.register(deferred_refresh, first_interval=0.1)
    else:
         print("Server address cleared, cannot refresh lists.")
         global _cached_checkpoint_list, _cached_lora_list
         _cached_checkpoint_list = [("NO_SERVER", "Set Server Address", "...")]
         _cached_lora_list = [("NO_SERVER", "Set Server Address", "...")]

    # Checkpoint model reset
    current_checkpoint = context.scene.model_name
    # update_model_list is now the API version
    checkpoint_items = update_model_list(self, context)
    valid_checkpoint_ids = {item[0] for item in checkpoint_items}
    placeholder_id = next((item[0] for item in checkpoint_items if item[0].startswith("NO_") or item[0] == "NONE_FOUND"), None)

    if current_checkpoint not in valid_checkpoint_ids:
        if placeholder_id:
            context.scene.model_name = placeholder_id
        elif checkpoint_items: # If no placeholder but other items, pick first valid one
            context.scene.model_name = checkpoint_items[0][0]
        # else: # No models found at all, leave it potentially invalid or set to a default if possible

    # LoRA unit reset (uses API now)
    if hasattr(context.scene, 'lora_units'):
        lora_items = get_lora_models(self, context) # API version
        valid_lora_ids = {item[0] for item in lora_items}
        placeholder_lora_id = next((item[0] for item in lora_items if item[0].startswith("NO_") or item[0] == "NONE_FOUND"), None)

        # Iterate safely while removing
        indices_to_remove = []
        for i, lora_unit in enumerate(context.scene.lora_units):
            if lora_unit.model_name not in valid_lora_ids or lora_unit.model_name == placeholder_lora_id:
                indices_to_remove.append(i)

        # Remove invalid units in reverse order to maintain indices
        for i in sorted(indices_to_remove, reverse=True):
             context.scene.lora_units.remove(i)

    # Check/reset current LoRA unit index
    num_loras = len(context.scene.lora_units)
    if context.scene.lora_units_index >= num_loras:
         context.scene.lora_units_index = max(0, num_loras - 1)
    elif num_loras == 0:
         context.scene.lora_units_index = 0 # Or -1 if appropriate, ensure index is valid

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
        # Use the existing check function with a short timeout
        is_online = check_server_availability(server_addr, timeout=0.75) # Use slightly longer than 0.5?

        prefs.server_online = is_online # Update the preference property

        if is_online:
            self.report({'INFO'}, f"ComfyUI server is online at {server_addr}.")
            # Trigger full refresh here if desired on manual check success
            bpy.ops.stablegen.refresh_checkpoint_list('INVOKE_DEFAULT')
            bpy.ops.stablegen.refresh_lora_list('INVOKE_DEFAULT')
            bpy.ops.stablegen.refresh_controlnet_mappings('INVOKE_DEFAULT')
            load_handler(None)
        else:
            self.report({'ERROR'}, f"ComfyUI server unreachable or timed out at {server_addr}.")

        # The update function on the property handles UI redraw

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
        prefs = context.preferences.addons[__package__].preferences
        server_models = fetch_from_comfyui_api(context, "/models/controlnet")

        if server_models is None: # Indicates connection/config error
             self.report({'ERROR'}, "Could not fetch models. Check server address and ensure ComfyUI is running.")
             return {'CANCELLED'}

        if not server_models:
             self.report({'WARNING'}, "No ControlNet models found on the server.")
             # Clear existing list if server returns empty
             prefs.controlnet_model_mappings.clear()
             return {'FINISHED'}

        # Synchronization Logic
        current_mappings = {item.name: item for item in prefs.controlnet_model_mappings}
        server_model_set = set(server_models)
        current_model_set = set(current_mappings.keys())

        # 1. Remove models from prefs that are no longer on the server
        models_to_remove = current_model_set - server_model_set
        indices_to_remove = []
        for i, item in enumerate(prefs.controlnet_model_mappings):
            if item.name in models_to_remove:
                indices_to_remove.append(i)

        # Remove in reverse order to avoid index issues
        for i in sorted(indices_to_remove, reverse=True):
             prefs.controlnet_model_mappings.remove(i)
             # Adjust index if necessary
             if prefs.controlnet_mapping_index >= len(prefs.controlnet_model_mappings):
                  prefs.controlnet_mapping_index = max(0, len(prefs.controlnet_model_mappings) - 1)


        # 2. Add new models found on the server
        models_to_add = server_model_set - current_model_set
        for model_name in sorted(list(models_to_add)):
            new_item = prefs.controlnet_model_mappings.add()
            new_item.name = model_name

            # Guessing Logic
            name_lower = model_name.lower()
            is_union_guess = 'union' in name_lower or 'promax' in name_lower

            # Guess based on keywords, prioritizing union
            if is_union_guess:
                new_item.supports_depth = True
                new_item.supports_canny = True
                new_item.supports_normal = True # Assume union supports all current types
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

            # If no specific type keyword found (and not union), leave all as False
            if not is_union_guess and not new_item.supports_depth and not new_item.supports_canny and not new_item.supports_normal:
                 print(f"  Could not guess type for '{model_name}'. Please assign manually.")

        self.report({'INFO'}, f"Refreshed ControlNet list: {len(models_to_add)} added, {len(models_to_remove)} removed.")
        return {'FINISHED'}
    
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
    
    if not check_server_availability(server_address, timeout=0.5): # Use a strict timeout here
         # Error message printed by check_server_availability
         return None # Server unreachable or timed out on initial check
    
    # Ensure endpoint starts with a slash
    if not endpoint.startswith('/'):
        endpoint = '/' + endpoint

    url = f"http://{server_address}{endpoint}"

    try:
        response = requests.get(url, timeout=5) # Add a timeout
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
        global _cached_checkpoint_list, _cached_checkpoint_architecture
        items = []
        model_list = None # Initialize to None

        architecture = getattr(context.scene, "model_architecture", "sdxl")

        # Determine endpoint based on current architecture setting
        if architecture == 'sdxl':
            model_list = fetch_from_comfyui_api(context, "/models/checkpoints")
            model_type_desc = "Checkpoint"
        elif architecture == 'flux1':
            model_list = fetch_from_comfyui_api(context, "/models/unet_gguf")
            if model_list is not None:
                to_extend = fetch_from_comfyui_api(context, "/models/diffusion_models")
                if to_extend:
                    model_list.extend(to_extend)
            model_type_desc = "UNET"
        elif architecture == 'qwen_image_edit':
            model_list = fetch_from_comfyui_api(context, "/models/unet_gguf")
            if model_list is not None:
                to_extend = fetch_from_comfyui_api(context, "/models/diffusion_models")
                if to_extend:
                    model_list.extend(to_extend)
            model_type_desc = "UNET (GGUF/Safetensors)"

        if model_list is None: # Config error
            _cached_checkpoint_list = [("NO_SERVER", "Set Server Address", "Cannot fetch")]
            _cached_checkpoint_architecture = None
            self.report({'ERROR'}, "Cannot fetch models. Check server address.")
            # Force UI update if possible
            if context.area:
                context.area.tag_redraw()
            return {'CANCELLED'}
        elif not model_list: # API ok, but empty list
             _cached_checkpoint_list = [("NONE_FOUND", f"No {model_type_desc}s Found", "Server list is empty")]
             _cached_checkpoint_architecture = architecture
             self.report({'WARNING'}, f"No {model_type_desc} models found on server.")
        else: # Models found
            for model_name in sorted(model_list):
                items.append((model_name, model_name, f"{model_type_desc}: {model_name}"))
            _cached_checkpoint_list = items
            _cached_checkpoint_architecture = architecture
            self.report({'INFO'}, f"Refreshed {model_type_desc} list ({len(items)} found).")

        # Reset Logic after refresh
        current_checkpoint = context.scene.model_name
        valid_checkpoint_ids = {item[0] for item in _cached_checkpoint_list}
        placeholder_id = next((item[0] for item in _cached_checkpoint_list if item[0].startswith("NO_") or item[0] == "NONE_FOUND"), None)

        if current_checkpoint not in valid_checkpoint_ids:
            if placeholder_id:
                context.scene.model_name = placeholder_id
            elif _cached_checkpoint_list:
                context.scene.model_name = _cached_checkpoint_list[0][0]

        # Force UI update if possible (e.g., redraw panels)
        if context.area:
            context.area.tag_redraw()
        # A more robust redraw might be needed depending on context
        # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

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
        global _cached_lora_list
        items = []
        lora_list = fetch_from_comfyui_api(context, "/models/loras")

        if lora_list is None: # Config error
            _cached_lora_list = [("NO_SERVER", "Set Server Address", "Cannot fetch")]
            self.report({'ERROR'}, "Cannot fetch LoRAs. Check server address.")
            context.area.tag_redraw()
            return {'CANCELLED'}
        elif not lora_list: # API ok, but empty
             _cached_lora_list = [("NONE_FOUND", "No LoRAs Found", "Server list is empty")]
             self.report({'WARNING'}, "No LoRA models found on server.")
        else: # LoRAs found
            for lora_name in sorted(lora_list):
                items.append((lora_name, lora_name, f"LoRA: {lora_name}"))
            _cached_lora_list = items
            self.report({'INFO'}, f"Refreshed LoRA list ({len(items)} found).")

        # Reset Logic after refresh
        if hasattr(context.scene, 'lora_units'):
            valid_lora_ids = {item[0] for item in _cached_lora_list}
            placeholder_lora_id = next((item[0] for item in _cached_lora_list if item[0].startswith("NO_") or item[0] == "NONE_FOUND"), None)
            indices_to_remove = []
            for i, lora_unit in enumerate(context.scene.lora_units):
                if lora_unit.model_name not in valid_lora_ids or lora_unit.model_name == placeholder_lora_id:
                    indices_to_remove.append(i)
            for i in sorted(indices_to_remove, reverse=True):
                context.scene.lora_units.remove(i)

            num_loras = len(context.scene.lora_units)
            if context.scene.lora_units_index >= num_loras:
                context.scene.lora_units_index = max(0, num_loras - 1)
            elif num_loras == 0:
                context.scene.lora_units_index = 0

        if context.area:
            context.area.tag_redraw()
        # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
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
    bpy.types.Scene.sequential_ipadapter_mode = bpy.props.EnumProperty(
        name="IPAdapter Mode",
        description="Mode for IPAdapter in sequential generation",
        items=[
            ('first', 'Use first generated image', ''),
            ('recent', 'Use most recent generated image', ''),
        ],
        default='first',
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
        default=1.0,
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
        description="Exponent shaping feather falloff (<1 = softer, >1 = sharper)",
        default=0.5,
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
        default=False,
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
        default=0.6,
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
    del bpy.types.Scene.output_timestamp
    del bpy.types.Scene.camera_prompts
    del bpy.types.Scene.use_camera_prompts
    del bpy.types.Scene.show_core_settings
    del bpy.types.Scene.show_lora_settings
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

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
        
    # Remove the load handler for default controlnet unit
    if load_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(load_handler)
   

if __name__ == "__main__":
    register()
