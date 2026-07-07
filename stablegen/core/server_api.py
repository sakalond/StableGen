"""ComfyUI server communication helpers.

Network functions that check server availability, query model lists,
and fetch data from ComfyUI's REST API.  All functions are safe to call
from a background thread (they do not access ``bpy.context``).
"""

import json
import os

import requests

from ..timeout_config import get_timeout
from . import ADDON_PKG, get_addon_prefs


# ── Server availability checks ────────────────────────────────────────────

def check_server_availability(server_address, timeout=0.5):
    """Quickly checks if the ComfyUI server is responding.

    Args:
        server_address (str): The address:port of the ComfyUI server.
        timeout (float): Strict timeout in seconds for this check.

    Returns:
        bool: True if the server responds quickly, False otherwise.
    """
    if not server_address:
        return False

    url = f"http://{server_address}/queue"
    try:
        response = requests.head(url, timeout=timeout)
        response.raise_for_status()
        return True
    except requests.exceptions.Timeout:
        print(f"[StableGen]   Initial server check failed: Timeout ({timeout}s).")
        return False
    except requests.exceptions.ConnectionError:
        print("[StableGen]   Initial server check failed: Connection Error.")
        return False


def check_trellis2_available(server_address, timeout=1.0):
    """Checks if TRELLIS.2 custom nodes are installed in ComfyUI.

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
        print(f"[StableGen]   Initial server check failed: Request Error ({e}).")
        return False
    except Exception as e:
        print(f"[StableGen]   Initial server check failed: Unexpected Error ({e}).")
        return False


def check_pbr_available(server_address, timeout=1.0):
    """Checks if PBR decomposition custom nodes are installed in ComfyUI.

    Looks for MarigoldModelLoader (ComfyUI-Marigold) and
    LoadStableDelightModel (ComfyUI_StableDelight_ll).

    Args:
        server_address (str): The address:port of the ComfyUI server.
        timeout (float): Timeout in seconds.

    Returns:
        bool: True if both node packages are detected, False otherwise.
    """
    if not server_address:
        return False
    required = ['MarigoldModelLoader', 'LoadStableDelightModel']
    try:
        for node_class in required:
            url = f"http://{server_address}/object_info/{node_class}"
            response = requests.get(url, timeout=timeout)
            if response.status_code != 200 or node_class not in response.json():
                print(f"[PBR] Auto-detect: node '{node_class}' NOT found in ComfyUI.")
                return False
        print("[PBR] Auto-detect: PBR decomposition nodes found in ComfyUI.")
        return True
    except Exception as e:
        print(f"[PBR] Auto-detect failed: {e}")
        return False


def check_skintokens_autorig_available(server_address, timeout=1.0):
    """Checks if the StableGen SkinTokens auto-rig custom node is installed."""
    if not server_address:
        return False
    node_class = 'StableGenSkinTokensAutoRig'
    try:
        url = f"http://{server_address}/object_info/{node_class}"
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200 and node_class in response.json():
            print("[SkinTokens] Auto-detect: StableGen SkinTokens node found in ComfyUI.")
            return True
        print("[SkinTokens] Auto-detect: StableGen SkinTokens node NOT found in ComfyUI.")
        return False
    except Exception as e:
        print(f"[SkinTokens] Auto-detect failed: {e}")
        return False


# ── API data retrieval ─────────────────────────────────────────────────────

def fetch_from_comfyui_api(context, endpoint):
    """Fetches data from a specified ComfyUI API endpoint.

    Args:
        context: Blender context to access addon preferences.
        endpoint (str): The API endpoint path (e.g., "/models/checkpoints").

    Returns:
        list: A list of items returned by the API (usually filenames),
              or an empty list if the request fails or returns invalid data.
              Returns None if the server address is not set.
    """
    addon_prefs = context.preferences.addons.get(ADDON_PKG)
    if not addon_prefs:
        print("[StableGen] Error: Could not access StableGen addon preferences.")
        return None

    server_address = addon_prefs.preferences.server_address
    if not server_address:
        print("[StableGen] Error: ComfyUI Server Address is not set in preferences.")
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
            if all(isinstance(item, str) for item in data):
                return data
            elif data:
                 print(f"[StableGen]   Warning: API endpoint {endpoint} returned a list, but it contains non-string items: {data[:5]}...")
                 string_items = [item for item in data if isinstance(item, str)]
                 if string_items:
                      return string_items
                 else:
                      print(f"[StableGen]   Error: No valid string filenames found in list from {endpoint}.")
                      return []
            else:
                 return []
        else:
            print(f"[StableGen]   Error: API endpoint {endpoint} did not return a JSON list. Received: {type(data)}")
            return []

    except requests.exceptions.Timeout:
        print(f"[StableGen]   Error: Timeout connecting to {url}.")
    except requests.exceptions.ConnectionError:
        print(f"[StableGen]   Error: Connection failed to {url}. Is ComfyUI running and accessible?")
    except requests.exceptions.RequestException as e:
        print(f"[StableGen]   Error fetching from {url}: {e}")
    except json.JSONDecodeError:
        print(f"[StableGen]   Error: Could not decode JSON response from {url}. Response text: {response.text}")
    except Exception as e:
        print(f"[StableGen]   An unexpected error occurred fetching from {url}: {e}")

    return []


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
            print(f"[StableGen]   Error: API endpoint {endpoint} did not return a JSON list.")
            return []
    except requests.exceptions.Timeout:
        print(f"[StableGen]   Error: Timeout connecting to {url}.")
    except requests.exceptions.ConnectionError:
        print(f"[StableGen]   Error: Connection failed to {url}.")
    except requests.exceptions.RequestException as e:
        print(f"[StableGen]   Error fetching from {url}: {e}")
    except json.JSONDecodeError:
        print(f"[StableGen]   Error: Could not decode JSON from {url}.")
    except Exception as e:
        print(f"[StableGen]   Unexpected error fetching from {url}: {e}")

    return []


# ── Local model scanning helpers ───────────────────────────────────────────

def get_models_from_directory(scan_root_path: str, valid_extensions: tuple,
                              type_for_description: str, path_prefix_for_id: str = ""):
    """Scans a given root directory for model files.

    Returns paths relative to *scan_root_path*, optionally prefixed.
    """
    items = []
    if not (scan_root_path and os.path.isdir(scan_root_path)):
        return items

    try:
        for root, _, files in os.walk(scan_root_path):
            for f_name in files:
                if f_name.lower().endswith(valid_extensions):
                    full_path = os.path.join(root, f_name)
                    relative_path = os.path.relpath(full_path, scan_root_path)
                    identifier = path_prefix_for_id + relative_path
                    display_name = identifier
                    items.append((identifier, display_name, f"{type_for_description}: {display_name}"))
    except PermissionError:
        print(f"[StableGen] Permission Denied for {scan_root_path}")
    except Exception as e:
        print(f"[StableGen] Error Scanning {scan_root_path}: {e}")
    
    return items


def merge_and_deduplicate_models(model_lists: list):
    """Merges multiple lists of model items and de-duplicates by identifier."""
    merged_items = []
    seen_identifiers = set()
    for model_list in model_lists:
        for identifier, name, description in model_list:
            if identifier.startswith("NO_") or identifier.startswith("PERM_") or identifier.startswith("SCAN_") or identifier == "NONE_FOUND":
                continue
            if identifier not in seen_identifiers:
                merged_items.append((identifier, name, description))
                seen_identifiers.add(identifier)
    
    if not merged_items:
        merged_items.append(("NONE_AVAILABLE", "No Models Found", "Check ComfyUI and External Directories in Preferences"))
    
    merged_items.sort(key=lambda x: x[1])
    return merged_items
