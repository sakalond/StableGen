"""Architecture-mode and server-address callbacks.

Property ``update=`` callbacks that react to architecture / mode changes
and the server address preference.  Extracted from the original
``__init__.py`` to keep it slim.
"""

import bpy  # pylint: disable=import-error
from urllib.parse import urlparse

from ..ui.presets import update_parameters
from . import ADDON_PKG
from .state import (
    _cached_checkpoint_list,
    _cached_lora_list,
    _run_async,
)
from .server_api import (
    check_pbr_available,
    check_server_availability,
    check_skintokens_autorig_available,
    check_trellis2_available,
)
from ..timeout_config import get_timeout


# ── Architecture-mode helpers ──────────────────────────────────────────────

def update_architecture_mode(self, context):
    """Called when the user changes the architecture_mode dropdown."""
    scene = context.scene
    mode = scene.architecture_mode

    if mode != 'trellis2':
        if scene.model_architecture != mode:
            scene.model_architecture = mode          # triggers update_combined
    else:
        _sync_trellis2_backbone(scene)

    update_parameters(self, context)


def _sync_trellis2_backbone(scene):
    """Pick the right diffusion backbone for the current TRELLIS.2 state."""
    tex_mode = getattr(scene, 'trellis2_texture_mode', 'native')
    if tex_mode in ('sdxl', 'flux1', 'qwen_image_edit', 'flux2_klein'):
        target = tex_mode
    elif getattr(scene, 'trellis2_generate_from', 'image') == 'prompt':
        target = getattr(scene, 'trellis2_initial_image_arch', 'sdxl')
    else:
        return
    if scene.model_architecture != target:
        scene.model_architecture = target


def update_trellis2_texture_mode(self, context):
    """Called when the user changes the texture generation mode inside TRELLIS.2."""
    scene = context.scene
    if getattr(scene, 'architecture_mode', '') != 'trellis2':
        return

    tex_mode = scene.trellis2_texture_mode
    scene.trellis2_skip_texture = (tex_mode != 'native')

    _sync_trellis2_backbone(scene)
    update_parameters(self, context)


def update_trellis2_initial_image_arch(self, context):
    """Called when the user changes the initial-image architecture for TRELLIS.2."""
    scene = context.scene
    if getattr(scene, 'architecture_mode', '') != 'trellis2':
        return
    _sync_trellis2_backbone(scene)
    update_parameters(self, context)


def update_trellis2_generate_from(self, context):
    """Called when the user switches between Image and Prompt input mode."""
    scene = context.scene
    if getattr(scene, 'architecture_mode', '') != 'trellis2':
        return

    if scene.trellis2_generate_from == 'prompt':
        _sync_trellis2_backbone(scene)

    update_parameters(self, context)


# ── Server-address change callback ────────────────────────────────────────

def update_combined(self, context):
    """Master callback for server_address changes — pings server, refreshes models."""
    # Import lazily to avoid circular ref at module level
    from .load_handlers import load_handler

    prefs = context.preferences.addons[ADDON_PKG].preferences
    raw_address = prefs.server_address

    if raw_address:
        if not raw_address.startswith(('http://', 'https://')):
            parsed_url = urlparse(f"http://{raw_address}")
        else:
            parsed_url = urlparse(raw_address)

        clean_address = parsed_url.netloc

        if clean_address and raw_address != clean_address:
            prefs.server_address = clean_address
            return None

    server_address = prefs.server_address

    if not server_address:
        prefs.server_online = False
        # Must mutate the lists in-place to keep existing references valid
        import stablegen.core.state as _state
        _state._cached_checkpoint_list = [("NO_SERVER", "Set Server Address", "...")]
        _state._cached_lora_list = [("NO_SERVER", "Set Server Address", "...")]
        return None

    print("[StableGen] Server address changed, checking asynchronously...")

    def _bg_work():
        result = {}
        result['online'] = check_server_availability(server_address, timeout=get_timeout('ping'))
        if result['online']:
            result['trellis2'] = check_trellis2_available(server_address, timeout=get_timeout('api'))
            result['pbr'] = check_pbr_available(server_address, timeout=get_timeout('api'))
            result['skintokens'] = check_skintokens_autorig_available(server_address, timeout=get_timeout('api'))
        else:
            result['trellis2'] = False
            result['pbr'] = False
            result['skintokens'] = False
        return result

    def _on_done(result):
        if result is None:
            return
        _prefs = bpy.context.preferences.addons[ADDON_PKG].preferences
        _prefs.server_online = result.get('online', False)

        if hasattr(bpy.context, 'scene') and bpy.context.scene:
            bpy.context.scene.trellis2_available = result.get('trellis2', False)
            bpy.context.scene.pbr_nodes_available = result.get('pbr', False)
            bpy.context.scene.skintokens_autorig_available = result.get('skintokens', False)

        if not result.get('online', False):
            print("[StableGen] ComfyUI server is not reachable.")
            return

        update_parameters(None, bpy.context)
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

    _run_async(_bg_work, _on_done, track_generation=True)

    update_parameters(self, context)
    load_handler(None)

    return None


def update_print_preset(self, context):
    from . import state as _state
    if _state._updating_print_preset:
        return
    
    scene = context.scene
    preset = scene.stablegen_print_preset
    if preset == 'CUSTOM':
        return
        
    _state._updating_print_preset = True
    try:
        scene.stablegen_print_palette.clear()
        
        cyan = (0.0, 1.0, 1.0)
        magenta = (1.0, 0.0, 1.0)
        yellow = (1.0, 1.0, 0.0)
        white = (1.0, 1.0, 1.0)
        black = (0.0, 0.0, 0.0)
        
        if preset == 'CMYW_DITHERED':
            scene.stablegen_print_dithered = True
            colors = [("Cyan", cyan), ("Magenta", magenta), ("Yellow", yellow), ("White", white)]
        elif preset == 'CMYK_DITHERED':
            scene.stablegen_print_dithered = True
            colors = [("Cyan", cyan), ("Magenta", magenta), ("Yellow", yellow), ("Black", black)]
        elif preset in ('CMYW_MIXES_SOLID', 'CMYK_MIXES_SOLID'):
            scene.stablegen_print_dithered = False
            fourth_color = white if preset == 'CMYW_MIXES_SOLID' else black
            fourth_name = "White" if preset == 'CMYW_MIXES_SOLID' else "Black"
            
            base_colors = [
                ("Cyan", cyan),
                ("Magenta", magenta),
                ("Yellow", yellow),
                (fourth_name, fourth_color)
            ]
            
            colors = list(base_colors)
            n_base = len(base_colors)
            for idx1 in range(n_base):
                for idx2 in range(idx1 + 1, n_base):
                    name1, col1 = base_colors[idx1]
                    name2, col2 = base_colors[idx2]
                    
                    def mix(c1, c2, w1, w2):
                        tot = w1 + w2
                        return (
                            (c1[0]*w1 + c2[0]*w2) / tot,
                            (c1[1]*w1 + c2[1]*w2) / tot,
                            (c1[2]*w1 + c2[2]*w2) / tot
                        )
                    
                    colors.append((f"{name1}:{name2} 1:1", mix(col1, col2, 1.0, 1.0)))
                    colors.append((f"{name1}:{name2} 2:1", mix(col1, col2, 2.0, 1.0)))
                    colors.append((f"{name1}:{name2} 1:2", mix(col1, col2, 1.0, 2.0)))
                    
        for name, col in colors:
            item = scene.stablegen_print_palette.add()
            item.name = name
            item.color = col
            
    finally:
        _state._updating_print_preset = False


def check_print_preset_match(scene):
    from . import state as _state
    if _state._updating_print_preset:
        return
        
    preset = scene.stablegen_print_preset
    if preset == 'CUSTOM':
        return
        
    cyan = (0.0, 1.0, 1.0)
    magenta = (1.0, 0.0, 1.0)
    yellow = (1.0, 1.0, 0.0)
    white = (1.0, 1.0, 1.0)
    black = (0.0, 0.0, 0.0)
    
    import math
    def is_close(c1, c2):
        return all(math.isclose(x, y, abs_tol=1e-4) for x, y in zip(c1, c2))
        
    palette = scene.stablegen_print_palette
    
    if preset == 'CMYW_DITHERED':
        if not scene.stablegen_print_dithered or len(palette) != 4:
            scene.stablegen_print_preset = 'CUSTOM'
            return
        expected = [cyan, magenta, yellow, white]
        for item, exp in zip(palette, expected):
            if not is_close(item.color, exp):
                scene.stablegen_print_preset = 'CUSTOM'
                return
                
    elif preset == 'CMYK_DITHERED':
        if not scene.stablegen_print_dithered or len(palette) != 4:
            scene.stablegen_print_preset = 'CUSTOM'
            return
        expected = [cyan, magenta, yellow, black]
        for item, exp in zip(palette, expected):
            if not is_close(item.color, exp):
                scene.stablegen_print_preset = 'CUSTOM'
                return
                
    elif preset in ('CMYW_MIXES_SOLID', 'CMYK_MIXES_SOLID'):
        if scene.stablegen_print_dithered or len(palette) != 22:
            scene.stablegen_print_preset = 'CUSTOM'
            return
        fourth_color = white if preset == 'CMYW_MIXES_SOLID' else black
        base_colors = [cyan, magenta, yellow, fourth_color]
        
        expected = list(base_colors)
        n_base = len(base_colors)
        for idx1 in range(n_base):
            for idx2 in range(idx1 + 1, n_base):
                col1 = base_colors[idx1]
                col2 = base_colors[idx2]
                def mix(c1, c2, w1, w2):
                    tot = w1 + w2
                    return (
                        (c1[0]*w1 + c2[0]*w2) / tot,
                        (c1[1]*w1 + c2[1]*w2) / tot,
                        (c1[2]*w1 + c2[2]*w2) / tot
                    )
                expected.append(mix(col1, col2, 1.0, 1.0))
                expected.append(mix(col1, col2, 2.0, 1.0))
                expected.append(mix(col1, col2, 1.0, 2.0))
                
        for item, exp in zip(palette, expected):
            if not is_close(item.color, exp):
                scene.stablegen_print_preset = 'CUSTOM'
                return


def update_palette_color(self, context):
    check_print_preset_match(self.id_data)


def update_print_dithered(self, context):
    check_print_preset_match(self.id_data)
