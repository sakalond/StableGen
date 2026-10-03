"""Preference-gated architecture menus, retaining saved enum identifiers."""

DEPRECATED_ARCHITECTURES = frozenset({'flux1', 'flux2_klein'})
DEPRECATION_NOTICE = 'Deprecated: no longer receiving further support'

# Explicit numbers match the original static enums, including their ordering.
# Retain these numbers when filtering so existing .blend files keep their choice.
MODEL_ARCHITECTURE_ITEMS = (
    ('sdxl', 'SDXL', 'Stable Diffusion XL', 0),
    ('flux1', 'FLUX.1 (Deprecated)', DEPRECATION_NOTICE, 1),
    ('qwen_image_edit', 'Qwen Image Edit', 'Qwen-Image-Edit checkpoints', 2),
    ('qwen_image_21', 'Qwen Image 2.1', 'Qwen-Image 2.1 checkpoints', 3),
    ('flux2_klein', 'FLUX.2 Klein (Deprecated)', DEPRECATION_NOTICE, 4),
)
ARCHITECTURE_ITEMS = MODEL_ARCHITECTURE_ITEMS + (
    ('trellis2', 'TRELLIS.2', 'Image to 3D mesh generation with TRELLIS.2', 5),
)
TRELLIS_TEXTURE_ITEMS = (
    ('none', 'None', 'Shape only, no texture generation', 0),
    ('native', 'Native (TRELLIS.2)', 'Use TRELLIS.2 built-in texture generation', 1),
) + tuple((key, label, description, number + 2)
          for key, label, description, number in MODEL_ARCHITECTURE_ITEMS)


def deprecated_architectures_enabled(context=None):
    try:
        from .core import get_addon_prefs
        prefs = get_addon_prefs(context)
        return bool(prefs and getattr(prefs, 'enable_deprecated_architectures', False))
    except (AttributeError, ImportError, KeyError, ReferenceError):
        return False


_READING_SELECTION = set()


def _architecture_items(scene, context, property_name, items):
    if deprecated_architectures_enabled(context) or property_name in _READING_SELECTION:
        return items
    # Read through RNA with the complete menu during the nested callback. This
    # avoids depending on ID-property storage, which changed in Blender 5.1.
    _READING_SELECTION.add(property_name)
    try:
        selected = getattr(scene, property_name, None)
    finally:
        _READING_SELECTION.remove(property_name)
    # A saved deprecated selection stays readable until the user replaces it or
    # opts in. This exception never offers another deprecated architecture.
    return tuple(item for item in items
                 if item[0] not in DEPRECATED_ARCHITECTURES or item[0] == selected)


def model_architecture_items(scene, context):
    return _architecture_items(scene, context, 'model_architecture', MODEL_ARCHITECTURE_ITEMS)


def architecture_items(scene, context):
    return _architecture_items(scene, context, 'architecture_mode', ARCHITECTURE_ITEMS)


def trellis_texture_items(scene, context):
    return _architecture_items(scene, context, 'trellis2_texture_mode', TRELLIS_TEXTURE_ITEMS)


def trellis_initial_image_items(scene, context):
    return _architecture_items(scene, context, 'trellis2_initial_image_arch', MODEL_ARCHITECTURE_ITEMS)


def active_deprecated_architectures(scene):
    if scene.architecture_mode != 'trellis2':
        choices = (scene.architecture_mode, scene.model_architecture)
    elif scene.trellis2_texture_mode in ('native', 'none'):
        choices = ((scene.trellis2_initial_image_arch,)
                   if scene.trellis2_generate_from == 'prompt' else ())
    else:
        choices = (scene.trellis2_texture_mode,)
    return DEPRECATED_ARCHITECTURES.intersection(choices)


def preset_uses_deprecated_architecture(preset):
    if preset.get('architecture_mode') == 'trellis2':
        texture = preset.get('trellis2_texture_mode', 'native')
        if texture in ('none', 'native'):
            return (preset.get('trellis2_generate_from', 'image') == 'prompt'
                    and preset.get('trellis2_initial_image_arch') in DEPRECATED_ARCHITECTURES)
        return texture in DEPRECATED_ARCHITECTURES
    return (preset.get('architecture_mode') in DEPRECATED_ARCHITECTURES
            or preset.get('model_architecture') in DEPRECATED_ARCHITECTURES)


def redraw_architecture_menus(self, context):
    # No scene mutation: switching the preference preserves all saved settings.
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()
