"""Preference-gated architecture menus, retaining saved enum identifiers."""

DEPRECATED_ARCHITECTURES = frozenset({'flux1', 'flux2_klein'})
DEPRECATION_NOTICE = 'Deprecated: no longer receiving further support'
# Shown by default; the add-on preferences can hide them like the deprecated ones.
EXPERIMENTAL_ARCHITECTURES = frozenset({'qwen_image_21_turbo', 'moss_image', 'fibo_edit'})
EXPERIMENTAL_NOTICE = 'Experimental: results and settings may still change'

# Explicit numbers match the original static enums, including their ordering.
# Retain these numbers when filtering so existing .blend files keep their choice.
# Number 5 belongs to TRELLIS.2 in ARCHITECTURE_ITEMS; later models start at 6.
MODEL_ARCHITECTURE_ITEMS = (
    ('sdxl', 'SDXL', 'Stable Diffusion XL', 0),
    ('moss_image', 'MOSS Image', 'MOSS-Image diffusion models, using the SDXL inpainting workflow. ' + EXPERIMENTAL_NOTICE, 7),
    ('flux1', 'FLUX.1', DEPRECATION_NOTICE, 1),
    ('qwen_image_edit', 'Qwen Image Edit', 'Qwen-Image-Edit checkpoints', 2),
    ('qwen_image_21', 'Qwen Image 2.1', 'Qwen-Image 2.1 checkpoints', 3),
    ('qwen_image_21_turbo', 'Qwen Image 2.1 Turbo', 'Qwen-Image 2.1 Turbo checkpoints with few-step sampling. ' + EXPERIMENTAL_NOTICE, 6),
    ('fibo_edit', 'FIBO 2.0 Edit', 'Bria FIBO 2.0 Edit, using the Qwen Image Edit workflow. ' + EXPERIMENTAL_NOTICE, 8),
    ('flux2_klein', 'FLUX.2 Klein', DEPRECATION_NOTICE, 4),
)
ARCHITECTURE_ITEMS = MODEL_ARCHITECTURE_ITEMS + (
    ('trellis2', 'TRELLIS.2', 'Image to 3D mesh generation with TRELLIS.2', 5),
)
DIFFUSION_ARCHITECTURES = tuple(item[0] for item in MODEL_ARCHITECTURE_ITEMS)

# Architecture families. Each new backbone reuses one established StableGen
# pipeline, so dispatch and UI checks use these groups instead of single ids.
QWEN_21_ARCHITECTURES = frozenset({'qwen_image_21', 'qwen_image_21_turbo'})
# Image-edit models driven by the shared Qwen guidance settings and prompts.
QWEN_ARCHITECTURES = QWEN_21_ARCHITECTURES | {'qwen_image_edit', 'fibo_edit'}
# Native-graph edit models that VAE-encode their first image as the output
# canvas, so projected context moves into image 1 and guides use gray.
CANVAS_EDIT_ARCHITECTURES = QWEN_21_ARCHITECTURES | {'fibo_edit'}
# Reference-image edit pipelines, including the deprecated FLUX.2 Klein.
EDIT_ARCHITECTURES = QWEN_ARCHITECTURES | {'flux2_klein'}
# Checkpoint-style models that run the SDXL txt2img / img2img / inpaint graphs.
SDXL_GRAPH_ARCHITECTURES = frozenset({'sdxl', 'moss_image'})
# Models loaded through dedicated native nodes, without LoRA or CLIP skip.
NATIVE_GRAPH_ARCHITECTURES = CANVAS_EDIT_ARCHITECTURES | {'moss_image'}
MASKED_EDIT_ARCHITECTURES = SDXL_GRAPH_ARCHITECTURES | QWEN_ARCHITECTURES
TRELLIS_TEXTURE_ITEMS = (
    ('none', 'None', 'Shape only, no texture generation', 0),
    ('native', 'Native (TRELLIS.2)', 'Use TRELLIS.2 built-in texture generation', 1),
) + tuple((key, label, description, number + 2)
          for key, label, description, number in MODEL_ARCHITECTURE_ITEMS)


def _preference(context, name, default):
    try:
        from . import get_addon_prefs
        prefs = get_addon_prefs(context)
        return bool(getattr(prefs, name, default)) if prefs else default
    except (AttributeError, ImportError, KeyError, ReferenceError):
        return default


def deprecated_architectures_enabled(context=None):
    return _preference(context, 'enable_deprecated_architectures', False)


def experimental_architectures_enabled(context=None):
    return _preference(context, 'enable_experimental_architectures', True)


def hidden_architectures(context=None):
    hidden = set()
    if not deprecated_architectures_enabled(context):
        hidden |= DEPRECATED_ARCHITECTURES
    if not experimental_architectures_enabled(context):
        hidden |= EXPERIMENTAL_ARCHITECTURES
    return hidden


def architecture_status(identifier):
    """'Deprecated', 'Experimental' or '' for an architecture identifier."""
    return ('Deprecated' if identifier in DEPRECATED_ARCHITECTURES
            else 'Experimental' if identifier in EXPERIMENTAL_ARCHITECTURES else '')


_READING_SELECTION = set()


def _architecture_items(scene, context, property_name, items):
    hidden = hidden_architectures(context)
    if not hidden or property_name in _READING_SELECTION:
        return items
    # Read through RNA with the complete menu during the nested callback. This
    # avoids depending on ID-property storage, which changed in Blender 5.1.
    _READING_SELECTION.add(property_name)
    try:
        selected = getattr(scene, property_name, None)
    finally:
        _READING_SELECTION.remove(property_name)
    # A saved hidden selection stays readable until the user replaces it or
    # opts in. This exception never offers another hidden architecture.
    return tuple(item for item in items if item[0] not in hidden or item[0] == selected)


def model_architecture_items(scene, context):
    return _architecture_items(scene, context, 'model_architecture', MODEL_ARCHITECTURE_ITEMS)


def architecture_items(scene, context):
    return _architecture_items(scene, context, 'architecture_mode', ARCHITECTURE_ITEMS)


def trellis_texture_items(scene, context):
    return _architecture_items(scene, context, 'trellis2_texture_mode', TRELLIS_TEXTURE_ITEMS)


def trellis_initial_image_items(scene, context):
    return _architecture_items(scene, context, 'trellis2_initial_image_arch', MODEL_ARCHITECTURE_ITEMS)


def active_deprecated_architectures(scene):
    return DEPRECATED_ARCHITECTURES.intersection(_active_architectures(scene))


def active_experimental_architectures(scene):
    return EXPERIMENTAL_ARCHITECTURES.intersection(_active_architectures(scene))


def _active_architectures(scene):
    if scene.architecture_mode != 'trellis2':
        choices = (scene.architecture_mode, scene.model_architecture)
    elif scene.trellis2_texture_mode in ('native', 'none'):
        choices = ((scene.trellis2_initial_image_arch,)
                   if scene.trellis2_generate_from == 'prompt' else ())
    else:
        choices = (scene.trellis2_texture_mode,)
    return choices


def _preset_architectures(preset):
    if preset.get('architecture_mode') == 'trellis2':
        texture = preset.get('trellis2_texture_mode', 'native')
        if texture in ('none', 'native'):
            return ({preset.get('trellis2_initial_image_arch')}
                    if preset.get('trellis2_generate_from', 'image') == 'prompt' else set())
        return {texture}
    return {preset.get('architecture_mode'), preset.get('model_architecture')}


def preset_uses_deprecated_architecture(preset):
    return bool(_preset_architectures(preset) & DEPRECATED_ARCHITECTURES)


def preset_uses_experimental_architecture(preset):
    return bool(_preset_architectures(preset) & EXPERIMENTAL_ARCHITECTURES)


def redraw_architecture_menus(self, context):
    # No scene mutation: switching the preference preserves all saved settings.
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()
