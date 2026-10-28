"""Context-aware presentation help; never changes parameter values or commands."""
import bpy

from . import studio_model as model
from ..core.architectures import EDIT_ARCHITECTURES


def property_hint(binding, context):
    prop, scene = binding.rna(context), context.scene
    key = binding.property
    if binding.path == 'sg_studio_diagnostics' and key == 'mode':
        from .studio_diagnostics import MODES
        description = next((description for identifier, _label, description in MODES
                            if identifier == binding.value(context)), '')
        return prop.description + '\n' + description
    if key == 'comfyui_prompt':
        if scene.architecture_mode != 'trellis2':
            return 'Describe the desired texture appearance. View-specific wording belongs in each camera’s view prompt.'
        if scene.trellis2_generate_from == 'image':
            return ('Mesh shape follows the selected input image. This prompt applies to diffusion textures '
                    'unless Separate texture prompt is enabled; native textures follow the image.')
        shared = not scene.use_separate_texture_prompt
        return ('Describe the image that will guide mesh generation. ' +
                ('This also drives per-camera diffusion textures.' if shared else
                 'Diffusion textures use the separate prompt on the Texture tab.'))
    if key == 'sequential_ipadapter':
        if scene.model_architecture in EDIT_ARCHITECTURES:
            return ('Use the first or most recent generated view as an image reference for style continuity. '
                    'An enabled external style reference can take precedence.')
        return ('Use IPAdapter with the first or most recent generated view for style continuity. '
                'An external IPAdapter image takes precedence when enabled.')
    if key == 'pbr_emission_threshold':
        return ('Cutoff on the blurred HSV detection mask (0–1). Higher values remove more weak '
                'detections after the saturation/brightness tests.' if scene.pbr_emission_method == 'hsv'
                else 'Soft cutoff for lighting-residual intensity (0–1), after metallic suppression '
                     'when available. Higher values remove more weak emission candidates.')
    return prop.description


class _PollMessage:
    """Read a Python poll explanation without changing Blender's poll message."""
    def __init__(self, owner):
        self.owner, self.message = owner, ''

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def poll_message_set(self, message, *args):
        self.message = str(message(*args) if callable(message) else message)


def operator_hint(name, context, available=True, *, busy=False):
    op = model.operator(name)
    hint = op.get_rna_type().description
    if available:
        return hint
    if busy:
        return hint + '\nUnavailable while StableGen is running.'
    cls = bpy.types.Operator.bl_rna_get_subclass_py(op.get_rna_type().identifier)
    poll = getattr(getattr(cls, 'poll', None), '__func__', None)
    if poll:
        probe = _PollMessage(cls)
        try:
            poll(probe, context)
        except Exception:
            # Some third-party polls need a real class (e.g. super()). The
            # original operator poll remains authoritative for enablement.
            return hint
        if probe.message:
            return hint + '\nUnavailable: ' + probe.message
    return hint


def native_help(control, label=None):
    """Identify native editors instead of asking users to open them to find out."""
    if control.method == 'template_list':
        name = control.args[3] if len(control.args) >= 6 else ''
        title = {'sg_camera_order': 'Edit camera order', 'sg_scene_queue': 'Edit scene queue',
                 'stablegen_print_palette': 'Edit filament palette'}.get(name, 'Edit list')
        return title, 'Open Blender’s native editor for ' + title.removeprefix('Edit ').lower() + '.'
    if control.method == 'operator_menu_enum':
        op = model.operator(control.args[0])
        title = label or control.kwargs.get('text') or op.get_rna_type().name
        return title, op.get_rna_type().description
    if control.method in {'prop', 'template_curve_mapping'} and len(control.args) >= 2:
        data, key = control.args[:2]
        prop = data.bl_rna.properties.get(key)
        if prop:
            title = label or control.kwargs.get('text') or prop.name
            return title, prop.description or 'Open Blender’s native editor for ' + title.lower() + '.'
    return label or 'Open editor', 'Open Blender’s native editor for these settings.'
