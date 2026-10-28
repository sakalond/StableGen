"""Capture the legacy UI and replay its live controls in alternate layouts.

This is a presentation adapter: it retains RNA bindings, operator properties,
conditional visibility, enable states, and progress without copying workflow logic.
"""
from dataclasses import dataclass, field
from types import SimpleNamespace

from .panel import StableGenPanel, StableGenPrintPanel


CONTAINERS = {'row', 'column', 'box', 'split', 'grid_flow', 'column_flow'}
LAYOUT_DEFAULTS = {
    'enabled': True, 'active': True, 'alert': False, 'alignment': 'EXPAND',
    'scale_x': 1.0, 'scale_y': 1.0, 'use_property_split': False,
    'use_property_decorate': True, 'operator_context': 'INVOKE_REGION_WIN',
}
SECTION_INFO = {
    'show_core_settings': ('Sampling & Resolution', 'GENERATE', 'SETTINGS'),
    'show_lora_settings': ('LoRA Management', 'GENERATE', 'MODIFIER'),
    'show_camera_options': ('Camera Settings', 'CAMERAS', 'CAMERA_DATA'),
    'show_scene_understanding_settings': ('Viewpoint Blending', 'BLEND', 'ZOOM_IN'),
    'show_output_material_settings': ('Output & Materials', 'OUTPUT', 'MATERIAL'),
    'show_image_guidance_settings': ('Image Guidance', 'GENERATE', 'MODIFIER'),
    'show_masking_inpainting_settings': ('Inpainting Options', 'GENERATE', 'MOD_MASK'),
    'show_mode_specific_settings': ('Generation Mode Specifics', 'GENERATE', 'OPTIONS'),
    'show_texturing_experiments': ('Experimental Texturing', 'DEBUG', 'EXPERIMENTAL'),
    'show_trellis2_mesh_settings': ('Mesh & Post-Processing', 'MESH', 'MESH_DATA'),
    'show_trellis2_autorig_settings': ('Rigging & Animation', 'MESH', 'ARMATURE_DATA'),
    'show_trellis2_texture_settings': ('Native Textures', 'MESH', 'TEXTURE'),
    'show_trellis2_camera_settings': ('Camera Placement', 'CAMERAS', 'CAMERA_DATA'),
}
STRUCTURAL_PROPERTIES = {
    *SECTION_INFO, 'show_generation_params', 'show_advanced_params', 'sg_show_queue',
}

# Group utilities by the data they operate on within the Tools tab.
# Studio cards and native alternate panels share these groups.
TOOL_SECTIONS = {
    'mesh_tools': ('Mesh & rig utilities', 'TOOLS', 'MESH_DATA', (
        ('object.apply_all_mesh_modifiers', 'Apply all modifiers', 'modifier'),
        ('object.curves_to_mesh', 'Convert curves to mesh', 'curve'),
        ('object.stablegen_local_postprocess', 'Local post-processing', 'modifier'),
        ('object.stablegen_make_solid', 'Make solid', 'modifier'),
        ('object.stablegen_autorig_selected_mesh', 'Auto-rig selected character', 'rig'),
    )),
    'material_tools': ('Material utilities', 'TOOLS', 'MATERIAL', (
        ('object.switch_material', 'Switch material', 'material'),
    )),
    'projection_tools': ('Projection utilities', 'TOOLS', 'MOD_MIRROR', (
        ('object.stablegen_mirror_reproject', 'Mirror last projection', 'mirror'),
    )),
    'presentation_tools': ('Presentation export', 'TOOLS', 'RENDER_ANIMATION', (
        ('object.export_orbit_gif', 'Export orbit GIF / MP4', 'camera'),
    )),
}


class ExpandedData:
    """Override only presentation expansion flags; forward all real scene data."""
    def __init__(self, data):
        object.__setattr__(self, '_data', data)

    def __getattr__(self, name):
        if name == 'windows':
            return [ExpandedWindow(window) for window in self._data.windows]
        if name == 'qwen_refine_mode' and not hasattr(self._data, name):
            # The legacy renderer still reads this retired presentation alias;
            # sequential/separate is now stored in generation_method.
            return self._data.generation_method
        value = getattr(self._data, name)
        if name in STRUCTURAL_PROPERTIES:
            return True
        return value

    def __setattr__(self, name, value):
        setattr(self._data, name, value)


class ExpandedWindow:
    """Built-in and retiring modal handlers can appear as None in Blender RNA."""
    def __init__(self, window):
        self._window = window

    def __getattr__(self, name):
        if name == 'modal_operators':
            return [op for op in self._window.modal_operators if op is not None]
        return getattr(self._window, name)


class ExpandedContext:
    def __init__(self, context):
        self._context = context
        self.scene = ExpandedData(context.scene)
        self.window_manager = ExpandedData(context.window_manager)

    def __getattr__(self, name):
        return getattr(self._context, name)


def _unwrap(value):
    return value._data if isinstance(value, ExpandedData) else value


@dataclass
class Control:
    method: str
    args: tuple
    kwargs: dict
    properties: dict = field(default_factory=dict)
    tooltip: str = ''

    def render(self, layout, *, narrow=False):
        if (narrow and self.method == 'prop' and self.args[1]
                in {'sg_projection_guard_scale', 'sg_guard_feather_method'}):
            column = layout.column()
            column.label(text=self.kwargs.get('text', self.args[1]))
            kwargs = dict(self.kwargs, text='')
            column.prop(*self.args, **kwargs)
            return
        result = getattr(layout, self.method)(*self.args, **self.kwargs)
        for name, value in self.properties.items():
            setattr(result, name, value)


class OperatorProperties:
    def __init__(self, control):
        object.__setattr__(self, '_control', control)

    def __setattr__(self, name, value):
        self._control.properties[name] = value


class Layout:
    def __init__(self, method='root', args=(), kwargs=None):
        object.__setattr__(self, 'method', method)
        object.__setattr__(self, 'args', args)
        object.__setattr__(self, 'kwargs', kwargs or {})
        object.__setattr__(self, 'attributes', {})
        object.__setattr__(self, 'children', [])

    def __setattr__(self, name, value):
        self.attributes[name] = value

    def __getattr__(self, name):
        if name in LAYOUT_DEFAULTS:
            return self.attributes.get(name, LAYOUT_DEFAULTS[name])

        def record(*args, **kwargs):
            args = tuple(_unwrap(value) for value in args)
            kwargs = {key: _unwrap(value) for key, value in kwargs.items()}
            if name in CONTAINERS:
                child = Layout(name, args, kwargs)
                self.children.append(child)
                return child
            child = Control(name, args, kwargs)
            self.children.append(child)
            if name == 'operator':
                return OperatorProperties(child)
            return child if name == 'progress' else None
        return record

    def controls(self):
        for child in self.children:
            if isinstance(child, Layout):
                yield from child.controls()
            else:
                yield child

    def has_property(self, name):
        return any(control.method == 'prop' and len(control.args) > 1
                   and control.args[1] == name for control in self.controls())

    def has_operator(self, *names):
        return any(control.method == 'operator' and control.args[0] in names
                   for control in self.controls())

    def render(self, parent, *, contents=False, omit_properties=(), narrow=False):
        method, kwargs = self.method, self.kwargs
        if (narrow and method == 'row' and len(self.children) >= 3
                and all(isinstance(c, Control) and c.method == 'operator' for c in self.children)):
            method, kwargs = 'grid_flow', dict(row_major=True, columns=2,
                                              even_columns=True, even_rows=True, align=False)
        if narrow and method == 'split' and kwargs.get('factor') == 0.5:
            # The legacy two-column sliders remain side by side at wide sizes.
            # Reflow them at compact sidebar sizes instead of truncating names.
            if sum(c.method in ('prop', 'operator') for c in self.controls()) >= 2:
                method, kwargs = 'column', {'align': False}
        layout = parent if contents or self.method == 'root' else getattr(parent, method)(
            *self.args, **kwargs)
        for name, value in self.attributes.items():
            setattr(layout, name, value)
        for child in self.children:
            if isinstance(child, Layout):
                if any(isinstance(control, Control) and control.method == 'prop'
                       and control.args[1] in omit_properties for control in child.children):
                    continue
                child.render(layout, omit_properties=omit_properties, narrow=narrow)
                continue
            if (isinstance(child, Control) and child.method == 'prop'
                    and len(child.args) > 1 and child.args[1] in STRUCTURAL_PROPERTIES):
                continue
            child.render(layout, narrow=narrow)


@dataclass
class Section:
    key: str
    title: str
    page: str
    icon: str
    content: Layout

    def search_text(self):
        parts = [self.title]
        for control in self.content.controls():
            parts.append(str(control.kwargs.get('text', '')))
            if control.method == 'prop' and len(control.args) > 1:
                data, name = control.args[:2]
                parts.append(name.replace('_', ' '))
                prop = data.bl_rna.properties.get(name)
                if prop:
                    parts.extend((prop.name, prop.description))
        return ' '.join(parts).casefold()


@dataclass
class Document:
    sections: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    notices: list = field(default_factory=list)
    presets: Layout | None = None
    architecture_alert: bool = False
    architecture_targets: list = field(default_factory=list)


def _section_parts(node):
    """Find a legacy section header and its content, preserving its real title."""
    for child in node.children:
        if isinstance(child, Control) and child.method == 'prop' and len(child.args) > 1:
            key = child.args[1]
            if key in SECTION_INFO:
                content = next((sibling for sibling in node.children
                                if isinstance(sibling, Layout) and sibling.method == 'box'), None)
                if content:
                    title, page, icon = SECTION_INFO[key]
                    return Section(key, child.kwargs.get('text') or title, page, icon, content)
        elif isinstance(child, Layout):
            result = _section_parts(child)
            if result:
                return result
    return None


def _advanced_sections(node):
    # The advanced column contains one box per section. Descending into an
    # already matched section would duplicate its conditional feature subboxes.
    for child in node.children:
        if not isinstance(child, Layout):
            continue
        section = _section_parts(child)
        if section and not child.has_property('show_advanced_params'):
            yield section
        else:
            yield from _advanced_sections(child)


def _main_content(node):
    for child in node.children:
        if not isinstance(child, Layout):
            continue
        if child.method == 'box' and not child.has_property('show_generation_params'):
            return child
        result = _main_content(child)
        if result:
            return result
    return node


def _tool_sections(source):
    """Split utility controls without losing inherited states or invoke context."""
    moved = {name for _, _, _, commands in TOOL_SECTIONS.values()
             for name, _, _ in commands}

    def select(node, names, remaining=False):
        result = Layout(node.method, node.args, dict(node.kwargs))
        result.attributes.update(node.attributes)
        for child in node.children:
            if isinstance(child, Layout):
                selected = select(child, names, remaining)
                if selected.children:
                    result.children.append(selected)
            elif child.method == 'operator' and child.args[0] in names:
                if not remaining:
                    result.children.append(child)
            elif remaining:
                result.children.append(child)
        return result

    yield Section('tools', 'Tools', 'TOOLS', 'TOOL_SETTINGS',
                  select(source, moved, remaining=True))
    for key, (title, page, icon, commands) in TOOL_SECTIONS.items():
        content = select(source, {name for name, _, _ in commands})
        if list(content.controls()):
            yield Section(key, title, page, icon, content)


def capture_panel(context):
    root = Layout()
    StableGenPanel.draw(SimpleNamespace(layout=root), ExpandedContext(context))
    document = Document()
    camera_tools = Layout()
    action_ops = ('object.test_stable', 'object.stablegen_regenerate',
                  'object.stablegen_autorig_selected_mesh',
                  'object.trellis2_generate', 'object.trellis2_batch_generate',
                  'object.trellis2_batch_cancel', 'stablegen.masked_edit',
                  'stablegen.stop_masked_edit', 'stablegen.resolve_masked_edit',
                  'object.bake_textures')
    for child in root.children:
        if not isinstance(child, Layout):
            continue
        if child.has_property('stablegen_preset'):
            document.presets = child
        elif child.has_property('show_generation_params'):
            document.sections.append(Section('main', 'Main Parameters', 'GENERATE', 'OPTIONS',
                                             _main_content(child)))
        elif child.has_property('show_advanced_params'):
            document.sections.extend(_advanced_sections(child))
        elif child.has_property('sg_show_queue'):
            document.sections.append(Section('queue', 'Scene Queue', 'TOOLS', 'SEQ_SEQUENCER', child))
        elif child.has_operator('stablegen.debug_solid_colors'):
            document.sections.append(Section('debug', 'Debug / Diagnostics', 'DEBUG', 'GHOST_ENABLED', child))
        elif child.has_operator('object.switch_material'):
            document.sections.extend(_tool_sections(child))
        elif child.has_operator(*action_ops) or any(c.method == 'progress' for c in child.controls()):
            document.actions.append(child)
        elif child.has_operator('object.add_cameras', 'object.clone_camera', 'object.collect_camera_prompts',
                                'object.mirror_camera', 'object.apply_auto_aspect', 'object.toggle_camera_labels') and not child.has_operator('stablegen.switch_to_mesh_generation'):
            camera_tools.children.append(child)
        elif child.has_operator('object.export_game_engine'):
            document.sections.append(Section('export', 'Game Engine Export', 'OUTPUT', 'EXPORT', child))
        else:
            # Inline prerequisites and deprecated-architecture notices retain
            # their original controls. They remain visible across all pages.
            controls = list(child.controls())
            if any(c.method == 'label' and c.kwargs.get('text') == 'Widen panel for side-by-side layout' for c in controls):
                continue
            if any(c.method == 'label' and c.kwargs.get('icon') in ('ERROR', 'INFO') for c in controls):
                document.notices.append(child)
            elif controls:
                # Informational status (model refresh, masked edit readiness)
                # should not disappear when moving the legacy action area.
                document.actions.append(child)
    if camera_tools.children:
        document.sections.insert(0, Section('camera_tools', 'Camera Tools', 'CAMERAS', 'CAMERA_DATA', camera_tools))
    for section in document.sections:
        if section.key == 'main':
            for row in section.content.children:
                if isinstance(row, Layout) and row.has_property('architecture_mode'):
                    document.architecture_alert = row.attributes.get('alert', False)
                    document.architecture_targets = [c.kwargs['text'] for c in row.controls()
                                                     if c.method == 'label' and c.kwargs.get('text', '').startswith('→')]
    prefs = context.preferences.addons.get(__package__.rsplit('.', 1)[0])
    if prefs and prefs.preferences.enable_print_tab:
        print_root = Layout()
        StableGenPrintPanel.draw(SimpleNamespace(layout=print_root), ExpandedContext(context))
        document.sections.append(Section('print', '3D Printing', 'PRINT', 'OUTLINER_OB_MESH', print_root))
    return document
