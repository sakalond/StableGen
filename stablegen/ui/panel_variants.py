"""Three live UI alternatives alongside the unchanged legacy sidebar."""
import bpy

from ..utils import sg_modal_active
from .presentation import capture_panel

PAGES = (
    ('GENERATE', 'Generate', 'Generation, models and guidance', 'RENDER_STILL', 0),
    ('CAMERAS', 'Cameras', 'Viewpoints, framing and order', 'CAMERA_DATA', 1),
    ('BLEND', 'Blending', 'Projection, visibility and DPPG', 'MOD_MASK', 2),
    ('OUTPUT', 'Output', 'Materials, PBR and export', 'MATERIAL', 3),
    ('MESH', 'Mesh & Rig', 'TRELLIS.2 geometry and rigging', 'MESH_DATA', 4),
    ('TOOLS', 'Tools & Batch', 'Utilities and queued scenes', 'TOOL_SETTINGS', 5),
    ('DEBUG', 'Diagnostics', 'Debug tools and experimental settings', 'GHOST_ENABLED', 6),
    ('PRINT', 'Printing', 'Palette and print exports', 'OUTLINER_OB_MESH', 7),
)
PAGE_LABELS = {item[0]: (item[1], item[3]) for item in PAGES}
STAGES = (
    ('SETUP', 'Set up', 'Model, prompt and generation settings', 'SETTINGS', 0),
    ('VIEWS', 'Arrange views', 'Camera setup and view descriptions', 'CAMERA_DATA', 1),
    ('RUN', 'Generate', 'Generation mode and execution', 'PLAY', 2),
    ('PROJECT', 'Blend & inspect', 'Projection and diagnostics', 'MOD_MASK', 3),
    ('DELIVER', 'Bake & export', 'Materials and deliverables', 'EXPORT', 4),
    ('UTILITY', 'Utilities', 'Batch workflows and remaining tools', 'TOOL_SETTINGS', 5),
)
STATE_NAMES = {'STUDIO': 'sg_ui_studio', 'WORKBENCH': 'sg_ui_workbench', 'FOCUS': 'sg_ui_focus'}
DEFAULT_OPEN = 'main|show_core_settings|show_mode_specific_settings|show_scene_understanding_settings|show_output_material_settings|camera_tools|export'


def _redraw(self, context):
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class SGPanelVariantState(bpy.types.PropertyGroup):
    page: bpy.props.EnumProperty(items=PAGES, default='GENERATE', update=_redraw)
    stage: bpy.props.EnumProperty(items=STAGES, default='SETUP', update=_redraw)
    query: bpy.props.StringProperty(name='Find a setting', description='Search available section titles, controls and tooltips', update=_redraw)
    open_sections: bpy.props.StringProperty(default=DEFAULT_OPEN)
    pinned_sections: bpy.props.StringProperty(default='')
    overview: bpy.props.BoolProperty(default=True)
    show_pins: bpy.props.BoolProperty(name='Pinned Settings', default=True, update=_redraw)
    camera_index: bpy.props.IntProperty(default=0)


class SG_OT_PanelNavigate(bpy.types.Operator):
    bl_idname = 'stablegen.panel_navigate'
    bl_label = 'Open Settings'
    bl_description = 'Open this part of the StableGen panel'
    bl_options = {'INTERNAL'}
    variant: bpy.props.EnumProperty(items=[(key, key.title(), '') for key in STATE_NAMES])
    destination: bpy.props.StringProperty()
    section: bpy.props.StringProperty()
    mode: bpy.props.EnumProperty(items=[('PAGE', 'Page', ''), ('STAGE', 'Stage', ''), ('OVERVIEW', 'Overview', '')])

    def execute(self, context):
        state = getattr(context.window_manager, STATE_NAMES[self.variant])
        if self.mode == 'STAGE':
            state.stage = self.destination
        elif self.mode == 'OVERVIEW':
            state.overview = True
            state.query = ''
        else:
            state.page = self.destination
            state.overview = False
            state.query = ''
        if self.section:
            keys = set(filter(None, state.open_sections.split('|')))
            keys.add(self.section)
            state.open_sections = '|'.join(sorted(keys))
        elif self.mode == 'PAGE' and self.destination == 'DEBUG':
            keys = set(filter(None, state.open_sections.split('|')))
            keys.add('debug')
            state.open_sections = '|'.join(sorted(keys))
        _redraw(self, context)
        return {'FINISHED'}


class SG_OT_PanelSection(bpy.types.Operator):
    bl_idname = 'stablegen.panel_section'
    bl_label = 'Expand Section'
    bl_description = 'Expand or collapse this section, or pin it to the Workbench overview'
    bl_options = {'INTERNAL'}
    variant: bpy.props.EnumProperty(items=[(key, key.title(), '') for key in STATE_NAMES])
    section: bpy.props.StringProperty()
    pin: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        state = getattr(context.window_manager, STATE_NAMES[self.variant])
        field = 'pinned_sections' if self.pin else 'open_sections'
        keys = set(filter(None, getattr(state, field).split('|')))
        if self.section in keys:
            keys.remove(self.section)
        else:
            keys.add(self.section)
        setattr(state, field, '|'.join(sorted(keys)))
        _redraw(self, context)
        return {'FINISHED'}


class SG_OT_PanelCamera(bpy.types.Operator):
    bl_idname = 'stablegen.panel_camera'
    bl_label = 'View Camera'
    bl_description = 'Select this camera and look through it in the viewport'
    bl_options = {'INTERNAL'}
    camera_name: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        return not sg_modal_active(context)

    def execute(self, context):
        camera = context.scene.objects.get(self.camera_name)
        if camera is None or camera.type != 'CAMERA':
            self.report({'ERROR'}, 'Camera is no longer in this scene')
            return {'CANCELLED'}
        from ..cameras.operators import switch_viewport_to_camera
        switch_viewport_to_camera(context, camera)
        if camera.name in context.view_layer.objects:
            for obj in context.selected_objects:
                obj.select_set(False)
            camera.select_set(True)
            context.view_layer.objects.active = camera
        return {'FINISHED'}


class SG_UL_PanelCameras(bpy.types.UIList):
    """A compact view selector with editable per-view descriptions."""
    bl_idname = 'SG_UL_PanelCameras'

    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        camera = item
        cameras = [obj for obj in context.scene.objects if obj.type == 'CAMERA']
        prompt = context.scene.camera_prompts.get(camera.name)
        row = layout.row(align=True)
        row.label(text=f'{cameras.index(camera) + 1:02}', icon='CAMERA_DATA')
        row.label(text=item.name)
        button = row.operator('stablegen.panel_camera', text='', icon='HIDE_OFF')
        button.camera_name = item.name
        if context.region.width >= 430 and prompt:
            row.prop(prompt, 'prompt', text='')

    def filter_items(self, context, data, property):
        return [self.bitflag_filter_item if obj.type == 'CAMERA' else 0
                for obj in getattr(data, property)], []


def _state(context, variant):
    return getattr(context.window_manager, STATE_NAMES[variant])


def _nav_button(layout, variant, destination, text, icon, *, active=False, mode='PAGE', section=''):
    op = layout.operator('stablegen.panel_navigate', text=text, icon=icon, depress=active)
    op.variant, op.destination, op.mode, op.section = variant, destination, mode, section


def _section(layout, section, state, variant, *, force_open=False, pinnable=False, title=None):
    expanded = force_open or section.key in state.open_sections.split('|')
    box = layout.box()
    header = box.row(align=True)
    if force_open:
        header.label(text='', icon='VIEWZOOM')
    else:
        button = header.operator('stablegen.panel_section', text='',
                                 icon='TRIA_DOWN' if expanded else 'TRIA_RIGHT', emboss=False)
        button.variant, button.section = variant, section.key
    header.label(text=title or section.title, icon=section.icon)
    if pinnable:
        pin = header.operator('stablegen.panel_section', text='',
                              icon='SOLO_ON' if section.key in state.pinned_sections.split('|') else 'SOLO_OFF',
                              emboss=False)
        pin.variant, pin.section, pin.pin = variant, section.key, True
    if expanded:
        section.content.render(box.column(), contents=True,
                               omit_properties={'architecture_mode'} if section.key == 'main' else (),
                               narrow=bpy.context.region.width < 420 if bpy.context.region else False)


def _available_pages(document):
    keys = {section.page for section in document.sections}
    return [item for item in PAGES if item[0] in keys]


def _effective_page(document, requested):
    pages = [item[0] for item in _available_pages(document)]
    return requested if requested in pages else ('GENERATE' if 'GENERATE' in pages else next(iter(pages), None))


def _header(layout, context, variant, subtitle, document):
    scene = context.scene
    row = layout.row()
    row.scale_y = 1.15
    row.label(text=variant.title(), icon={'STUDIO': 'WORKSPACE', 'WORKBENCH': 'NODETREE', 'FOCUS': 'SEQ_SEQUENCER'}[variant])
    right = row.row()
    right.alignment = 'RIGHT'
    right.label(text=subtitle)
    row = layout.row(align=True)
    architecture = row.row()
    architecture.alert = document.architecture_alert
    architecture.prop(scene, 'architecture_mode', text='')
    cameras = sum(obj.type == 'CAMERA' for obj in scene.objects)
    row.label(text=f'{cameras} views', icon='CAMERA_DATA')
    for target in document.architecture_targets:
        layout.label(text=target)


def _shared(layout, context, document, *, idle_generation=True):
    if document.presets:
        document.presets.render(layout, contents=True)
    for notice in document.notices:
        notice.render(layout)
    for action in document.actions:
        # Baking remains available on Output; active baking progress stays
        # shared across pages, like generation and cancellation.
        if action.has_operator('object.bake_textures'):
            continue
        active = any(c.method == 'progress' or (c.method == 'operator' and
                     (c.kwargs.get('icon') in {'CANCEL', 'TIME'} or c.args[0] == 'stablegen.resolve_masked_edit'))
                     for c in action.controls())
        if not idle_generation and not active and action.has_operator(
                'object.test_stable', 'object.stablegen_regenerate', 'object.trellis2_generate',
                'object.trellis2_batch_generate', 'stablegen.masked_edit'):
            continue
        action.render(layout)


def _output_actions(layout, document):
    for action in document.actions:
        if action.has_operator('object.bake_textures'):
            action.render(layout)


def _deliver(layout, document, state, variant, sections):
    # Configure material output before baking, then choose export formats.
    ordered = list(sections)
    for section in ordered:
        if section.key == 'show_output_material_settings':
            _section(layout, section, state, variant, pinnable=variant == 'WORKBENCH')
    _output_actions(layout, document)
    for section in ordered:
        if section.key != 'show_output_material_settings':
            _section(layout, section, state, variant, pinnable=variant == 'WORKBENCH')


def _camera_browser(layout, context, state):
    if any(obj.type == 'CAMERA' for obj in context.scene.objects):
        box = layout.box()
        row = box.row()
        row.label(text='View Navigator', icon='CAMERA_DATA')
        box.template_list('SG_UL_PanelCameras', '', context.scene, 'objects', state, 'camera_index', rows=4, maxrows=6)


class SG_MT_WorkbenchPages(bpy.types.Menu):
    bl_idname = 'SG_MT_WorkbenchPages'
    bl_label = 'Workbench Pages'

    def draw(self, context):
        for key, label, description, icon, number in _available_pages(capture_panel(context)):
            _nav_button(self.layout, 'WORKBENCH', key, label, icon)


class _VariantPanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = set()

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'PAINT_TEXTURE'}


class SG_PT_Studio(_VariantPanel, bpy.types.Panel):
    bl_idname = 'SG_PT_Studio'
    bl_label = 'StableGen Studio'
    bl_category = 'SG Studio'

    def draw(self, context):
        layout = self.layout
        state = _state(context, 'STUDIO')
        document = capture_panel(context)
        _header(layout, context, 'STUDIO', 'Task workspace', document)
        _shared(layout, context, document)
        available = _available_pages(document)
        page = _effective_page(document, state.page)
        nav = layout.grid_flow(row_major=True, columns=3 if context.region.width >= 360 else 2,
                               even_columns=True, even_rows=True, align=True)
        for key, label, description, icon, number in available:
            _nav_button(nav, 'STUDIO', key, label, icon, active=page == key)
        layout.separator(factor=0.6)
        if page == 'CAMERAS':
            _camera_browser(layout, context, state)
        if page == 'OUTPUT':
            _deliver(layout, document, state, 'STUDIO', (s for s in document.sections if s.page == page))
            return
        for section in document.sections:
            if section.page == page:
                _section(layout, section, state, 'STUDIO')


def _card_summary(section, scene):
    summaries = {
        'main': 'Prompt, model & generation mode',
        'camera_tools': 'Create, clone & frame views',
        'export': 'Materials & game engine formats',
        'show_texturing_experiments': 'Optional research controls',
        'print': 'Palette, slices & print exports',
        'show_trellis2_texture_settings': 'Native texture generation',
        'show_trellis2_camera_settings': 'Place views for texturing',
        'show_masking_inpainting_settings': 'Masks & texture repair',
        'show_core_settings': f'{scene.steps} steps · CFG {scene.cfg:g}',
        'show_lora_settings': f'{len(scene.lora_units)} LoRA units',
        'show_image_guidance_settings': 'References & context render',
        'show_mode_specific_settings': 'Controls for the current mode',
        'show_scene_understanding_settings': f'Angle {scene.discard_factor:g}° · exponent {scene.weight_exponent:g}',
        'show_camera_options': 'Prompts & generation order',
        'show_output_material_settings': 'PBR, display & resolution',
        'show_trellis2_mesh_settings': 'Geometry & post-processing',
        'show_trellis2_autorig_settings': 'Rigging, poses & animation',
        'queue': 'Queued scenes & recording',
        'tools': 'Import, prepare & process',
        'debug': 'Materials, measurements & tests',
    }
    return summaries.get(section.key, 'Open settings')


class SG_PT_Workbench(_VariantPanel, bpy.types.Panel):
    bl_idname = 'SG_PT_Workbench'
    bl_label = 'StableGen Workbench'
    bl_category = 'SG Workbench'

    def draw(self, context):
        layout = self.layout
        state = _state(context, 'WORKBENCH')
        document = capture_panel(context)
        _header(layout, context, 'WORKBENCH', 'Feature browser', document)
        _shared(layout, context, document)
        page = _effective_page(document, state.page)
        row = layout.row(align=True)
        _nav_button(row, 'WORKBENCH', '', 'Overview', 'HOME', active=state.overview, mode='OVERVIEW')
        label, icon = ('Pages', 'SHORTDISPLAY') if state.overview else PAGE_LABELS.get(page, ('Pages', 'SHORTDISPLAY'))
        row.menu('SG_MT_WorkbenchPages', text=label, icon=icon)
        layout.prop(state, 'query', text='', icon='VIEWZOOM')
        query = state.query.strip().casefold()
        if query:
            matches = [section for section in document.sections if query in section.search_text()]
            layout.label(text=f'{len(matches)} matching {"section" if len(matches) == 1 else "sections"}', icon='FILTER')
            for section in matches:
                _section(layout, section, state, 'WORKBENCH', force_open=True, pinnable=True)
            if not matches:
                layout.label(text='No matching settings.', icon='INFO')
                layout.label(text='Check the architecture or mode.')
            return
        if not state.overview:
            if page == 'CAMERAS':
                _camera_browser(layout, context, state)
            if page == 'OUTPUT':
                _deliver(layout, document, state, 'WORKBENCH', (s for s in document.sections if s.page == page))
                return
            for section in document.sections:
                if section.page == page:
                    _section(layout, section, state, 'WORKBENCH', pinnable=True)
            return
        pinned = [section for section in document.sections if section.key in state.pinned_sections.split('|')]
        if pinned:
            layout.prop(state, 'show_pins', text=f'Pinned Settings ({len(pinned)})',
                        icon='TRIA_DOWN' if state.show_pins else 'TRIA_RIGHT', emboss=False)
            if state.show_pins:
                for section in pinned:
                    _section(layout, section, state, 'WORKBENCH', pinnable=True)
        layout.label(text='Browse Features', icon='SHORTDISPLAY')
        for page, label, description, icon, number in _available_pages(document):
            layout.label(text=label, icon=icon)
            grid = layout.grid_flow(row_major=True, columns=2 if context.region.width >= 410 else 1,
                                    even_columns=True, even_rows=False, align=False)
            for section in document.sections:
                if section.page != page:
                    continue
                box = grid.box()
                row = box.row(align=True)
                _nav_button(row, 'WORKBENCH', page, section.title, section.icon, section=section.key)
                box.label(text=_card_summary(section, context.scene))


def _focus_sections(document, stage):
    pages = {
        'SETUP': {'GENERATE', 'MESH'}, 'VIEWS': {'CAMERAS'},
        'RUN': {'GENERATE'}, 'PROJECT': {'BLEND', 'DEBUG'},
        'DELIVER': {'OUTPUT', 'PRINT'}, 'UTILITY': {'TOOLS', 'DEBUG'},
    }[stage]
    for section in document.sections:
        if section.page not in pages:
            continue
        if stage == 'SETUP' and section.key in {'show_mode_specific_settings', 'show_masking_inpainting_settings'}:
            continue
        if stage == 'RUN' and section.key not in {'show_mode_specific_settings', 'show_masking_inpainting_settings', 'show_image_guidance_settings', 'main'}:
            continue
        if stage == 'PROJECT' and section.key == 'show_texturing_experiments':
            continue
        yield section


class SG_PT_Focus(_VariantPanel, bpy.types.Panel):
    bl_idname = 'SG_PT_Focus'
    bl_label = 'StableGen Focus'
    bl_category = 'SG Focus'

    def draw(self, context):
        layout = self.layout
        state = _state(context, 'FOCUS')
        document = capture_panel(context)
        _header(layout, context, 'FOCUS', 'Guided workflow', document)
        _shared(layout, context, document, idle_generation=state.stage == 'RUN')
        rail = layout.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=True)
        for key, label, description, icon, number in STAGES:
            _nav_button(rail, 'FOCUS', key, f'{number + 1}  {label}', icon,
                        active=state.stage == key, mode='STAGE')
        layout.separator(factor=0.6)
        instructions = {
            'SETUP': 'Choose models and generation settings.',
            'VIEWS': 'Arrange cameras and view prompts.',
            'RUN': 'Review the prompt, then generate.',
            'PROJECT': 'Inspect coverage and blending.',
            'DELIVER': 'Set materials, bake, then export.',
            'UTILITY': 'Tools, scene queue and experiments.',
        }
        layout.label(text=instructions[state.stage], icon='INFO')
        if state.stage == 'VIEWS':
            _camera_browser(layout, context, state)
        sections = list(_focus_sections(document, state.stage))
        if state.stage == 'DELIVER':
            _deliver(layout, document, state, 'FOCUS', sections)
        else:
            for section in sections:
                title = None
                if section.key == 'main':
                    title = 'Model & Generation Setup' if state.stage == 'SETUP' else 'Review Prompt & Model'
                _section(layout, section, state, 'FOCUS', title=title)
        if not sections:
            layout.label(text='No additional settings for this stage in the current mode.', icon='INFO')
        navigation = layout.row(align=True)
        keys = [item[0] for item in STAGES]
        index = keys.index(state.stage)
        if index:
            _nav_button(navigation, 'FOCUS', keys[index - 1], 'Previous', 'BACK', mode='STAGE')
        if index < len(keys) - 1:
            _nav_button(navigation, 'FOCUS', keys[index + 1], 'Next', 'FORWARD', mode='STAGE')


VARIANT_UI_CLASSES = [SGPanelVariantState, SG_OT_PanelNavigate, SG_OT_PanelSection,
                      SG_OT_PanelCamera, SG_UL_PanelCameras,
                      SG_MT_WorkbenchPages,
                      SG_PT_Studio, SG_PT_Workbench, SG_PT_Focus]


def register_variant_properties():
    for name in STATE_NAMES.values():
        setattr(bpy.types.WindowManager, name, bpy.props.PointerProperty(type=SGPanelVariantState, options={'SKIP_SAVE'}))


def unregister_variant_properties():
    for name in STATE_NAMES.values():
        if hasattr(bpy.types.WindowManager, name):
            delattr(bpy.types.WindowManager, name)
