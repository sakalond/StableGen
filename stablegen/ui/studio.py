"""Custom Studio sidebar, alongside the existing native StableGen panels.

GPU drawing lives only in this tab's UI region. A lightweight input operator
passes viewport and native-tab events through. Every workflow command invokes
its existing Blender operator; every setting writes its original RNA property.
"""
import json
import logging
from pathlib import Path
import re
import time

import bpy
import blf
from bpy.app.handlers import persistent

from . import studio_model as model
from . import studio_diagnostics as diagnostics
from . import studio_hierarchy as hierarchy_model
from . import studio_cards as cards
from . import studio_help as help_text
from . import studio_actions as action_style
from . import studio_lora as lora
from . import studio_tips as tips
from .presentation import Layout, Control, Section, STRUCTURAL_PROPERTIES, TOOL_SECTIONS
from .studio_text import TextBuffer, wrap
from .studio_widgets import (Painter, FONT_SIZE, CONTROL_HEIGHT, CONTROL_GAP,
                             TEXT_LINE_HEIGHT, TEXT_PADDING, font_size)
from .presets import PRESETS
from ..utils import sg_modal_active
from ..core import ADDON_PKG, legacy_panel_enabled
from ..core.generation_status import WAITING_STAGES
from ..core.architectures import ARCHITECTURE_ITEMS, QWEN_ARCHITECTURES, architecture_status

TAB = 'StableGen'
PAGES = [('GENERATE', 'Texture'), ('CAMERAS', 'Cameras'), ('BLEND', 'Blending')]
MORE = [('TOOLS', 'Tools'), ('QUEUE', 'Scene queue'), ('DEBUG', 'Diagnostics'),
        ('MESH', 'Mesh & rig'), ('PRINT', '3D printing'), ('EXPERIMENTS', 'Experiments')]
# Architectures that generate the asset itself rather than texturing an existing mesh.
ASSET_ARCHITECTURES = {'trellis2'}


ARCHITECTURE_PROPERTIES = {'architecture_mode', 'model_architecture', 'trellis2_texture_mode', 'trellis2_initial_image_arch'}


def _architecture_tags(prop, identifier):
    """Tags for an architecture choice, drawn right to left: its role, then its support status."""
    if prop not in ARCHITECTURE_PROPERTIES:
        return []
    tags = ['Asset generation' if identifier in ASSET_ARCHITECTURES else 'Texturing'] if prop == 'architecture_mode' else []
    status = architecture_status(identifier)
    return tags + [status] if status else tags


# Preset names repeat their model and variant, which the panel shows as tags instead: it lists only the current
# architecture's presets, so the model name is redundant there.
PRESET_MODEL_NAMES = {'qwen_image_edit': 'qwen', 'qwen_image_21': 'qwen 2.1', 'qwen_image_21_turbo': 'qwen 2.1 turbo',
                      'fibo_edit': 'fibo', 'moss_image': 'moss', 'flux2_klein': 'klein'}
PRESET_VARIANTS = {'nunchaku': 'Nunchaku', 'experimental': 'Experimental'}
PRESET_WORKFLOWS = ('New texture', 'Refine', 'Local edit', 'UV inpaint', 'Mesh + texture', 'Mesh only')


def _preset_workflow(preset):
    if preset.get('architecture_mode') == 'trellis2':
        return 'Mesh only' if preset.get('trellis2_texture_mode') == 'none' else 'Mesh + texture'
    if preset.get('generation_method') == 'uv_inpaint':
        return 'UV inpaint'
    if 'local_edit' in (preset.get('generation_method'), preset.get('qwen_generation_method')):
        return 'Local edit'
    if preset.get('qwen_generation_method') == 'refine':
        return 'Refine'
    return 'New texture'


def _preset_model(preset):
    if preset.get('architecture_mode') == 'trellis2':
        return preset.get('trellis2_texture_mode', 'none')
    return preset.get('model_architecture', 'sdxl')


def _strip(label, word):
    lower = label.casefold()
    if lower.endswith(' (' + word + ')'):
        return label[:-len(word) - 3], True
    if lower.startswith(word + ' '):
        return label[len(word) + 1:], True
    return label, False


def _preset_label(identifier, label):
    preset = PRESETS.get(identifier)
    if not preset:
        return label
    trellis = preset.get('architecture_mode') == 'trellis2'
    for word in ('mesh + texture', 'mesh only', *PRESET_VARIANTS) if trellis else PRESET_VARIANTS:
        label = _strip(label, word)[0]
    name = PRESET_MODEL_NAMES.get(_preset_model(preset))
    if name:
        label = _strip(label, name)[0]
    if label.startswith('Uv '):         # title() case of UV
        label = 'UV ' + label[3:]
    # TRELLIS.2 (MESH ONLY) is the stock mesh preset once its tag says what it does.
    return 'Default' if trellis and label.casefold() == 'trellis.2' else label


def _preset_tags(identifier):
    """Tags for a preset, drawn right to left: its workflow, then its texture model (TRELLIS.2) and variant."""
    preset = PRESETS.get(identifier)
    if not preset:
        return []
    tags = [_preset_workflow(preset)]
    if preset.get('architecture_mode') == 'trellis2' and preset.get('trellis2_texture_mode') not in (None, 'none', 'native'):
        tags.append(_architecture_label(preset['trellis2_texture_mode']))
    tags += [tag for word, tag in PRESET_VARIANTS.items() if _strip(identifier, word)[1]]
    return tags


def _architecture_label(identifier):
    return next((item[1] for item in ARCHITECTURE_ITEMS if item[0] == identifier), identifier)


def _choice_tags(prop, identifier):
    return _preset_tags(identifier) if prop == 'stablegen_preset' else _architecture_tags(prop, identifier)


def _draw_tags(p, tags, right, y, enabled=True):
    """Draw *tags* ending at *right*; returns the width they take, gaps included."""
    used = 0
    for tag in tags:
        used += p.tag(tag, right - used, y, enabled=enabled) + 4
    return used


PROJECTION_KEYS = {'discard_factor', 'weight_exponent', 'early_priority', 'early_priority_strength',
                   'bake_visibility_weights', 'sg_texturing_frame', 'sg_allow_backface_projection',
                   'sg_rework_acquisition_quality', 'sg_rework_projection_guard', 'sg_guard_directional',
                   'sg_silhouette_margin', 'sg_guard_foreground_margin', 'sg_silhouette_depth',
                   'sg_guard_coverage_rescue', 'sg_silhouette_feather', 'sg_projection_guard_scale',
                   'sg_guard_feather_method'}
SECTIONS = {'show_core_settings': 'Sampling & resolution', 'show_lora_settings': 'LoRA',
            'show_image_guidance_settings': 'Image guidance', 'show_mode_specific_settings': 'Mode settings',
            'show_scene_understanding_settings': 'Viewpoint blending',
            'show_output_material_settings': 'Output & materials', 'debug': 'Diagnostics'}
_sessions = {}
_cursor_owners = {}
_window_mouse = {}
_pending = set()
_handler = None
_registered = False
_font_id = None
_font_path = ''
_log = logging.getLogger(__name__)


def _context(session):
    return bpy.context.temp_override(window=session.window, area=session.area, region=session.region)


def _viewport(session):
    region = next(r for r in session.area.regions if r.type == 'WINDOW')
    return bpy.context.temp_override(window=session.window, area=session.area, region=region)


def _undo(message):
    try:
        bpy.ops.ed.undo_push(message=message)
    except RuntimeError:
        # File loading and a closing area can invalidate an otherwise valid edit.
        _log.debug('Undo push unavailable', exc_info=True)


def _binding_key(binding):
    return f'{binding.root}:{binding.name}:{binding.path}:{binding.property}'


class StudioSession:
    def __init__(self, context):
        self.window, self.area, self.region = context.window, context.area, context.region
        self.key = (self.window.as_pointer(), self.area.as_pointer())
        self.page = 'MESH' if context.scene.architecture_mode == 'trellis2' else 'GENERATE'
        self.architecture = context.scene.architecture_mode
        self.open = {'show_scene_understanding_settings', 'show_core_settings'}
        self.scroll = {}
        self.text_scroll = {}
        self.text_fields = {}
        self.prompt_editor = None
        self.targets = []
        self.pointer = (-1, -1)
        self.mouse = (-1, -1)
        self.pointer_valid = False
        self.hover_key, self.hover_since = '', time.monotonic()
        self.hover_signature = None
        self.hover_origin = self.pointer
        self.tooltip_key = ''
        self.tooltip_rect = None
        self.cursor = ''
        self.last_refresh = 0
        self.edit = None
        self.drag = None
        self.camera_drag = None
        self.camera_rows = []
        self.camera_clip = None
        self.menu = None
        self.query = ''
        self.popup = False
        self.alive = True
        self.error = ''
        self.notice = ''
        self.notice_until = 0
        self.dirty_projection = False
        self.last_projection = None
        self.scene_pointer = context.scene.as_pointer()
        self.document = None
        self.hierarchies = {}
        self.busy = False
        self.ctrl = False
        self.camera_images = {}
        self.card_bounds = {}
        # Narrowest width while the widen callout was shown in this session.
        self.callout_width = None

    def valid(self):
        try:
            # Choosing the legacy panel in preferences ends every Studio session.
            return (self.alive and not legacy_panel_enabled()
                    and self.window in bpy.context.window_manager.windows[:]
                    and self.area in self.window.screen.areas[:] and self.area.type == 'VIEW_3D'
                    and self.region in self.area.regions[:])
        except ReferenceError:
            return False

    def active(self):
        return self.valid() and self.area.spaces.active.show_region_ui and self.region.active_panel_category == TAB

    def native_resize_active(self):
        """Blender owns hover and the cursor while moving a region or area edge."""
        return any(op is not None and op.bl_idname in {'SCREEN_OT_region_scale', 'SCREEN_OT_area_move'}
                   for op in getattr(self.window, 'modal_operators', ()))

    def release_cursor(self):
        """Restore the native cursor only if this sidebar owns the window override."""
        if _cursor_owners.get(self.key[0]) is self:
            try:
                self.window.cursor_modal_restore()
            except (ReferenceError, RuntimeError):
                pass
            _cursor_owners.pop(self.key[0], None)
        self.cursor = ''

    def set_cursor(self, cursor):
        owner = _cursor_owners.get(self.key[0])
        if owner is not None and owner is not self:
            owner.release_cursor()
        if cursor == 'DEFAULT':
            self.release_cursor()
            # The canvas consumes native hover events, so Blender cannot clear
            # its resize cursor when the pointer enters our controls or gaps.
            self.window.cursor_set('DEFAULT')
            return
        if self.cursor != cursor:
            # Modal restore must not revive a resize cursor saved on entry.
            self.window.cursor_set('DEFAULT')
            self.window.cursor_modal_set(cursor)
            self.cursor = cursor
            _cursor_owners[self.key[0]] = self

    def clear_hover(self, *, leave=False):
        changed = bool(self.hover_key or self.tooltip_key)
        self.hover_key, self.hover_signature = '', None
        self.hover_since = time.monotonic()
        self.hover_origin = self.pointer
        self.tooltip_key, self.tooltip_rect = '', None
        if leave:
            self.pointer_valid = False
            self.release_cursor()
        if changed and self.valid():
            self.area.tag_redraw()

    def pointer_inside(self):
        if not self.pointer_valid:
            return False
        # A modal handler in another Studio area may consume the move first.
        self.mouse = _window_mouse.get(self.key[0], self.mouse)
        scale = max(.8, bpy.context.preferences.system.ui_scale)
        self.pointer = ((self.mouse[0] - self.region.x) / scale,
                        (self.region.y + self.region.height - self.mouse[1]) / scale)
        x, y = self.pointer
        # Leave native category tabs and the sidebar resize border to Blender.
        return 8 < x < self.region.width / scale - 28 and 0 <= y < self.region.height / scale

    def hit_target(self, targets=None):
        return next((t for t in reversed(self.targets if targets is None else targets)
                     if t.contains(*self.pointer) and (not self.menu or t.key.startswith('choice:'))), None)

    def update_hover(self, targets=None, *, moved=False, cursor=True):
        if not self.active() or self.popup or self.native_resize_active():
            self.clear_hover(leave=True)
            return None
        inside = self.pointer_inside()
        if self.camera_drag or self.drag or self.edit and self.edit['selecting']:
            self.clear_hover()
            if cursor:
                self.set_cursor('MOVE_Y' if self.camera_drag else 'MOVE_X' if self.drag else 'TEXT')
            return None
        if not inside:
            self.clear_hover()
            if cursor:
                self.release_cursor()
            return None
        target = self.hit_target(targets)
        signature = (target.key, target.rect, target.clip, target.hint, target.enabled) if target else None
        distance = ((self.pointer[0] - self.hover_origin[0]) ** 2
                    + (self.pointer[1] - self.hover_origin[1]) ** 2)
        if signature != self.hover_signature or moved and distance > 9:
            self.clear_hover()
            self.hover_key = target.key if target else ''
            self.hover_signature = signature
            if target:
                self.area.tag_redraw()
        if cursor:
            self.set_cursor('MOVE_X' if self.drag else target.cursor if target else 'DEFAULT')
        return target

    def tooltip_target(self, targets=None):
        if (not bpy.context.preferences.view.show_tooltips or self.camera_drag or self.drag or self.popup
                or self.notice and time.monotonic() < self.notice_until):
            return None
        target = self.hit_target(targets) if self.pointer_inside() else None
        if (target and target.key == self.hover_key and target.hint
                and not (self.edit and self.edit['key'] == target.key)
                and time.monotonic() - self.hover_since >= .65):
            return target
        return None

    def tick(self, now):
        if not self.active() or self.popup:
            self.camera_drag = None
            self.clear_hover(leave=True)
            return
        if now - self.last_refresh >= .25:
            self.refresh()
        self.update_hover()
        if self.camera_drag and self.camera_clip and self.pointer_inside():
            _, top, _, height = self.camera_clip
            dy = -10 if self.pointer[1] < top + 24 else 10 if self.pointer[1] > top + height - 24 else 0
            if dy:
                self.scroll['CAMERAS'] = max(0, min(getattr(self, 'max_scroll', 0), self.scroll.get('CAMERAS', 0) + dy))
                self.area.tag_redraw()
        tooltip = self.tooltip_target()
        if (tooltip.key if tooltip else '') != self.tooltip_key:
            self.area.tag_redraw()

    def refresh(self):
        self.last_refresh = time.monotonic()
        with _context(self):
            ctx = bpy.context
            if ctx.scene.architecture_mode != self.architecture:
                previous_architecture = self.architecture
                self.architecture = ctx.scene.architecture_mode
                if self.architecture == 'trellis2' or previous_architecture == 'trellis2':
                    self.page = 'MESH' if self.architecture == 'trellis2' else 'GENERATE'
            if ctx.scene.as_pointer() != self.scene_pointer:
                self.edit = self.drag = self.menu = None
                self.camera_drag = None
                self.scene_pointer = ctx.scene.as_pointer()
                self.last_projection = None
                self.dirty_projection = False
                self.clear_hover(leave=True)
            self.document = model.document(ctx)
            self.hierarchies.clear()
            if self.page == 'CAMERAS':
                self.camera_images = model.projected_images(ctx)
            self.busy = sg_modal_active(ctx) or any(c.control.method == 'progress'
                                                  for n in self.document.actions for c in model.controls(n))
            values = tuple(getattr(ctx.scene, key, None) for key in sorted(PROJECTION_KEYS))
            if self.last_projection is not None and values != self.last_projection and not self.busy:
                self.dirty_projection = bool(ctx.scene.output_timestamp)
            self.last_projection = values
            if self.busy and self.edit:
                self.edit = None
            if (self.edit and 'lora_snapshot' in self.edit
                    and self.edit['lora_snapshot'] != lora.snapshot(ctx.scene)):
                self.edit = None
            if self.menu and self.menu.get('kind') == 'lora' and not lora.valid_menu(self, self.menu):
                self.menu = None
            if self.camera_drag and (self.busy or self.camera_drag['names'] != tuple(c.name for c in self.ordered_cameras())):
                self.camera_drag = None
        self.area.tag_redraw()

    def tell(self, message):
        self.notice, self.notice_until = message, time.monotonic() + 5
        self.area.tag_redraw()

    def call(self, name, properties=None, *, mode='INVOKE_DEFAULT'):
        display = next((key for key, operator in diagnostics.OPERATORS.items()
                        if name == 'stablegen.' + operator), None)
        if display:
            return self.call('stablegen.studio_diagnostic', {'value': json.dumps(display)}, mode='EXEC_DEFAULT')
        self.commit()
        self.clear_hover(leave=True)
        try:
            with _viewport(self):
                op = model.operator(name)
                if not op.poll():
                    self.tell('This command is unavailable for the current scene or operation.')
                    return
                if diagnostics.active(bpy.context.scene) and name not in {
                        'stablegen.studio_diagnostic', 'stablegen.debug_open_projection_line',
                        'stablegen.studio_preferences', 'stablegen.export_retarget_debug_bundle'}:
                    diagnostics.restore(bpy.context.scene)
                result = op(mode, **(properties or {}))
                if name == 'stablegen.studio_diagnostic' and 'FINISHED' in result:
                    _undo('Change diagnostic display')
                if name == 'stablegen.studio_lora' and 'FINISHED' in result:
                    operation = (properties or {}).get('operation', 'REPLACE').title()
                    _undo(operation + ' LoRA adapter')
                if name in {'object.stablegen_reproject', 'object.test_stable', 'object.stablegen_regenerate'} and 'CANCELLED' not in result:
                    self.dirty_projection = False
            self.refresh()
            return result
        except (RuntimeError, TypeError, ValueError) as exc:
            self.tell(str(exc).split('\n')[0])
            _log.warning('Studio command %s: %s', name, exc)

    def native(self, section='', binding=None, index=-1):
        self.commit()
        self.menu = None
        self.clear_hover(leave=True)
        with _context(self):
            bpy.ops.stablegen.studio_native('INVOKE_DEFAULT', section=section,
                                         binding=json.dumps(binding.__dict__) if binding else '',
                                         control_index=index)

    def commit(self, cancel=False):
        edit, self.edit = self.edit, None
        if not edit or cancel:
            return
        if edit['key'] == 'search':
            self.query = edit['buffer'].text
            return
        try:
            with _context(self):
                if 'lora_snapshot' in edit and edit['lora_snapshot'] != lora.snapshot(bpy.context.scene):
                    raise ReferenceError('The adapter list changed while this value was being edited.')
                binding, text = edit['binding'], edit['buffer'].text
                prop = binding.rna(bpy.context)
                value = float(text) if prop.type == 'FLOAT' else int(text) if prop.type == 'INT' else text
                if binding.value(bpy.context) != value:
                    self.write_binding(binding, value)
            self.refresh()
        except (ValueError, TypeError, ReferenceError, RuntimeError) as exc:
            self.tell('Could not apply this value: ' + str(exc).split('\n')[0])

    def set_value(self, binding, value, undo=True):
        with _context(self):
            if self.busy:
                return
            self.write_binding(binding, value, undo)
        self.refresh()

    def write_binding(self, binding, value, undo=True):
        if binding.path == 'sg_studio_diagnostics':
            with _viewport(self):
                bpy.ops.stablegen.studio_diagnostic('EXEC_DEFAULT', setting=binding.property,
                                                   value=json.dumps(value))
                _undo('Change diagnostic display')
        else:
            binding.set(bpy.context, value)
            if undo:
                _undo('Change ' + binding.rna(bpy.context).name)

    def begin_edit(self, target, select=False):
        self.commit()
        binding = target.action[1]
        with _context(self):
            value = self.query if binding is None else binding.value(bpy.context)
        buffer = TextBuffer(str(value))
        if select:
            buffer.anchor = 0
        self.edit = {'key': target.key, 'binding': binding, 'buffer': buffer,
                     'rect': target.rect, 'lines': self.text_fields.get(target.key, ([],0))[0],
                     'scroll': self.text_scroll.get(target.key, 0), 'follow_caret': False,
                     'multiline': bool(target.action[2]), 'selecting': False}
        if binding and binding.path.startswith('lora_units['):
            self.edit['lora_snapshot'] = lora.snapshot(bpy.context.scene)
        self.menu = None
        self.area.tag_redraw()

    def enum_menu(self, target):
        self.commit()
        binding = target.action[1]
        with _context(self):
            items = list(model.enum_items(binding, bpy.context))
            if binding.property == 'stablegen_preset':
                arch = bpy.context.scene.architecture_mode
                items = [item for item in items if item and item[0] and
                         (item[0] == 'CUSTOM' or (item[0] in PRESETS and
                          (PRESETS[item[0]].get('architecture_mode') or PRESETS[item[0]].get('model_architecture', 'sdxl')) == arch))]
            value = binding.value(bpy.context)
        items = [(item[0], item[1], item[2] if len(item) > 2 else '') for item in items if item and item[0]]
        if binding.property == 'stablegen_preset':
            # Group by workflow (new texture, refine, local edit, ...); Custom stays last.
            order = {w: n for n, w in enumerate(PRESET_WORKFLOWS)}
            items = sorted(((i, _preset_label(i, label), d) for i, label, d in items),
                           key=lambda item: order.get(_preset_workflow(PRESETS[item[0]]), 99) if item[0] in PRESETS else 100)
        if not items:
            self.native(binding=binding)
            return
        self.menu = {'anchor': target.rect, 'items': items, 'query': '', 'binding': binding,
                     'index': next((i for i, item in enumerate(items) if item[0] == value), 0), 'scroll': 0}

    def navigate(self, page):
        self.commit()
        self.camera_drag = None
        self.page, self.menu = page, None
        self.clear_hover()
        if page == 'SEARCH':
            buffer = TextBuffer(self.query)
            buffer.anchor = 0
            self.edit = {'key': 'search', 'binding': None, 'buffer': buffer,
                         'rect': (0,0,0,0), 'lines': [], 'scroll': 0,
                         'multiline': False, 'selecting': False}
        self.area.tag_redraw()

    def expand_prompt(self, binding, label):
        previous = self.page
        self.navigate('EDITOR')
        self.prompt_editor = (binding, label, previous)
        self.scroll['EDITOR'] = 0

    def activate(self, target):
        self.clear_hover()
        if not target or not target.enabled:
            return
        action = target.action
        if callable(action):
            self.commit()
            action()
        elif action[0] == 'operator':
            self.call(action[1], action[2])
        elif action[0] == 'toggle':
            self.commit()
            with _context(self):
                value = not action[1].value(bpy.context)
            self.set_value(action[1], value)
        elif action[0] == 'edit':
            self.begin_edit(target, select=not action[2])
        elif action[0] == 'enum':
            self.enum_menu(target)
        elif action[0] == 'section':
            self.commit()
            if action[1] in self.open:
                self.open.remove(action[1])
            else:
                self.open.add(action[1])
        elif action[0] == 'native':
            self.native(action[1], index=action[2])
        elif action[0] == 'slider':
            self.commit()
            with _context(self):
                before = action[1].value(bpy.context)
            self.drag = {'binding': action[1], 'min': action[2], 'max': action[3],
                         'rect': target.rect, 'before': before}
            self.slide(self.pointer[0])
        elif action[0] == 'camera-drag':
            self.commit()
            self.camera_drag = {'name': action[1], 'names': tuple(c.name for c in self.ordered_cameras())}
        elif action[0] == 'menu-item':
            menu = self.menu
            if action[1] in menu.get('disabled', ()):
                return
            self.menu = None
            if menu.get('binding'):
                self.set_value(menu['binding'], action[1])
                if menu['binding'].property == 'stablegen_preset' and action[1] != 'CUSTOM':
                    self.call('stablegen.apply_preset', mode='EXEC_DEFAULT')
            elif menu.get('callback'):
                menu['callback'](action[1])
            else:
                self.navigate(action[1])
        elif action[0] == 'menu-lora-close':
            self.menu = None
        elif action[0] == 'menu-lora-search':
            lora.select_search(self)
        elif action[0] == 'menu-lora-folder':
            lora.folders(self)
        self.area.tag_redraw()

    def slide(self, x):
        drag = self.drag
        if not drag or self.busy:
            return
        left, _, width, _ = drag['rect']
        value = drag['min'] + max(0, min(1, (x - left) / width)) * (drag['max'] - drag['min'])
        self.set_value(drag['binding'], value, undo=False)

    def ordered_cameras(self):
        from ..texturing.view_plan import order_cameras
        scene = bpy.context.scene
        return order_cameras((o for o in scene.objects if o.type == 'CAMERA'),
                             [item.name for item in scene.sg_camera_order] if scene.sg_use_custom_camera_order else ())

    def camera_drop(self, y):
        """Insertion boundary among the remaining rows; no scene edits during drag."""
        rows = [(name, rect) for name, rect in self.camera_rows if name != self.camera_drag['name']]
        for name, rect in rows:
            if y < rect[1] + rect[3] / 2:
                return name, rect[1] - 4
        return '', rows[-1][1][1] + rows[-1][1][3] + 4 if rows else y

    def finish_camera_drag(self, cancel=False):
        drag = self.camera_drag
        if not drag:
            return
        before, _ = self.camera_drop(self.pointer[1])
        self.camera_drag = None
        # Leaving the canvas cancels, as do scene edits while a drag is pending.
        if (not cancel and not self.busy and self.pointer_inside() and self.camera_clip
                and self.camera_clip[1] <= self.pointer[1] <= self.camera_clip[1] + self.camera_clip[3]
                and drag['names'] == tuple(c.name for c in self.ordered_cameras())):
            self.call('stablegen.studio_reorder_camera', {'camera_name': drag['name'], 'before_name': before}, mode='EXEC_DEFAULT')
        self.update_hover()
        self.area.tag_redraw()

    def camera_order_menu(self, rect):
        self.commit()
        prop = model.operator('stablegen.apply_camera_order_preset').get_rna_type().properties['strategy']
        self.menu = {'anchor': rect, 'items': [(i.identifier, i.name, i.description) for i in prop.enum_items],
                     'query': '', 'binding': None, 'index': 0, 'scroll': 0,
                     'callback': lambda strategy: self.call('stablegen.studio_sort_cameras', {'strategy': strategy}, mode='EXEC_DEFAULT')}

    def select_caret(self, x, y, extend=False):
        edit = self.edit
        if not edit or not edit['lines']:
            return
        rx, ry, width, height = edit['rect']
        row = int((y - ry - TEXT_PADDING) / TEXT_LINE_HEIGHT) if edit['multiline'] else 0
        index = max(0, min(len(edit['lines']) - 1, row + edit['scroll']))
        start, end, line = edit['lines'][index]
        # BLF uses the same loaded font and pixel scale as the displayed editor.
        blf.size(_font_id, font_size() * self.scale)
        distances = [blf.dimensions(_font_id, line[:i])[0] / self.scale for i in range(len(line) + 1)]
        caret = min(range(len(distances)), key=lambda i: abs(distances[i] - (x - rx - TEXT_PADDING))) + start
        edit['buffer'].move(caret, select=extend)
        edit['follow_caret'] = True

    def text_event(self, event):
        edit, key = self.edit, event.type
        if not edit or event.value != 'PRESS':
            return False
        buffer = edit['buffer']
        ctrl = event.ctrl or event.oskey
        if key == 'ESC':
            self.commit(cancel=True)
        elif key in {'RET', 'NUMPAD_ENTER'}:
            if edit['multiline'] and not ctrl:
                buffer.replace('\n')
            else:
                self.commit()
        elif ctrl and key == 'A':
            buffer.anchor, buffer.caret = 0, len(buffer.text)
        elif ctrl and key in {'C', 'X'}:
            start, end = buffer.selection
            if start != end:
                self.window_manager.clipboard = buffer.text[start:end]
                if key == 'X':
                    buffer.replace('')
        elif ctrl and key == 'V':
            text = self.window_manager.clipboard
            buffer.replace(text if edit['multiline'] else text.replace('\n', ' ').replace('\r', ''))
        elif ctrl and key in {'Z', 'Y'}:
            buffer.undo(redo=key == 'Y' or event.shift)
        elif key in {'LEFT_ARROW', 'RIGHT_ARROW'}:
            direction = -1 if key == 'LEFT_ARROW' else 1
            start, end = buffer.selection
            destination = (start if direction < 0 else end) if start != end and not event.shift else buffer.word(direction) if ctrl else buffer.caret + direction
            buffer.move(destination, event.shift)
        elif key in {'HOME', 'END'}:
            if ctrl:
                destination = 0 if key == 'HOME' else len(buffer.text)
            else:
                destination = buffer.text.rfind('\n', 0, buffer.caret) + 1 if key == 'HOME' else buffer.text.find('\n', buffer.caret)
                if destination < 0:
                    destination = len(buffer.text)
            buffer.move(destination, event.shift)
        elif key in {'UP_ARROW', 'DOWN_ARROW'} and edit['lines']:
            lines = edit['lines']
            row = next((i for i, (a, b, _) in enumerate(lines) if a <= buffer.caret <= b), len(lines) - 1)
            column = buffer.caret - lines[row][0]
            target = max(0, min(len(lines) - 1, row + (-1 if key == 'UP_ARROW' else 1)))
            buffer.move(min(lines[target][1], lines[target][0] + column), event.shift)
        elif key in {'BACK_SPACE', 'DEL'}:
            buffer.delete(backward=key == 'BACK_SPACE', word=ctrl)
        elif key == 'TAB':
            candidates = [t for t in self.targets if t.enabled and isinstance(t.action, tuple) and t.action[0] in {'edit', 'enum', 'toggle'}]
            index = next((i for i, target in enumerate(candidates) if target.key == edit['key']), -1)
            self.commit()
            if candidates:
                self.activate(candidates[(index + (-1 if event.shift else 1)) % len(candidates)])
        elif getattr(event, 'unicode', '') and not ctrl and not event.alt:
            buffer.replace(event.unicode)
        else:
            return False
        if self.edit:
            self.edit['follow_caret'] = True
        self.clear_hover()
        self.area.tag_redraw()
        return True

    @property
    def window_manager(self):
        return bpy.context.window_manager

    def event(self, event):
        self.ctrl = event.ctrl or event.oskey
        self.scale = max(.8, bpy.context.preferences.system.ui_scale)
        mouse_event = event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE', 'LEFTMOUSE', 'RIGHTMOUSE',
                                    'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN'}
        if event.type == 'WINDOW_DEACTIVATE':
            self.camera_drag = None
            _window_mouse.pop(self.key[0], None)
            self.clear_hover(leave=True)
            return {'PASS_THROUGH'}
        if mouse_event:
            self.mouse = (event.mouse_x, event.mouse_y)
            _window_mouse[self.key[0]] = self.mouse
            self.pointer_valid = True
        inside = self.pointer_inside()
        x, y = self.pointer
        if not self.active() or self.popup:
            self.camera_drag = None
            self.clear_hover(leave=True)
            if not self.active():
                self.commit()
                self.menu = None
            return {'PASS_THROUGH'}
        if self.camera_drag:
            if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
                self.finish_camera_drag(cancel=True)
                return {'RUNNING_MODAL'}
            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
                self.finish_camera_drag()
                return {'RUNNING_MODAL'}
            if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
                self.clear_hover()
                self.set_cursor('MOVE_Y')
                self.area.tag_redraw()
                return {'RUNNING_MODAL'}
            # Keep shortcuts and other commands from changing the scene mid-drag.
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and event.type == 'ESC' and self.drag:
            drag, self.drag = self.drag, None
            self.set_value(drag['binding'], drag['before'], undo=False)
            self.update_hover()
            return {'RUNNING_MODAL'}
        if self.edit and event.value == 'PRESS' and (event.ctrl or event.oskey) and event.type == 'S':
            self.commit()
            self.clear_hover(leave=True)
            return {'PASS_THROUGH'}
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            if self.drag:
                self.slide(x)
                self.clear_hover()
                self.set_cursor('MOVE_X')
                return {'RUNNING_MODAL'}
            if self.edit and self.edit['selecting']:
                self.select_caret(x, y, extend=True)
                self.set_cursor('TEXT')
                self.area.tag_redraw()
                return {'RUNNING_MODAL'}
            target = self.update_hover(moved=True)
            if target and target.enabled and self.menu and isinstance(target.action, tuple) and target.action[0] == 'menu-item':
                self.menu['index'] = next((i for i, item in enumerate(self.menu_items())
                                           if item[0] == target.action[1]), self.menu['index'])
            # The GPU canvas covers native controls; do not let those steal hover.
            return {'RUNNING_MODAL'} if inside else {'PASS_THROUGH'}
        if mouse_event and event.value in {'PRESS', 'DOUBLE_CLICK'}:
            self.clear_hover()
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if self.drag:
                drag, self.drag = self.drag, None
                with _context(self):
                    if drag['binding'].value(bpy.context) != drag['before']:
                        _undo('Adjust ' + drag['binding'].rna(bpy.context).name)
                self.update_hover()
                return {'RUNNING_MODAL'}
            if self.edit:
                self.edit['selecting'] = False
            self.update_hover()
            return {'RUNNING_MODAL'} if inside else {'PASS_THROUGH'}
        if self.edit and self.text_event(event):
            return {'RUNNING_MODAL'}
        if not inside:
            self.clear_hover()
            self.release_cursor()
            if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                self.commit()
                self.menu = None
            return {'PASS_THROUGH'}
        if self.menu:
            if event.value == 'PRESS' and event.type == 'ESC':
                self.menu = self.menu.get('parent')
                self.area.tag_redraw()
                self.update_hover()
                return {'RUNNING_MODAL'}
            if self.menu.get('kind') == 'lora' and lora.keyboard(self, event):
                return {'RUNNING_MODAL'}
            if event.value == 'PRESS' and event.type in {'UP_ARROW', 'DOWN_ARROW', 'RET', 'NUMPAD_ENTER', 'BACK_SPACE'} or (event.value == 'PRESS' and getattr(event, 'unicode', '')):
                self.clear_hover()
                items = self.menu_items()
                if event.type in {'UP_ARROW', 'DOWN_ARROW'}:
                    direction = -1 if event.type == 'UP_ARROW' else 1
                    available = [i for i, item in enumerate(items) if item[0] not in self.menu.get('disabled', ())]
                    choices = [i for i in available if (i - self.menu['index']) * direction > 0]
                    self.menu['index'] = (min(choices) if direction > 0 else max(choices)) if choices else self.menu['index']
                    self.menu['scroll'] = max(0, self.menu['index'] - self.menu.get('rows', 6) + 1)
                elif event.type in {'RET', 'NUMPAD_ENTER'} and items:
                    item = items[min(self.menu['index'], len(items) - 1)]
                    from .studio_widgets import Target
                    self.activate(Target('menu', (0, 0, 0, 0), ('menu-item', item[0]), item[0] not in self.menu.get('disabled', ())))
                elif event.type == 'BACK_SPACE':
                    self.menu['query'] = self.menu['query'][:-1]
                    self.menu['index'] = self.menu['scroll'] = 0
                elif getattr(event, 'unicode', ''):
                    self.menu['query'] += event.unicode
                    self.menu['index'] = self.menu['scroll'] = 0
                self.area.tag_redraw()
                return {'RUNNING_MODAL'}
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN'}:
            self.clear_hover()
            change = -45 if event.type == 'WHEELUPMOUSE' else 45
            if event.type == 'TRACKPADPAN':
                change = (event.mouse_prev_y - event.mouse_y) / self.scale
            if self.menu:
                maximum = max(0, len(self.menu_items()) - self.menu.get('rows', 6))
                self.menu['scroll'] = max(0, min(maximum, self.menu['scroll'] + (-1 if change < 0 else 1)))
            else:
                target = next((t for t in reversed(self.targets) if t.contains(x, y) and t.key in self.text_fields), None)
                # A prompt owns scrolling only while it is the active editor.
                if target and self.edit and self.edit['key'] == target.key:
                    lines, count = self.text_fields[target.key]
                    before = self.edit['scroll']
                    offset = max(0, min(len(lines) - count, before + (-2 if change < 0 else 2)))
                    self.text_scroll[target.key] = offset
                    self.edit.update(scroll=offset, follow_caret=False)
                else:
                    self.scroll[self.page] = max(0, min(self.max_scroll, self.scroll.get(self.page, 0) + change))
            self.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value in {'PRESS', 'DOUBLE_CLICK'}:
            target = self.hit_target()
            if self.menu and (target is None or not isinstance(target.action, tuple) or not target.action[0].startswith('menu-')):
                self.menu = self.menu.get('parent')
                self.area.tag_redraw()
                return {'RUNNING_MODAL'}
            if self.edit and target and target.key == self.edit['key']:
                self.select_caret(x, y, extend=event.shift)
                self.edit['selecting'] = True
            else:
                self.commit()
                self.activate(target)
                # Only a field's own click places the caret; buttons such as Search also start an edit.
                if (self.edit and target and isinstance(target.action, tuple) and target.action[0] == 'edit'
                        and target.action[2]):
                    self.select_caret(x, y)
                    self.edit['selecting'] = True
            self.area.tag_redraw()
            self.update_hover()
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and event.type == 'ESC' and self.edit:
            self.commit(cancel=True)
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and (event.ctrl or event.oskey) and event.type == 'K':
            self.navigate('SEARCH')
            return {'RUNNING_MODAL'}
        # Keep the canvas panel's covered header from collapsing (A, Enter)
        # or being pinned to other tabs from its context menu.
        if event.value == 'PRESS' and (event.type == 'RIGHTMOUSE' or event.type in {'A', 'RET', 'NUMPAD_ENTER'}
                                       and not (event.ctrl or event.oskey or event.alt or event.shift)):
            return {'RUNNING_MODAL'}
        if (event.value == 'PRESS' and not mouse_event and event.type not in
                {'LEFT_SHIFT', 'RIGHT_SHIFT', 'LEFT_CTRL', 'RIGHT_CTRL', 'LEFT_ALT', 'RIGHT_ALT', 'OSKEY'}):
            # Native shortcuts can open dialogs that have their own cursor handling.
            self.clear_hover(leave=True)
        return {'PASS_THROUGH'}

    def hovered(self, key):
        return key == self.hover_key

    def draw_binding(self, painter, binding, label, x, y, width, *, enabled=True, multiline=False, height=112, corners='ALL'):
        ctx = bpy.context
        prop, value = binding.rna(ctx), binding.value(ctx)
        hint = help_text.property_hint(binding, ctx)
        label_y = y
        key = _binding_key(binding)
        enabled = enabled and not self.busy and not prop.is_readonly
        if prop.type == 'BOOLEAN':
            lines = wrap(label, width - 40, painter.measure)
            height = max(CONTROL_HEIGHT, len(lines) * 16 + 6)
            painter.switch(key, (x, y, width, height), label, value, ('toggle', binding),
                           enabled=enabled, hint=hint, hover=self.hovered(key))
            return height + CONTROL_GAP
        inline = (prop.type in {'INT', 'FLOAT'} and not prop.is_array and width >= 210
                  and not multiline and painter.measure(label) <= width - 100)
        label_height = 0
        if inline:
            painter.text(label, x, y + 5, FONT_SIZE, 'text' if enabled else 'disabled', width - 100)
        else:
            label_width = max(30, width - 56 if multiline else width - 12)
            labels = wrap(label, label_width, painter.measure)
            for i, (_, _, line) in enumerate(labels):
                painter.text(line, x, y + i * 14, FONT_SIZE, 'text' if enabled else 'disabled')
            label_height = len(labels) * 14 + 3
        if multiline and self.page != 'EDITOR':
            painter.button(key + ':expand', (x + width - 22, y - 4, 22, CONTROL_HEIGHT), '',
                           lambda: self.expand_prompt(binding, label), icon='expand', enabled=enabled,
                           hover=self.hovered(key + ':expand'), hint='Expand this prompt editor')
        y += label_height
        editing = self.edit if self.edit and self.edit['key'] == key else None
        if multiline:
            display = editing['buffer'].text if editing else value
            lines, scroll = painter.text_field(key, (x, y, width, height), display,
                                              ('edit', binding, True), enabled=enabled,
                                              hint=hint, editing=editing['buffer'] if editing else None,
                                              scroll=editing['scroll'] if editing else self.text_scroll.get(key, 0), multiline=True,
                                              follow_caret=editing.get('follow_caret', True) if editing else False)
            self.text_scroll[key] = scroll
            self.text_fields[key] = (lines, max(1, int((height - TEXT_PADDING * 2) / TEXT_LINE_HEIGHT)))
            if editing:
                editing.update(rect=(x, y, width, height), lines=lines, scroll=scroll, follow_caret=False)
            painter.text(f'{len(str(display).split())} words', x + width - 55, y + height + 5, 10, 'muted')
            painter.text('Ctrl Enter to finish' if editing else 'Click to edit · Enter adds a line', x, y + height + 5, 10, 'muted', width - 68)
            painter.target(key + ':label', (x, label_y, label_width, label_height),
                           ('hint', None, False), False, hint)
            return height + label_height + 24
        if prop.type == 'ENUM':
            label_value = model.enum_label(binding, ctx)
            if binding.property == 'stablegen_preset':
                label_value = _preset_label(value, label_value)
            painter.outline(x, y, width, CONTROL_HEIGHT, fill='menu',
                            border='hover_border' if enabled and self.hovered(key) else 'menu_border', corners=corners)
            text_width = width - 33
            tags = _choice_tags(binding.property, value)
            # In a narrow field an architecture drops its role tag before its support status; a preset keeps its
            # workflow tag longest.
            while tags and painter.measure(label_value) + sum(painter.measure(t, 9) + 14 for t in tags) + 40 > width:
                tags.pop(-1 if binding.property == 'stablegen_preset' else 0)
            text_width -= _draw_tags(painter, tags, x + width - 24, y + 4, enabled)
            painter.text(label_value, x + 8, y + 5, FONT_SIZE, 'menu_text' if enabled else 'disabled', text_width)
            painter.icon('down', x + width - 16, y + 6, 10, 'menu_text' if enabled else 'disabled')
            painter.target(key, (x, y, width, CONTROL_HEIGHT), ('enum', binding), enabled, hint)
        elif prop.type in {'STRING', 'INT', 'FLOAT'} and not getattr(prop, 'is_array', False):
            display = editing['buffer'].text if editing else f'{value:.{prop.precision}f}'.rstrip('0').rstrip('.') if isinstance(value, float) and prop.precision > 0 else str(value)
            filepath = prop.type == 'STRING' and prop.subtype in {'FILE_PATH', 'DIR_PATH'}
            field_width = 90 if inline else width - 27 if filepath else width
            field_x = x + width - field_width if inline else x
            lines, scroll = painter.text_field(key, (field_x, y, field_width, CONTROL_HEIGHT), display,
                                              ('edit', binding, False), enabled=enabled, hint=hint,
                                              editing=editing['buffer'] if editing else None,
                                              kind='number' if prop.type in {'INT','FLOAT'} else 'field')
            if editing:
                editing.update(rect=(field_x, y, field_width, CONTROL_HEIGHT), lines=lines, scroll=scroll)
            if filepath:
                painter.button(key + ':browse', (x + width - 24, y, 24, CONTROL_HEIGHT), '',
                               lambda: self.native(binding=binding), icon='folder', enabled=enabled,
                               hover=self.hovered(key + ':browse'), hint='Browse with Blender\'s file picker')
        else:
            color = getattr(prop, 'subtype', '') in {'COLOR', 'COLOR_GAMMA'} and prop.is_array
            painter.button(key, (x, y, width - 34 if color else width, CONTROL_HEIGHT), 'Open color picker' if color else 'Edit ' + label, lambda: self.native(binding=binding),
                           icon='settings', style='outlined', enabled=enabled, hint=hint)
            if color:
                rgb = [max(0, min(1, c)) for c in value[:3]]
                if prop.subtype == 'COLOR':
                    rgb = [12.92*c if c <= .0031308 else 1.055*c**(1/2.4)-.055 for c in rgb]
                painter.rect(x + width - 27, y + 2, 24, CONTROL_HEIGHT - 4, (*rgb, 1), 3)
                painter.target(key + ':color', (x + width - 30, y, 30, CONTROL_HEIGHT),
                               lambda: self.native(binding=binding), enabled, hint)
        # Help-only bounds are disjoint from editable fields. They do not start
        # edits, join keyboard traversal, or capture prompt scrolling.
        painter.target(key + ':label', (x, label_y, width - 100 if inline else label_width,
                                       CONTROL_HEIGHT if inline else label_height),
                       ('hint', None, False), False, hint)
        return label_height + CONTROL_HEIGHT + CONTROL_GAP

    def binding(self, name):
        return model.Binding.from_data(bpy.context.scene, name)

    def draw_slider(self, p, name, label, x, y, width, minimum, maximum, unit='', *, enabled=True):
        binding = self.binding(name)
        value = binding.value(bpy.context)
        key = _binding_key(binding)
        enabled = enabled and not self.busy and not binding.rna(bpy.context).is_readonly
        narrow = width < 160
        p.text(label, x, y, FONT_SIZE, 'text' if enabled else 'disabled', width if narrow else width - 75)
        edit_width = 60
        ex, ey = x + width - edit_width, y + 16 if narrow else y - 4
        editing = self.edit if self.edit and self.edit['key'] == key else None
        precision = binding.rna(bpy.context).precision
        number = f'{value:.{precision}f}'
        display = editing['buffer'].text if editing else (number.rstrip('0').rstrip('.') if precision else number) + unit
        lines, scroll = p.text_field(key, (ex, ey, edit_width, CONTROL_HEIGHT), display, ('edit', binding, False),
                                    enabled=enabled, hint=binding.rna(bpy.context).description,
                                    editing=editing['buffer'] if editing else None, kind='number')
        if editing:
            editing.update(rect=(ex, ey, edit_width, CONTROL_HEIGHT), lines=lines, scroll=scroll)
        track_y = y + (46 if narrow else 27)
        p.rect(x, track_y, width, 3, 'line', 1)
        fill = max(0, min(1, (value - minimum) / (maximum - minimum)))
        p.rect(x, track_y, width * fill, 3, 'blue', 1)
        radius = 5 if enabled and (self.hovered(key + ':slider') or self.drag and self.drag['binding'] == binding) else 4
        p.rect(x + width * fill - radius, track_y + 1 - radius, radius * 2, radius * 2,
               'accent' if enabled else 'disabled', radius)
        p.target(key + ':slider', (x, track_y - 7, width, 17), ('slider', binding, minimum, maximum),
                 enabled, binding.rna(bpy.context).description)
        p.target(key + ':label', (x, y, width if narrow else width - edit_width - 8, 16),
                 ('hint', None, False), False, binding.rna(bpy.context).description)
        return 65 if narrow else 46

    def section_summary(self, key):
        scene = bpy.context.scene
        if key == 'show_core_settings':
            return f'{scene.steps} steps · CFG {scene.cfg:g} · {scene.sampler.replace("_", " ")}'
        if key == 'show_lora_settings':
            return f'{len(scene.lora_units)} adapters'
        if key == 'show_mode_specific_settings':
            return 'Controls for the current generation mode'
        if key == 'show_image_guidance_settings':
            return 'References, context and prompt templates'
        return ''

    def draw_section(self, p, section, x, y, width, force=False, omit=()):
        key = section.key
        if key in TOOL_SECTIONS:
            title, _page, _icon, commands = TOOL_SECTIONS[key]
            return self.draw_actions(p, title, commands, x, y, width)
        expanded = force or key in self.open
        if key == 'show_lora_settings' and expanded:
            return lora.draw_section(self, p, section, x, y, width)
        summary = self.section_summary(key)
        h = 40 if summary and not expanded else 29
        if self.hovered('section:' + key):
            p.rect(x, y, width, h - 3, 'hover_soft', 2)
        p.icon('down' if expanded else 'right', x + 2, y + 9, 10, 'text')
        p.text(SECTIONS.get(key, section.title), x + 19, y + 6, FONT_SIZE, 'text', width - 27)
        if summary and not expanded:
            p.text(summary, x + 19, y + 23, 10, 'secondary', width - 27)
        p.target('section:' + key, (x, y, width, h), ('section', key), True, section.title)
        y += h
        if expanded:
            omit = set(omit)
            if key == 'show_output_material_settings' and any(s.key == 'show_core_settings' for s in self.document.sections):
                omit.update(cards.RESOLUTION_KEYS)
            if key == 'show_core_settings':
                omit.add('qwen_image_21_native_rgba')
            y = cards.sections(self, p, section, x, y, width, omit)
            if key == 'show_core_settings':
                output = next((s for s in self.document.sections if s.key == 'show_output_material_settings'), None)
                if output:
                    ids = {id(c) for c in output.content.controls() if cards.property_name(c) in cards.RESOLUTION_KEYS}
                    node = cards.select(output.content, ids)
                    if node:
                        y = cards.draw(self, p, cards.partition(node, cards.SCHEMAS['resolution']),
                                       output.key, x, y, width)
                rgba = cards.select(section.content, {id(c) for c in section.content.controls()
                                                     if cards.property_name(c) == 'qwen_image_21_native_rgba'})
                if rgba:
                    y = cards.draw(self, p, cards.partition(rgba, cards.SCHEMAS[key]), section.key, x, y, width)
        return y + 6

    def draw_dependent(self, p, x, y, width, draw, anchors=None):
        """Indent a dependency scope and extend its rail through all descendants."""
        start = y
        before = len(p.targets)
        end = draw(x + 18, y + 2, width - 18)
        if end > start + 2:
            targets = p.targets[before:]
            # Include nested children and the entire final control, even when
            # its dropdown or text field is below a separate caption.
            last = max((t.rect[1] + t.rect[3] for t in targets),
                       default=end - CONTROL_GAP)
            p.dependency_rail(x + 6, start - 6, last)
            return end + 6
        return start

    def draw_control(self, p, live, x, y, width, section, index, label=None, hierarchy=None, anchors=None, card_body=False):
        before = len(p.targets)
        if card_body and live.control.method == 'prop':
            name = live.control.args[1]
            label = cards.LABELS.get(name) or getattr(hierarchy, 'card_labels', {}).get(id(live.control)) or label
        if card_body and live.control.method == 'prop' and live.control.args[1] in {'discard_factor', 'weight_exponent'}:
            name = live.control.args[1]
            end = y + self.draw_slider(p, name, label or ('Discard angle' if name == 'discard_factor' else 'Exponent'),
                                       x, y, width, 0 if name == 'discard_factor' else .1,
                                       180 if name == 'discard_factor' else 15, '°' if name == 'discard_factor' else '',
                                       enabled=live.enabled)
        else:
            end = self.draw_plain_control(p, live, x, y, width, section, index, label)
        if anchors is not None and end > y:
            targets = p.targets[before:]
            anchors.append(targets[0].rect[1] + targets[0].rect[3] / 2 if targets else (y + end - CONTROL_GAP) / 2)
        child = hierarchy.children.get(id(live.control)) if hierarchy else None
        if child:
            child_anchors = []
            end = self.draw_dependent(p, x, end, width, lambda cx, cy, cw:
                self.draw_node(p, child, cx, cy, cw, section, hierarchy=hierarchy, anchors=child_anchors,
                               card_body=card_body), child_anchors)
        return end

    def draw_plain_control(self, p, live, x, y, width, section, index, label=None, *, corners='ALL'):
        control = live.control
        enabled = live.enabled
        if control.method == 'prop':
            data, name = control.args[:2]
            if name in STRUCTURAL_PROPERTIES:
                return y
            label = label or control.kwargs.get('text') or data.bl_rna.properties[name].name
            try:
                binding = model.Binding.from_data(data, name)
                multiline = binding.rna(bpy.context).type == 'STRING' and 'prompt' in name and 'use_' not in name
                return y + self.draw_binding(p, binding, label, x, y, width, enabled=enabled, multiline=multiline,
                                             height=48 if name == 'comfyui_negative_prompt' else 90 if multiline else 112)
            except (TypeError, ReferenceError, ValueError):
                pass
        elif control.method == 'label':
            text = control.kwargs.get('text', '')
            if text:
                # Instructions and warnings wrap rather than disappearing at narrow widths.
                lines = wrap(text, width, lambda s: p.measure(s, 11))
                for _, _, line in lines:
                    p.text(line, x, y + 3, 11, 'warning' if live.alert else 'muted')
                    y += 16
                return y + 3
            return y
        elif control.method == 'separator':
            return y + 6
        elif control.method == 'operator':
            name = control.args[0]
            op = model.operator(name)
            label = label or control.kwargs.get('text') or op.get_rna_type().name
            if not control.kwargs.get('text') and name == 'wm.context_set_int':
                index_value = control.properties.get('value', 0)
                path = control.properties.get('data_path', '')
                label = f'Filament {index_value + 1}' if path == 'scene.stablegen_print_palette_index' else 'Select adapter' if path == 'scene.lora_units_index' else label
            if name == 'stablegen.move_camera_order':
                label = 'Move earlier' if control.properties.get('direction') == 'UP' else 'Move later'
            with _viewport(self):
                available = enabled and op.poll()
                hint = help_text.operator_hint(name, bpy.context, available, busy=self.busy)
            key = f'op:{section}:{index}:{name}'
            selection = action_style.selection(control)
            if selection:
                prop, value, kind = selection
                p.selection(key, (x, y, width, CONTROL_HEIGHT), f'{kind} {value + 1}',
                            ('operator', name, control.properties), getattr(bpy.context.scene, prop) == value,
                            enabled=available, hint=hint)
            else:
                if section == 'print' and name == 'object.export_textured_print_obj':
                    label = 'Textured OBJ (Image Map)'
                p.button(key, (x, y, width, CONTROL_HEIGHT), label,
                         ('operator', name, control.properties), icon=action_style.icon(control), style='outlined',
                         enabled=available, hover=self.hovered(key), hint=hint, corners=corners)
            return y + CONTROL_HEIGHT + CONTROL_GAP
        elif control.method == 'progress':
            factor = max(0, min(1, control.kwargs.get('factor', 0)))
            text = re.sub(r'\s*\(\s*\d+(?:\.\d+)?%\)\s*$', '', control.kwargs.get('text', ''))
            p.text(text, x, y, 11, 'secondary', width - 38)
            waiting = any(text.startswith(stage + ' · ') for stage in WAITING_STAGES)
            if not waiting or re.search(r'\(\s*\d+(?:\.\d+)?%\)\s*$', control.kwargs.get('text', '')):
                p.text(f'{factor:.0%}', x + width - 34, y, 11, 'secondary')
            p.rect(x, y + 17, width, 3, 'line', 1)
            p.rect(x, y + 17, width * factor, 3, 'blue', 1)
            if control.tooltip:
                p.target(f'progress:{section}:{index}', (x, y, width, 22), None,
                         enabled=False, hint=control.tooltip)
            return y + 28
        elif control.method == 'template_list' and len(control.args) >= 6:
            owner, name, index_owner, index_name = control.args[2:6]
            items = getattr(owner, name)
            selected = getattr(index_owner, index_name)
            p.text(f'{len(items)} item' + ('' if len(items) == 1 else 's'), x, y, 11, 'muted')
            y += 18
            if 0 <= selected < len(items):
                item = items[selected]
                title = getattr(item, 'label', '') or getattr(item, 'scene_name', '') or getattr(item, 'name', '')
                p.text(f'{selected + 1}. {title}', x, y, FONT_SIZE, 'text', width)
                y += 19
                status = getattr(item, 'status', '')
                if status:
                    p.text(status.replace('_', ' ').capitalize(), x, y, 10, 'muted', width)
                    y += 17
        label, hint = help_text.native_help(control, label)
        p.button(f'native:{section}:{index}', (x, y, width, CONTROL_HEIGHT), label,
                 ('native', section, index), icon='settings', style='outlined', enabled=enabled and not self.busy,
                 hint=hint)
        return y + CONTROL_HEIGHT + CONTROL_GAP

    def draw_operator_row(self, p, controls, x, y, width, section, enabled, alert):
        """Join related native commands; reflow without truncating their labels."""
        heading, captions = action_style.print_group(controls, section)
        if heading:
            p.text(heading, x, y, FONT_SIZE, 'secondary', width)
            y += 19
        labels = [captions.get(c.args[0]) or c.kwargs.get('text') or model.operator(c.args[0]).get_rna_type().name
                  for c in controls]
        columns = len(controls)
        while columns > 1 and any(p.measure(label) + (35 if action_style.icon(c) else 20) > width / columns
                                  for c, label in zip(controls, labels)):
            columns -= 1
        for first in range(0, len(controls), columns):
            row = controls[first:first + columns]
            cw = width / len(row)
            for i, c in enumerate(row):
                corners = 'ALL' if len(row) == 1 else 'LEFT' if i == 0 else 'RIGHT' if i == len(row) - 1 else 'NONE'
                self.draw_plain_control(p, model.LiveControl(c, enabled, alert), x + i * cw, y, cw,
                                        section, self.control_index(section, c), labels[first + i], corners=corners)
            y += CONTROL_HEIGHT + CONTROL_GAP
        return y

    def draw_color_selection(self, p, selector, color, x, y, width, section, enabled, alert):
        """Keep native filament selection and color editing in one clear row."""
        prop, value, kind = action_style.selection(selector)
        op = model.operator(selector.args[0])
        with _viewport(self):
            available = enabled and op.poll()
            hint = help_text.operator_hint(selector.args[0], bpy.context, available, busy=self.busy)
        key = f'op:{section}:{self.control_index(section, selector)}:{selector.args[0]}'
        picker_width = min(112, width * .47)
        p.selection(key, (x, y, width, 28), f'{kind} {value + 1}',
                    ('operator', selector.args[0], selector.properties), getattr(bpy.context.scene, prop) == value,
                    enabled=available, hint=hint, accessory_width=picker_width + 10)
        binding = model.Binding.from_data(color.args[0], color.args[1])
        key = _binding_key(binding)
        field = binding.rna(bpy.context)
        editable = enabled and not self.busy and not field.is_readonly
        cx, cy = x + width - picker_width - 3, y + 3
        p.outline(cx, cy, picker_width, CONTROL_HEIGHT, 'raised',
                  'hover_border' if editable and self.hovered(key) else 'tool_border', 2)
        p.text('Edit color', cx + 8, cy + 4, FONT_SIZE, 'text' if editable else 'disabled', picker_width - 39)
        rgb = [max(0, min(1, c)) for c in binding.value(bpy.context)[:3]]
        if field.subtype == 'COLOR':
            rgb = [12.92*c if c <= .0031308 else 1.055*c**(1/2.4)-.055 for c in rgb]
        p.outline(cx + picker_width - 26, cy + 2, 24, CONTROL_HEIGHT - 4, (*rgb, 1), 'tool_border', 2)
        p.target(key, (cx, cy, picker_width, CONTROL_HEIGHT), lambda: self.native(binding=binding),
                 editable, help_text.property_hint(binding, bpy.context))
        return y + 28 + CONTROL_GAP

    def draw_node(self, p, node, x, y, width, section, enabled=True, alert=False, omit=None, hierarchy=None, anchors=None, card_body=False):
        omit = omit or set()
        if hierarchy is None:
            key = (id(node), frozenset(omit))
            if key not in self.hierarchies:
                self.hierarchies[key] = hierarchy_model.build(node, omit)
            hierarchy = self.hierarchies[key]
            node = hierarchy.root
        enabled = enabled and node.attributes.get('enabled', True) and node.attributes.get('active', True)
        alert = alert or node.attributes.get('alert', False)
        children = []
        def omitted(c):
            return (c.method == 'prop' and c.args[1] in omit or c.method == 'operator' and 'op:' + c.args[0] in omit
                    or id(c) in hierarchy.redundant_labels
                    or section == 'main' and c.method == 'operator' and c.args[0] == 'stablegen.refresh_checkpoint_list'
                    or section in {'tools', 'debug'} and c.method == 'label'
                    and c.kwargs.get('text', '') in {'Tools', 'Debug / Diagnostics'})
        for child in node.children:
            if isinstance(child, Control) and omitted(child):
                continue
            if isinstance(child, Layout):
                child_controls = list(child.controls())
                if any(omitted(c) for c in child_controls) and all(omitted(c) or c.method in {'label', 'separator'} for c in child_controls):
                    continue
            children.append(child)
        flat = list(node.controls())
        if node.method in {'row', 'split', 'column'} and len(children) >= 2 and all(
                isinstance(c, Control) and c.method == 'operator' and not action_style.selection(c)
                for c in children):
            return self.draw_operator_row(p, children, x, y, width, section, enabled, alert)
        if node.method in {'row', 'split'}:
            labels = [c for c in flat if c.method == 'label' and c.kwargs.get('text') and not omitted(c)]
            fields = [c for c in flat if c.method in {'prop', 'operator'} and not omitted(c)]
            selectors = [c for c in fields if action_style.selection(c)]
            colors = [c for c in fields if c.method == 'prop'
                      and c.args[0].bl_rna.properties[c.args[1]].subtype in {'COLOR', 'COLOR_GAMMA'}]
            if len(fields) == 2 and len(selectors) == len(colors) == 1 and not labels:
                return self.draw_color_selection(p, selectors[0], colors[0], x, y, width, section, enabled, alert)
            if (len(labels) == 1 and len(fields) == 1
                    and not labels[0].kwargs['text'].startswith('→')
                    and not (fields[0].method == 'operator'
                             and labels[0].kwargs.get('icon') in {'ERROR', 'INFO', 'WARNING'})):
                c = fields[0]
                caption = labels[0].kwargs['text'].rstrip(':')
                if c.method == 'prop' and c.args[0].bl_rna.properties[c.args[1]].type == 'BOOLEAN':
                    caption = c.kwargs.get('text') or c.args[0].bl_rna.properties[c.args[1]].name
                return self.draw_control(p, model.LiveControl(c, enabled, alert), x, y, width,
                                         section, self.control_index(section, c), caption, hierarchy, anchors, card_body)
            if not labels and len(children) in {2, 3} and all(isinstance(c, Control) and c.method in {'prop', 'operator'} for c in children):
                prompts = [c for c in children if c.method == 'prop' and 'prompt' in c.args[1]
                           and c.args[0].bl_rna.properties[c.args[1]].type == 'STRING']
                if len(prompts) == 1 and len(children) == 2:
                    other = next(c for c in children if c is not prompts[0])
                    if other.method == 'operator' and not other.kwargs.get('text'):
                        key = f'op:{section}:{self.control_index(section, other)}:{other.args[0]}'
                        op = model.operator(other.args[0])
                        with _viewport(self):
                            available = enabled and not self.busy and op.poll()
                        p.button(key, (x + width - 46, y - 4, 22, CONTROL_HEIGHT), '',
                                 ('operator',other.args[0],other.properties), icon='reset', style='outlined', enabled=available,
                                 hover=self.hovered(key), hint=op.get_rna_type().description)
                        return self.draw_control(p, model.LiveControl(prompts[0],enabled,alert),x,y,width,section,
                                                 self.control_index(section,prompts[0]), hierarchy=hierarchy, anchors=anchors, card_body=card_body)
                columns = len(children)
                # Small utilities and numeric fields share a row; long switches keep their full label.
                long_bool = any(c.method == 'prop' and c.args[0].bl_rna.properties[c.args[1]].type == 'BOOLEAN'
                                and len(c.kwargs.get('text') or c.args[0].bl_rna.properties[c.args[1]].name) > 24 for c in children)
                long_field = bool(prompts)
                long_command = any(c.method == 'operator' and p.measure(c.kwargs.get('text') or model.operator(c.args[0]).get_rna_type().name)
                                   > (width - 10*(columns-1))/columns - 32 for c in children)
                if width / columns >= 130 and not long_bool and not long_field and not long_command:
                    cw = (width - 10 * (columns - 1)) / columns
                    return max(self.draw_control(p, model.LiveControl(c, enabled, alert), x + i * (cw + 10), y, cw,
                                                 section, self.control_index(section, c), hierarchy=hierarchy, anchors=anchors, card_body=card_body) for i, c in enumerate(children))
        grouped = set()
        for index, child in enumerate(children):
            if index in grouped:
                continue
            if node.method == 'column':
                commands = []
                for following in children[index:]:
                    if not (isinstance(following, Control) and following.method == 'operator'
                            and not action_style.selection(following)):
                        break
                    commands.append(following)
                if len(commands) > 1:
                    y = self.draw_operator_row(p, commands, x, y, width, section, enabled, alert)
                    grouped.update(range(index + 1, index + len(commands)))
                    continue
            if isinstance(child, Control) and child.method == 'label' and index + 1 < len(children) and isinstance(children[index+1],Layout):
                fields = [c for c in children[index+1].controls() if c.method == 'prop']
                caption = child.kwargs.get('text','').rstrip(':').casefold()
                if len(fields) == 1 and fields[0].args[0].bl_rna.properties[fields[0].args[1]].name.casefold() == caption:
                    continue
            if isinstance(child, Layout):
                inset = 8 if child.method == 'box' and node.method != 'root' and not card_body else 0
                start = y
                if inset:
                    p.begin_group()
                y = self.draw_node(p, child, x + inset, y + (6 if inset else 0),
                                   width - inset * 2, section, enabled, alert, omit, hierarchy, anchors, card_body)
                if inset:
                    if y > start + 6:
                        y += 5
                        p.end_group((x, start, width, y - start), 'subbox')
                        y += 6
                    else:
                        p.end_group((x, start, width, 0), 'subbox')
                        y = start
            else:
                y = self.draw_control(p, model.LiveControl(child, enabled, alert), x, y, width, section,
                                      self.control_index(section, child), hierarchy=hierarchy, anchors=anchors, card_body=card_body)
        return y

    def control_index(self, section, control):
        node = next((s.content for s in self.document.sections if s.key == section), None)
        if node is None:
            return -1
        return next((i for i, c in enumerate(node.controls()) if c is control), -1)

    def section_page(self, section):
        if bpy.context.scene.architecture_mode == 'trellis2':
            if section.key == 'show_trellis2_texture_settings':
                return 'GENERATE'
            if (section.key in {'show_core_settings', 'show_lora_settings'} and
                    bpy.context.scene.trellis2_texture_mode in {'native', 'none'}):
                return 'MESH'
        return ('QUEUE' if section.key == 'queue' else
                'EXPERIMENTS' if section.key == 'show_texturing_experiments' else
                'GENERATE' if section.page == 'OUTPUT' else section.page)

    def pages(self):
        if bpy.context.scene.architecture_mode == 'trellis2':
            return [('MESH', 'Mesh'), *PAGES]
        return PAGES + [('TOOLS', 'Tools')]

    def more_menu(self, rect):
        available = {self.section_page(s) for s in self.document.sections}
        primary = dict(self.pages())
        self.menu = {'anchor': rect, 'items': [(key, name, '') for key, name in MORE if key in available and key not in primary],
                     'query': '', 'binding': None, 'index': 0, 'scroll': 0}

    def draw_header(self, p, x, width):
        p.logo(x, 12, 18)
        p.text('StableGen', x + 26, 12, 13, 'text')
        p.button('search-button', (x + width - 50, 9, 22, CONTROL_HEIGHT), '', lambda: self.navigate('SEARCH'),
                 icon='search', hover=self.hovered('search-button'), hint='Find a setting · Ctrl K')
        p.button('preferences', (x + width - 23, 9, 23, CONTROL_HEIGHT), '', lambda: self.call('stablegen.studio_preferences'),
                 icon='settings', enabled=not self.busy, hover=self.hovered('preferences'), hint='StableGen add-on preferences')
        current_page = self.prompt_editor[2] if self.page == 'EDITOR' and self.prompt_editor else self.page
        if current_page not in dict(self.pages()):
            p.text(dict(MORE).get(current_page, 'Search'), x + 26, 29, 10, 'muted', width - 80)
        scene = bpy.context.scene
        y = 42
        p.outline(x, y, width, 46, 'raised')
        tools = [('object.add_cameras', 'Add cameras', 'camera'),
                 ('object.collect_camera_prompts', 'Collect camera prompts' if width >= 360 else 'Collect prompts', 'document')]
        for i, (name, text, icon) in enumerate(tools):
            with _viewport(self):
                enabled = model.operator(name).poll() and not self.busy
            key = 'toolbar:' + name
            p.button(key, (x + i * width / 2 + 1, y + 1, width / 2 - 2, CONTROL_HEIGHT), text,
                     ('operator', name, {}), icon=icon, enabled=enabled, hover=self.hovered(key),
                     hint=help_text.operator_hint(name, bpy.context, enabled, busy=self.busy))
        p.rect(x + width / 2, y + 1, 1, CONTROL_HEIGHT, 'line')
        p.rect(x + 1, y + 24, width - 2, 1, 'divider')
        tools = [('object.clone_camera', 'Clone', 'clone'), ('object.mirror_camera', 'Mirror', 'mirror'),
                 ('object.apply_auto_aspect', 'Auto aspect' if width >= 360 else 'Aspect', 'expand'), ('stablegen.camera_display', 'Display', 'labels')]
        for i, (name, text, icon) in enumerate(tools):
            with _viewport(self):
                enabled = model.operator(name).poll() and (not self.busy or name == 'stablegen.camera_display')
            key = 'toolbar:' + name
            p.button(key, (x + i * width / 4 + 1, y + 25, width / 4 - 2, 20), text,
                     ('operator', name, {}), icon=icon, enabled=enabled, hover=self.hovered(key),
                     hint=help_text.operator_hint(name, bpy.context, enabled, busy=self.busy))
        y += 56
        cw = (width - 12) / 2
        left = self.draw_binding(p, self.binding('architecture_mode'), 'Architecture', x, y, cw)
        actions = [item for item in model.controls(self.document.presets)
                   if item.control.method == 'operator'] if self.document.presets else []
        compact_actions = len(actions) > 1 or cw < 130
        buttons = []
        for item in actions:
            name = item.control.args[0]
            label, icon = {'stablegen.save_preset': ('Save', 'plus'),
                           'stablegen.apply_preset': ('Apply', 'right'),
                           'stablegen.delete_preset': ('', 'x')}.get(name, ('', 'settings'))
            buttons.append((item, '' if compact_actions else label, icon,
                            60 if label and not compact_actions else 26))
        button_space = sum(button[3] for button in buttons)
        preset_x = x + cw + 12
        right = self.draw_binding(p, self.binding('stablegen_preset'), 'Preset', preset_x, y, cw - button_space,
                                  corners='LEFT' if buttons else 'ALL')
        button_x = preset_x + cw - button_space
        for i, (item, label, icon, button_width) in enumerate(buttons):
            c = item.control
            with _viewport(self):
                op = model.operator(c.args[0])
                enabled = item.enabled and not self.busy and op.poll()
                hint = help_text.operator_hint(c.args[0], bpy.context, enabled, busy=self.busy)
            p.button('preset:' + c.args[0], (button_x, y + right - CONTROL_HEIGHT - CONTROL_GAP,
                                            button_width, CONTROL_HEIGHT), label,
                     ('operator', c.args[0], c.properties), icon=icon, style='outlined', enabled=enabled, hint=hint,
                     corners='RIGHT' if i == len(buttons) - 1 else 'NONE')
            button_x += button_width
        y += max(left, right)
        pages = self.pages() + [('MORE', 'More')]
        p.outline(x, y, width, 28, 'card', 'section_border', 4)
        compact = width < 360
        widths = [(width - 30) / 4] * 4 + [30] if compact else [width * n for n in (.22, .24, .25, .19, .1)]
        px = x
        for (page, label), w in zip(pages, widths):
            active = current_page == page or page == 'MORE' and current_page not in dict(self.pages())
            key = 'tab:' + page
            rect = (px, y, w, 27)
            action = (lambda r=rect: self.more_menu(r)) if page == 'MORE' else lambda page=page: self.navigate(page)
            caption = '' if page == 'MORE' else (
                {'CAMERAS': 'Views', 'BLEND': 'Blend'}.get(page, label) if compact else label)
            if active or self.hovered(key):
                # Use the shared frame for both the tint and border so selection
                # reaches its existing edges and rounded outer corners.
                previous_clip = p.clip_rect
                left = round(px * p.scale) / p.scale
                right = round((px + w) * p.scale) / p.scale
                try:
                    p.clip((left, y - 1, right - left, 30))
                    p.outline(x, y, width, 28, 'tab_selected' if active else 'card',
                              'accent' if active else 'hover_border', 4)
                finally:
                    p.clip(previous_clip)
            p.button(key, rect, caption, action, icon='more' if page == 'MORE' else None,
                     style='tab', active=active, hover=self.hovered(key), hint=label)
            px += w
        return y + 29

    def clear_camera_selection(self):
        with _viewport(self):
            for obj in bpy.context.selected_objects:
                if obj.type == 'CAMERA':
                    obj.select_set(False)
            if bpy.context.view_layer.objects.active and bpy.context.view_layer.objects.active.type == 'CAMERA':
                bpy.context.view_layer.objects.active = next((o for o in bpy.context.selected_objects if o.type == 'MESH'), None)
        self.refresh()

    def dismiss_tip(self, name):
        tips.dismiss(name)
        self.area.tag_redraw()

    def draw_widen_hint(self, p, x, width, bottom):
        """One-line reminder above the footer while the sidebar is too narrow."""
        y = bottom - 26
        p.rect(1, y, p.width - 2, 1, 'divider')
        p.icon('widen', x + 2, y + 6, 14, 'muted')
        p.text('Drag the sidebar edge to widen', x + 24, y + 6, 10, 'muted', width - 50)
        p.target('tip:widen-hint', (x, y, width - 24, 26), ('hint', None, False), False,
                 'Drag the left edge of the sidebar outward. Wider panels place settings side by side.')
        key = 'tip:widen-hint-dismiss'
        p.button(key, (x + width - 20, y + 3, 20, 20), '', lambda: self.dismiss_tip(tips.WIDEN_HINT),
                 icon='x', hover=self.hovered(key), hint="Don't show this tip again")
        return y

    def draw_widen_callout(self, p, top, width):
        """First-run tip pointing at the sidebar edge; dismissal lasts across files."""
        x, w, y = 16, min(width - 8, 240), top + 10
        body = 'StableGen has more room for settings when the sidebar is wider. Drag this edge to the left.'
        lines = wrap(body, w - 24, lambda s: p.measure(s, 10))
        h = 34 + len(lines) * 15 + 40
        p.rect(x - 1, y + 3, w + 2, h + 2, 'panel_shadow', 6)
        p.outline(x, y, w, h, 'tooltip', 'tooltip_line', 5)
        p.polygon([(x + 1, y + 16), (x - 9, y + 24), (x + 1, y + 32)], 'tooltip')
        p.rect(0, y + 8, 3, 32, 'blue', 1)
        p.icon('widen', x + 12, y + 10, 15, 'tooltip_text')
        p.text('Widen the sidebar', x + 34, y + 10, 11, 'tooltip_text', w - 44)
        for i, (_, _, line) in enumerate(lines):
            p.text(line, x + 12, y + 32 + i * 15, 10, 'tooltip_text', w - 24)
        # Keep clicks on the bubble from reaching the controls underneath.
        p.target('tip:widen-callout', (x - 9, y, w + 9, h), ('hint', None, False), False)
        key = 'tip:widen-callout-ok'
        p.button(key, (x + w - 72, y + h - 30, 62, 22), 'Got it', lambda: self.dismiss_tip(tips.WIDEN_CALLOUT),
                 style='primary', hover=self.hovered(key))

    def draw_footer(self, p, x, width):
        progress = [item for n in self.document.actions for item in model.controls(n) if item.control.method == 'progress']
        decision = [item for n in self.document.actions for item in model.controls(n)
                    if item.control.method == 'operator' and item.control.args[0] == 'stablegen.resolve_masked_edit']
        utilities = self.page in {'TOOLS', 'DEBUG'} and not decision
        preview = diagnostics.active(bpy.context.scene)
        # Keep the last action row twelve pixels clear of the panel's bottom edge.
        bottom_padding = 12
        if progress:
            height = max(92, 10 + len(progress) * 28 + 24 + bottom_padding)
        elif utilities:
            height = 10 + (28 if preview else 0) + CONTROL_HEIGHT + bottom_padding
        else:
            height = 10 + 20 + 35 + CONTROL_HEIGHT + bottom_padding
        y = p.height - height
        p.rect(0, y, p.width, height, 'card')
        p.rect(0, y, p.width, 1, 'section_border')
        top = y
        y += 10
        if not self.document.actions and not utilities:
            p.text('Choose a supported architecture to continue.', x, y, 11, 'muted', width)
            return top
        if progress:
            for index, item in enumerate(progress):
                y = self.draw_control(p, item, x, y, width, 'actions', index)
            cancel = next((item for n in self.document.actions for item in model.controls(n)
                           if item.control.method == 'operator' and
                           (item.control.kwargs.get('icon') == 'CANCEL' or 'Cancel' in item.control.kwargs.get('text', '') or item.control.args[0] == 'stablegen.stop_masked_edit')), None)
            if cancel:
                c = cancel.control
                p.button('cancel', (x, y, width, 24), c.kwargs.get('text', 'Stop'), ('operator', c.args[0], c.properties),
                         icon='stop', style='outlined', enabled=cancel.enabled)
            return top
        if utilities:
            if preview:
                with _viewport(self):
                    available = model.operator('stablegen.studio_diagnostic').poll()
                p.button('return-textures', (x, y, width, CONTROL_HEIGHT), 'Return to textures',
                         lambda: self.call('stablegen.studio_diagnostic', {'value': '"TEXTURES"'}, mode='EXEC_DEFAULT'),
                         icon='reset', style='outlined', enabled=available,
                         hint='Restore the original mesh, materials and render settings' if available else 'Switch to Object Mode to restore textures')
                y += 28
            self.draw_footer_commands(p, x, y, width)
            return top
        scene = bpy.context.scene
        from ..core.preferences import CheckServerStatus
        prefs = bpy.context.preferences.addons[ADDON_PKG].preferences
        checking_server = CheckServerStatus._checking
        refresh_server = checking_server or bool(prefs.server_address and not prefs.server_online)
        selected = [o for o in bpy.context.selected_objects if o.type == 'CAMERA']
        count = sum(o.type == 'CAMERA' for o in scene.objects)
        native = scene.architecture_mode == 'trellis2'
        primary = model.action(self.document, 'object.test_stable', 'object.stablegen_regenerate',
                               'object.trellis2_generate', 'object.trellis2_batch_generate', 'stablegen.masked_edit')
        disabled_reason = ''
        if primary:
            text = primary.control.kwargs.get('text', '')
            if any(word in text.lower() for word in ('cannot', 'enable online', 'select an image')):
                disabled_reason = text.replace('Cannot generate: ', '').replace('Cannot edit: ', '')
            if prefs.server_address and not prefs.server_online and not checking_server:
                disabled_reason = 'ComfyUI offline · ' + prefs.server_address
        scope = 'TRELLIS.2 · Shape and texture' if native else f'{len(selected)} selected views' if selected else f'{count} views · ' + ('Selected objects' if scene.texture_objects == 'selected' else 'All visible objects')
        p.text('Checking server…' if checking_server else disabled_reason or ('Reproject changes before baking' if self.dirty_projection else scope), x, y, 10,
               'warning' if disabled_reason or self.dirty_projection else 'muted', width - 64)
        if selected and not native:
            p.button('all-views', (x + width - 65, y - 3, 65, 18), 'All views', self.clear_camera_selection,
                     enabled=not self.busy, hint='Clear only the selected cameras; retain selected meshes')
        else:
            p.text(f'Seed {scene.trellis2_seed if native else scene.seed}', x + width - 64, y, 10, 'muted', 64)
        y += 20
        if decision:
            for i, item in enumerate(decision):
                c = item.control
                p.button('decision:' + str(i), (x + i * (width + 10) / 2, y, (width - 10) / 2, 28),
                         c.kwargs.get('text', ''), ('operator', c.args[0], c.properties), style='primary' if i == 0 else 'outlined')
        elif primary:
            c = primary.control
            label = 'Generate asset' if native else 'Regenerate selected views' if c.args[0] == 'object.stablegen_regenerate' else 'Edit painted selection' if c.args[0] == 'stablegen.masked_edit' else 'Generate textures'
            with _viewport(self):
                available = primary.enabled and not disabled_reason and model.operator(c.args[0]).poll()
            refresh_width = 34 if refresh_server else 0
            p.button('generate', (x, y, width - refresh_width, 28), label, ('operator', c.args[0], c.properties),
                     icon='play', style='primary', enabled=available and not self.busy and not checking_server,
                     corners='LEFT' if refresh_server else 'ALL',
                     hint=disabled_reason or help_text.operator_hint(c.args[0], bpy.context, available, busy=self.busy))
            if refresh_server:
                with _viewport(self):
                    refresh_available = model.operator('stablegen.check_server_status').poll() and not self.busy
                p.button('refresh-server', (x + width - refresh_width, y, refresh_width, 28), '',
                         ('operator', 'stablegen.check_server_status', {}), icon='reset', style='outlined',
                         enabled=refresh_available, corners='RIGHT', hover=self.hovered('refresh-server'),
                         hint='Checking server connection…' if checking_server else
                         'Refresh server connection and available models · ' + prefs.server_address)
        y += 35
        self.draw_footer_commands(p, x, y, width)
        return top

    def draw_footer_commands(self, p, x, y, width):
        commands = [('object.stablegen_reproject', 'Reproject', 'layers'),
                    ('object.bake_textures', 'Bake', 'grid'), ('object.export_game_engine', 'Export', 'download')]
        cw = (width - 16) / 3
        for i, (name, label, icon) in enumerate(commands):
            with _viewport(self):
                enabled = model.operator(name).poll() and not self.busy
            if name == 'object.bake_textures' and self.dirty_projection:
                enabled = False
            key = 'footer:' + name
            p.button(key, (x + i * (cw + 8), y, cw, CONTROL_HEIGHT), label, ('operator', name, {}), icon=icon,
                     style='outlined', enabled=enabled, active=name == 'object.stablegen_reproject' and self.dirty_projection,
                     hover=self.hovered(key), hint='Reproject changed settings before baking' if name == 'object.bake_textures' and self.dirty_projection else help_text.operator_hint(name, bpy.context, enabled, busy=self.busy))

    def live_operator(self, name):
        return next((item for section in self.document.sections for item in model.controls(section.content)
                     if item.control.method == 'operator' and item.control.args[0] == name), None)

    def draw_actions(self, p, title, commands, x, y, width, *, collapsible=False):
        """Task cards preserve each original operator's poll and arguments."""
        key = 'utilities:' + title
        if collapsible:
            if self.hovered('section:' + key):
                p.rect(x, y, width, 26, 'hover_soft', 2)
            p.icon('down' if key in self.open else 'right', x + 8, y + 8, 10, 'text')
            p.text(title, x + 25, y + 7, FONT_SIZE, 'text', width - 33)
            p.target('section:' + key, (x, y, width, 26), ('section', key), True, title)
            y += 30
            if key not in self.open:
                return y
        else:
            p.text(title, x, y + 3, FONT_SIZE, 'secondary', width)
            y += 23
        rows = []
        for name, label, icon in commands:
            item = self.live_operator(name)
            if item is None:
                continue
            with _viewport(self):
                op = model.operator(name)
                enabled = item.enabled and op.poll()
                hint = help_text.operator_hint(name, bpy.context, enabled, busy=self.busy)
            cancellation = name in {'stablegen.batch_texture_dae_cancel', 'stablegen.stop_texturing_regression',
                                    'stablegen.debug_force_generation_idle', 'stablegen.interactive_discard_cancel',
                                    'stablegen.interactive_discard_apply'}
            enabled = enabled and (not self.busy or cancellation)
            if not enabled and self.busy and not cancellation:
                hint = help_text.operator_hint(name, bpy.context, False, busy=True)
            rows.append((name, label, icon, item.control.properties, enabled, hint))
        if not rows:
            return y
        # Tool cards stay a single column at every width, also in search results;
        # only Diagnostics may go side by side.
        return cards.actions(self, p, rows, x, y, width, max_columns=2 if self.page == 'DEBUG' else 1)

    def draw_tools(self, p, x, y, width):
        y = self.draw_actions(p, 'Scene setup', [
            ('object.add_hdri', 'Add HDRI lighting', 'light')], x, y, width)
        for section in self.document.sections:
            if section.key in TOOL_SECTIONS:
                y = self.draw_section(p, section, x, y, width)
        y = self.draw_actions(p, 'Import & batch', [
            ('stablegen.import_dae', 'Import DAE', 'download'),
            ('stablegen.batch_import_dae', 'Batch import DAE', 'folder'),
            ('stablegen.batch_texture_dae', 'Batch texture DAE / GLB', 'layers')], x, y, width)
        wm = bpy.context.window_manager
        if getattr(wm, 'sg_batch_tex_running', False):
            total = max(1, wm.sg_batch_tex_total)
            p.text(f'{wm.sg_batch_tex_index} / {total} models · {wm.sg_batch_tex_phase.replace("_", " ")}',
                   x, y, FONT_SIZE, 'text', width)
            p.rect(x, y + 20, width, 3, 'line', 1)
            p.rect(x, y + 20, width * min(1, wm.sg_batch_tex_index / total), 3, 'blue', 1)
            y += 31
            y = self.draw_actions(p, 'Running batch', [
                ('stablegen.batch_texture_dae_cancel', 'Stop batch', 'stop')], x, y, width)
        return y

    def diagnostic_camera_menu(self, rect):
        state = bpy.context.scene.sg_studio_diagnostics
        items = [(obj.name, obj.name, 'Inspect this camera') for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
        items.sort()
        self.menu = {'anchor': rect, 'items': items, 'query': '', 'binding': None,
                     'callback': lambda name: self.call('stablegen.studio_diagnostic',
                         {'setting': 'camera', 'value': json.dumps(name)}, mode='EXEC_DEFAULT'),
                     'index': next((i for i, item in enumerate(items) if state.camera and item[0] == state.camera.name), 0), 'scroll': 0}

    def draw_diagnostics(self, p, x, y, width):
        scene, state = bpy.context.scene, bpy.context.scene.sg_studio_diagnostics
        if not any(s.key == 'debug' for s in self.document.sections):
            p.text('Enable Debug in add-on preferences to use diagnostics.', x, y, FONT_SIZE, 'secondary', width)
            return y + 28
        start = y
        p.begin_group()
        y = cards.heading(p, 'Viewport inspection', x + 12, y + 11, width - 24, 'eye')
        p.rect(x + 12, y - 3, width - 24, 1, 'divider')
        with _viewport(self):
            available = model.operator('stablegen.studio_diagnostic').poll()
        y += self.draw_binding(p, model.Binding.from_data(state, 'mode'), 'Display', x + 8, y + 8, width - 16, enabled=available) + 8
        description = diagnostics.DISPLAY_SUMMARIES[state.mode]
        for _, _, line in wrap(description, width - 16, p.measure):
            p.text(line, x + 8, y, 10, 'secondary')
            y += 15
        y += 6
        if diagnostics.active(scene):
            count = sum(record.object is not None for record in state.meshes)
            p.text(f'Previewing {count} mesh' + ('es' if count != 1 else '') + ' · Originals preserved',
                   x + 8, y, 10, 'accent', width - 16)
            y += 22
        options = {'GRID': ['grid_size'], 'COVERAGE': ['coverage'], 'CONFIDENCE': ['confidence'],
                   'DPPG': ['scope', 'stage']}.get(state.mode, [])
        for key in options:
            binding = model.Binding.from_data(state, key)
            y += self.draw_binding(p, binding, binding.rna(bpy.context).name, x + 8, y, width - 16, enabled=available)
        if state.mode in {'WEIGHT', 'FEATHER', 'VISIBILITY', 'CONFIDENCE'} or state.mode == 'DPPG' and state.scope == 'ACTIVE':
            p.text('Camera', x + 8, y, FONT_SIZE, 'text')
            y += 17
            rect = (x + 8, y, width - 16, CONTROL_HEIGHT)
            p.outline(*rect, 'menu', 'hover_border' if self.hovered('diagnostic-camera') else 'menu_border')
            p.text(state.camera.name if state.camera else 'Choose camera', rect[0] + 8, y + 5, FONT_SIZE, 'menu_text', width - 49)
            p.icon('down', x + width - 24, y + 6, 10, 'menu_text')
            p.target('diagnostic-camera', rect, lambda rect=rect: self.diagnostic_camera_menu(rect), available and not self.busy,
                     'Choose the camera whose signal is displayed')
            y += CONTROL_HEIGHT + 8
        y += 2
        p.end_group((x, start, width, y - start), 'subbox', 'section_border', 3)
        y += 12
        y = self.draw_actions(p, 'Projection analysis', [
            ('stablegen.debug_projection_line', 'Measure a surface line', 'graph'),
            ('stablegen.debug_open_projection_line', 'Open last line report', 'document'),
            ('stablegen.interactive_discard_check', 'Interactive discard check', 'eye')], x, y, width)
        from ..debug_tools import SG_OT_InteractiveDiscardCheck, SG_OT_DebugProjectionLine
        if SG_OT_DebugProjectionLine.status:
            for _, _, line in wrap(SG_OT_DebugProjectionLine.status, width, p.measure):
                p.text(line, x, y, 10, 'secondary')
                y += 15
            y += 6
        if SG_OT_InteractiveDiscardCheck._active is not None:
            tools = next((s for s in self.document.sections if s.key == 'tools'), None)
            if tools:
                block = next((child for child in tools.content.children if isinstance(child, Layout) and
                              any(item.control.method == 'operator' and item.control.args[0] == 'stablegen.interactive_discard_apply'
                                  for item in model.controls(child))), None)
                if block:
                    y = self.draw_node(p, block, x, y, width, 'tools')
        from ..texturing.regression import SG_OT_TexturingRegression
        if SG_OT_TexturingRegression._active is not None:
            p.text('Workflow tests', x, y, FONT_SIZE, 'secondary')
            y += 23
            for _, _, line in wrap(SG_OT_TexturingRegression.status, width, p.measure):
                p.text(line, x, y, FONT_SIZE, 'text')
                y += 16
            y = self.draw_actions(p, 'Running test', [
                ('stablegen.stop_texturing_regression', 'Stop workflow test', 'stop')], x, y + 6, width)
        else:
            y = self.draw_actions(p, 'Workflow tests', [
                ('stablegen.texturing_regression', 'Run texturing regression', 'play'),
                ('stablegen.export_retarget_debug_bundle', 'Export retarget test bundle', 'download')], x, y, width)
        y = self.draw_actions(p, 'Rig diagnostics', [
            ('stablegen.debug_skinning_weights', 'Inspect skin weights', 'rig'),
            ('stablegen.debug_clear_skinning_weights', 'Restore skin colors', 'reset'),
            ('stablegen.debug_convert_fortune_rig', 'Convert rig to Fortune', 'rig'),
            ('stablegen.debug_make_solid', 'Inspect Make Solid', 'material')], x, y, width, collapsible=True)
        return self.draw_actions(p, 'Recovery', [
            ('stablegen.debug_restore_material', 'Clear mesh materials', 'x'),
            ('stablegen.debug_force_generation_idle', 'Reset stalled generation', 'stop')], x, y, width, collapsible=True)

    def draw_generate(self, p, x, y, width):
        scene = bpy.context.scene
        main = next((s for s in self.document.sections if s.key == 'main'), None)
        if main is None:
            return y
        start = y
        p.begin_group()
        y = cards.heading(p, 'Texture input', x + 12, y + 11, width - 24, 'layers')
        p.rect(x + 12, y - 3, width - 24, 1, 'divider')
        outer_x, outer_width = x, width
        x, width, y = x + 12, width - 24, y + 6
        native = scene.architecture_mode == 'trellis2' and scene.trellis2_texture_mode in {'native', 'none'}
        trellis = scene.architecture_mode == 'trellis2'
        if trellis:
            y += self.draw_binding(p, self.binding('trellis2_texture_mode'), 'Texture mode', x, y, width)
            y += self.draw_binding(p, self.binding('use_separate_texture_prompt'), 'Use separate texture prompt', x, y + 3, width,
                                   enabled=not native) + 3
            if scene.use_separate_texture_prompt:
                y = self.draw_dependent(p, x, y, width, lambda cx, cy, cw:
                    cy + self.draw_binding(p, self.binding('texture_prompt'), 'Texture prompt', cx, cy, cw,
                                           multiline=True, height=112, enabled=not native)) + 6
            if not native:
                if not scene.use_separate_texture_prompt:
                    for _, _, line in wrap('Texturing uses the prompt on the Mesh tab.', width, p.measure):
                        p.text(line, x, y, 10, 'secondary')
                        y += 15
                    y += 10
            else:
                p.text('Native textures follow the input image.' if scene.trellis2_texture_mode == 'native' else 'Texture generation is disabled.',
                       x, y, 10, 'secondary', width)
                y += 25
        checkpoint = model.first_property(self.document, 'model_name')
        if checkpoint and not (trellis and native):
            y += self.draw_binding(p, self.binding('model_name'), 'Checkpoint', x, y, width - 31)
            p.button('refresh-models', (x + width - 25, y - CONTROL_HEIGHT - CONTROL_GAP, 25, CONTROL_HEIGHT), '',
                     ('operator', 'stablegen.refresh_checkpoint_list', {}), icon='reset', enabled=not self.busy,
                     hint='Refresh the available ComfyUI models after connecting the server or adding model files')
        masked = (scene.qwen_generation_method if scene.model_architecture in QWEN_ARCHITECTURES else scene.generation_method) == 'masked_edit' and scene.architecture_mode != 'trellis2'
        prompt = 'sg_masked_prompt' if masked else 'comfyui_prompt'
        if hasattr(scene, prompt) and not trellis:
            y += self.draw_binding(p, self.binding(prompt), 'Edit instructions' if masked else 'Object prompt' if native else 'Texture prompt', x, y + 4, width, multiline=True, height=112) + 6
        handled = {'architecture_mode', 'stablegen_preset', 'model_name', 'comfyui_prompt', 'sg_masked_prompt',
                   'generation_method', 'qwen_generation_method', 'texture_objects'}
        if masked:
            handled.discard('comfyui_prompt')
        mask_commands = {'stablegen.paint_edit_mask', 'stablegen.finish_edit_mask', 'stablegen.clear_edit_mask'}
        if not native:
            mode_key = 'qwen_generation_method' if scene.model_architecture in QWEN_ARCHITECTURES else 'generation_method'
            cw = (width - 12) / 2
            left = self.draw_binding(p, self.binding(mode_key), 'Texture workflow', x, y, cw)
            right = self.draw_binding(p, self.binding('texture_objects'), 'Target objects', x + cw + 12, y, cw)
            y += max(left, right)
            if masked:
                mode_section = next((s for s in self.document.sections if s.key == 'show_mode_specific_settings'), None)
                commands = [item for item in model.controls(mode_section.content)
                            if item.control.method == 'operator' and item.control.args[0] in mask_commands] if mode_section else []
                for i, item in enumerate(commands):
                    c = item.control
                    with _viewport(self):
                        available = item.enabled and not self.busy and model.operator(c.args[0]).poll()
                    columns = 2 if width >= 320 else 1
                    command_width = (width - 12 * (columns - 1)) / columns
                    p.button('mask:' + c.args[0], (x + i % columns * (command_width + 12),
                                                y + i // columns * (CONTROL_HEIGHT + CONTROL_GAP), command_width, CONTROL_HEIGHT),
                             c.kwargs.get('text') or model.operator(c.args[0]).get_rna_type().name,
                             ('operator', c.args[0], c.properties), style='outlined', enabled=available,
                             icon='settings' if i == 0 else 'x', hint=model.operator(c.args[0]).get_rna_type().description)
                if commands:
                    y += ((len(commands) + columns - 1) // columns) * (CONTROL_HEIGHT + CONTROL_GAP)
        y += 8
        p.end_group((outer_x, start, outer_width, y - start), 'subbox', 'section_border', 3)
        x, width, y = outer_x, outer_width, y + 12
        if main and scene.architecture_mode != 'trellis2':
            extras = Layout()
            extras.children.extend(item.control for item in model.controls(main.content)
                               if item.control.method == 'prop' and item.control.args[1] not in handled
                               and item.control.args[1] not in STRUCTURAL_PROPERTIES)
            if extras.children:
                y = self.draw_section(p, Section('main_extra', 'Additional prompts', 'GENERATE', '', extras), x, y, width)
        for section in self.document.sections:
            if self.section_page(section) == 'GENERATE' and section.key not in {'main', 'export'}:
                omit = {prompt} | {'op:' + name for name in mask_commands} if masked and section.key == 'show_mode_specific_settings' else {prompt}
                y = self.draw_section(p, section, x, y, width, force=masked and section.key == 'show_mode_specific_settings', omit=omit)
        return y

    def draw_mesh(self, p, x, y, width):
        scene = bpy.context.scene
        main = next((s for s in self.document.sections if s.key == 'main'), None)
        if main:
            start = y
            p.begin_group()
            y = cards.heading(p, 'Mesh input', x + 12, y + 11, width - 24, 'layers')
            p.rect(x + 12, y - 3, width - 24, 1, 'divider')
            outer_x, outer_width = x, width
            x, width, y = x + 12, width - 24, y + 6
            y += self.draw_binding(p, self.binding('trellis2_generate_from'), 'Generate mesh from', x, y, width)
            if scene.trellis2_generate_from == 'prompt':
                shared = scene.trellis2_texture_mode != 'none' and (
                    not scene.use_separate_texture_prompt or scene.trellis2_texture_mode == 'native')
                y += self.draw_binding(p, self.binding('comfyui_prompt'), 'Mesh image & texture prompt' if shared else 'Mesh image prompt', x, y + 4, width,
                                       multiline=True, height=112) + 6
                hint = ('Generates the input image for the mesh.' if scene.trellis2_texture_mode == 'none' else
                        'Generates the mesh’s input image. Texturing uses its separate prompt.'
                        if scene.use_separate_texture_prompt and scene.trellis2_texture_mode != 'native' else
                        'Generates the input image for the mesh and also guides texturing.')
                for _, _, line in wrap(hint, width, p.measure):
                    p.text(line, x, y, 10, 'secondary')
                    y += 15
                y += 10
                y += self.draw_binding(p, self.binding('trellis2_preview_gallery_enabled'), 'Preview gallery', x, y, width)
                binding = self.binding('trellis2_preview_gallery_count')
                y = self.draw_dependent(p, x, y, width, lambda cx, cy, cw:
                    cy + self.draw_binding(p, binding, 'Image count', cx, cy, cw,
                                          enabled=scene.trellis2_preview_gallery_enabled)) + 5
            else:
                shared = scene.trellis2_texture_mode not in {'native', 'none'} and not scene.use_separate_texture_prompt
                y += self.draw_binding(p, self.binding('comfyui_prompt'), 'Shared texture prompt' if shared else 'Mesh image prompt', x, y + 4, width,
                                       multiline=True, height=112, enabled=shared) + 6
                p.text('Mesh generation uses the selected input image.', x, y, 10, 'secondary', width)
                y += 25
            omit = {'architecture_mode', 'stablegen_preset', 'comfyui_prompt', 'texture_prompt',
                    'use_separate_texture_prompt', 'trellis2_texture_mode', 'trellis2_generate_from',
                    'trellis2_preview_gallery_enabled', 'trellis2_preview_gallery_count',
                    'generation_method', 'qwen_generation_method', 'texture_objects'}
            if scene.trellis2_texture_mode not in {'native', 'none'}:
                omit.add('model_name')
                omit.add('op:stablegen.refresh_checkpoint_list')
            y = self.draw_node(p, main.content, x, y, width, 'main', omit=omit)
            y += 8
            p.end_group((outer_x, start, outer_width, y - start), 'subbox', 'section_border', 3)
            x, width, y = outer_x, outer_width, y + 12
        for section in self.document.sections:
            if self.section_page(section) == 'MESH':
                y = self.draw_section(p, section, x, y, width)
        return y

    def draw_cameras(self, p, x, y, width):
        scene = bpy.context.scene
        cameras = self.ordered_cameras()
        self.camera_rows = []
        self.camera_clip = p.clip_rect
        p.text('Scene cameras', x, y, 13)
        p.text(f'{len(cameras)} views', x + width - 58, y + 2, 10, 'muted')
        y += 29
        def setup(cx, cy, cw):
            cy += self.draw_binding(p, self.binding('use_camera_prompts'), 'Use view prompts', cx, cy, cw)
            return cy + self.draw_binding(p, self.binding('sg_use_custom_camera_order'), 'Custom generation order', cx, cy, cw)
        y = cards.group(p, 'View setup', x, y, width, setup, 'camera')
        if cameras:
            # The displayed camera cards are the order editor. Avoid a second,
            # disconnected native list and its up/down-button cluster.
            sort_rect = (x + width - 124, y, 124, CONTROL_HEIGHT)
            p.text('Generation order', x, y + 5, FONT_SIZE)
            p.button('camera-sort', sort_rect, 'Sort…', lambda: self.camera_order_menu(sort_rect),
                     icon='down', style='outlined', enabled=not self.busy,
                     hint='Choose an ordering preset. Enables custom generation order.')
            y += CONTROL_HEIGHT + 8
        if not cameras:
            p.text('Add cameras using the toolbar above.', x, y, FONT_SIZE, 'muted', width)
            y += 27
        for i, camera in enumerate(cameras):
            start = y
            p.begin_group()
            selected = camera.select_get()
            hovered = self.hovered('camera:' + camera.name) and not self.busy
            tx = x + 32 if selected else x + 103
            p.text(f'{i + 1:02} · {camera.name}', tx, y + 16, FONT_SIZE, 'text', x + width - tx - 44)
            image = self.camera_images.get(camera.name)
            prompt = scene.camera_prompts.get(camera.name)
            if not selected:
                title = prompt.prompt if prompt and prompt.prompt else 'Generated' if image else 'No generated image'
                p.text(title, tx, y + 36, 10, 'secondary', width - 146)
            p.target('camera:' + camera.name, (x + 25, y + 1, width - 67, 38 if selected else 82),
                     lambda name=camera.name: self.call('stablegen.studio_select_camera', {'camera_name': name, 'extend': self.ctrl}),
                     not self.busy, camera.name + ' · Ctrl click to add to selection')
            grip = (x + 4, y + 5, 23, 28)
            p.icon('grip', grip[0] + 4, grip[1] + 7, 14, 'disabled' if self.busy else 'accent' if self.hovered('camera-drag:' + camera.name) else 'secondary')
            p.target('camera-drag:' + camera.name, grip, ('camera-drag', camera.name), not self.busy,
                              'Drag to change generation order. Dropping enables custom order; Esc cancels.')
            p.button('look:' + camera.name, (x + width - 37, y + (10 if selected else 28), 25, CONTROL_HEIGHT), '',
                     ('operator', 'stablegen.studio_look_camera', {'camera_name': camera.name}), icon='eye', enabled=not self.busy,
                     hint='Look through ' + camera.name)
            image_rect = (x + 12, y + 43, 70, 160) if selected else (x + 30, y + 12, 60, 60)
            p.outline(*image_rect, 'menu', 'menu_border', 2)
            if image:
                p.image(image, image_rect)
            else:
                ix, iy, iw, ih = image_rect
                p.icon('camera', ix + (iw - 25) / 2, iy + (ih - 25) / 2, 25)
            if selected:
                y += 43
                y += self.draw_binding(p, model.Binding.from_data(camera, 'name'), 'Camera name', x + 98, y, width - 110)
                if prompt and scene.use_camera_prompts:
                    y += self.draw_binding(p, model.Binding.from_data(prompt, 'prompt'), 'View prompt', x + 98, y, width - 110,
                                           multiline=True, height=93)
                y = max(y, image_rect[1] + image_rect[3]) + 12
            else:
                y += 84
            self.camera_rows.append((camera.name, (x, start, width, y - start)))
            p.end_group((x, start, width, y - start), 'subbox', 'accent' if selected else 'hover_border' if hovered else 'section_border', 3)
            y += 8
        if self.camera_drag and self.camera_rows:
            _, line_y = self.camera_drop(self.pointer[1])
            p.rect(x + 3, line_y - 1, width - 6, 2, 'accent')
            source = next((rect for name, rect in self.camera_rows if name == self.camera_drag['name']), None)
            if source:
                sx, sy, sw, sh = source
                p.line([(sx, sy), (sx + sw, sy), (sx + sw, sy + sh), (sx, sy + sh), (sx, sy)], 'accent')
        for section in self.document.sections:
            if section.page == 'CAMERAS' and section.key not in {'camera_tools', 'show_camera_options'}:
                y = self.draw_section(p, section, x, y + 8, width, omit={'use_camera_prompts','sg_use_custom_camera_order'})
        return y

    def menu_items(self):
        if self.menu.get('kind') == 'lora':
            return lora.menu_items(self.menu)
        query = self.menu['query'].casefold()
        return [item for item in self.menu['items'] if query in item[1].casefold()]

    def draw_menu(self, p):
        if self.menu.get('kind') == 'lora':
            return lora.draw_menu(self, p)
        menu = self.menu
        items = self.menu_items()
        ax, ay, aw, ah = menu['anchor']
        width = max(220, aw)
        if menu.get('binding'):
            # Room for each label and its tags.
            width = max([width] + [p.measure(label) + sum(p.measure(t, 9) + 14 for t in _choice_tags(menu['binding'].property, i)) + 40
                                   for i, label, _ in items])
        width = min(p.width - 30, width)
        x = min(max(15, ax), p.width - width - 15)
        y = ay + ah + 5
        if menu.get('drop_down'):
            # Always open below the anchor and fill the space to the bottom; the rest scrolls.
            rows = min(len(items), max(1, int((p.height - 10 - y - 36) / 24)))
        else:
            rows = min(9, len(items), max(1, int((p.height - 100) / 24)))
        menu['rows'] = rows
        height = 30 + max(1, rows) * 24 + 6
        if not menu.get('drop_down') and y + height > p.height - 10:
            y = max(10, ay - height - 5)
        menu['rect'] = (x, y, width, height)
        p.outline(x, y, width, height, 'raised', 'line', 6)
        p.text(menu['query'] or 'Type to filter…', x + 10, y + 8, FONT_SIZE, 'text' if menu['query'] else 'muted', width - 20)
        p.rect(x + 8, y + 25, width - 16, 1, 'line')
        if not items:
            p.text('No matching choices', x + 10, y + 35, FONT_SIZE, 'muted')
        start = min(menu['scroll'], max(0, len(items) - rows))
        for i, (identifier, label, description) in enumerate(items[start:start + rows], start):
            rect = (x + 5, y + 30 + (i - start) * 24, width - 10, CONTROL_HEIGHT)
            key = 'choice:' + identifier
            if i == menu['index'] or self.hovered(key):
                p.rect(*rect, 'selected')
            label_width = width - 26
            if menu.get('binding'):
                label_width -= _draw_tags(p, _choice_tags(menu['binding'].property, identifier),
                                          rect[0] + rect[2] - 6, rect[1] + 4)
            p.text(label, rect[0] + 8, rect[1] + 5, FONT_SIZE, 'text', label_width)
            p.target(key, rect, ('menu-item', identifier), True, description)
        if len(items) > rows:
            p.rect(x + width - 5, y + 31 + start / len(items) * (height - 36), 2,
                   max(12, rows / len(items) * (height - 36)), 'muted', 1)

    def draw_search(self, p, x, y, width):
        editing = self.edit if self.edit and self.edit['key'] == 'search' else None
        value = editing['buffer'].text if editing else self.query
        lines, scroll = p.text_field('search', (x, y, width, CONTROL_HEIGHT), value, ('edit', None, False),
                                    editing=editing['buffer'] if editing else None, hint='Search visible settings and tooltips')
        if editing:
            editing.update(rect=(x, y, width, CONTROL_HEIGHT), lines=lines, scroll=scroll)
        y += 36
        query = value.strip().casefold()
        if not query:
            p.text('Find a setting, workflow or tool.', x, y, FONT_SIZE, 'muted')
            return y + 30
        found = False
        for section in self.document.sections:
            if section.key == 'show_lora_settings':
                searchable = 'LoRA adapters add remove refresh models'
                for item in model.controls(section.content):
                    control = item.control
                    if control.method == 'prop':
                        prop = control.args[0].bl_rna.properties[control.args[1]]
                        searchable += ' ' + prop.name + ' ' + prop.description
                if query in searchable.casefold():
                    found = True
                    y = lora.draw_section(self, p, section, x, y, width) + 6
                continue
            for i, item in enumerate(model.controls(section.content)):
                control = item.control
                if control.method not in {'prop', 'operator'}:
                    continue
                text = control.kwargs.get('text', '')
                if control.method == 'prop':
                    prop = control.args[0].bl_rna.properties.get(control.args[1])
                    if prop:
                        text += ' ' + prop.name + ' ' + prop.description + ' ' + control.args[1]
                elif control.method == 'operator':
                    rna = model.operator(control.args[0]).get_rna_type()
                    text += ' ' + rna.name + ' ' + rna.description
                if query not in text.casefold():
                    continue
                found = True
                p.text(section.title, x, y, 10, 'muted')
                y = self.draw_control(p, item, x, y + 17, width, section.key, i) + 6
        if not found:
            p.text('No matching controls in this architecture or mode.', x, y, 11, 'muted', width)
            y += 30
        return y

    def draw_prompt_editor(self, p, x, y, width, bottom):
        if not self.prompt_editor:
            return y
        binding, label, previous = self.prompt_editor
        p.button('close-editor', (x, y, width, CONTROL_HEIGHT), 'Back to ' + dict(self.pages() + MORE).get(previous, 'settings'),
                 lambda: self.navigate(previous), style='outlined')
        y += 34
        return y + self.draw_binding(p, binding, label, x, y, width, multiline=True,
                                    height=max(100, bottom - y - 60))

    def draw(self):
        global _font_id
        if _font_id is None:
            _font_id = blf.load(_font_path) if _font_path else 0
            if _font_id < 0:
                _font_id = 0
        self.scale = max(.8, bpy.context.preferences.system.ui_scale)
        p = Painter(self.region, self.scale, _font_id)
        p.hover_key = self.hover_key
        try:
            self.text_fields = {}
            self.card_bounds = {}
            p.panel()
            x, width = 12, p.width - 24
            top = self.draw_header(p, x, width)
            bottom = self.draw_footer(p, x, width)
            narrow = width < cards.SIDE_BY_SIDE_WIDTH
            if narrow and not tips.dismissed(tips.WIDEN_HINT):
                bottom = self.draw_widen_hint(p, x, width, bottom)
            clip = (1, top + 1, p.width - 2, max(0, bottom - top - 2))
            p.clip(clip)
            offset = self.scroll.get(self.page, 0)
            y = top + 12 - offset
            for node in self.document.notices:
                y = self.draw_node(p, node, x, y, width, 'notices')
            if self.page == 'GENERATE':
                y = self.draw_generate(p, x, y, width)
            elif self.page == 'CAMERAS':
                y = self.draw_cameras(p, x, y, width)
            elif self.page == 'TOOLS':
                y = self.draw_tools(p, x, y, width)
            elif self.page == 'DEBUG':
                y = self.draw_diagnostics(p, x, y, width)
            elif self.page == 'MESH' and bpy.context.scene.architecture_mode == 'trellis2':
                y = self.draw_mesh(p, x, y, width)
            elif self.page == 'SEARCH':
                y = self.draw_search(p, x, y, width)
            elif self.page == 'EDITOR':
                y = self.draw_prompt_editor(p, x, y, width, bottom)
            else:
                sections = [section for section in self.document.sections if self.section_page(section) == self.page]
                for section in sections:
                    y = self.draw_section(p, section, x, y, width)
            self.max_scroll = max(0, y + offset + 20 - bottom)
            if offset > self.max_scroll:
                self.scroll[self.page] = self.max_scroll
                self.area.tag_redraw()
            if self.max_scroll:
                available = bottom - top
                thumb = max(24, available * available / (available + self.max_scroll))
                p.rect(p.width - 5, top + offset / self.max_scroll * max(0, available - thumb), 2, thumb, 'muted', 1)
            p.clip(None)
            if not tips.dismissed(tips.WIDEN_CALLOUT):
                if self.callout_width is not None and width > self.callout_width + 20:
                    # Dragging the edge wider shows the tip has done its job.
                    tips.dismiss(tips.WIDEN_CALLOUT)
                elif narrow:
                    self.callout_width = width if self.callout_width is None else min(self.callout_width, width)
                    self.draw_widen_callout(p, top, width)
            if self.menu:
                parent = self.menu.get('parent')
                if parent:
                    # A submenu (the LoRA folder filter) opens over its picker, which stays visible but inert.
                    menu, before = self.menu, len(p.targets)
                    self.menu = parent
                    try:
                        self.draw_menu(p)
                    finally:
                        self.menu = menu
                    del p.targets[before:]
                self.draw_menu(p)
            # Layout changes can put a different control under a stationary mouse.
            self.update_hover(p.targets, cursor=False)
            self.tooltip_key, self.tooltip_rect = '', None
            if self.notice and time.monotonic() < self.notice_until:
                lines = wrap(self.notice, width - 24, lambda s: p.measure(s, 11))[:4]
                h = len(lines) * 18 + 18
                p.outline(x, bottom - h - 8, width, h, 'raised', 'line', 5)
                for i, (_, _, line) in enumerate(lines):
                    p.text(line, x + 12, bottom - h + i * 18, 11, 'warning')
            else:
                target = self.tooltip_target(p.targets)
                if target:
                    tw = min(width, max(180, min(420, p.measure(target.hint, 11) + 24)))
                    lines = wrap(target.hint, tw - 24, lambda s: p.measure(s, 11))
                    maximum = max(1, int((p.height - 46) / 18))
                    if len(lines) > maximum:
                        lines = lines[:maximum]
                        start, end, text = lines[-1]
                        lines[-1] = (start, end, p.fit(text, tw - 38, 11) + '…')
                    h = len(lines) * 18 + 18
                    ax, ay, aw, ah = target.rect
                    if target.clip:
                        ay, ah = max(ay, target.clip[1]), min(ay + ah, target.clip[1] + target.clip[3]) - max(ay, target.clip[1])
                    if self.menu:
                        ax, ay, aw, ah = self.menu['rect']
                    tx = max(14, min(ax, p.width - tw - 12))
                    ty = ay + ah + 8
                    if ty + h > p.height - 10:
                        ty = max(10, ay - h - 8)
                    p.rect(tx - 1, ty + 2, tw + 2, h + 2, 'panel_shadow', 5)
                    p.outline(tx, ty, tw, h, 'tooltip', 'tooltip_line', 4)
                    for i, (_, _, line) in enumerate(lines):
                        p.text(line, tx + 12, ty + 8 + i * 18, 11, 'tooltip_text')
                    self.tooltip_key, self.tooltip_rect = target.key, (tx, ty, tw, h)
            self.targets = p.targets
        finally:
            p.finish()


class SG_OT_StudioSession(bpy.types.Operator):
    bl_idname = 'stablegen.studio_session'
    bl_label = 'StableGen Studio Input'
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D' or context.region.type != 'UI':
            return {'CANCELLED'}
        key = (context.window.as_pointer(), context.area.as_pointer())
        if key in _sessions and _sessions[key].alive:
            return {'CANCELLED'}
        self.session = StudioSession(context)
        _sessions[key] = self.session
        try:
            self.session.refresh()
        except Exception as exc:
            _sessions.pop(key, None)
            self.report({'ERROR'}, 'Could not open Studio: ' + str(exc))
            _log.exception('Studio startup error')
            return {'CANCELLED'}
        context.window_manager.modal_handler_add(self)
        if not bpy.app.timers.is_registered(_tick):
            bpy.app.timers.register(_tick, first_interval=.05, persistent=True)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        session = self.session
        if not _registered or not session.valid():
            session.clear_hover(leave=True)
            if _sessions.get(session.key) is session:
                del _sessions[session.key]
            return {'FINISHED'}
        try:
            return session.event(event)
        except (ReferenceError, RuntimeError, ValueError, TypeError) as exc:
            session.edit = session.drag = session.menu = None
            session.camera_drag = None
            session.clear_hover(leave=True)
            session.tell(str(exc).split('\n')[0])
            _log.exception('Studio input error')
            return {'PASS_THROUGH'}

    def cancel(self, context):
        if hasattr(self, 'session'):
            self.session.alive = False
            self.session.clear_hover(leave=True)


class SG_OT_StudioNative(bpy.types.Operator):
    """Retain Blender's native controls for specialized editors and lists."""
    bl_idname = 'stablegen.studio_native'
    bl_label = 'StableGen Settings'
    bl_options = {'INTERNAL'}
    section: bpy.props.StringProperty()
    binding: bpy.props.StringProperty()
    control_index: bpy.props.IntProperty(default=-1)

    def invoke(self, context, event):
        self.session_key = (context.window.as_pointer(), context.area.as_pointer())
        session = _sessions.get(self.session_key)
        if session:
            session.clear_hover(leave=True)
            session.popup = True
        return context.window_manager.invoke_props_dialog(self, width=max(380, min(620, context.region.width)))

    def draw(self, context):
        layout = self.layout
        layout.use_property_decorate = False
        if self.binding:
            binding = model.Binding(**json.loads(self.binding))
            data = binding.data(context)
            layout.prop(data, binding.property)
            return
        document = model.document(context)
        section = next((s for s in document.sections if s.key == self.section), None)
        if section:
            layout.label(text=section.title)
            if self.control_index >= 0:
                items = list(section.content.controls())
                if self.control_index < len(items):
                    items[self.control_index].render(layout)
            else:
                section.content.render(layout, contents=True, narrow=context.region.width < 420)

    def finish(self):
        session = _sessions.get(getattr(self, 'session_key', None))
        if session:
            session.popup = False
            session.refresh()

    def execute(self, context):
        self.finish()
        return {'FINISHED'}

    def cancel(self, context):
        self.finish()


class SG_OT_StudioReorderCamera(bpy.types.Operator):
    bl_idname = 'stablegen.studio_reorder_camera'
    bl_label = 'Reorder Camera'
    bl_description = 'Move this camera to the indicated position in generation order'
    bl_options = {'UNDO', 'INTERNAL'}
    camera_name: bpy.props.StringProperty()
    before_name: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        return not sg_modal_active(context)

    def execute(self, context):
        from ..texturing.view_plan import order_cameras
        scene = context.scene
        cameras = order_cameras((o for o in scene.objects if o.type == 'CAMERA'),
                                [item.name for item in scene.sg_camera_order] if scene.sg_use_custom_camera_order else ())
        names = [c.name for c in cameras]
        if self.camera_name not in names or self.before_name == self.camera_name:
            return {'CANCELLED'}
        if self.before_name and self.before_name not in names:
            return {'CANCELLED'}
        ordered = [name for name in names if name != self.camera_name]
        ordered.insert(ordered.index(self.before_name) if self.before_name else len(ordered), self.camera_name)
        if ordered == names:
            return {'CANCELLED'}
        scene.sg_camera_order.clear()
        for name in ordered:
            scene.sg_camera_order.add().name = name
        scene.sg_camera_order_index = ordered.index(self.camera_name)
        scene.sg_use_custom_camera_order = True
        return {'FINISHED'}


class SG_OT_StudioSortCameras(bpy.types.Operator):
    bl_idname = 'stablegen.studio_sort_cameras'
    bl_label = 'Sort Cameras'
    bl_description = 'Apply a generation-order preset and enable custom order'
    bl_options = {'UNDO', 'INTERNAL'}
    strategy: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        return not sg_modal_active(context)

    def execute(self, context):
        prop = bpy.ops.stablegen.apply_camera_order_preset.get_rna_type().properties['strategy']
        if self.strategy not in {item.identifier for item in prop.enum_items}:
            return {'CANCELLED'}
        result = bpy.ops.stablegen.apply_camera_order_preset('EXEC_DEFAULT', strategy=self.strategy)
        if 'FINISHED' in result:
            context.scene.sg_use_custom_camera_order = True
        return result


class SG_OT_StudioSelectCamera(bpy.types.Operator):
    bl_idname = 'stablegen.studio_select_camera'
    bl_label = 'Select Camera'
    bl_description = 'Select this view; Ctrl click adds or removes it from the selection'
    bl_options = {'UNDO', 'INTERNAL'}
    camera_name: bpy.props.StringProperty()
    extend: bpy.props.BoolProperty(default=False)

    @classmethod
    def poll(cls, context):
        return not sg_modal_active(context)

    def execute(self, context):
        camera = context.scene.objects.get(self.camera_name)
        if camera is None or camera.type != 'CAMERA' or camera.name not in context.view_layer.objects:
            return {'CANCELLED'}
        if not self.extend:
            for obj in context.selected_objects:
                if obj.type == 'CAMERA':
                    obj.select_set(False)
        camera.select_set(not camera.select_get() if self.extend else True)
        if camera.select_get():
            context.view_layer.objects.active = camera
        return {'FINISHED'}


class SG_OT_StudioLookCamera(bpy.types.Operator):
    bl_idname = 'stablegen.studio_look_camera'
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


class SG_OT_StudioPreferences(bpy.types.Operator):
    bl_idname = 'stablegen.studio_preferences'
    bl_label = 'StableGen Preferences'
    bl_options = {'INTERNAL'}

    def execute(self, context):
        bpy.ops.preferences.addon_show('INVOKE_DEFAULT', module=ADDON_PKG)
        return {'FINISHED'}


def _ensure_session(context):
    if bpy.app.background or not _registered:
        return
    window, area = context.window, context.area
    key = (window.as_pointer(), area.as_pointer())
    if key in _sessions or key in _pending:
        return
    _pending.add(key)

    def start():
        _pending.discard(key)
        if not _registered:
            return None
        try:
            if window not in bpy.context.window_manager.windows[:] or area not in window.screen.areas[:]:
                return None
            region = next((r for r in area.regions if r.type == 'UI'), None)
            if region and region.active_panel_category == TAB:
                with bpy.context.temp_override(window=window, area=area, region=region):
                    bpy.ops.stablegen.studio_session('INVOKE_DEFAULT')
        except ReferenceError:
            pass
        return None
    bpy.app.timers.register(start, first_interval=.05)


class SG_PT_StudioCanvas(bpy.types.Panel):
    bl_idname = 'SG_PT_StudioCanvas'
    bl_label = 'StableGen'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = TAB
    # Keep the native header: Blender sorts header-less panels, and with them
    # their tab, above its own sidebar tabs. Studio draws over the header.
    bl_options = set()

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'PAINT_TEXTURE'} and not legacy_panel_enabled(context)

    def draw_header(self, context):
        # Blender skips draw() while the hidden panel is collapsed.
        _ensure_session(context)

    def draw(self, context):
        _ensure_session(context)
        # Provide a native entry point even if custom drawing failed to start.
        self.layout.operator('stablegen.studio_session', text='Open StableGen Studio', icon='WORKSPACE')
        # Native sidebar resize zones follow the panel's layout footprint.
        # Reserve the canvas height so its lower border remains draggable too.
        spacer = self.layout.row()
        spacer.scale_y = max(1, (context.region.height / context.preferences.system.ui_scale - 80) / 20)
        spacer.label(text='')


def _draw():
    context = bpy.context
    if context.area is None or context.region.type != 'UI' or context.region.active_panel_category != TAB:
        return
    session = _sessions.get((context.window.as_pointer(), context.area.as_pointer()))
    if session and session.active() and session.document:
        try:
            session.draw()
            session.error = ''
        except Exception as exc:
            if str(exc) != session.error:
                _log.exception('Studio drawing error')
            session.error = str(exc)


def _tick():
    if not _registered:
        return None
    for key, session in list(_sessions.items()):
        if not session.valid():
            session.clear_hover(leave=True)
            session.alive = False
            _sessions.pop(key, None)
        else:
            try:
                session.tick(time.monotonic())
            except (ReferenceError, RuntimeError):
                _log.debug('Studio refresh temporarily unavailable', exc_info=True)
    return .05


@persistent
def _load_pre(_):
    for session in _sessions.values():
        session.clear_hover(leave=True)
        session.alive = False
    _sessions.clear()
    _window_mouse.clear()
    _pending.clear()


STUDIO_UI_CLASSES = diagnostics.CLASSES + [lora.SG_OT_StudioLoRA, SG_OT_StudioSession, SG_OT_StudioNative, SG_OT_StudioSelectCamera, SG_OT_StudioLookCamera,
                     SG_OT_StudioReorderCamera, SG_OT_StudioSortCameras,
                     SG_OT_StudioPreferences, tips.SG_OT_ResetTips, SG_PT_StudioCanvas]


def register_studio_ui():
    global _registered, _handler, _font_id, _font_path
    _registered = True
    diagnostics.register()
    _font_id = None
    if not bpy.app.background:
        # Load after Blender initializes its UI fonts, at the first actual draw.
        fonts = bpy.utils.system_resource('DATAFILES', path='fonts')
        for path in (Path(fonts) / 'Inter.woff2' if fonts else Path('__missing_font__'),
                     Path('C:/Windows/Fonts/segoeui.ttf'),
                     Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                     Path('/System/Library/Fonts/Supplemental/Arial.ttf')):
            if path.is_file():
                _font_path = str(path)
                break
        _handler = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'UI', 'POST_PIXEL')
        bpy.app.timers.register(_tick, first_interval=.05, persistent=True)
    if _load_pre not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_load_pre)


def unregister_studio_ui():
    global _registered, _handler
    _registered = False
    diagnostics.unregister()
    _load_pre(None)
    if _handler:
        bpy.types.SpaceView3D.draw_handler_remove(_handler, 'UI')
        _handler = None
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    if _load_pre in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_load_pre)
