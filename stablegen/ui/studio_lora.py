"""Direct LoRA cards and a shared model picker for the Studio sidebar."""
from pathlib import PurePosixPath

import bpy

from . import studio_model as model
from .model_units import get_lora_models
from .studio_text import TextBuffer, wrap
from .studio_widgets import CONTROL_HEIGHT, FONT_SIZE
from ..utils import sg_modal_active

PLACEHOLDERS = {'NONE_AVAILABLE', 'NO_COMFYUI_DIR_LORA', 'NO_LORAS_SUBDIR',
                'PERM_ERROR', 'SCAN_ERROR', 'NONE_FOUND', 'NO_SERVER', 'NO_PREFS'}


def models(context):
    return [tuple(item[:3]) for item in get_lora_models(context.scene, context)
            if item and item[0] and item[0] not in PLACEHOLDERS]


def path_name(value):
    return str(value).replace('\\', '/')


def title(value):
    return PurePosixPath(path_name(value)).stem or 'Choose a model'


def snapshot(scene):
    return tuple((str(unit.as_pointer()), unit.model_name) for unit in scene.lora_units)


class SG_OT_StudioLoRA(bpy.types.Operator):
    bl_idname = 'stablegen.studio_lora'
    bl_label = 'Edit LoRA Adapter'
    bl_description = 'Add, replace or remove the adapter on this card'
    bl_options = {'UNDO', 'INTERNAL'}
    operation: bpy.props.EnumProperty(items=[('ADD', 'Add', ''), ('REPLACE', 'Replace', ''), ('REMOVE', 'Remove', '')])
    model_name: bpy.props.StringProperty()
    index: bpy.props.IntProperty(default=-1)
    token: bpy.props.StringProperty()
    scene_name: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        return not sg_modal_active(context)

    def execute(self, context):
        scene = context.scene
        if scene.name != self.scene_name:
            return {'CANCELLED'}
        if self.operation != 'ADD':
            if not (0 <= self.index < len(scene.lora_units)
                    and str(scene.lora_units[self.index].as_pointer()) == self.token):
                self.report({'WARNING'}, 'The adapter list changed. Choose the card again.')
                return {'CANCELLED'}
        if self.operation == 'REMOVE':
            scene.lora_units_index = self.index
            return bpy.ops.stablegen.remove_lora_unit('EXEC_DEFAULT')
        if self.model_name not in {item[0] for item in models(context)}:
            self.report({'WARNING'}, 'This model is no longer available. Refresh the model list.')
            return {'CANCELLED'}
        if self.operation == 'REPLACE':
            unit = scene.lora_units[self.index]
            if unit.model_name == self.model_name:
                return {'CANCELLED'}
            unit.model_name = self.model_name
            return {'FINISHED'}
        if any(unit.model_name == self.model_name for unit in scene.lora_units):
            return {'CANCELLED'}
        if not bpy.ops.stablegen.add_lora_unit.poll():
            return {'CANCELLED'}
        result = bpy.ops.stablegen.add_lora_unit('EXEC_DEFAULT')
        if 'FINISHED' in result:
            scene.lora_units[-1].model_name = self.model_name
        return result


def open_picker(s, rect, index=None):
    s.commit()
    if s.busy:
        return
    scene = bpy.context.scene
    current = scene.lora_units[index].model_name if index is not None else ''
    items = models(bpy.context)
    if not items:
        s.tell('Refresh the LoRA model list to choose an adapter.')
        return
    token = str(scene.lora_units[index].as_pointer()) if index is not None else ''
    operation = 'REPLACE' if index is not None else 'ADD'
    used = {unit.model_name for unit in scene.lora_units}
    menu = {'kind': 'lora', 'anchor': rect, 'items': items, 'query': '',
            'buffer': TextBuffer(''), 'binding': None, 'index': 0, 'scroll': 0,
            'folder': None, 'scene': scene.name, 'architecture': scene.architecture_mode,
            'snapshot': snapshot(scene), 'current': current,
            'catalog': tuple(item[0] for item in items),
            'disabled': used if operation == 'ADD' else set(), 'operation': operation}

    def choose(identifier):
        if not valid_menu(s, menu) or identifier in menu['disabled']:
            return
        s.call('stablegen.studio_lora', {'operation': operation, 'model_name': identifier,
               'index': index if index is not None else -1, 'token': token,
               'scene_name': menu['scene']}, mode='EXEC_DEFAULT')

    menu['callback'] = choose
    menu['index'] = next((i for i, item in enumerate(items) if item[0] == current),
                         next((i for i, item in enumerate(items) if item[0] not in menu['disabled']), 0))
    s.menu = menu
    s.clear_hover()
    s.area.tag_redraw()


def valid_menu(s, menu):
    scene = bpy.context.scene
    return (not s.busy and not sg_modal_active(bpy.context)
            and scene.name == menu['scene'] and scene.architecture_mode == menu['architecture']
            and snapshot(scene) == menu['snapshot']
            and tuple(item[0] for item in models(bpy.context)) == menu['catalog'])


def menu_items(menu):
    terms = path_name(menu['query']).casefold().split()
    folder = menu['folder']
    return [item for item in menu['items']
            if all(term in path_name(item[0]).casefold() for term in terms)
            and (folder is None or str(PurePosixPath(path_name(item[0])).parent) == folder)]


def folders(s):
    parent = s.menu
    values = sorted({str(PurePosixPath(path_name(item[0])).parent) for item in parent['items']}, key=str.casefold)

    def choose(value):
        if not valid_menu(s, parent):
            return
        parent.update(folder=None if value == '__all__' else value, index=0, scroll=0)
        parent['index'] = next((i for i, item in enumerate(menu_items(parent)) if item[0] not in parent['disabled']), 0)
        s.menu = parent

    s.menu = {'anchor': parent['folder_rect'], 'items': [('__all__', 'All folders', '')]
              + [(value, 'Root folder' if value == '.' else value, value) for value in values],
              'query': '', 'binding': None, 'index': 0, 'scroll': 0, 'callback': choose}


def keyboard(s, event):
    """Edit the picker query without turning it into a scene property edit."""
    if event.value != 'PRESS':
        return False
    menu, key = s.menu, event.type
    buffer = menu['buffer']
    ctrl = event.ctrl or event.oskey
    if ctrl and key == 'A':
        buffer.anchor, buffer.caret = 0, len(buffer.text)
    elif ctrl and key in {'C', 'X'}:
        a, b = buffer.selection
        if a != b:
            s.window_manager.clipboard = buffer.text[a:b]
            if key == 'X':
                buffer.replace('')
    elif ctrl and key == 'V':
        buffer.replace(s.window_manager.clipboard.replace('\n', ' ').replace('\r', ''))
    elif ctrl and key in {'Z', 'Y'}:
        buffer.undo(redo=key == 'Y' or event.shift)
    elif key in {'LEFT_ARROW', 'RIGHT_ARROW', 'HOME', 'END'}:
        direction = -1 if key == 'LEFT_ARROW' else 1
        a, b = buffer.selection
        destination = (0 if key == 'HOME' else len(buffer.text) if key == 'END'
                       else (a if direction < 0 else b) if a != b and not event.shift
                       else buffer.word(direction) if ctrl else buffer.caret + direction)
        buffer.move(destination, event.shift)
    elif key in {'BACK_SPACE', 'DEL'}:
        buffer.delete(backward=key == 'BACK_SPACE', word=ctrl)
    elif getattr(event, 'unicode', '') and not ctrl and not event.alt:
        buffer.replace(event.unicode)
    else:
        return False
    if menu['query'] != buffer.text:
        menu.update(query=buffer.text, index=0, scroll=0)
        menu['index'] = next((i for i, item in enumerate(menu_items(menu)) if item[0] not in menu['disabled']), 0)
    s.clear_hover()
    s.area.tag_redraw()
    return True


def select_search(s):
    """Use Studio's caret hit testing without committing the query to RNA."""
    previous = s.edit
    if not s.menu.get('editor'):
        return
    try:
        s.edit = s.menu['editor']
        s.select_caret(*s.pointer)
    finally:
        s.edit = previous


def number(s, p, binding, label, x, y, width, enabled):
    from .studio import _binding_key
    prop = binding.rna(bpy.context)
    enabled = enabled and not prop.is_readonly and not s.busy
    key = _binding_key(binding)
    editing = s.edit if s.edit and s.edit['key'] == key else None
    value = binding.value(bpy.context)
    display = editing['buffer'].text if editing else f'{value:.{prop.precision}f}'.rstrip('0').rstrip('.') or '0'
    inline = width >= p.measure(label) + 72
    p.text(label, x, y + (5 if inline else 0), FONT_SIZE, 'text' if enabled else 'disabled')
    rect = (x + width - 62, y, 62, CONTROL_HEIGHT) if inline else (x, y + 17, width, CONTROL_HEIGHT)
    hint = prop.description
    lines, scroll = p.text_field(key, rect, display, ('edit', binding, False), enabled=enabled,
                                  hint=hint, kind='number', editing=editing['buffer'] if editing else None)
    if editing:
        editing.update(rect=rect, lines=lines, scroll=scroll)
    return rect[1] + rect[3]


def draw_section(s, p, section, x, y, width):
    """Use only the per-unit properties present in the native panel capture."""
    scene, start = bpy.context.scene, y
    fields = {(c.control.args[0].as_pointer(), c.control.args[1]): c for c in model.controls(section.content)
              if c.control.method == 'prop'}
    add = next((c for c in model.controls(section.content)
                if c.control.method == 'operator' and c.control.args[0] == 'stablegen.add_lora_unit'), None)
    can_add = bool(add and add.enabled and not s.busy and models(bpy.context)
                   and bpy.ops.stablegen.add_lora_unit.poll())
    p.begin_group()
    cx, cw = x + 12, width - 24
    p.icon('down', cx, y + 16, 10, 'text')
    p.text('LoRA', cx + 18, y + 12, FONT_SIZE)
    if cw >= 270:
        p.text(f'{len(scene.lora_units)} adapters', cx + 57, y + 13, 10, 'secondary')
    p.target('section:' + section.key, (cx, y + 5, cw - 104, 26), ('section', section.key), hint=section.title)
    refresh = bpy.ops.stablegen.refresh_lora_list.poll() and not s.busy
    p.button('lora:refresh', (cx + cw - 96, y + 8, 22, 22), '',
             ('operator', 'stablegen.refresh_lora_list', {}), icon='reset', enabled=refresh,
             hover=s.hovered('lora:refresh'), hint='Refresh the available LoRA models from ComfyUI')
    rect = (cx + cw - 70, y + 8, 70, 22)
    anchor = (cx, y + 8, cw, 22)
    p.button('lora:add', rect, 'Add', lambda r=anchor: open_picker(s, r), icon='plus', style='outlined',
             enabled=can_add, hint='Choose a LoRA model to add' if can_add else
             'Unavailable while StableGen is running' if s.busy else
             'All available models are already added' if models(bpy.context) else 'Refresh to fetch the available LoRA models')
    p.rect(cx, y + 40, cw, 1, 'divider')
    y += 52
    for index, unit in enumerate(scene.lora_units):
        top, token = y, str(unit.as_pointer())
        live = fields.get((unit.as_pointer(), 'model_name'))
        enabled = bool(live and live.enabled and not s.busy)
        name = unit.model_name
        key = 'lora:model:' + token
        rect = (cx + 10, y + 8, cw - 43, 26)
        p.begin_group()
        if s.hovered(key) and enabled:
            p.rect(*rect, 'hover_soft', 2)
        p.icon('document', cx + 17, y + 15, 12, 'secondary' if enabled else 'disabled')
        p.text(title(name), cx + 35, y + 15, FONT_SIZE, 'text' if enabled else 'disabled', cw - 97)
        p.icon('down', cx + cw - 52, y + 17, 11, 'secondary' if enabled else 'disabled')
        anchor = (cx, y + 8, cw, 26)
        p.target(key, rect, lambda r=anchor, i=index: open_picker(s, r, i), enabled,
                 'Change this adapter\n' + path_name(name))
        p.button('lora:remove:' + token, (cx + cw - 29, y + 10, 22, 22), '',
                 ('operator', 'stablegen.studio_lora', {'operation': 'REMOVE', 'index': index,
                  'token': token, 'scene_name': scene.name}), icon='x', enabled=not s.busy,
                 hover=s.hovered('lora:remove:' + token), hint='Remove ' + title(name))
        p.text(path_name(name), cx + 11, y + 41, 10, 'secondary', cw - 22)
        props = [(key, label) for key, label in [('model_strength', 'Model'), ('clip_strength', 'CLIP')]
                 if (unit.as_pointer(), key) in fields]
        nw = (cw - 22 - 14 * (len(props) - 1)) / max(1, len(props))
        ends = [number(s, p, model.Binding.from_data(unit, prop), label,
                       cx + 11 + i * (nw + 14), y + 59, nw, fields[(unit.as_pointer(), prop)].enabled)
                for i, (prop, label) in enumerate(props)]
        y = max(ends, default=y + 59) + 11
        s.card_bounds[section.key + ':unit:' + str(index)] = (cx, top, cw, y - top)
        p.end_group((cx, top, cw, y - top), 'subbox', 'section_border', 3)
        y += 8
    if not scene.lora_units:
        message = 'Add an adapter to adjust its model strength.' if models(bpy.context) else 'Refresh to fetch LoRA models from ComfyUI.'
        for _, _, line in wrap(message, cw, lambda value: p.measure(value, 10)):
            p.text(line, cx, y, 10, 'secondary')
            y += 15
        y += 8
    y += 4
    s.card_bounds[section.key + ':adapters'] = (x, start, width, y - start)
    p.end_group((x, start, width, y - start), 'card', 'section_border', 3)
    return y + 8


def draw_menu(s, p):
    menu, items = s.menu, s.menu_items()
    ax, ay, aw, ah = menu['anchor']
    width = min(p.width - 30, max(280, aw))
    rows = min(8, len(items), max(1, int((p.height - 150) / 45)))
    menu['rows'] = rows
    height = 109 + max(1, rows) * 45 + 29
    x = min(max(15, ax), p.width - width - 15)
    y = ay + ah + 5
    if y + height > p.height - 10:
        y = max(10, ay - height - 5)
    menu['rect'] = (x, y, width, height)
    p.rect(x - 1, y + 2, width + 2, height + 2, 'panel_shadow', 4)
    p.outline(x, y, width, height, 'tooltip', 'section_border', 4)
    p.text('Add adapter' if menu['operation'] == 'ADD' else 'Change adapter', x + 12, y + 12, FONT_SIZE)
    p.button('choice:lora:close', (x + width - 31, y + 6, 22, 22), '', ('menu-lora-close',), icon='x')
    rect = (x + 12, y + 38, width - 24, 27)
    lines, scroll = p.text_field('choice:lora:search', rect, menu['query'], ('menu-lora-search',),
                                 editing=menu['buffer'], hint='Search model filenames and folder paths')
    menu['editor'] = {'buffer': menu['buffer'], 'rect': rect, 'lines': lines, 'scroll': scroll,
                      'multiline': False, 'follow_caret': False}
    if not menu['query']:
        p.text('Search adapters…', x + 22, y + 45, FONT_SIZE, 'secondary', width - 45)
    menu['folder_rect'] = (x + 12, y + 74, min(160, width - 100), 22)
    label = 'All folders' if menu['folder'] is None else 'Root folder' if menu['folder'] == '.' else menu['folder']
    fx, fy, fw, fh = menu['folder_rect']
    p.outline(fx, fy, fw, fh, 'menu', 'hover_border' if s.hovered('choice:lora:folder') else 'menu_border', 3)
    p.text(label, fx + 8, fy + 5, FONT_SIZE, 'menu_text', fw - 30)
    p.icon('down', fx + fw - 16, fy + 6, 10, 'menu_text')
    p.target('choice:lora:folder', menu['folder_rect'], ('menu-lora-folder',), hint='Filter models by folder')
    p.text(f'{len(items)} results', x + width - 66, y + 79, 10, 'secondary', 57)
    start = min(menu['scroll'], max(0, len(items) - rows))
    for i, (identifier, label, description) in enumerate(items[start:start + rows], start):
        rect = (x + 4, y + 109 + (i - start) * 45, width - 8, 45)
        key, disabled = 'choice:' + identifier, identifier in menu['disabled']
        if not disabled and (i == menu['index'] or s.hovered(key)):
            p.rect(*rect, 'hover_soft')
        tint = 'muted' if disabled else 'text'
        p.icon('document', rect[0] + 10, rect[1] + 8, 12, tint)
        p.text(title(identifier), rect[0] + 31, rect[1] + 6, FONT_SIZE, tint, width - 92)
        p.text(path_name(identifier), rect[0] + 31, rect[1] + 24, 10, 'secondary', width - 48)
        if disabled:
            p.text('Added', x + width - 48, rect[1] + 8, 10, 'muted')
        else:
            p.icon('check' if identifier == menu['current'] else 'plus' if menu['operation'] == 'ADD' else 'right',
                   x + width - 24, rect[1] + 10, 12, 'secondary')
        p.target(key, rect, ('menu-item', identifier), not disabled, path_name(identifier) + '\n' + description)
    if not items:
        p.text('No matching models', x + 12, y + 120, FONT_SIZE, 'secondary', width - 24)
    if len(items) > rows:
        track = rows * 45
        p.rect(x + width - 5, y + 109 + start / len(items) * track, 2,
               max(12, rows / len(items) * track), 'muted', 1)
    p.rect(x + 12, y + height - 26, width - 24, 1, 'divider')
    p.text('Enter to add · Esc to close' if menu['operation'] == 'ADD' else 'Enter to replace · Esc to close',
           x + 12, y + height - 18, 10, 'secondary', width - 24)
