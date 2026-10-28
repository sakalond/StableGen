"""Viewport label prompt expansion and focused editing of saved view prompts."""
from dataclasses import dataclass, field
import logging
import time

import bpy
import blf

from ..ui.studio_text import TextBuffer
from ..utils import sg_modal_active

_views = {}
_keymaps = []
_registered = False
_log = logging.getLogger(__name__)


@dataclass
class PromptView:
    window: object
    area: object
    region: object
    scene: object
    expanded: set = field(default_factory=set)
    targets: list = field(default_factory=list)
    fields: dict = field(default_factory=dict)
    scrolls: dict = field(default_factory=dict)
    edit: dict | None = None
    hover: int | None = None


def view_key(context):
    return (context.window.as_pointer(), context.area.as_pointer(),
            context.region.as_pointer(), context.scene.as_pointer())


def begin_draw(context, cameras):
    key = view_key(context)
    view = _views.get(key)
    if view is None:
        view = _views[key] = PromptView(context.window, context.area, context.region, context.scene)
    view.targets.clear()
    view.fields.clear()
    view.expanded.intersection_update(c.as_pointer() for c in cameras)
    if not context.space_data.show_gizmo:
        view.hover = None
    return view


def clear():
    _views.clear()


def prune():
    live = {(w.as_pointer(), a.as_pointer(), r.as_pointer(), w.scene.as_pointer())
            for w in bpy.context.window_manager.windows for a in w.screen.areas if a.type == 'VIEW_3D'
            for r in a.regions if r.type == 'WINDOW'}
    for key in set(_views) - live:
        del _views[key]


def available(context):
    return (context.area is not None and context.area.type == 'VIEW_3D'
            and context.region is not None and context.region.type == 'WINDOW'
            and context.region_data is not None and context.region_data.view_perspective != 'CAMERA'
            and context.space_data.overlay.show_overlays
            and context.scene.sg_camera_display.labels != 'NONE')


def pointer(view, event):
    scale = bpy.context.preferences.system.ui_scale
    return ((event.mouse_x - view.region.x) / scale,
            (view.region.height - (event.mouse_y - view.region.y)) / scale)


def hit(view, point):
    x, y = point
    return next((t for t in reversed(view.targets)
                 if t['rect'][0] <= x <= t['rect'][0] + t['rect'][2]
                 and t['rect'][1] <= y <= t['rect'][1] + t['rect'][3]), None)


def prompt_height(painter, value):
    from ..ui.studio_text import wrap
    from ..ui.studio_widgets import TEXT_LINE_HEIGHT, TEXT_PADDING
    width = min(320, painter.width - 16)
    rows = len(wrap(value, width - TEXT_PADDING * 2, painter.measure))
    return width, min(118, max(67, rows * TEXT_LINE_HEIGHT + TEXT_PADDING * 2))


class SG_GT_CameraPrompt(bpy.types.Gizmo):
    """Native hover target; the existing overlay paints its visible control."""
    bl_idname = 'SG_GT_camera_prompt'
    __slots__ = ('rect', 'camera_key')

    def setup(self):
        self.rect = (0, 0, 0, 0)
        self.camera_key = 0

    def draw(self, context):
        pass

    def test_select(self, context, location):
        scale = context.preferences.system.ui_scale
        x, y = location[0] / scale, (context.region.height - location[1]) / scale
        rx, ry, width, height = self.rect
        return 0 if rx <= x <= rx + width and ry <= y <= ry + height else -1


class SG_GGT_CameraPrompts(bpy.types.GizmoGroup):
    bl_idname = 'SG_GGT_camera_prompts'
    bl_label = 'Camera view prompts'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'PERSISTENT', 'SHOW_MODAL_ALL'}

    @classmethod
    def poll(cls, context):
        return _registered and available(context) and not sg_modal_active(context)

    def setup(self, context):
        self.widgets = {}

    def draw_prepare(self, context):
        view = _views.get(view_key(context))
        targets = {t['camera'].as_pointer(): t for t in view.targets if t['kind'] == 'toggle'} if view else {}
        for key in set(self.widgets) - targets.keys():
            self.gizmos.remove(self.widgets.pop(key))
        for key, target in targets.items():
            if key not in self.widgets:
                widget = self.gizmos.new(SG_GT_CameraPrompt.bl_idname)
                widget.target_set_operator('stablegen.camera_prompt_input')
                widget.use_draw_modal = True
                self.widgets[key] = widget
            self.widgets[key].rect = target['rect']
            self.widgets[key].camera_key = key
        hovered = next((key for key, widget in self.widgets.items() if widget.is_highlight), None)
        if view and view.hover != hovered:
            view.hover = hovered
            context.area.tag_redraw()


def draw_icon(painter, view, camera, rect, has_prompt, border='section_border'):
    x, y, width, height = rect
    expanded = camera.as_pointer() in view.expanded
    hovered = view.hover == camera.as_pointer()
    fill = 'selected' if expanded else 'hover' if hovered else 'raised'
    edge = 'accent' if expanded else 'hover_border' if hovered else border
    painter.outline(x, y, width, height, fill, edge, 3, corners='RIGHT')
    painter.rect(x, y, 1, height, edge)
    # The filled page denotes saved wording; an empty page remains actionable.
    tint = 'accent' if expanded else 'text'
    u = 15 / 24
    px, py = x + (width - 15) / 2, y + (height - 15) / 2
    points = [(5, 3), (14, 3), (19, 8), (19, 21), (5, 21)]
    fold = [(14, 3), (14, 8), (19, 8)]
    if has_prompt:
        painter.polygon([(px + a*u, py + b*u) for a, b in points], tint)
        painter.polygon([(px + a*u, py + b*u) for a, b in fold], fill)
        for top, line_width in ((11, 8), (15, 6)):
            painter.rect(px + 8*u, py + top*u, line_width*u, 1.7*u, fill)
    else:
        painter.line([(px + a*u, py + b*u) for a, b in [*points, points[0]]], tint, 1.2)
        painter.line([(px + a*u, py + b*u) for a, b in fold], tint, 1.2)
    view.targets.append(dict(kind='toggle', camera=camera, rect=rect))


def draw_field(painter, view, camera, prompt, rect):
    key = camera.as_pointer()
    edit = view.edit if view.edit and view.edit['camera'] == camera else None
    value = edit['buffer'].text if edit else prompt.prompt if prompt else ''
    lines, scroll = painter.text_field(str(key), rect, value, ('edit', None),
        enabled=not sg_modal_active(bpy.context), multiline=True,
        editing=edit['buffer'] if edit else None, scroll=view.scrolls.get(key, 0),
        follow_caret=edit is not None and edit['follow_caret'])
    view.scrolls[key] = scroll
    view.fields[key] = dict(rect=rect, lines=lines, scroll=scroll)
    view.targets.append(dict(kind='edit', camera=camera, rect=rect))
    if not value and not edit:
        painter.text('View prompt', rect[0] + 8, rect[1] + 8, 11, 'muted')


def commit(view, cancel=False):
    edit, view.edit = view.edit, None
    if not edit or cancel:
        return False
    try:
        camera, prompt = edit['camera'], edit['prompt']
        # Cancel safely if a camera or its prompt was removed/replaced externally.
        if view.scene.objects.get(camera.name) != camera:
            return False
        value = edit['buffer'].text
        if prompt is None:
            if (camera.name != edit['camera_name'] or view.scene.camera_prompts.get(camera.name) is not None
                    or not value.strip()):
                return False
            prompt = view.scene.camera_prompts.add()
            prompt.name = camera.name
        elif (camera.name != prompt.name or view.scene.camera_prompts.get(camera.name) != prompt
              or prompt.prompt != edit['original']):
            return False
        if prompt.prompt == value:
            return False
        prompt.prompt = value
        try:
            with bpy.context.temp_override(window=view.window, area=view.area, region=view.region):
                bpy.ops.ed.undo_push(message='Edit camera view prompt')
        except RuntimeError:
            _log.debug('Undo push unavailable for camera prompt edit', exc_info=True)
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        return True
    except (ReferenceError, RuntimeError):
        _log.debug('Camera prompt editor lost its scene context', exc_info=True)
        return False


class SG_OT_CameraPromptInput(bpy.types.Operator):
    bl_idname = 'stablegen.camera_prompt_input'
    bl_label = 'Edit Camera View Prompt'
    bl_description = 'Show or hide the camera view prompt below its label; click the field to add or edit wording'
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return _registered and available(context) and not sg_modal_active(context)

    def invoke(self, context, event):
        view = _views.get(view_key(context))
        target = hit(view, pointer(view, event)) if view else None
        if not target:
            return {'PASS_THROUGH'}
        self.view, self.key = view, view_key(context)
        self.cursor_active = False
        self.selecting = False
        self.exit_timer = None
        self.exit_deadline = 0
        self.activate(context, target, event)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def activate(self, context, target, event):
        view, camera = self.view, target['camera']
        key = camera.as_pointer()
        if target['kind'] == 'toggle':
            commit(view)
            if key in view.expanded:
                view.expanded.remove(key)
            else:
                view.expanded.add(key)
        elif not sg_modal_active(context):
            if not view.edit or view.edit['camera'] != camera:
                commit(view)
                prompt = view.scene.camera_prompts.get(camera.name)
                value = prompt.prompt if prompt else ''
                view.edit = dict(camera=camera, camera_name=camera.name, prompt=prompt, original=value,
                                 buffer=TextBuffer(value), follow_caret=False)
            self.place_caret(pointer(view, event), extend=event.shift)
            self.selecting = True
        self.update_cursor(pointer(view, event))
        view.area.tag_redraw()

    def update_hover(self, point=None):
        target = hit(self.view, point) if point is not None else None
        hovered = target['camera'].as_pointer() if target and target['kind'] == 'toggle' else None
        if self.view.hover != hovered:
            self.view.hover = hovered
            try:
                self.view.area.tag_redraw()
            except ReferenceError:
                pass

    def place_caret(self, point, extend=False):
        from ..ui.studio_widgets import TEXT_PADDING, TEXT_LINE_HEIGHT, font_size
        edit = self.view.edit
        field = self.view.fields.get(edit['camera'].as_pointer()) if edit else None
        if not field or not field['lines']:
            return
        x, y = point
        rx, ry, width, height = field['rect']
        row = max(0, min(len(field['lines']) - 1,
                        int((y - ry - TEXT_PADDING) / TEXT_LINE_HEIGHT) + field['scroll']))
        start, end, line = field['lines'][row]
        scale = bpy.context.preferences.system.ui_scale
        blf.size(0, font_size() * scale)
        distances = [blf.dimensions(0, line[:i])[0] / scale for i in range(len(line) + 1)]
        caret = start + min(range(len(distances)), key=lambda i: abs(distances[i] - (x - rx - TEXT_PADDING)))
        edit['buffer'].move(caret, extend)
        edit['follow_caret'] = True

    def update_cursor(self, point=None):
        target = hit(self.view, point) if point is not None else None
        active = bool(self.view.edit and target and target['kind'] == 'edit')
        try:
            if active and not self.cursor_active:
                self.view.window.cursor_modal_set('TEXT')
            elif self.cursor_active and not active:
                self.view.window.cursor_modal_restore()
        except ReferenceError:
            active = False
        self.cursor_active = active

    def finish(self, cancel=False):
        commit(self.view, cancel)
        self.update_hover()
        self.update_cursor()
        try:
            self.view.area.tag_redraw()
        except ReferenceError:
            pass

    def defer_finish(self, context):
        self.finish()
        # Finishing on an outside mouse press can suppress Blender's derived
        # CLICK event even with PASS_THROUGH. Release input ownership first and
        # finish on our own timer, leaving native selection/navigation intact.
        self.exit_timer = context.window_manager.event_timer_add(.05, window=self.view.window)
        self.exit_deadline = time.monotonic() + .05
        return {'PASS_THROUGH'}

    def remove_timer(self, context):
        if self.exit_timer is not None:
            context.window_manager.event_timer_remove(self.exit_timer)
            self.exit_timer = None

    def modal(self, context, event):
        view = self.view
        try:
            if self.exit_timer is not None:
                if event.type == 'TIMER' and time.monotonic() >= self.exit_deadline:
                    self.remove_timer(context)
                    return {'FINISHED'}
                return {'PASS_THROUGH'}
            stale = _views.get(self.key) is not view or not _registered
            if (stale or not available(context) or view_key(context) != self.key or sg_modal_active(context)):
                self.finish(cancel=stale or context.scene != view.scene or sg_modal_active(context))
                return {'FINISHED', 'PASS_THROUGH'}
            point = pointer(view, event)
            target = hit(view, point)
            key, pressed = event.type, event.value == 'PRESS'
            if key == 'WINDOW_DEACTIVATE':
                self.finish()
                return {'FINISHED', 'PASS_THROUGH'}
            if key in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
                self.update_hover(point)
                if self.selecting and view.edit:
                    self.place_caret(point, extend=True)
                    view.area.tag_redraw()
                    self.update_cursor(point)
                    return {'RUNNING_MODAL'}
                self.update_cursor(point)
                return {'RUNNING_MODAL', 'PASS_THROUGH'}
            if key == 'LEFTMOUSE' and event.value == 'RELEASE':
                self.selecting = False
                if view.edit:
                    return {'RUNNING_MODAL'}
                self.finish()
                return {'FINISHED'}
            if key == 'LEFTMOUSE' and event.value in {'PRESS', 'DOUBLE_CLICK'}:
                if target:
                    self.activate(context, target, event)
                    return {'RUNNING_MODAL'}
                return self.defer_finish(context)
            if key in {'MIDDLEMOUSE', 'RIGHTMOUSE'} and pressed:
                return self.defer_finish(context)
            if key in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE', 'WHEELOUTMOUSE', 'TRACKPADPAN'}:
                if view.edit and target and target['kind'] == 'edit' and target['camera'] == view.edit['camera']:
                    camera_key = view.edit['camera'].as_pointer()
                    change = (-2 if key in {'WHEELUPMOUSE', 'WHEELINMOUSE'} else 2)
                    if key == 'TRACKPADPAN':
                        change = round((event.mouse_prev_y - event.mouse_y) / 17)
                    view.scrolls[camera_key] = max(0, view.scrolls.get(camera_key, 0) + change)
                    view.edit['follow_caret'] = False
                    view.area.tag_redraw()
                    return {'RUNNING_MODAL'}
                return {'PASS_THROUGH'}
            if not view.edit or not pressed:
                return {'PASS_THROUGH'}
            buffer, ctrl = view.edit['buffer'], event.ctrl or event.oskey
            if key == 'ESC':
                self.finish(cancel=True)
                return {'FINISHED'}
            if key in {'RET', 'NUMPAD_ENTER'}:
                if ctrl:
                    self.finish()
                    return {'FINISHED'}
                buffer.replace('\n')
            elif ctrl and key == 'A':
                buffer.anchor, buffer.caret = 0, len(buffer.text)
            elif ctrl and key in {'C', 'X'}:
                start, end = buffer.selection
                if start != end:
                    context.window_manager.clipboard = buffer.text[start:end]
                    if key == 'X':
                        buffer.replace('')
            elif ctrl and key == 'V':
                buffer.replace(context.window_manager.clipboard.replace('\r\n', '\n').replace('\r', '\n'))
            elif ctrl and key in {'Z', 'Y'}:
                buffer.undo(redo=key == 'Y' or event.shift)
            elif key in {'LEFT_ARROW', 'RIGHT_ARROW'}:
                direction = -1 if key == 'LEFT_ARROW' else 1
                start, end = buffer.selection
                destination = ((start if direction < 0 else end) if start != end and not event.shift
                               else buffer.word(direction) if ctrl else buffer.caret + direction)
                buffer.move(destination, event.shift)
            elif key in {'HOME', 'END'}:
                if ctrl:
                    destination = 0 if key == 'HOME' else len(buffer.text)
                else:
                    destination = (buffer.text.rfind('\n', 0, buffer.caret) + 1 if key == 'HOME'
                                   else buffer.text.find('\n', buffer.caret))
                    if destination < 0:
                        destination = len(buffer.text)
                buffer.move(destination, event.shift)
            elif key in {'UP_ARROW', 'DOWN_ARROW'}:
                lines = view.fields[view.edit['camera'].as_pointer()]['lines']
                row = next((i for i, (a, b, _) in enumerate(lines) if a <= buffer.caret <= b), len(lines) - 1)
                column = buffer.caret - lines[row][0]
                row = max(0, min(len(lines) - 1, row + (-1 if key == 'UP_ARROW' else 1)))
                buffer.move(min(lines[row][1], lines[row][0] + column), event.shift)
            elif key in {'BACK_SPACE', 'DEL'}:
                buffer.delete(backward=key == 'BACK_SPACE', word=ctrl)
            elif key == 'TAB':
                self.finish()
                return {'FINISHED'}
            elif ctrl and key == 'S':
                return self.defer_finish(context)
            elif getattr(event, 'unicode', '') and not ctrl and not event.alt:
                buffer.replace(event.unicode)
            else:
                # Keyboard shortcuts must not mutate the scene behind a focused field.
                return {'RUNNING_MODAL'}
            view.edit['follow_caret'] = True
            view.area.tag_redraw()
            return {'RUNNING_MODAL'}
        except (ReferenceError, RuntimeError, KeyError):
            self.finish(cancel=True)
            return {'FINISHED', 'PASS_THROUGH'}

    def cancel(self, context):
        if hasattr(self, 'view'):
            self.remove_timer(context)
            self.finish(cancel=True)


def register():
    global _registered
    _registered = True
    if not bpy.app.background:
        config = bpy.context.window_manager.keyconfigs.addon
        if config:
            keymap = config.keymaps.new(name='3D View', space_type='VIEW_3D', region_type='WINDOW')
            item = keymap.keymap_items.new(SG_OT_CameraPromptInput.bl_idname, 'LEFTMOUSE', 'PRESS', any=True, head=True)
            _keymaps.append((keymap, item))


def unregister():
    global _registered
    _registered = False
    for keymap, item in _keymaps:
        keymap.keymap_items.remove(item)
    _keymaps.clear()
    clear()
