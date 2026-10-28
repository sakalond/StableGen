"""GPU widgets for Studio's sidebar. Bounds also define their hit targets."""
from dataclasses import dataclass
import math

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np

from . import brand
from .studio_text import wrap

FONT_SIZE = 11
CONTROL_HEIGHT = 22
CONTROL_GAP = 5
TEXT_LINE_HEIGHT = 17
TEXT_PADDING = 8

def _arc_arrow(cx, cy, radius, start, end, length=4.5, width=5.5):
    """Arc from start to end degrees (clockwise on screen when end > start) and its filled head.

    The arc stops at the head's base so a thin stroke never runs into the arrowhead.
    """
    steps = max(10, int(abs(end - start) / 8))
    arc = [(cx + radius * math.cos(math.radians(start + (end - start) * i / steps)),
            cy + radius * math.sin(math.radians(start + (end - start) * i / steps))) for i in range(steps + 1)]
    a, sign = math.radians(end), 1 if end > start else -1
    bx, by = cx + radius * math.cos(a), cy + radius * math.sin(a)
    tx, ty, nx, ny = -math.sin(a) * sign, math.cos(a) * sign, math.cos(a), math.sin(a)
    head = [(bx + tx * length, by + ty * length),
            (bx + nx * width / 2, by + ny * width / 2), (bx - nx * width / 2, by - ny * width / 2)]
    return arc, head


_SYNC = (_arc_arrow(10.5, 10.5, 7, 200, 320), _arc_arrow(10.5, 10.5, 7, 20, 140))

# Line icons on a 21-unit grid. Like Blender's SVG icons, they are rasterized
# once per pixel size into anti-aliased coverage bitmaps and drawn as textures.
ICON_PATHS = {
    'camera': [[(2, 5), (6, 5), (8, 2), (13, 2), (15, 5), (19, 5), (19, 17), (2, 17), (2, 5)]],
    'document': [[(13, 2), (4, 2), (4, 19), (17, 19), (17, 6), (13, 2), (13, 6), (17, 6)], [(7, 10), (14, 10)], [(7, 14), (12, 14)]],
    'clone': [[(8, 7), (19, 7), (19, 18), (8, 18), (8, 7)], [(14, 4), (3, 4), (3, 14), (5, 14)]],
    'mirror': [[(10, 2), (10, 19)], [(2, 6), (6, 6), (6, 15), (2, 15), (2, 6)], [(14, 6), (18, 6), (18, 15), (14, 15), (14, 6)]],
    'expand': [[(12, 2), (19, 2), (19, 9)], [(19, 2), (12, 9)], [(8, 19), (2, 19), (2, 12)], [(2, 19), (9, 12)]],
    'labels': [[(3, 3), (18, 3)], [(10, 3), (10, 18)], [(6, 18), (14, 18)]],
    'down': [[(5, 8), (10, 13), (15, 8)]],
    'up': [[(5, 13), (10, 8), (15, 13)]],
    'right': [[(7, 5), (12, 10), (7, 15)]],
    'reset': [arc for arc, _ in _SYNC],
    'search': [[(14, 14), (19, 19)]],
    'settings': [[(2, 5), (19, 5)], [(2, 10), (19, 10)], [(2, 15), (19, 15)], [(7, 3), (7, 7)], [(14, 8), (14, 12)], [(9, 13), (9, 17)]],
    'play': [[(6, 3), (18, 10), (6, 18), (6, 3)]],
    'stop': [[(5, 5), (16, 5), (16, 16), (5, 16), (5, 5)]],
    'layers': [[(10, 2), (19, 7), (10, 12), (1, 7), (10, 2)], [(1, 11), (10, 16), (19, 11)], [(1, 15), (10, 20), (19, 15)]],
    'grid': [[(3, 3), (18, 3), (18, 18), (3, 18), (3, 3)], [(3, 8), (18, 8)], [(3, 13), (18, 13)], [(8, 3), (8, 18)], [(13, 3), (13, 18)]],
    'download': [[(10, 2), (10, 13)], [(5, 8), (10, 13), (15, 8)], [(2, 14), (2, 19), (18, 19), (18, 14)]],
    'eye': [[(1, 10), (6, 5), (14, 5), (19, 10), (14, 15), (6, 15), (1, 10)]],
    'x': [[(5, 5), (16, 16)], [(5, 16), (16, 5)]],
    'plus': [[(10, 3), (10, 18)], [(3, 10), (18, 10)]],
    'minus': [[(3, 10), (18, 10)]],
    'check': [[(3, 10), (8, 15), (18, 5)]],
    'arrow': [[(3, 10), (18, 10)], [(12, 4), (18, 10), (12, 16)]],
    'widen': [[(3, 3), (3, 18)], [(18, 10.5), (7, 10.5)], [(11, 6.5), (7, 10.5), (11, 14.5)]],
    'folder': [[(2, 6), (2, 18), (19, 18), (19, 6), (10, 6), (8, 3), (2, 3), (2, 6)]],
    'light': [[(7, 15), (7, 18), (14, 18), (14, 15), (17, 11), (17, 6), (13, 2), (8, 2), (4, 6), (4, 11), (7, 15)], [(7, 15), (14, 15)]],
    'modifier': [[(3, 3), (8, 3), (8, 8), (13, 8), (13, 13), (18, 13), (18, 18), (3, 18), (3, 3)]],
    'curve': [[(2, 17), (5, 17), (8, 14), (10, 6), (13, 3), (19, 3)], [(2, 15), (2, 19)], [(19, 1), (19, 5)]],
    'material': [[(10, 2), (18, 6), (18, 15), (10, 19), (2, 15), (2, 6), (10, 2)], [(2, 6), (10, 10), (18, 6)], [(10, 10), (10, 19)]],
    'rig': [[(10, 2), (10, 9), (4, 6)], [(10, 9), (17, 5)], [(10, 9), (6, 19)], [(10, 9), (16, 18)]],
    'graph': [[(2, 2), (2, 18), (19, 18)], [(4, 14), (8, 8), (12, 11), (18, 3)]],
    'shield': [[(10, 2), (18, 5), (17, 13), (10, 19), (3, 13), (2, 5), (10, 2)]],
}
ICON_CIRCLES = {'camera': (10.5, 11, 4), 'eye': (10, 10, 2.5), 'search': (9, 9, 6)}
# Filled convex shapes, such as arrowheads that would blur into a thin stroke.
ICON_FILLS = {'reset': [head for _, head in _SYNC]}
ICON_PAD = 2
_icon_textures = {}


def _hint(path, stroke):
    """Put axis-aligned strokes on whole pixels, as Blender's pixel-grid icon designs are."""
    odd = round(stroke) % 2 == 1
    snap = (lambda v: math.floor(v) + .5) if odd else round
    hinted = [list(point) for point in path]
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(path, path[1:])):
        if abs(x1 - x2) < 1e-4:
            hinted[i][0] = hinted[i + 1][0] = snap(x1)
        if abs(y1 - y2) < 1e-4:
            hinted[i][1] = hinted[i + 1][1] = snap(y1)
    return hinted


def _rasterize(paths, extent, half_width):
    """Analytic per-pixel stroke coverage, the anti-aliasing nanosvg gives Blender's icons."""
    ys, xs = np.mgrid[0:extent, 0:extent].astype(np.float32) + .5
    coverage = np.zeros((extent, extent), np.float32)
    for path in paths:
        for (x1, y1), (x2, y2) in zip(path, path[1:]):
            dx, dy = x2 - x1, y2 - y1
            length = dx * dx + dy * dy
            t = np.clip(((xs - x1) * dx + (ys - y1) * dy) / length, 0, 1) if length else 0
            distance = np.hypot(xs - (x1 + t * dx), ys - (y1 + t * dy))
            np.maximum(coverage, np.clip(half_width + .5 - distance, 0, 1), out=coverage)
    return coverage


def _fill(coverage, polygons):
    """Add convex polygons, anti-aliased by their signed distance to the nearest edge."""
    extent = coverage.shape[0]
    ys, xs = np.mgrid[0:extent, 0:extent].astype(np.float32) + .5
    for polygon in polygons:
        # Wind counter-clockwise on screen so every edge normal points outward.
        if sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1])) > 0:
            polygon = polygon[::-1]
        distance = np.full((extent, extent), -np.inf, np.float32)
        for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
            length = math.hypot(x2 - x1, y2 - y1)
            np.maximum(distance, ((xs - x1) * -(y2 - y1) + (ys - y1) * (x2 - x1)) / length, out=distance)
        np.maximum(coverage, np.clip(.5 - distance, 0, 1), out=coverage)
    return coverage


def _icon_coverage(name, size, scale):
    """Anti-aliased coverage of an icon at this device pixel size, top row first."""
    stroke = scale  # one UI pixel, the stroke weight of Blender's own icon set
    unit = size * scale / 21
    to_pixels = lambda points: [(px * unit + ICON_PAD, py * unit + ICON_PAD) for px, py in points]
    paths = [_hint(to_pixels(path), stroke) for path in ICON_PATHS.get(name, ICON_PATHS['settings'])]
    if name in ICON_CIRCLES:
        cx, cy, radius = ICON_CIRCLES[name]
        paths.append(to_pixels([(cx + math.cos(i * math.pi / 12) * radius,
                                 cy + math.sin(i * math.pi / 12) * radius) for i in range(25)]))
    extent = math.ceil(size * scale) + ICON_PAD * 2
    coverage = _rasterize(paths, extent, stroke / 2)
    return _fill(coverage, [to_pixels(polygon) for polygon in ICON_FILLS.get(name, ())])


def _icon_texture(name, size, scale):
    """Return a cached (texture, extent) for an icon at this device pixel size."""
    key = (name, round(size * scale, 3))
    if key not in _icon_textures:
        coverage = _icon_coverage(name, size, scale)
        extent = coverage.shape[0]
        rgba = np.ones((extent, extent, 4), np.float32)
        rgba[..., 3] = coverage[::-1]  # GPU rows start at the bottom
        data = gpu.types.Buffer('FLOAT', rgba.size, rgba.ravel().tolist())
        _icon_textures[key] = gpu.types.GPUTexture((extent, extent), format='RGBA16F', data=data), extent
    return _icon_textures[key]


def _logo_texture(size, scale):
    """Return a cached (texture, extent) for the full-colour logo at this device pixel size."""
    extent = round(size * scale)
    key = ('logo', extent)
    if key not in _icon_textures:
        rgba = brand.rasterize(brand.LOGO, extent, brand.LOGO_BOX)[::-1]  # GPU rows start at the bottom
        data = gpu.types.Buffer('FLOAT', rgba.size, np.ascontiguousarray(rgba).ravel().tolist())
        _icon_textures[key] = gpu.types.GPUTexture((extent, extent), format='RGBA16F', data=data), extent
    return _icon_textures[key]


def _text_shadow(scale):
    """Blender's widget font-style shadow as (blf level, value, alpha, x, y), or None."""
    styles = bpy.context.preferences.ui_styles
    if not styles or styles[0].widget.shadow <= 0:
        return None
    style = styles[0].widget
    # The text shader blurs types 1-4 with 3x3, 5 and 7+ with 5x5; 6 is an outline.
    level = 3 if style.shadow <= 4 else 6 if style.shadow == 6 else 5
    return (level, style.shadow_value, style.shadow_alpha,
            int(style.shadow_offset_x * scale), int(style.shadow_offset_y * scale))


def font_size(size=FONT_SIZE):
    """Use Blender's widget font size; coordinates already follow its UI scale."""
    styles = bpy.context.preferences.ui_styles
    return size * styles[0].widget.points / FONT_SIZE if styles else size


def color(value):
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (1,)


COLORS = {key: color(value) for key, value in {
    'panel': '3d3d3d', 'field': '1d1d1d', 'field_text': 'e6e6e6',
    'number': '545454', 'number_text': 'e6e6e6', 'menu': '282828', 'menu_text': 'e6e6e6',
    'raised': '545454', 'line': '494949', 'divider': '464646', 'text': 'e6e6e6',
    'secondary': 'c4c4c4', 'muted': '999999', 'accent': '8fa6ca',
    'blue': '4772b3', 'on_accent': 'ffffff', 'hover': '545454', 'disabled': '787878',
    'warning': 'c0a673', 'selected': '404c5e', 'card': '353535',
    'text_selected': '4a4a4a', 'number_selected': '6e6e6e',
    'cursor': '71a8ff', 'switch_off': '545454', 'switch_knob': 'ffffff',
    'outer_edge': '494949',
    'hover_border': '858585', 'hover_soft': '454545', 'blue_hover': '5980bd',
    'switch_off_hover': '666666', 'tooltip': '191919',
    'tooltip_text': 'e6e6e6', 'tooltip_line': '555555',
}.items()}


def _rgba(value):
    value = tuple(value)
    return value if len(value) == 4 else value + (1,)


def _mix(a, b, factor):
    return tuple(a[i] * (1 - factor) + b[i] * factor for i in range(3)) + (1,)


def _over(foreground, background):
    foreground = _rgba(foreground)
    return _mix(background, foreground, foreground[3])


def theme_colors(context):
    """Read the active native palette each draw, including unsaved theme edits."""
    theme = context.preferences.themes[0]
    ui = theme.user_interface
    space = theme.view_3d.space
    # Panel colors moved from editor spaces to User Interface in Blender 5.0.
    panels = getattr(space, 'panelcolors', None)
    back = getattr(ui, 'panel_back', getattr(panels, 'back', COLORS['panel']))
    panel = _over(back, COLORS['panel'])
    text = _rgba(getattr(ui, 'panel_text', space.text))
    border = _over(getattr(ui, 'panel_outline', (1, 1, 1, .067)), panel)
    tool, field, number, menu = ui.wcol_tool, ui.wcol_text, ui.wcol_num, ui.wcol_menu
    blue = _over(tool.inner_sel, panel)
    field_back = _over(field.inner, panel)
    card = _over(getattr(ui, 'panel_sub_back', (0, 0, 0, .12)), panel)
    subbox = _mix(card, field_back, .2)
    section_border = _mix(border, text, .075)
    return {
        'panel': panel, 'field': field_back, 'field_text': _rgba(field.text),
        'number': _over(number.inner, panel), 'number_text': _rgba(number.text),
        'menu': _over(menu.inner, panel), 'menu_text': _rgba(menu.text),
        'field_border': _over(field.outline, panel), 'number_border': _over(number.outline, panel),
        'menu_border': _over(menu.outline, panel), 'tool_border': _over(tool.outline, panel),
        'raised': _over(tool.inner, panel), 'line': border, 'divider': _mix(panel, border, .7),
        'outer_edge': border, 'panel_shadow': _rgba(ui.widget_emboss),
        'text': text, 'secondary': _mix(text, panel, .2),
        'muted': _mix(text, panel, .3), 'disabled': _mix(text, panel, .65),
        'blue': blue, 'on_accent': _rgba(tool.text_sel),
        # Colored text needs contrast on both Blender's dark and light themes.
        'accent': _mix(blue, text, .45), 'cursor': _rgba(ui.widget_text_cursor),
        'hover': _mix(_over(tool.inner, panel), text, .08),
        'hover_soft': _mix(panel, text, .055), 'hover_border': _mix(border, text, .3),
        'blue_hover': _mix(blue, _rgba(tool.text_sel), .12),
        'selected': _mix(panel, blue, .28),
        'tab_selected': _mix(card, blue, .14),
        'card': card, 'section_header': _mix(panel, text, .085), 'section_body': card,
        'section_border': section_border, 'subbox': subbox,
        'group_body': _mix(panel, subbox, .3), 'group_border': _mix(section_border, text, .16),
        'dependency_rail': _mix(panel, text, .23),
        'text_selected': _over(field.item, field_back),
        'number_selected': _mix(_over(number.inner, panel), _rgba(number.text), .18),
        'warning': _mix(_rgba(getattr(ui.wcol_state, 'warning', COLORS['warning'])), text, .35),
        'switch_off': _over(ui.wcol_toggle.inner, panel), 'switch_knob': _rgba(ui.wcol_option.item),
        'switch_off_hover': _mix(_over(ui.wcol_toggle.inner, panel), text, .14),
        'tooltip': _over(ui.wcol_tooltip.inner, panel),
        'tooltip_text': _rgba(ui.wcol_tooltip.text),
        'tooltip_line': _over(ui.wcol_tooltip.outline, _over(ui.wcol_tooltip.inner, panel)),
    }


@dataclass
class Target:
    key: str
    rect: tuple
    action: object
    enabled: bool = True
    hint: str = ''
    clip: tuple | None = None

    @property
    def cursor(self):
        if not self.enabled:
            return 'DEFAULT'
        kind = self.action[0] if isinstance(self.action, tuple) else ''
        return 'TEXT' if kind in {'edit', 'menu-lora-search'} else 'MOVE_X' if kind == 'slider' else 'MOVE_Y' if kind == 'camera-drag' else 'DEFAULT'

    def contains(self, x, y):
        rx, ry, width, height = self.rect
        inside = rx <= x <= rx + width and ry <= y <= ry + height
        if inside and self.clip:
            cx, cy, cw, ch = self.clip
            inside = cx <= x <= cx + cw and cy <= y <= cy + ch
        return inside


class Painter:
    def __init__(self, region, scale=1, font=0):
        self.region, self.scale, self.font = region, scale, font
        self.colors = theme_colors(bpy.context)
        self.text_shadow = _text_shadow(scale)
        self.width = region.width / scale - 28
        self.height = region.height / scale
        self.shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        self.targets = []
        self.hover_key = ''
        self.layers = []
        self.clip_rect = None
        self.canvas_clip = None
        self.previous_blend = gpu.state.blend_get()
        self.previous_scissor = gpu.state.scissor_get()
        self.viewport = gpu.state.viewport_get()
        gpu.state.blend_set('ALPHA')
        gpu.state.scissor_test_set(False)

    def finish(self):
        self.canvas_clip = None
        self.clip(None)
        gpu.state.scissor_set(*self.previous_scissor)
        gpu.state.blend_set(self.previous_blend)

    def clip(self, rect):
        if self.canvas_clip:
            if rect is None:
                rect = self.canvas_clip
            else:
                x, y = max(rect[0], self.canvas_clip[0]), max(rect[1], self.canvas_clip[1])
                right = min(rect[0] + rect[2], self.canvas_clip[0] + self.canvas_clip[2])
                bottom = min(rect[1] + rect[3], self.canvas_clip[1] + self.canvas_clip[3])
                rect = (x, y, max(0, right - x), max(0, bottom - y))
        self.clip_rect = rect
        gpu.state.scissor_test_set(rect is not None)
        if rect:
            x, y, w, h = rect
            gpu.state.scissor_set(int(self.viewport[0] + x * self.scale),
                                 int(self.viewport[1] + (self.height - y - h) * self.scale),
                                 max(0, int(w * self.scale)), max(0, int(h * self.scale)))

    def panel(self):
        """Cover the native backing and bring the frame to the sidebar's left edge."""
        # Cover the native backing panel first, including the rounded corners and
        # gutter; its offset outline must not show through beside our own frame.
        self.rect(0, 0, self.width + 1, self.height, 'panel')
        x, y, width, height = 0, 2, self.width, self.height - 4
        ui = bpy.context.preferences.themes[0].user_interface
        radius = max(0, min(8, ui.panel_roundness * 10))
        self.rect(x, y + 1, width, height + 1, 'panel_shadow', radius + 1)
        self.outline(x, y, width, height, 'panel', 'outer_edge', radius)
        # Header/footer painting must not overwrite the frame or its curved corners.
        inset = max(1, radius)
        self.canvas_clip = (x + 1, y + inset, width - 2, height - inset * 2)
        self.clip(None)

    def begin_group(self):
        """Collect drawing once, so the measured group's backdrop can go behind it."""
        self.layers.append([])

    def end_group(self, rect, fill='section_body', border='section_border', radius=4):
        self.end_group_row([rect], fill, border, radius)

    def end_group_row(self, rects, fill='section_body', border='section_border', radius=4, *, styles=None):
        """Give adjacent cards a shared measured height without drawing twice."""
        commands = self.layers.pop()
        previous_clip = self.clip_rect
        for i, rect in enumerate(rects):
            card_fill, card_border = styles[i] if styles else (fill, border)
            self.outline(*rect, card_fill, card_border, radius)
        if self.layers:
            self.layers[-1].extend(commands)
        else:
            try:
                for method, args, clip in commands:
                    if clip != self.clip_rect:
                        self.clip(clip)
                    method(*args)
            finally:
                if previous_clip != self.clip_rect:
                    self.clip(previous_clip)

    def emit(self, method, *args):
        if self.layers:
            self.layers[-1].append((method, args, self.clip_rect))
        else:
            method(*args)

    def polygon(self, points, tint):
        self.emit(self.draw_polygon, tuple(points), tint)

    def draw_polygon(self, points, tint):
        if len(points) < 3:
            return
        vertices = [(x * self.scale, (self.height - y) * self.scale) for x, y in points]
        self.shader.bind()
        self.shader.uniform_float('color', self.colors.get(tint, tint))
        batch_for_shader(self.shader, 'TRIS', {'pos': vertices},
                         indices=[(0, i, i + 1) for i in range(1, len(vertices) - 1)]).draw(self.shader)

    def rect(self, x, y, width, height, tint, radius=0, *, corners='ALL'):
        if width <= 0 or height <= 0:
            return
        radius = min(radius, width / 2, height / 2)
        if radius:
            points = []
            right = radius if corners in {'ALL', 'RIGHT'} else 0
            left = radius if corners in {'ALL', 'LEFT'} else 0
            for cx, cy, angle, r in ((x + width - right, y + right, -90, right),
                                     (x + width - right, y + height - right, 0, right),
                                     (x + left, y + height - left, 90, left),
                                     (x + left, y + left, 180, left)):
                if not r:
                    points.append((cx, cy))
                    continue
                for j in range(5):
                    a = math.radians(angle + j * 22.5)
                    points.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
        else:
            points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        self.polygon(points, tint)

    def outline(self, x, y, width, height, fill='field', border='line', radius=4, *, corners='ALL'):
        self.rect(x, y, width, height, border, radius, corners=corners)
        self.rect(x + 1, y + 1, width - 2, height - 2, fill, max(0, radius - 1), corners=corners)

    def line(self, points, tint='muted', thickness=1):
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            length = math.hypot(x2 - x1, y2 - y1)
            if length:
                dx, dy = -(y2 - y1) / length * thickness / 2, (x2 - x1) / length * thickness / 2
                self.polygon([(x1 + dx, y1 + dy), (x2 + dx, y2 + dy),
                              (x2 - dx, y2 - dy), (x1 - dx, y1 - dy)], tint)

    def dependency_rail(self, x, top, last):
        """One straight rail through the dependency scope, without branches."""
        self.line([(x, top), (x, last)], 'dependency_rail', 1 / self.scale)

    def measure(self, text, size=FONT_SIZE):
        blf.size(self.font, font_size(size) * self.scale)
        return blf.dimensions(self.font, text)[0] / self.scale

    def fit(self, text, width, size=FONT_SIZE):
        if self.measure(text, size) <= width:
            return text
        while text and self.measure(text + '…', size) > width:
            text = text[:-1]
        return text + '…' if text else ''

    def tag(self, text, right, y, *, enabled=True):
        """A small rounded label ending at *right*; returns its width."""
        width = self.measure(text, 9) + 10
        self.outline(right - width, y, width, 15, 'hover_soft', 'line', 3)
        self.text(text, right - width + 5, y + 2, 9, 'secondary' if enabled else 'disabled')
        return width

    def text(self, text, x, y, size=FONT_SIZE, tint='text', width=None):
        text = str(text)
        if width is not None:
            text = self.fit(text, width, size)
        self.emit(self.draw_text, text, x, y, size, tint)

    def draw_text(self, text, x, y, size, tint):
        size = font_size(size)
        blf.size(self.font, size * self.scale)
        blf.color(self.font, *self.colors.get(tint, tint))
        blf.position(self.font, x * self.scale, (self.height - y - size) * self.scale, 0)
        if self.text_shadow:
            level, value, alpha, offset_x, offset_y = self.text_shadow
            blf.enable(self.font, blf.SHADOW)
            blf.shadow(self.font, level, value, value, value, alpha)
            blf.shadow_offset(self.font, offset_x, offset_y)
        blf.draw(self.font, text)
        if self.text_shadow:
            blf.disable(self.font, blf.SHADOW)
        # blf.draw leaves blending off; translucent shapes and icon edges need it.
        gpu.state.blend_set('ALPHA')

    def image(self, image, rect):
        self.emit(self.draw_image, image, rect)

    def draw_image(self, image, rect):
        x, y, width, height = rect
        iw, ih = image.size
        if iw <= 0 or ih <= 0:
            return
        ratio = min(width / iw, height / ih)
        w, h = iw * ratio, ih * ratio
        x, y = x + (width - w) / 2, y + (height - h) / 2
        shader = gpu.shader.from_builtin('IMAGE')
        texture = gpu.texture.from_image(image)
        shader.bind()
        shader.uniform_sampler('image', texture)
        positions = [(px * self.scale, (self.height - py) * self.scale)
                     for px, py in ((x, y+h), (x+w, y+h), (x+w, y), (x, y))]
        batch_for_shader(shader, 'TRI_FAN', {'pos': positions,
                         'texCoord': [(0,0), (1,0), (1,1), (0,1)]}).draw(shader)

    def icon(self, name, x, y, size=15, tint='secondary'):
        if name == 'grip':
            diameter = max(2, round(size * self.scale / 7)) / self.scale
            gap = 2 * diameter
            for row in range(3):
                for column in range(2):
                    self.rect(x + size / 2 + (column - .5) * gap - diameter / 2,
                              y + size / 2 + (row - 1) * gap - diameter / 2,
                              diameter, diameter, tint, diameter / 2)
            return
        if name == 'more':
            diameter = max(2, round(size * self.scale / 4))
            gap = max(1, round(size * self.scale / 8))
            left = round((x + size / 2) * self.scale - (3 * diameter + 2 * gap) / 2)
            top = round((y + size / 2) * self.scale - diameter / 2)
            for i in range(3):
                self.rect((left + i * (diameter + gap)) / self.scale, top / self.scale,
                          diameter / self.scale, diameter / self.scale, tint, diameter / (2 * self.scale))
            return
        self.emit(self.draw_icon, name, x, y, size, tint)

    def draw_icon(self, name, x, y, size, tint):
        texture, extent = _icon_texture(name, size, self.scale)
        r, g, b, a = self.colors.get(tint, tint)
        a *= bpy.context.preferences.themes[0].user_interface.icon_alpha
        # Whole device pixels, like Blender's widget icons at 1:1 zoom.
        left = round(x * self.scale) - ICON_PAD
        top = self.region.height - round(y * self.scale) + ICON_PAD
        shader = gpu.shader.from_builtin('IMAGE_COLOR')
        shader.bind()
        shader.uniform_sampler('image', texture)
        shader.uniform_float('color', (r, g, b, a))
        batch_for_shader(shader, 'TRI_FAN', {
            'pos': [(left, top - extent), (left + extent, top - extent), (left + extent, top), (left, top)],
            'texCoord': [(0, 0), (1, 0), (1, 1), (0, 1)]}).draw(shader)

    def logo(self, x, y, size):
        self.emit(self.draw_logo, x, y, size)

    def draw_logo(self, x, y, size):
        texture, extent = _logo_texture(size, self.scale)
        alpha = bpy.context.preferences.themes[0].user_interface.icon_alpha
        left = round(x * self.scale)
        top = self.region.height - round(y * self.scale)
        shader = gpu.shader.from_builtin('IMAGE_COLOR')
        shader.bind()
        shader.uniform_sampler('image', texture)
        shader.uniform_float('color', (1, 1, 1, alpha))
        batch_for_shader(shader, 'TRI_FAN', {
            'pos': [(left, top - extent), (left + extent, top - extent), (left + extent, top), (left, top)],
            'texCoord': [(0, 0), (1, 0), (1, 1), (0, 1)]}).draw(shader)

    def target(self, key, rect, action, enabled=True, hint=''):
        target = Target(key, rect, action, enabled, hint, self.clip_rect)
        self.targets.append(target)
        return target

    def button(self, key, rect, text, action, *, icon=None, style='quiet', enabled=True, hover=False, active=False, hint='', text_size=FONT_SIZE, corners='ALL'):
        hover = enabled and (hover or key == self.hover_key)
        x, y, w, h = rect
        tint = 'disabled' if not enabled else 'on_accent' if style == 'primary' else 'accent' if active else 'text'
        if style == 'primary':
            self.rect(x, y, w, h, 'blue_hover' if hover else 'blue' if enabled else 'raised', 4, corners=corners)
        elif style == 'outlined':
            self.outline(x, y, w, h, 'hover' if hover else 'raised', 'hover_border' if hover else 'tool_border', corners=corners)
        elif hover and enabled and style != 'tab':
            self.rect(x + 2, y + 2, w - 4, h - 4, 'hover')
        icon_size = 16 if icon == 'more' else 13
        icon_advance = icon_size + 4
        tw = self.measure(text, text_size) if text else 0
        available = w - (icon_advance + 9 if icon and text else 18)
        tw = min(tw, max(0, available))
        start = x + (w - tw - (icon_advance if icon and text else icon_size if icon else 0)) / 2
        if icon:
            self.icon(icon, start, y + (h - icon_size) / 2, icon_size, tint=tint)
        if text:
            self.text(text, start + (icon_advance if icon else 0), y + (h - font_size(text_size)) / 2 - 1, text_size, tint, max(0, available))
        self.target(key, rect, action, enabled, hint)

    def action_row(self, key, rect, label, action, *, icon='settings', enabled=True, hint=''):
        """A left-aligned command in a utility list, with one full-row hit area."""
        x, y, width, height = rect
        hover = enabled and key == self.hover_key
        if hover:
            self.rect(x, y, width, height, 'hover_soft')
        tint = 'text' if enabled else 'disabled'
        self.icon(icon, x + 8, y + (height - 14) / 2, 14, 'secondary' if enabled else 'disabled')
        self.text(label, x + 30, y + (height - font_size()) / 2 - 1, FONT_SIZE, tint, width - 52)
        self.icon('right', x + width - 18, y + (height - 10) / 2, 10,
                  'accent' if hover else 'muted' if enabled else 'disabled')
        self.target(key, rect, action, enabled, hint)

    def selection(self, key, rect, label, action, selected, *, enabled=True, hint='', accessory_width=0):
        """A selected item is a row with a radio marker, separate from actions."""
        x, y, width, height = rect
        hover = enabled and key == self.hover_key
        self.outline(x, y, width, height, 'selected' if selected else 'hover_soft' if hover else 'menu',
                     'accent' if selected else 'hover_border' if hover else 'menu_border', 3)
        tint = 'accent' if selected and enabled else 'secondary' if enabled else 'disabled'
        self.rect(x + 10, y + (height - 12) / 2, 12, 12, tint, 6)
        self.rect(x + 11, y + (height - 10) / 2, 10, 10,
                  'selected' if selected else 'hover_soft' if hover else 'menu', 5)
        if selected:
            self.rect(x + 13, y + (height - 6) / 2, 6, 6, tint, 3)
        self.text(label, x + 30, y + (height - font_size()) / 2 - 1,
                  FONT_SIZE, 'text' if enabled else 'disabled', width - accessory_width - 38)
        self.target(key, (x, y, width - accessory_width, height), action, enabled, hint)

    def switch(self, key, rect, label, checked, action, *, enabled=True, hint='', hover=False):
        hover = enabled and (hover or key == self.hover_key)
        x, y, w, h = rect
        tint = 'text' if enabled else 'disabled'
        if hover:
            self.rect(x, y, w, h, 'hover_soft')
        lines = wrap(label, w - 40, self.measure)
        for i, (_, _, line) in enumerate(lines):
            self.text(line, x, y + (h - len(lines) * 16) / 2 + i * 16, FONT_SIZE, tint)
        sx, sy = x + w - 28, y + (h - 14) / 2
        self.rect(sx, sy, 28, 14, ('blue_hover' if hover else 'blue') if checked and enabled
                  else 'switch_off_hover' if hover else 'switch_off', 7)
        self.rect(sx + (16 if checked else 2), sy + 2, 10, 10, 'switch_knob' if enabled else 'muted', 5)
        self.target(key, rect, action, enabled, hint)

    def text_field(self, key, rect, value, action, *, enabled=True, hint='', editing=None, scroll=0, multiline=False, follow_caret=True, kind='field'):
        x, y, w, h = rect
        self.outline(x, y, w, h, fill=kind, border='cursor' if editing else 'hover_border'
                     if enabled and key == self.hover_key else kind + '_border')
        self.target(key, rect, action, enabled, hint)
        if not multiline and editing is None:
            self.text(value, x + TEXT_PADDING, y + (h - font_size()) / 2 - 1, FONT_SIZE,
                      kind + '_text' if enabled else 'disabled', w - TEXT_PADDING * 2)
            return [], 0
        lines = wrap(value, w - TEXT_PADDING * 2, self.measure)
        line_height = TEXT_LINE_HEIGHT
        padding_y = TEXT_PADDING if multiline else (h - font_size()) / 2 - 1
        count = max(1, int((h - TEXT_PADDING * 2) / line_height)) if multiline else 1
        if editing:
            caret_row = next((i for i, (start, end, _) in enumerate(lines)
                              if start <= editing.caret <= end and (editing.caret != end or i == len(lines) - 1 or lines[i + 1][0] != end)), len(lines) - 1)
            if follow_caret and caret_row < scroll:
                scroll = caret_row
            elif follow_caret and caret_row >= scroll + count:
                scroll = caret_row - count + 1
        scroll = max(0, min(max(0, len(lines) - count), scroll))
        for index, (start, end, line) in enumerate(lines[scroll:scroll + count], scroll):
            ly = y + padding_y + (index - scroll) * line_height
            if editing:
                sel_start, sel_end = editing.selection
                a, b = max(start, sel_start), min(end, sel_end)
                if b > a:
                    lx = x + TEXT_PADDING + self.measure(value[start:a])
                    self.rect(lx, ly - 1, self.measure(value[a:b]), line_height,
                              'number_selected' if kind == 'number' else 'text_selected')
            self.text(line.rstrip('\r'), x + TEXT_PADDING, ly, FONT_SIZE, kind + '_text' if enabled else 'disabled')
            if editing and index == caret_row:
                cx = x + TEXT_PADDING + self.measure(value[start:editing.caret])
                self.rect(cx, ly, 1, font_size() + 3, 'cursor')
        if len(lines) > count:
            track = h - 12
            self.rect(x + w - 5, y + 6 + scroll / len(lines) * track, 2,
                      max(10, count / len(lines) * track), 'muted', 1)
        return lines, scroll
