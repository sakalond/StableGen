"""GPU widgets for Studio's sidebar. Bounds also define their hit targets."""
from dataclasses import dataclass
import math
import os

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
    'left': [[(13, 5), (8, 10), (13, 15)]],
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
    'rename': [[(3, 18), (3, 14), (14, 3), (18, 7), (7, 18), (3, 18)], [(12, 5), (16, 9)]],
    'server': [[(3, 3), (18, 3), (18, 9), (3, 9), (3, 3)], [(3, 12), (18, 12), (18, 18), (3, 18), (3, 12)],
               [(11, 6), (15, 6)], [(11, 15), (15, 15)]],
    'image': [[(2, 3), (19, 3), (19, 18), (2, 18), (2, 3)], [(2, 15), (7, 10), (11, 14), (14, 11), (19, 16)]],
    'cube': [[(10, 2), (18, 6), (18, 15), (10, 19), (2, 15), (2, 6), (10, 2)], [(2, 6), (10, 10), (18, 6)],
             [(10, 10), (10, 19)]],
    'trash': [[(3, 5), (18, 5)], [(8, 5), (8, 2.5), (13, 2.5), (13, 5)], [(5, 5), (6, 19), (15, 19), (16, 5)],
              [(8.5, 9), (8.5, 15.5)], [(12.5, 9), (12.5, 15.5)]],
    'warning': [[(10.5, 2), (19.5, 18.5), (1.5, 18.5), (10.5, 2)], [(10.5, 8), (10.5, 12.5)]],
}
ICON_CIRCLES = {'camera': (10.5, 11, 4), 'eye': (10, 10, 2.5), 'search': (9, 9, 6), 'server': (6.5, 6, 1),
                'image': (14, 7.5, 1.6), 'warning': (10.5, 15.5, .6)}
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


_thumbnail_sources = {}     # (image name, size cap) -> (image key, premultiplied RGBA uint16, rows bottom first)
_thumbnail_textures = {}
_overlay_textures = {}
THUMBNAIL_SOURCE = 1024     # longest side kept for an image's thumbnails; a full-size preview keeps up to 4096


def _image_key(image):
    """Changes when the image's pixels may have: another file version, an edit, a resize."""
    path = bpy.path.abspath(image.filepath, library=image.library) if image.filepath else ''
    try:
        stamp = os.path.getmtime(path) if path else 0.0
    except OSError:
        stamp = 0.0
    return image.name, image.as_pointer(), tuple(image.size), image.is_dirty, stamp


def _thumbnail_source(image, cap=THUMBNAIL_SOURCE):
    """The image as premultiplied, display-referred RGBA at most *cap* pixels long. Premultiplied, so
    resampling never pulls in the colour a generator leaves under transparent pixels (Qwen's purple)."""
    import cv2
    key = _image_key(image)
    cached = _thumbnail_sources.get((image.name, cap))
    if cached and cached[0] == key:
        return cached[1]
    width, height = image.size
    channels = image.channels
    if width <= 0 or height <= 0 or channels not in (1, 3, 4):
        return None
    pixels = np.empty(width * height * channels, np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(height, width, channels)
    if channels == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    rgb, alpha = pixels[..., :3], (pixels[..., 3:4] if channels == 4 else np.ones((height, width, 1), np.float32))
    if image.is_float and 'srgb' not in image.colorspace_settings.name.lower():
        # Float buffers hold scene-linear values; byte buffers are already display-encoded.
        rgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(np.maximum(rgb, 0), 1 / 2.4) - 0.055)
    rgba = np.concatenate([np.clip(rgb, 0, 1) * alpha, np.clip(alpha, 0, 1)], axis=2)
    scale = min(1.0, cap / max(width, height))
    if scale < 1:
        rgba = cv2.resize(rgba, (max(1, round(width * scale)), max(1, round(height * scale))),
                          interpolation=cv2.INTER_AREA)
    source = (rgba * 65535).round().astype(np.uint16)
    if len(_thumbnail_sources) > 64:
        _thumbnail_sources.clear()
    _thumbnail_sources[(image.name, cap)] = (key, source)
    return source


def _file_key(path):
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return os.path.normcase(os.path.abspath(path)), stat.st_mtime_ns, stat.st_size


_file_sizes = {}    # file path -> (its _file_key, (width, height))


def image_size(image):
    """(width, height) of a Blender image or an image file: a PNG's header, else the decoded file."""
    if not isinstance(image, str):
        return tuple(image.size)
    key = _file_key(image)
    if key is None:
        return 0, 0
    cached = _file_sizes.get(key[0])
    if cached is None or cached[0] != key:
        size = (0, 0)
        try:
            with open(image, 'rb') as handle:
                head = handle.read(24)
            if head[:8] == b'\x89PNG\r\n\x1a\n':
                size = int.from_bytes(head[16:20], 'big'), int.from_bytes(head[20:24], 'big')
        except OSError:
            pass
        if size == (0, 0):
            source = _file_source(image)
            if source is not None:
                size = source.shape[1], source.shape[0]
        cached = _file_sizes[key[0]] = (key, size)
    return cached[1]


def _file_source(path, cap=THUMBNAIL_SOURCE):
    """An image file as _thumbnail_source gives a Blender image (premultiplied RGBA, bottom row first), read
    straight from the file: drawing may not load image datablocks. None when it can't be read."""
    import cv2
    key = _file_key(path)
    if key is None:
        return None
    cached = _thumbnail_sources.get((key[0], cap))
    if cached and cached[0] == key:
        return cached[1]
    try:
        pixels = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)     # works with any path's letters
    except (OSError, ValueError, cv2.error):
        pixels = None
    if pixels is None:
        return None
    depth = 65535.0 if pixels.dtype == np.uint16 else 1.0 if pixels.dtype.kind == 'f' else 255.0
    pixels = pixels.astype(np.float32) / depth
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    if pixels.shape[2] == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    if pixels.shape[2] == 3:
        pixels = np.concatenate([pixels, np.ones(pixels.shape[:2] + (1,), np.float32)], axis=2)
    rgba = np.clip(pixels[::-1, :, [2, 1, 0, 3]], 0, 1)        # OpenCV's BGRA, top row first
    rgba = np.concatenate([rgba[..., :3] * rgba[..., 3:4], rgba[..., 3:4]], axis=2)
    height, width = rgba.shape[:2]
    scale = min(1.0, cap / max(width, height))
    if scale < 1:
        rgba = cv2.resize(rgba, (max(1, round(width * scale)), max(1, round(height * scale))),
                          interpolation=cv2.INTER_AREA)
    source = (rgba * 65535).round().astype(np.uint16)
    if len(_thumbnail_sources) > 64:
        _thumbnail_sources.clear()
    _thumbnail_sources[(key[0], cap)] = (key, source)
    return source


def _thumbnail_texture(image, width, height, check):
    """A width x height device-pixel texture of the image (a Blender image, or an image file's path) over the
    theme's transparency checkerboard."""
    import cv2
    ui = bpy.context.preferences.themes[0].user_interface
    primary = tuple(ui.transparent_checker_primary)
    secondary = tuple(ui.transparent_checker_secondary)
    from_file = isinstance(image, str)
    key = (_file_key(image) if from_file else _image_key(image), width, height, check, primary, secondary)
    if key not in _thumbnail_textures:
        source = (_file_source if from_file else _thumbnail_source)(
            image, THUMBNAIL_SOURCE if max(width, height) <= THUMBNAIL_SOURCE else 4096)
        if source is None:
            return None
        small = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 65535
        rows, columns = np.mgrid[0:height, 0:width]
        board = np.where((((rows // check) + (columns // check)) % 2 == 0)[..., None],
                         np.float32(primary), np.float32(secondary))
        rgba = np.ones((height, width, 4), np.float32)
        rgba[..., :3] = small[..., :3] + board * (1 - small[..., 3:4])
        if len(_thumbnail_textures) > 128:
            _thumbnail_textures.clear()
        data = _float_buffer(rgba)
        _thumbnail_textures[key] = gpu.types.GPUTexture((width, height), format='RGBA16F', data=data)
    return _thumbnail_textures[key]


def _float_buffer(array):
    """A GPU float buffer of a numpy array, without the per-element list a full-size preview would stall on."""
    flat = np.ascontiguousarray(array, np.float32).ravel()
    try:
        return gpu.types.Buffer('FLOAT', flat.size, flat)
    except (TypeError, ValueError):
        return gpu.types.Buffer('FLOAT', flat.size, flat.tolist())


def _array_texture(rgba):
    """A texture of a small RGBA float array (rows bottom first), cached while the same array is drawn."""
    key = (id(rgba), rgba.shape)
    cached = _overlay_textures.get(key)
    if cached is None or cached[0] is not rgba:
        if len(_overlay_textures) > 16:
            _overlay_textures.clear()
        height, width = rgba.shape[:2]
        data = _float_buffer(rgba)
        cached = _overlay_textures[key] = (rgba, gpu.types.GPUTexture((width, height), format='RGBA16F', data=data))
    return cached[1]


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
    'number': '545454', 'number_text': 'e6e6e6', 'number_hover': '646464', 'number_hot': '797979',
    'menu': '282828', 'menu_text': 'e6e6e6', 'scroll_track': '2f2f2f',
    'raised': '545454', 'line': '494949', 'divider': '464646', 'text': 'e6e6e6',
    'secondary': 'c4c4c4', 'muted': '999999', 'accent': '8fa6ca',
    'blue': '4772b3', 'on_accent': 'ffffff', 'hover': '545454', 'disabled': '787878',
    'warning': 'c0a673', 'selected': '404c5e', 'card': '353535', 'ok': '7dba82',
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


def _shade(color, amount):
    """Lighten like Blender's widget hover shading, which adds to each channel."""
    return tuple(max(0, min(1, c + amount)) for c in color[:3]) + (1,)


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
    warning = _mix(_rgba(getattr(ui.wcol_state, 'warning', COLORS['warning'])), text, .35)
    return {
        'panel': panel, 'field': field_back, 'field_text': _rgba(field.text),
        'number': _over(number.inner, panel), 'number_text': _rgba(number.text),
        # A hovered number field lightens, and the part under the pointer more (measured on Blender's own).
        'number_hover': _shade(_over(number.inner, panel), 16 / 255),
        'number_hot': _shade(_over(number.inner, panel), 37 / 255),
        'scroll_track': _mix(panel, (0, 0, 0, 1), .22),
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
        'warning': warning,
        # a model that isn't installed: the tab switcher's selected tab in amber, and its hovered segment
        'warning_tab': _mix(card, warning, .17), 'warning_strong': _mix(panel, warning, .3),
        'ok': _mix(COLORS['ok'], text, .2),     # installed, running: the theme has no success color
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

    def number_zone(self, x):
        """Blender's arrow zones: min(a third of the width, 0.7 x the height) at each end."""
        rx, _, width, height = self.rect
        handle = min(width / 3, height * .7)
        return 'left' if x < rx + handle else 'right' if x > rx + width - handle else 'middle'

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

    def image(self, image, rect, check=8):
        """Draw *image* (a Blender image, or an image file's path) fitted inside *rect*, transparent parts over the
        theme's checkerboard (*check* UI pixels a square); returns the rectangle it fills, None for a file it can't
        read."""
        x, y, width, height = rect
        if isinstance(image, str):
            source = _file_source(image)
            if source is None:
                return None
            ih, iw = source.shape[:2]
        else:
            iw, ih = image.size
        if iw <= 0 or ih <= 0:
            return None
        ratio = min(width / iw, height / ih)
        w = max(1, round(iw * ratio * self.scale)) / self.scale
        h = max(1, round(ih * ratio * self.scale)) / self.scale
        x = round((x + (width - w) / 2) * self.scale) / self.scale
        y = round((y + (height - h) / 2) * self.scale) / self.scale
        self.emit(self.draw_image, image, (x, y, w, h), check)
        return x, y, w, h

    def draw_image(self, image, rect, check):
        x, y, w, h = rect
        texture = _thumbnail_texture(image, round(w * self.scale), round(h * self.scale),
                                     max(1, round(check * self.scale)))
        if texture is None:
            return
        shader = gpu.shader.from_builtin('IMAGE')
        shader.bind()
        shader.uniform_sampler('image', texture)
        positions = [(px * self.scale, (self.height - py) * self.scale)
                     for px, py in ((x, y+h), (x+w, y+h), (x+w, y), (x, y))]
        batch_for_shader(shader, 'TRI_FAN', {'pos': positions,
                         'texCoord': [(0,0), (1,0), (1,1), (0,1)]}).draw(shader)

    def overlay(self, rgba, rect):
        """Draw an RGBA float array (rows bottom first, straight alpha) stretched over *rect*."""
        self.emit(self.draw_overlay, rgba, rect)

    def draw_overlay(self, rgba, rect):
        x, y, w, h = rect
        texture = _array_texture(rgba)
        shader = gpu.shader.from_builtin('IMAGE')
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

    def number_field(self, key, rect, display, action, *, enabled=True, hint='', zone=None, dragging=False):
        """A number button drawn like Blender's: the value centered, arrows at both ends while hovered and the
        part under the pointer (*zone*: left, middle or right) lit."""
        x, y, w, h = rect
        target = self.target(key, rect, action, enabled, hint)
        hovered = enabled and (dragging or key == self.hover_key)
        self.outline(x, y, w, h, 'number_selected' if dragging else 'number_hover' if hovered else 'number',
                     'hover_border' if hovered else 'number_border')
        handle = min(w / 3, h * .7)
        if hovered and not dragging and zone:
            zx, zw, corners = {'left': (x + 1, handle - 1, 'LEFT'), 'middle': (x + handle, w - 2 * handle, 'NONE'),
                               'right': (x + w - handle, handle - 1, 'RIGHT')}[zone]
            self.rect(zx, y + 1, zw, h - 2, 'number_hot', 3, corners=corners)
        if hovered:
            for name, zx in (('left', x), ('right', x + w - handle)):
                self.icon(name, zx + (handle - 12) / 2, y + (h - 12) / 2, 12, 'number_text')
        text = self.fit(display, max(0, w - 2 * handle - 2))
        self.text(text, x + (w - self.measure(text)) / 2, y + (h - font_size()) / 2 - 1, FONT_SIZE,
                  'number_text' if enabled else 'disabled')
        return target

    def text_field(self, key, rect, value, action, *, enabled=True, hint='', editing=None, scroll=0, multiline=False, follow_caret=True, kind='field', placeholder=''):
        x, y, w, h = rect
        self.outline(x, y, w, h, fill=kind, border='cursor' if editing else 'hover_border'
                     if enabled and key == self.hover_key else kind + '_border')
        self.target(key, rect, action, enabled, hint)
        if placeholder and not value:
            # what belongs here, greyed, until something is typed (the caret still shows while editing)
            padding_y = TEXT_PADDING if multiline else (h - font_size()) / 2 - 1
            rows = int((h - TEXT_PADDING * 2) / TEXT_LINE_HEIGHT) if multiline else 1
            for index, (_, _, line) in enumerate(wrap(placeholder, w - TEXT_PADDING * 2, self.measure)[:max(1, rows)]):
                self.text(line, x + TEXT_PADDING, y + padding_y + index * TEXT_LINE_HEIGHT, FONT_SIZE, 'disabled')
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
