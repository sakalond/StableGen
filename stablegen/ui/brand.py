"""StableGen's logo: a lens projecting onto a cube, painting the faces it can see.

The geometry lives here once. The Studio header rasterizes it at the current
UI scale, and tools/export_brand.py writes the SVG and PNG assets from it.
Pure Python + numpy so it also runs outside Blender.
"""
import math

import numpy as np

GOLD, ORANGE, ROSE = '#FFC25C', '#FF8642', '#E8487A'
CLAY = '#8E93AD'
INK = '#14151F'
S3 = math.sqrt(3) / 2


def _iso(x, y, z, cx, cy, r):
    """Isometric unit cube; (1, 1, 1) is the front-top corner at (cx, cy)."""
    c = r * S3
    return (cx + c * (x - y), cy - r + r / 2 * (x + y) - r * (z - 1))


def _face(name, cx, cy, r):
    on = {'top': lambda s, t: (s, t, 1), 'left': lambda s, t: (s, 1, t), 'right': lambda s, t: (1, s, t)}[name]
    return [_iso(*on(s, t), cx, cy, r) for s, t in ((0, 0), (1, 0), (1, 1), (0, 1))]


def _projector(lens, centre, r, lens_r, unlit):
    """Layers of (shape, paint), back to front, in 0..100 units.

    Shapes: ('poly', points) for convex polygons, ('circle', (x, y, r)),
    ('rrect', (x, y, w, h, radius)). Paints: ('solid', hex, alpha) or
    ('linear', (x1, y1, x2, y2), [(offset, hex, alpha), ...]).
    """
    cx, cy = centre
    outline = [_iso(*v, cx, cy, r) for v in ((0, 0, 1), (1, 0, 1), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1))]
    angle = lambda v: math.atan2(v[1] - lens[1], v[0] - lens[0])
    lo, hi = min(outline, key=angle), max(outline, key=angle)
    ax, ay = cx - lens[0], cy - lens[1]
    length = math.hypot(ax, ay)
    ux, uy = ax / length, ay / length
    rim = lens_r * .62
    e1 = (lens[0] - uy * rim, lens[1] + ux * rim)
    e2 = (lens[0] + uy * rim, lens[1] - ux * rim)
    if angle(e1) > angle(e2):
        e1, e2 = e2, e1
    top, left, right = _face('top', cx, cy, r), _face('left', cx, cy, r), _face('right', cx, cy, r)
    return [
        # The beam ends on the cube's silhouette; the cube covers the rest of it.
        (('poly', [e1, lo, hi, e2]), ('linear', (*lens, cx, cy), [(0, GOLD, .04), (1, GOLD, .5)])),
        (('poly', top), ('linear', (*top[0], *top[2]), [(0, '#FFD98A', 1), (1, GOLD, 1)])),
        (('poly', left), ('linear', (*left[3], *left[1]), [(0, ORANGE, 1), (1, '#F0606A', 1)])),
        (('poly', right), ('solid', unlit, 1)),
        (('circle', (*lens, lens_r)), ('linear', (lens[0] - lens_r, lens[1] - lens_r, lens[0] + lens_r,
                                                  lens[1] + lens_r), [(0, GOLD, 1), (1, ORANGE, 1)])),
        (('circle', (*lens, lens_r * .5)), ('solid', '#FFF7E8', 1)),
        (('circle', (*lens, lens_r * .19)), ('solid', ORANGE, 1)),
    ]


LOGO = _projector((21.5, 21.5), (60, 58), 31, 10.5, CLAY)
# The square around the logo's visible extent, for small sizes like the Studio header.
LOGO_BOX = (9.5, 10.5, 88.5, 89.5)

ICON = [(('rrect', (4, 4, 92, 92, 22)), ('linear', (0, 4, 0, 96), [(0, '#23253A', 1), (1, INK, 1)]))] + \
    _projector((29, 29), (59.5, 57.5), 24.5, 8.5, '#4A4E68')
ICON_BOX = (0, 0, 100, 100)


def _rgb(value):
    return np.array([int(value[i:i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)


def _inside(shape, xs, ys):
    kind, data = shape
    if kind == 'circle':
        x, y, r = data
        return (xs - x) ** 2 + (ys - y) ** 2 <= r * r
    if kind == 'rrect':
        x, y, w, h, radius = data
        qx = np.abs(xs - (x + w / 2)) - (w / 2 - radius)
        qy = np.abs(ys - (y + h / 2)) - (h / 2 - radius)
        return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) <= radius
    points = data
    area = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))
    inside = np.ones(xs.shape, bool)
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        cross = (x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)
        inside &= cross * area >= 0
    return inside


def _paint(paint, xs, ys):
    if paint[0] == 'solid':
        return np.broadcast_to(_rgb(paint[1]), xs.shape + (3,)), np.full(xs.shape, paint[2], np.float32)
    x1, y1, x2, y2 = paint[1]
    dx, dy = x2 - x1, y2 - y1
    t = np.clip(((xs - x1) * dx + (ys - y1) * dy) / (dx * dx + dy * dy), 0, 1)
    stops = paint[2]
    offsets = [s[0] for s in stops]
    rgb = np.stack([np.interp(t, offsets, [_rgb(s[1])[k] for s in stops]) for k in range(3)], -1)
    return rgb, np.interp(t, offsets, [s[2] for s in stops]).astype(np.float32)


def rasterize(layers, size, box=(0, 0, 100, 100), samples=4, strip=128):
    """Straight-alpha RGBA float array (size x size x 4), top row first.

    Supersampled with exact inside tests, so faces that share an edge meet
    without the seams per-shape anti-aliasing leaves.
    """
    x0, y0, x1, y1 = box
    unit = size / (x1 - x0)
    out = np.zeros((size, size, 4), np.float32)
    sx = x0 + (np.arange(size * samples) + .5) / samples / unit
    for row in range(0, size, strip):
        rows = min(strip, size - row)
        sy = y0 + (row + (np.arange(rows * samples) + .5) / samples) / unit
        xs, ys = np.meshgrid(sx, sy)
        colour = np.zeros(xs.shape + (3,), np.float32)
        alpha = np.zeros(xs.shape, np.float32)
        for shape, paint in layers:
            rgb, a = _paint(paint, xs, ys)
            a = a * _inside(shape, xs, ys)
            colour = colour * (1 - a)[..., None] + rgb * a[..., None]
            alpha = alpha * (1 - a) + a
        colour = colour.reshape(rows, samples, size, samples, 3).mean((1, 3))
        alpha = alpha.reshape(rows, samples, size, samples).mean((1, 3))
        out[row:row + rows, :, :3] = colour / np.maximum(alpha, 1e-6)[..., None]
        out[row:row + rows, :, 3] = alpha
    return out


def to_svg(layers, box=(0, 0, 100, 100), mono=None):
    """SVG markup for the layers. *mono* maps layer index to opacity for a one-colour version."""
    defs, body = [], []
    for i, (shape, paint) in enumerate(layers):
        if mono is not None:
            if i not in mono:
                continue
            fill = f'fill="currentColor" opacity="{mono[i]}"'
        elif paint[0] == 'solid':
            fill = f'fill="{paint[1]}"' + (f' fill-opacity="{paint[2]}"' if paint[2] != 1 else '')
        else:
            x1, y1, x2, y2 = paint[1]
            stops = ''.join(f'<stop offset="{o}" stop-color="{c}"' + (f' stop-opacity="{a}"' if a != 1 else '') + '/>'
                            for o, c, a in paint[2])
            defs.append(f'<linearGradient id="g{i}" gradientUnits="userSpaceOnUse" x1="{x1:.2f}" y1="{y1:.2f}" '
                        f'x2="{x2:.2f}" y2="{y2:.2f}">{stops}</linearGradient>')
            fill = f'fill="url(#g{i})"'
        kind, data = shape
        if kind == 'circle':
            body.append(f'<circle cx="{data[0]:.2f}" cy="{data[1]:.2f}" r="{data[2]:.2f}" {fill}/>')
        elif kind == 'rrect':
            x, y, w, h, radius = data
            body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" {fill}/>')
        else:
            points = ' '.join(f'{x:.2f},{y:.2f}' for x, y in data)
            body.append(f'<polygon points="{points}" {fill}/>')
    x0, y0, x1, y1 = box
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0} {y0} {x1 - x0} {y1 - y0}">'
            f'<defs>{"".join(defs)}</defs>{"".join(body)}</svg>\n')
