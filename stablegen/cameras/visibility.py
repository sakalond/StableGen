"""Which objects each camera sees, for object prompts.

A grid of rays through the camera's own frame (its per-camera resolution, shift and sensor fit) counts the object
each ray hits first; an object's share is its fraction of the rays. A view's prompt then carries the prompts of the
described objects it sees, largest first. Sampling runs from a timer for the panel, so drawing never waits for it,
and synchronously when a generation builds its view plan.
"""
import math

import bpy  # pylint: disable=import-error
import numpy as np
from mathutils import Vector  # pylint: disable=import-error

from .geometry import _get_camera_resolution

RAYS = 6000             # per view: about 0.07 s on a 2M-triangle scene, an object from 0.2 % of the frame is 12 rays
DETAIL_RAYS = 40000     # the open view's outline of one object
MIN_SHARE = 0.002       # an object counts as seen from 0.2 % of the frame
# The reference categorical palette's dark steps in their fixed order (validated on the card colour, adjacent
# pairs); the described objects take them by how much of the views they fill.
COLORS = ('#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9')

_cache = {}             # (scene, camera, rays) -> (key, grid)
_pending = []           # (scene name, camera name, rays) waiting for the timer
_timer = None


class Grid:
    """One camera's sample: ``labels`` (rows bottom first) index ``names``; -1 where the ray hits nothing."""

    def __init__(self, names, labels):
        self.names, self.labels = names, labels
        counts = np.bincount(labels[labels >= 0].ravel(), minlength=len(names)) if len(names) else np.zeros(0)
        self.shares = sorted(((names[i], float(counts[i] / labels.size)) for i in range(len(names)) if counts[i]),
                             key=lambda item: -item[1])

    def mask(self, name):
        return self.labels == self.names.index(name) if name in self.names else np.zeros(self.labels.shape, bool)


def _camera_key(scene, camera):
    data = camera.data
    return (tuple(v for row in camera.matrix_world for v in row), data.type, data.lens, data.ortho_scale,
            data.shift_x, data.shift_y, data.sensor_fit, data.sensor_width, data.sensor_height,
            _get_camera_resolution(camera, scene))


def _scene_key(scene):
    """What else changes what a camera sees: the meshes, where they are and whether they show."""
    return hash(tuple((obj.name, obj.visible_get(), tuple(v for row in obj.matrix_world for v in row))
                      for obj in scene.objects if obj.type == 'MESH'))


def sample(scene, depsgraph, camera, rays=RAYS):
    """Cast about *rays* rays through the camera's frame; returns a Grid."""
    rx, ry = _get_camera_resolution(camera, scene)
    aspect = rx / max(1, ry)
    columns = max(8, round(math.sqrt(rays * aspect)))
    rows = max(8, round(rays / columns))
    inverse = camera.calc_matrix_camera(depsgraph, x=rx, y=ry, scale_x=scene.render.pixel_aspect_x,
                                        scale_y=scene.render.pixel_aspect_y).inverted_safe()
    world = camera.matrix_world
    ortho = camera.data.type == 'ORTHO'
    forward = (world.to_3x3() @ Vector((0, 0, -1))).normalized()
    origin = world.translation
    names, index = [], {}
    labels = np.full((rows, columns), -1, np.int32)
    for row in range(rows):
        for column in range(columns):
            near = inverse @ Vector(((column + .5) / columns * 2 - 1, (row + .5) / rows * 2 - 1, -1, 1))
            point = world @ (near.xyz / near.w)
            start, direction = (point, forward) if ortho else (origin, (point - origin).normalized())
            hit, _location, _normal, _face, obj, _matrix = scene.ray_cast(depsgraph, start, direction)
            if not hit or obj is None:
                continue
            name = getattr(obj, 'original', obj).name
            if name not in index:
                index[name] = len(names)
                names.append(name)
            labels[row, column] = index[name]
    return Grid(names, labels)


def cached(scene, camera, rays=RAYS, scene_key=None):
    """The camera's Grid if it is current, else None."""
    entry = _cache.get((scene.name, camera.name, rays))
    key = (_camera_key(scene, camera), scene_key if scene_key is not None else _scene_key(scene))
    return entry[1] if entry and entry[0] == key else None


def detect(context, cameras, rays=RAYS):
    """Sample every camera whose result is out of date, now; returns {camera name: Grid}."""
    scene = context.scene
    depsgraph = context.evaluated_depsgraph_get()
    scene_key = _scene_key(scene)
    result = {}
    for camera in cameras:
        grid = cached(scene, camera, rays, scene_key)
        if grid is None:
            grid = sample(scene, depsgraph, camera, rays)
            _cache[(scene.name, camera.name, rays)] = ((_camera_key(scene, camera), scene_key), grid)
        result[camera.name] = grid
    return result


def request(scene, cameras, rays=RAYS):
    """Cached Grids now, the rest sampled by a timer that redraws the 3D views when they are ready."""
    global _timer
    scene_key = _scene_key(scene)
    result = {}
    for camera in cameras:
        grid = cached(scene, camera, rays, scene_key)
        if grid is not None:
            result[camera.name] = grid
        elif (scene.name, camera.name, rays) not in _pending:
            _pending.append((scene.name, camera.name, rays))
    if _pending and _timer is None:
        _timer = _tick
        bpy.app.timers.register(_tick, first_interval=0.05)
    return result


def _tick():
    """Sample pending cameras for up to ~0.15 s per call, so the interface stays responsive."""
    global _timer
    import time
    deadline = time.perf_counter() + 0.15
    context = bpy.context
    try:
        while _pending and time.perf_counter() < deadline:
            scene_name, camera_name, rays = _pending.pop(0)
            scene, camera = bpy.data.scenes.get(scene_name), bpy.data.objects.get(camera_name)
            if scene is None or camera is None or camera.type != 'CAMERA' or context.scene != scene:
                continue
            detect(context, [camera], rays)
    except Exception as error:      # a timer must not raise; the next request retries
        print(f"[StableGen] Object prompts: could not sample the cameras' views: {error}")
        _pending.clear()
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    if _pending:
        return 0.02
    _timer = None
    return None


def colors(grids):
    """{object name: colour} for the described objects in these Grids, the largest overall first."""
    totals = {}
    for grid in grids:
        for name, share in grid.shares:
            totals[name] = totals.get(name, 0) + share
    described = [name for name in sorted(totals, key=lambda n: -totals[n]) if object_prompt(name)]
    return {name: COLORS[i % len(COLORS)] for i, name in enumerate(described)}


def object_prompt(name):
    obj = bpy.data.objects.get(name)
    return obj.sg_object_prompt.strip() if obj is not None and hasattr(obj, 'sg_object_prompt') else ''


def described(grid):
    """The described objects a view sees enough of, largest first: [(name, share, prompt)]."""
    return [(name, share, object_prompt(name)) for name, share in grid.shares
            if share >= MIN_SHARE and object_prompt(name)]


def view_prompt(scene, camera, grid=None):
    """What a view adds before the texture prompt: its view prompt (Use view prompts) and the prompts of the
    described objects it sees (Use object prompts)."""
    parts = []
    if scene.use_camera_prompts:
        item = scene.camera_prompts.get(camera.name)
        if item and item.prompt.strip():
            parts.append(item.prompt.strip())
    if getattr(scene, 'use_object_prompts', False) and grid is not None:
        parts += [prompt for _name, _share, prompt in described(grid)]
    return ', '.join(parts)


def clear():
    global _timer
    _cache.clear()
    _pending.clear()
    if _timer is not None and bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    _timer = None
